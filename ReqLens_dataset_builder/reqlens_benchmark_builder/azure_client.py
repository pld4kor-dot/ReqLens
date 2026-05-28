"""Thin Azure OpenAI gateway for the benchmark builder.

Design mirrors ReqInOne's AzureOpenAIClient:
- Single shared client, same env-key convention
- Centralized retry / rate-limit handling
- Token accounting across all calls
- JSON mode with fallback text extraction
"""

from __future__ import annotations

import json
import re
import time
from typing import Any

import structlog
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import (
    retry,
    retry_if_exception_type,
    stop_after_attempt,
    wait_exponential,
)

from reqlens_benchmark_builder.config import get_settings

logger = structlog.get_logger(__name__)

# Retryable error types (transient network / quota issues)
_RETRYABLE = (RateLimitError, APIConnectionError, APIError)

# GPT-5 family (gpt-5, gpt-5-mini, gpt-5-nano, gpt-5.4, gpt-5.4-nano) has two
# parameter differences vs older models:
#   1. Uses max_completion_tokens instead of max_tokens.
#   2. Does not accept a custom temperature — only the default (1) is supported.
_GPT5_PREFIXES = ("gpt-5",)


def _is_gpt5(model: str) -> bool:
    return any(model.startswith(p) for p in _GPT5_PREFIXES)


def _token_limit_kwarg(model: str, value: int) -> dict[str, int]:
    """Return the correct token-limit keyword argument for the given model.

    GPT-5 family requires ``max_completion_tokens``; older models use ``max_tokens``.
    """
    if _is_gpt5(model):
        return {"max_completion_tokens": value}
    return {"max_tokens": value}


def _temperature_kwarg(model: str, value: float) -> dict[str, float]:
    """Return the temperature keyword argument, or empty dict for GPT-5 family.

    GPT-5 family deployments reject any temperature other than the default (1).
    Omitting the parameter entirely lets the API use its default safely.
    """
    if _is_gpt5(model):
        return {}  # use model default; do not send temperature at all
    return {"temperature": value}


def _with_retry():
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )


def _extract_json(text: str) -> dict[str, Any]:
    """Try several strategies to parse JSON from a raw model response.

    Some deployments wrap the JSON in markdown code fences or produce
    a short preamble sentence before the actual object.
    """
    # 1. Plain parse
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    # 2. Fenced code block  ```json ... ```
    fence = re.search(r"```(?:json)?\s*(\{[\s\S]+?\})\s*```", text)
    if fence:
        try:
            return json.loads(fence.group(1))
        except json.JSONDecodeError:
            pass

    # 3. First balanced {...} block
    brace = re.search(r"\{[\s\S]+\}", text)
    if brace:
        try:
            return json.loads(brace.group(0))
        except json.JSONDecodeError:
            pass

    raise ValueError(
        f"Could not parse JSON from model response "
        f"(first 300 chars): {text[:300]!r}"
    )


class AzureOpenAIClient:
    """Single LLM gateway used by every module in the benchmark builder."""

    def __init__(self) -> None:
        settings = get_settings()
        if not settings.azure_openai_api_key:
            raise ValueError(
                "AZURE_OPENAI_API_KEY is not set. "
                "Copy .env.example to .env and fill in your credentials."
            )
        if not settings.azure_openai_base_url:
            raise ValueError("AZURE_OPENAI_BASE_URL is not set.")

        self.client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_base_url,
        )
        self.chat_model: str = settings.azure_openai_chat_deployment
        self.reasoning_model: str = (
            settings.azure_openai_reasoning_deployment or self.chat_model
        )
        self.extraction_model: str = (
            settings.azure_openai_extraction_deployment or self.chat_model
        )

        # Cumulative accounting
        self._call_count: int = 0
        self._total_input_tokens: int = 0
        self._total_output_tokens: int = 0

    # ── Free-text completion ──────────────────────────────────────────────────

    @_with_retry()
    def chat_text(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        model: str | None = None,
        max_tokens: int = 8192,
    ) -> str:
        """Plain text chat completion."""
        chosen = model or self.chat_model
        t0 = time.perf_counter()
        response = self.client.chat.completions.create(
            model=chosen,
            **_temperature_kwarg(chosen, temperature),
            **_token_limit_kwarg(chosen, max_tokens),
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        )
        content = response.choices[0].message.content or ""
        self._record(response.usage, chosen, time.perf_counter() - t0)
        return content.strip()

    # ── JSON completion ──────────────────────────────────────────────────────

    @_with_retry()
    def chat_json(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.2,
        model: str | None = None,
        max_tokens: int = 8192,
    ) -> dict[str, Any]:
        """Chat completion that guarantees a parsed dict response.

        Tries the ``json_object`` response_format first; falls back to plain
        text + heuristic JSON extraction if the deployment does not support it.
        """
        chosen = model or self.chat_model
        t0 = time.perf_counter()

        tok_kwarg  = _token_limit_kwarg(chosen, max_tokens)
        temp_kwarg = _temperature_kwarg(chosen, temperature)

        try:
            response = self.client.chat.completions.create(
                model=chosen,
                **temp_kwarg,
                **tok_kwarg,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
            )
        except Exception as exc:
            logger.warning(
                "azure_client.json_mode_fallback",
                model=chosen,
                error=str(exc)[:120],
                note="retrying as plain text (response_format or token param incompatibility)",
            )
            # Append a hard instruction so the model still outputs JSON
            augmented_sys = (
                system_prompt
                + "\n\nCRITICAL: Output ONLY raw valid JSON — no prose, no markdown fences."
            )
            response = self.client.chat.completions.create(
                model=chosen,
                **temp_kwarg,
                **tok_kwarg,
                messages=[
                    {"role": "system", "content": augmented_sys},
                    {"role": "user", "content": user_prompt},
                ],
            )

        content = response.choices[0].message.content or "{}"
        self._record(response.usage, chosen, time.perf_counter() - t0)
        return _extract_json(content)

    # ── Token accounting ─────────────────────────────────────────────────────

    def _record(self, usage: Any, model: str, elapsed_s: float) -> None:
        self._call_count += 1
        if usage:
            self._total_input_tokens += usage.prompt_tokens or 0
            self._total_output_tokens += usage.completion_tokens or 0
        logger.info(
            "llm.call",
            call=self._call_count,
            model=model,
            in_tok=getattr(usage, "prompt_tokens", "?"),
            out_tok=getattr(usage, "completion_tokens", "?"),
            latency_ms=int(elapsed_s * 1000),
        )

    def usage_summary(self) -> dict[str, int]:
        return {
            "total_calls": self._call_count,
            "total_input_tokens": self._total_input_tokens,
            "total_output_tokens": self._total_output_tokens,
        }
