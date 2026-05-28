"""LLM-as-judge client for the evaluation pipeline.

Wraps Azure OpenAI calls with retry logic and structured JSON output parsing.
Used by the judging router to call any of the five prompt families.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from openai import APIConnectionError, APIError, OpenAI, RateLimitError
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from reqlens_eval.config import get_settings

logger = structlog.get_logger(__name__)

_RETRYABLE = (RateLimitError, APIConnectionError, APIError)


def _with_retry():
    return retry(
        retry=retry_if_exception_type(_RETRYABLE),
        wait=wait_exponential(multiplier=1, min=2, max=60),
        stop=stop_after_attempt(4),
        reraise=True,
    )


class JudgeClient:
    """Thin Azure OpenAI client for LLM judge calls."""

    def __init__(self) -> None:
        settings = get_settings()
        self._client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_base_url,
        )
        self._model = settings.azure_openai_judge_deployment
        # self._temperature = settings.judge_temperature
        self._max_tokens = settings.max_judge_tokens

    @_with_retry()
    def call(
        self,
        system_prompt: str,
        user_prompt: str,
        *,
        max_tokens: int | None = None,
    ) -> dict[str, Any]:
        """Call the judge LLM and return a parsed JSON dict.

        Args:
            system_prompt: The judge's instruction prompt (one of families A-E).
            user_prompt:   The formatted user message for this specific judgment.
            max_tokens:    Override the default token limit.

        Returns:
            Parsed JSON dict from the model response.
            On parse failure, returns {"error": "<message>"}.
        """
        t0 = time.perf_counter()
        resp = self._client.chat.completions.create(
            model=self._model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format={"type": "json_object"},
            max_completion_tokens=max_tokens or self._max_tokens
            # temperature=self._temperature,
        )
        elapsed = time.perf_counter() - t0
        raw = resp.choices[0].message.content or "{}"

        try:
            result = json.loads(raw)
        except json.JSONDecodeError as exc:
            logger.warning(
                "judge_client.parse_failed",
                error=str(exc),
                raw=raw[:200],
            )
            result = {"error": f"JSON parse failed: {exc}", "raw": raw}

        logger.debug(
            "judge_client.call_done",
            model=self._model,
            elapsed=round(elapsed, 3),
        )
        return result