"""Centralized Azure OpenAI gateway.

All agents call this wrapper – never the SDK directly – so we get:
  • Centralized retry & rate-limit handling
  • Centralized logging & token accounting
  • Centralized structured-output parsing
  • Centralized prompt versioning
  • Optional response caching
"""

from __future__ import annotations

import hashlib
import time
from typing import TYPE_CHECKING, Type, TypeVar

import structlog
from openai import OpenAI
from pydantic import BaseModel

from reqlens.config.settings import get_settings
from reqlens.domain.ids import generate_id
from reqlens.domain.models import LLMCallLog
from reqlens.llm.retry import with_retry

if TYPE_CHECKING:
    pass

logger = structlog.get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


def _prompt_hash(text: str) -> str:
    """SHA-256 truncated to 16 hex chars."""
    return hashlib.sha256(text.encode()).hexdigest()[:16]


class AzureOpenAIClient:
    """Single LLM gateway shared by every agent."""

    def __init__(self) -> None:
        settings = get_settings()
        self.client = OpenAI(
            api_key=settings.azure_openai_api_key,
            base_url=settings.azure_openai_base_url,
        )
        self.chat_model = settings.azure_openai_chat_deployment
        self.reasoning_model = settings.azure_openai_reasoning_deployment or self.chat_model
        self.embedding_model = settings.azure_openai_embedding_deployment
        self._call_logs: list[LLMCallLog] = []

    # ── Structured chat (Pydantic response_format) ──────────────────
    @with_retry
    def structured_chat(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        response_model: Type[T],
        model: str | None = None,
        project_id: str | None = None,
        agent_name: str = "",
    ) -> T:
        """Call Azure OpenAI with Pydantic structured output.

        Uses ``client.beta.chat.completions.parse(...)`` so the
        response is guaranteed to match *response_model*.
        """
        chosen_model = model or self.chat_model
        t0 = time.perf_counter()

        completion = self.client.beta.chat.completions.parse(
            model=chosen_model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            response_format=response_model,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        usage = completion.usage

        parsed = completion.choices[0].message.parsed
        if parsed is None:
            raise ValueError("Azure OpenAI returned no parsed structured output.")

        # Log the call
        log = LLMCallLog(
            project_id=project_id,
            agent_name=agent_name,
            model=chosen_model,
            prompt_hash=_prompt_hash(system_prompt + user_prompt),
            input_tokens=usage.prompt_tokens if usage else 0,
            output_tokens=usage.completion_tokens if usage else 0,
            latency_ms=latency_ms,
            status="ok",
        )
        self._call_logs.append(log)
        logger.info(
            "llm.structured_chat",
            model=chosen_model,
            agent=agent_name,
            input_tokens=log.input_tokens,
            output_tokens=log.output_tokens,
            latency_ms=latency_ms,
        )

        return parsed

    # ── Free-text via Responses API ─────────────────────────────────
    @with_retry
    def response_text(
        self,
        *,
        instructions: str,
        input_text: str,
        model: str | None = None,
        project_id: str | None = None,
        agent_name: str = "",
    ) -> str:
        """Call Azure OpenAI Responses API for free-form text."""
        chosen_model = model or self.chat_model
        t0 = time.perf_counter()

        response = self.client.responses.create(
            model=chosen_model,
            instructions=instructions,
            input=input_text,
        )

        latency_ms = int((time.perf_counter() - t0) * 1000)
        text = response.output_text

        log = LLMCallLog(
            project_id=project_id,
            agent_name=agent_name,
            model=chosen_model,
            prompt_hash=_prompt_hash(instructions + input_text),
            latency_ms=latency_ms,
            status="ok",
        )
        self._call_logs.append(log)
        logger.info("llm.response_text", model=chosen_model, agent=agent_name, latency_ms=latency_ms)

        return text

    # ── Embeddings ──────────────────────────────────────────────────
    @with_retry
    def embed_texts(
        self,
        texts: list[str],
        *,
        project_id: str | None = None,
        agent_name: str = "",
    ) -> list[list[float]]:
        """Generate embeddings for a batch of texts."""
        settings = get_settings()
        all_embeddings: list[list[float]] = []
        batch_size = settings.embedding_batch_size

        for i in range(0, len(texts), batch_size):
            batch = texts[i : i + batch_size]
            t0 = time.perf_counter()

            response = self.client.embeddings.create(
                model=self.embedding_model,
                input=batch,
            )

            latency_ms = int((time.perf_counter() - t0) * 1000)
            batch_embeddings = [item.embedding for item in response.data]
            all_embeddings.extend(batch_embeddings)

            log = LLMCallLog(
                project_id=project_id,
                agent_name=agent_name,
                model=self.embedding_model,
                prompt_hash=_prompt_hash(str(len(batch))),
                input_tokens=response.usage.prompt_tokens if response.usage else 0,
                latency_ms=latency_ms,
                status="ok",
            )
            self._call_logs.append(log)

        logger.info(
            "llm.embed_texts",
            count=len(texts),
            batches=(len(texts) + batch_size - 1) // batch_size,
        )
        return all_embeddings

    # ── Utilities ───────────────────────────────────────────────────
    def drain_call_logs(self) -> list[LLMCallLog]:
        """Return and clear accumulated call logs (for persistence)."""
        logs = list(self._call_logs)
        self._call_logs.clear()
        return logs
