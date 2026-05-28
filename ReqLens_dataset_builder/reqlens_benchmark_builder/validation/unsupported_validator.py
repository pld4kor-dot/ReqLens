"""Unsupported-requirement detector.

Checks whether the generated source bundle implies significant system-capability
requirements that are NOT present in the gold set (leakage / over-generation).
"""

from __future__ import annotations

from typing import Any

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    VALIDATION_SYSTEM_PROMPT,
    build_unsupported_prompt,
)

logger = structlog.get_logger(__name__)


def validate_unsupported(
    llm: AzureOpenAIClient,
    unit_id: str,
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Identify implied requirements in the source bundle not in the gold set.

    Returns a dict with keys:
        unsupported_implied_requirements : list of dicts (text, source_snippet, reason)
        count                            : int
    """
    settings = get_settings()

    if not source_texts:
        return {"unsupported_implied_requirements": [], "count": 0}

    prompt = build_unsupported_prompt(unit_id, source_texts, gold_requirements)

    try:
        raw = llm.chat_json(
            system_prompt=VALIDATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=settings.temp_validation,
            model=settings.azure_openai_reasoning_deployment,
            max_tokens=settings.max_tokens_validation,
        )
    except Exception as exc:
        logger.error(
            "unsupported_validator.llm_error",
            unit_id=unit_id,
            error=str(exc)[:200],
        )
        # Optimistic fallback on failure (avoid unnecessary repair loops)
        return {"unsupported_implied_requirements": [], "count": 0}

    implied = raw.get("unsupported_implied_requirements", [])
    if not isinstance(implied, list):
        implied = []

    # Sanitise each entry
    clean: list[dict[str, Any]] = []
    for item in implied:
        text = (item.get("text") or "").strip()
        if not text:
            continue
        clean.append(
            {
                "text":          text,
                "source_snippet": (item.get("source_snippet") or "").strip()[:300],
                "source_type":   (item.get("source_type") or "").strip(),
                "reason":        (item.get("reason") or "").strip(),
            }
        )

    logger.info(
        "unsupported_validator.done",
        unit_id=unit_id,
        unsupported_count=len(clean),
    )
    return {"unsupported_implied_requirements": clean, "count": len(clean)}
