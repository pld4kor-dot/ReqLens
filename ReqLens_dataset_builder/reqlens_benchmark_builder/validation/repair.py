"""Source-bundle repair module.

When the coverage or leakage validator flags issues, this module sends the
current source bundle plus a diagnostic report back to the LLM for targeted
repair.  The repair preserves artifact types and realistic tone while:
- Adding coverage for missing gold requirements.
- Removing or softening passages that imply ungrounded capabilities.
"""

from __future__ import annotations

from typing import Any

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_repair_prompt,
)
from reqlens_benchmark_builder.schemas.benchmark_models import SourceArtifact

logger = structlog.get_logger(__name__)


def repair_source_bundle(
    llm: AzureOpenAIClient,
    unit_id: str,
    brief: dict[str, Any],
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    unsupported_report: dict[str, Any],
) -> list[SourceArtifact]:
    """Repair the source bundle to fix coverage and leakage issues.

    Returns a new list of ``SourceArtifact`` objects.  Falls back to the
    original (un-repaired) artifacts if the LLM call fails, to avoid losing
    a partially good bundle.
    """
    settings = get_settings()

    prompt = build_repair_prompt(
        unit_id=unit_id,
        brief=brief,
        source_texts=source_texts,
        gold_requirements=gold_requirements,
        coverage_report=coverage_report,
        unsupported_report=unsupported_report,
    )

    try:
        result = llm.chat_json(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=settings.temp_generation,
            model=settings.azure_openai_chat_deployment,
            max_tokens=settings.max_tokens_generation,
        )
    except Exception as exc:
        logger.error(
            "repair.llm_error",
            unit_id=unit_id,
            error=str(exc)[:200],
        )
        # Return original artifacts rather than crashing the pipeline
        return [SourceArtifact(**s) for s in source_texts]

    raw_artifacts = result.get("source_texts", [])
    if not raw_artifacts:
        logger.warning("repair.empty_response", unit_id=unit_id)
        return [SourceArtifact(**s) for s in source_texts]

    repaired: list[SourceArtifact] = []
    for item in raw_artifacts:
        art_type  = (item.get("type") or "").strip()
        art_title = (item.get("title") or f"{art_type} artifact").strip()
        art_text  = (item.get("text") or "").strip()
        if not art_type or not art_text:
            continue
        repaired.append(SourceArtifact(type=art_type, title=art_title, text=art_text))

    if not repaired:
        logger.warning("repair.no_valid_artifacts", unit_id=unit_id)
        return [SourceArtifact(**s) for s in source_texts]

    logger.info(
        "repair.done",
        unit_id=unit_id,
        artifact_count=len(repaired),
    )
    return repaired
