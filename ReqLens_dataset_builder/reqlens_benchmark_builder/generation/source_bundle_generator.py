"""Source bundle generator.

Given a scenario brief and the gold requirements, calls the LLM to produce
three realistic raw source artifacts: interview transcript, meeting notes,
and an email thread.
"""

from __future__ import annotations

from typing import Any

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_source_bundle_prompt,
)
from reqlens_benchmark_builder.schemas.benchmark_models import SourceArtifact

logger = structlog.get_logger(__name__)

_EXPECTED_TYPES = ("interview_transcript", "meeting_notes", "email_thread")


def _validate_artifacts(artifacts: list[SourceArtifact]) -> list[str]:
    """Return a list of warnings about artifact quality."""
    warnings: list[str] = []
    found_types = {a.type for a in artifacts}
    for expected in _EXPECTED_TYPES:
        if expected not in found_types:
            warnings.append(f"Missing artifact type: '{expected}'")
    for art in artifacts:
        if len(art.text.split()) < 150:
            warnings.append(
                f"Artifact '{art.type}' is very short ({len(art.text.split())} words)"
            )
    return warnings


def generate_source_bundle(
    llm: AzureOpenAIClient,
    unit_id: str,
    brief: dict[str, Any],
    gold_requirements: list[dict[str, Any]],
) -> list[SourceArtifact]:
    """Generate the 3-artifact source bundle for a benchmark unit.

    Args:
        llm:               Shared client.
        unit_id:           Identifier used in logging.
        brief:             Scenario brief from ``build_scenario_brief()``.
        gold_requirements: Serialized GoldRequirement dicts.

    Returns:
        A list of ``SourceArtifact`` objects (always 3 when successful).
        Falls back to placeholder artifacts on LLM error.
    """
    settings = get_settings()
    prompt = build_source_bundle_prompt(
        unit_id=unit_id,
        brief=brief,
        gold_requirements=gold_requirements,
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
            "source_bundle.llm_error",
            unit_id=unit_id,
            error=str(exc)[:200],
        )
        return _fallback_artifacts(unit_id, gold_requirements)

    raw_artifacts = result.get("source_texts", [])
    if not raw_artifacts:
        logger.warning(
            "source_bundle.empty_response",
            unit_id=unit_id,
        )
        return _fallback_artifacts(unit_id, gold_requirements)

    artifacts: list[SourceArtifact] = []
    for item in raw_artifacts:
        art_type  = (item.get("type") or "").strip()
        art_title = (item.get("title") or f"{art_type} artifact").strip()
        art_text  = (item.get("text") or "").strip()
        if not art_type or not art_text:
            continue
        artifacts.append(SourceArtifact(type=art_type, title=art_title, text=art_text))

    warnings = _validate_artifacts(artifacts)
    for w in warnings:
        logger.warning("source_bundle.quality_warning", unit_id=unit_id, warning=w)

    logger.info(
        "source_bundle.done",
        unit_id=unit_id,
        artifact_count=len(artifacts),
        total_words=sum(len(a.text.split()) for a in artifacts),
    )
    return artifacts


def _fallback_artifacts(
    unit_id: str,
    gold_requirements: list[dict[str, Any]],
) -> list[SourceArtifact]:
    """Minimal placeholder bundle when the LLM call fails outright."""
    req_list = "\n".join(f"- {r.get('text', '')}" for r in gold_requirements[:10])
    base_text = (
        f"[PLACEHOLDER — LLM generation failed for unit {unit_id}]\n\n"
        f"Gold requirements:\n{req_list}"
    )
    return [
        SourceArtifact(type=t, title=f"[PLACEHOLDER] {t}", text=base_text)
        for t in _EXPECTED_TYPES
    ]
