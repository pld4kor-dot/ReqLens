"""Scenario brief builder.

Generates a structured project brief from gold requirements (and optional raw
global context for PURE documents).  The brief is the intermediary between the
gold-requirement set and the source-bundle generator — it captures domain
vocabulary, stakeholders, and goals so that the generated artifacts are
internally consistent.
"""

from __future__ import annotations

from typing import Any

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    GENERATION_SYSTEM_PROMPT,
    build_scenario_brief_prompt,
)

logger = structlog.get_logger(__name__)


def build_scenario_brief(
    llm: AzureOpenAIClient,
    unit_id: str,
    gold_requirements: list[dict[str, Any]],
    *,
    global_context: str | None = None,
) -> dict[str, Any]:
    """Generate a project scenario brief from gold requirements.

    Args:
        llm:               Shared client.
        unit_id:           Identifier used in logging.
        gold_requirements: Serialized GoldRequirement dicts.
        global_context:    Optional raw text from PURE doc intro/scope sections;
                           helps ground the brief in the actual domain.

    Returns:
        A dict with keys: project_name, domain, users, stakeholders,
        business_goals, core_features, quality_concerns, constraints, terminology.
    """
    settings = get_settings()
    prompt = build_scenario_brief_prompt(
        unit_id=unit_id,
        gold_requirements=gold_requirements,
        global_context=global_context,
    )

    try:
        brief = llm.chat_json(
            system_prompt=GENERATION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=settings.temp_generation,
        )
    except Exception as exc:
        logger.error("scenario_brief.llm_error", unit_id=unit_id, error=str(exc)[:200])
        # Return a minimal fallback brief so the pipeline can continue
        brief = {
            "project_name": f"Project {unit_id}",
            "domain": "software system",
            "users": ["end user"],
            "stakeholders": ["product owner", "development team"],
            "business_goals": ["satisfy stated requirements"],
            "core_features": [r.get("text", "")[:80] for r in gold_requirements[:5]],
            "quality_concerns": [],
            "constraints": [],
            "terminology": [],
        }

    logger.info(
        "scenario_brief.done",
        unit_id=unit_id,
        project=brief.get("project_name"),
        domain=brief.get("domain"),
    )
    return brief
