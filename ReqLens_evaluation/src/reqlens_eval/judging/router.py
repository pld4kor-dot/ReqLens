"""Judging router — applies the correct prompt family for each evaluation task.

Routing logic:
  Track 1:
    - Uncertain decisions after normalization → family A (support_check)
    - Known hallucinations (seeded fakes) for HRR secondary check → family B
  Track 2:
    - Contradiction seeds → family C
    - Duplicate seeds    → family D
"""

from __future__ import annotations

from typing import Any

import structlog

from reqlens_eval.judging.client import JudgeClient
from reqlens_eval.judging.prompts import (
    FAMILY_A_SYSTEM,
    FAMILY_A_USER_TEMPLATE,
    FAMILY_B_SYSTEM,
    FAMILY_B_USER_TEMPLATE,
    FAMILY_C_SYSTEM,
    FAMILY_C_USER_TEMPLATE,
    FAMILY_D_SYSTEM,
    FAMILY_D_USER_TEMPLATE,
)
from reqlens_eval.models.artifacts import DefectSeedItem, HallucinationSeedItem
from reqlens_eval.models.experiment import JudgeVerdict

logger = structlog.get_logger(__name__)


def _confidence(raw: dict[str, Any], default: float = 0.5) -> float:
    """Safely coerce a possibly-null LLM confidence value to float."""
    v = raw.get("confidence", default)
    try:
        return float(v) if v is not None else default
    except (TypeError, ValueError):
        return default


def _source_block(source_texts: list[dict[str, Any]]) -> str:
    parts = []
    for st in source_texts:
        doc_type = st.get("type", "document").replace("_", " ").upper()
        title = st.get("title", "")
        parts.append(f"=== {doc_type}: {title} ===\n{st.get('text', '')}")
    return "\n\n".join(parts)


class JudgeRouter:
    """Routes each judge task to the appropriate prompt family and calls the LLM."""

    def __init__(self, client: JudgeClient | None = None) -> None:
        self._client = client or JudgeClient()

    # ── Family A: support_check (Track 1 normalisation) ──────────────────────

    def judge_support(
        self,
        candidate_id: str,
        candidate_text: str,
        source_texts: list[dict[str, Any]],
    ) -> JudgeVerdict:
        """Determine if a candidate is supported by source documents (family A)."""
        user_prompt = FAMILY_A_USER_TEMPLATE.format(
            source_block=_source_block(source_texts),
            candidate_id=candidate_id,
            candidate_text=candidate_text,
        )
        raw = self._client.call(FAMILY_A_SYSTEM, user_prompt)
        status = raw.get("status", "rejected")
        if status not in ("accepted", "rejected"):
            status = "rejected"
        verdict_val = "accepted" if status == "accepted" else "rejected"
        return JudgeVerdict(
            target_id=candidate_id,
            verdict=verdict_val,
            confidence=_confidence(raw),
            reasoning=raw.get("explanation", raw.get("reasoning", "")),
            prompt_family="A",
        )

    # ── Family B: hallucination_fate (Track 1 HRR secondary) ─────────────────

    def judge_hallucination_fate(
        self,
        seed: HallucinationSeedItem,
        extraction_text: str,
    ) -> JudgeVerdict:
        """Check whether a hallucinated requirement was correctly rejected (family B)."""
        user_prompt = FAMILY_B_USER_TEMPLATE.format(
            extraction_text=extraction_text,
            seed_item_id=seed.seed_item_id,
            hallucinated_text=seed.requirement_text,
            unsupported_reason=seed.unsupported_reason,
        )
        raw = self._client.call(FAMILY_B_SYSTEM, user_prompt)
        verdict = raw.get("verdict", "accepted")
        if verdict not in ("rejected", "accepted"):
            verdict = "accepted"
        return JudgeVerdict(
            target_id=seed.seed_item_id,
            verdict=verdict,
            confidence=_confidence(raw),
            reasoning=raw.get("reasoning", ""),
            prompt_family="B",
        )

    # ── Family C: contradiction_check (Track 2) ───────────────────────────────

    def judge_contradiction(
        self,
        seed: DefectSeedItem,
        extraction_text: str,
    ) -> JudgeVerdict:
        """Check whether a seeded contradiction leaked to the extraction (family C)."""
        extraction_count = extraction_text.count("\n") + 1 if extraction_text.strip() else 0
        user_prompt = FAMILY_C_USER_TEMPLATE.format(
            extraction_text=extraction_text,
            extraction_count=extraction_count,
            seed_item_id=seed.seed_item_id,
            injected_text=seed.injected_text,
            original_req_texts="; ".join(seed.original_req_texts),
            defect_description=seed.defect_description,
        )
        raw = self._client.call(FAMILY_C_SYSTEM, user_prompt)
        verdict = raw.get("verdict", "leaked")
        if verdict not in ("leaked", "detected"):
            verdict = "leaked"
        return JudgeVerdict(
            target_id=seed.seed_item_id,
            verdict=verdict,
            confidence=_confidence(raw),
            reasoning=raw.get("reasoning", ""),
            prompt_family="C",
        )

    # ── Family D: duplicate_check (Track 2) ──────────────────────────────────

    def judge_duplicate(
        self,
        seed: DefectSeedItem,
        extraction_text: str,
    ) -> JudgeVerdict:
        """Check whether a seeded duplicate leaked to the extraction (family D)."""
        extraction_count = extraction_text.count("\n") + 1 if extraction_text.strip() else 0
        user_prompt = FAMILY_D_USER_TEMPLATE.format(
            extraction_text=extraction_text,
            extraction_count=extraction_count,
            seed_item_id=seed.seed_item_id,
            injected_text=seed.injected_text,
            original_req_texts="; ".join(seed.original_req_texts),
            defect_description=seed.defect_description,
        )
        raw = self._client.call(FAMILY_D_SYSTEM, user_prompt)
        verdict = raw.get("verdict", "leaked")
        if verdict not in ("leaked", "detected"):
            verdict = "leaked"
        return JudgeVerdict(
            target_id=seed.seed_item_id,
            verdict=verdict,
            confidence=_confidence(raw),
            reasoning=raw.get("reasoning", ""),
            prompt_family="D",
        )

    # ── Convenience: judge all seeds in a Track 2 artifact ───────────────────

    def judge_all_seeds(
        self,
        seeds: list[DefectSeedItem],
        extraction_text: str,
    ) -> list[JudgeVerdict]:
        """Run the appropriate judge for each seed and return all verdicts."""
        verdicts: list[JudgeVerdict] = []
        for seed in seeds:
            try:
                if seed.defect_type == "contradiction":
                    verdicts.append(self.judge_contradiction(seed, extraction_text))
                elif seed.defect_type == "duplicate":
                    verdicts.append(self.judge_duplicate(seed, extraction_text))
                else:
                    logger.warning(
                        "judge_router.unknown_defect_type",
                        seed_id=seed.seed_item_id,
                        defect_type=seed.defect_type,
                    )
            except Exception as exc:
                logger.error(
                    "judge_router.seed_judgment_failed",
                    seed_id=seed.seed_item_id,
                    error=str(exc),
                )
                # Conservative: count as leaked on failure
                verdicts.append(
                    JudgeVerdict(
                        target_id=seed.seed_item_id,
                        verdict="leaked",
                        confidence=0.0,
                        reasoning=f"Judge error — defaulted to leaked: {exc}",
                        prompt_family="C" if seed.defect_type == "contradiction" else "D",
                    )
                )
        return verdicts
