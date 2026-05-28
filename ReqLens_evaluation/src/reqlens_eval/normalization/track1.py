"""Track 1 normalizer — resolves 'uncertain' decisions via LLM judge.

After a system adapter returns its ``Track1SystemOutput``, there may be
candidates with status = 'uncertain' (e.g. LLM call failed, or system gave
no clear signal).  The normalizer calls the LLM judge (prompt family A /
support_check) for those candidates and upgrades their status.

The output is a fully resolved list of ``CandidateDecision`` objects ready
for metrics computation (all statuses are either 'accepted' or 'rejected').
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import structlog

from reqlens_eval.models.experiment import CandidateDecision, Track1SystemOutput

if TYPE_CHECKING:
    from reqlens_eval.judging.router import JudgeRouter

logger = structlog.get_logger(__name__)


def normalize_track1(
    system_output: Track1SystemOutput,
    source_texts: list[dict[str, Any]],
    candidate_text_map: dict[str, str],
    judge: "JudgeRouter",
    candidate_evidence_map: dict[str, list[dict[str, Any]]] | None = None,
# OLD: ) -> list[CandidateDecision]:
# NEW: also returns the list of candidate_ids that were escalated to the LLM judge
) -> tuple[list[CandidateDecision], list[str]]:
    """Resolve 'uncertain' decisions using the shared JudgeRouter (prompt family A).

    Args:
        system_output:          Raw adapter output with possible 'uncertain' decisions.
        source_texts:           Unmodified source evidence documents (fallback if no
                                per-candidate evidence is available).
        candidate_text_map:     Mapping from candidate_id → requirement text for judge context.
        judge:                  Shared JudgeRouter (already has tenancy retry via JudgeClient).
        candidate_evidence_map: Optional per-candidate evidence override.  When present,
                                each uncertain candidate's judge call receives only that
                                candidate's evidence (e.g. the system's retrieved spans or
                                its extracted requirements) instead of the full raw source.
                                Falls back to ``source_texts`` when the candidate has no
                                entry or when the entry is empty.

    Returns:
        Tuple of:
          - List of fully resolved CandidateDecision (all 'accepted' or 'rejected').
          - List of candidate_ids that were escalated to the LLM judge.
    """
    resolved: list[CandidateDecision] = []
    uncertain: list[CandidateDecision] = []

    for d in system_output.decisions:
        if d.status in ("accepted", "rejected"):
            resolved.append(d)
        else:
            uncertain.append(d)

    if not uncertain:
        # OLD: return resolved
        return resolved, []

    escalated_ids = [d.candidate_id for d in uncertain]

    logger.info(
        "normalizer.track1.resolving_uncertain",
        system_id=system_output.system_id,
        unit_id=system_output.unit_id,
        count=len(uncertain),
        # NEW: log IDs so they appear in the structured log immediately
        escalated_ids=escalated_ids,
    )

    for d in uncertain:
        req_text = candidate_text_map.get(d.candidate_id, "")
        # Use per-candidate evidence when available; fall back to full source_texts.
        evidence = source_texts
        if candidate_evidence_map:
            per_candidate = candidate_evidence_map.get(d.candidate_id)
            if per_candidate:
                evidence = per_candidate
        try:
            verdict = judge.judge_support(
                candidate_id=d.candidate_id,
                candidate_text=req_text,
                source_texts=evidence,
            )
            # verdict.verdict is 'accepted' | 'rejected' (family A only returns these two)
            status = verdict.verdict if verdict.verdict in ("accepted", "rejected") else "rejected"
            resolved.append(
                CandidateDecision(
                    candidate_id=d.candidate_id,
                    status=status,
                    confidence=verdict.confidence,
                    signal_source="llm_judge",
                    explanation=verdict.reasoning,
                )
            )
            # NEW: one line per escalated candidate showing the final verdict
            logger.info(
                "normalizer.track1.escalation_resolved",
                candidate_id=d.candidate_id,
                verdict=status,
                confidence=verdict.confidence,
            )
        except Exception as exc:
            logger.error(
                "normalizer.track1.judge_failed",
                candidate_id=d.candidate_id,
                error=str(exc),
            )
            # Default to 'rejected' on error — conservative assumption
            resolved.append(
                CandidateDecision(
                    candidate_id=d.candidate_id,
                    status="rejected",
                    confidence=0.0,
                    signal_source="llm_judge",
                    explanation=f"Judge error — defaulted to rejected: {exc}",
                )
            )

    # OLD: return resolved
    return resolved, escalated_ids
