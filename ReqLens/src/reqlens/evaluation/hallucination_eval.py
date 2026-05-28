"""Hallucination / evidence-grounding benchmark."""

from __future__ import annotations

import structlog

from reqlens.domain.enums import EvidenceStatus
from reqlens.domain.models import EvidenceAssessment
from reqlens.evaluation.metrics import (
    hallucination_rejection_rate,
    precision_recall_f1,
    unsupported_acceptance_rate,
)

logger = structlog.get_logger(__name__)


def evaluate_evidence_grounding(
    assessments: list[EvidenceAssessment],
    gold_supported_ids: set[str],
    gold_hallucinated_ids: set[str],
) -> dict:
    """Evaluate the evidence agent's ability to block hallucinations.

    Args:
        assessments: Evidence assessments produced by the agent.
        gold_supported_ids: Candidate IDs that are genuinely supported.
        gold_hallucinated_ids: Candidate IDs that were injected hallucinations.
    """
    # Classify agent decisions
    accepted_ids = {
        a.requirement_candidate_id
        for a in assessments
        if a.status == EvidenceStatus.entailed
    }
    rejected_ids = {
        a.requirement_candidate_id
        for a in assessments
        if a.status in (EvidenceStatus.contradicted, EvidenceStatus.insufficient_evidence)
    }

    # Hallucination detection
    rejected_hallucinations = len(gold_hallucinated_ids & rejected_ids)
    accepted_hallucinations = len(gold_hallucinated_ids & accepted_ids)
    total_hallucinations = len(gold_hallucinated_ids)

    # Supported detection
    accepted_supported = len(gold_supported_ids & accepted_ids)
    rejected_supported = len(gold_supported_ids & rejected_ids)
    total_supported = len(gold_supported_ids)

    hrr = hallucination_rejection_rate(total_hallucinations, rejected_hallucinations)
    uar = unsupported_acceptance_rate(total_hallucinations, accepted_hallucinations)

    # Source span precision/recall
    gold_spans: list[str] = []
    pred_spans: list[str] = []
    for a in assessments:
        if a.requirement_candidate_id in gold_supported_ids:
            pred_spans.extend(a.supporting_span_ids)
            # Gold spans would need to come from benchmark data
            # This is a placeholder for the full implementation

    return {
        "hallucination_rejection_rate": hrr,
        "unsupported_acceptance_rate": uar,
        "total_hallucinations": total_hallucinations,
        "rejected_hallucinations": rejected_hallucinations,
        "accepted_hallucinations": accepted_hallucinations,
        "total_supported": total_supported,
        "accepted_supported": accepted_supported,
        "rejected_supported": rejected_supported,
        "evidence_accuracy": (rejected_hallucinations + accepted_supported) / max(len(assessments), 1),
    }
