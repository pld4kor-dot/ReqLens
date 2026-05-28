"""Review service – human review queue and decisions."""

from __future__ import annotations

import structlog

from reqlens.domain.enums import EvidenceStatus, RequirementStatus, ReviewStatus
from reqlens.domain.models import (
    EvidenceAssessment,
    Requirement,
    RequirementCandidate,
    ReviewDecision,
)
from reqlens.storage.repositories import (
    EvidenceAssessmentRepository,
    RequirementCandidateRepository,
    RequirementRepository,
    ReviewDecisionRepository,
)

logger = structlog.get_logger(__name__)


class ReviewService:
    def __init__(
        self,
        requirement_repo: RequirementRepository,
        candidate_repo: RequirementCandidateRepository,
        evidence_repo: EvidenceAssessmentRepository,
        decision_repo: ReviewDecisionRepository,
    ) -> None:
        self.requirement_repo = requirement_repo
        self.candidate_repo = candidate_repo
        self.evidence_repo = evidence_repo
        self.decision_repo = decision_repo

    def get_review_queue(self, project_id: str) -> list[dict]:
        """Return items needing human review.

        Includes:
          - Requirements with pending review status
          - Candidates with insufficient evidence
        """
        queue: list[dict] = []

        # Requirements pending review
        requirements = self.requirement_repo.list_by_project(project_id)
        for req in requirements:
            if req.review_status == ReviewStatus.pending:
                queue.append({
                    "type": "requirement",
                    "id": req.id,
                    "text": req.text,
                    "kind": req.kind.value,
                    "status": req.status.value,
                    "review_status": req.review_status.value,
                })

        # Candidates with insufficient evidence
        candidates = self.candidate_repo.list_by_project(project_id)
        for cand in candidates:
            assessment = self.evidence_repo.get_by_candidate(cand.id)
            if assessment and assessment.status == EvidenceStatus.insufficient_evidence:
                queue.append({
                    "type": "candidate_insufficient_evidence",
                    "id": cand.id,
                    "text": cand.text,
                    "kind": cand.requirement_kind.value,
                    "evidence_explanation": assessment.explanation,
                    "confidence": assessment.confidence,
                })

        return queue

    def submit_decision(
        self,
        requirement_id: str,
        decision: ReviewStatus,
        reviewer: str = "human",
        comment: str = "",
    ) -> ReviewDecision:
        """Record a review decision and update the requirement status."""
        review = ReviewDecision(
            requirement_id=requirement_id,
            decision=decision,
            reviewer=reviewer,
            comment=comment,
        )
        self.decision_repo.create(review)

        # Update requirement
        req = self.requirement_repo.get(requirement_id)
        if req:
            req.review_status = decision
            if decision == ReviewStatus.accepted:
                req.status = RequirementStatus.accepted
            elif decision == ReviewStatus.rejected:
                req.status = RequirementStatus.rejected
            self.requirement_repo.update(req)

        logger.info(
            "review.decision",
            requirement_id=requirement_id,
            decision=decision.value,
        )
        return review
