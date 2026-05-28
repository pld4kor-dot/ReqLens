"""Requirement service – extraction, evidence, promotion."""

from __future__ import annotations

import structlog

from reqlens.domain.enums import EvidenceStatus, RequirementStatus, ReviewStatus
from reqlens.domain.ids import generate_id
from reqlens.domain.models import (
    EvidenceAssessment,
    Requirement,
    RequirementCandidate,
    SourceSpan,
)
from reqlens.storage.repositories import (
    EvidenceAssessmentRepository,
    RequirementCandidateRepository,
    RequirementRepository,
)

logger = structlog.get_logger(__name__)


class RequirementService:
    def __init__(
        self,
        candidate_repo: RequirementCandidateRepository,
        requirement_repo: RequirementRepository,
        evidence_repo: EvidenceAssessmentRepository,
    ) -> None:
        self.candidate_repo = candidate_repo
        self.requirement_repo = requirement_repo
        self.evidence_repo = evidence_repo

    def store_candidates(self, candidates: list[RequirementCandidate]) -> None:
        self.candidate_repo.create_many(candidates)

    def store_assessments(self, assessments: list[EvidenceAssessment]) -> None:
        self.evidence_repo.create_many(assessments)

    def promote_entailed_candidates(
        self,
        project_id: str,
        candidates: list[RequirementCandidate],
        assessments: list[EvidenceAssessment],
    ) -> list[Requirement]:
        """Promote candidates with ENTAILED evidence to full requirements."""
        assessment_map = {a.requirement_candidate_id: a for a in assessments}
        promoted: list[Requirement] = []

        for cand in candidates:
            assessment = assessment_map.get(cand.id)
            if assessment and assessment.status == EvidenceStatus.entailed:
                req = Requirement(
                    id=generate_id("REQ"),
                    project_id=project_id,
                    text=cand.text,
                    kind=cand.requirement_kind,
                    nfr_subtype=cand.nfr_subtype,
                    status=RequirementStatus.evidence_checked,
                    review_status=ReviewStatus.pending,
                    created_from_candidate_id=cand.id,
                    source_span_ids=assessment.supporting_span_ids or cand.source_span_ids,
                )
                promoted.append(req)

                # Update candidate status
                self.candidate_repo.update_status(cand.id, RequirementStatus.evidence_checked.value)

        if promoted:
            self.requirement_repo.create_many(promoted)

        logger.info(
            "requirement_service.promoted",
            total_candidates=len(candidates),
            promoted=len(promoted),
        )
        return promoted

    def list_requirements(self, project_id: str) -> list[Requirement]:
        return self.requirement_repo.list_by_project(project_id)

    def get_requirement(self, requirement_id: str) -> Requirement | None:
        return self.requirement_repo.get(requirement_id)

    def update_requirement(self, req: Requirement) -> None:
        self.requirement_repo.update(req)

    def list_candidates(self, project_id: str) -> list[RequirementCandidate]:
        return self.candidate_repo.list_by_project(project_id)
