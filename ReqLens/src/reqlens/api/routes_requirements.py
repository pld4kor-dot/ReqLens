"""API routes – requirements."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.domain.enums import ReviewStatus
from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import (
    EvidenceAssessmentRepository,
    RequirementCandidateRepository,
    RequirementRepository,
)
from reqlens.services.requirement_service import RequirementService

router = APIRouter()


class RequirementResponse(BaseModel):
    id: str
    project_id: str
    text: str
    kind: str
    nfr_subtype: str
    status: str
    review_status: str
    quality_score: float | None
    source_span_ids: list[str]
    created_at: str


class RequirementPatchRequest(BaseModel):
    text: str | None = None
    review_status: str | None = None


def _service(session: Session) -> RequirementService:
    return RequirementService(
        RequirementCandidateRepository(session),
        RequirementRepository(session),
        EvidenceAssessmentRepository(session),
    )


def _resp(r) -> RequirementResponse:
    return RequirementResponse(
        id=r.id, project_id=r.project_id, text=r.text,
        kind=r.kind.value, nfr_subtype=r.nfr_subtype.value,
        status=r.status.value, review_status=r.review_status.value,
        quality_score=r.quality_score,
        source_span_ids=r.source_span_ids,
        created_at=r.created_at.isoformat(),
    )


@router.get("/projects/{project_id}/requirements", response_model=list[RequirementResponse])
async def list_requirements(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> list[RequirementResponse]:
    return [_resp(r) for r in _service(session).list_requirements(project_id)]


@router.get("/requirements/{requirement_id}", response_model=RequirementResponse)
async def get_requirement(
    requirement_id: str,
    session: Session = Depends(get_db_session),
) -> RequirementResponse:
    req = _service(session).get_requirement(requirement_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Requirement not found")
    return _resp(req)


@router.patch("/requirements/{requirement_id}", response_model=RequirementResponse)
async def patch_requirement(
    requirement_id: str,
    body: RequirementPatchRequest,
    session: Session = Depends(get_db_session),
) -> RequirementResponse:
    svc = _service(session)
    req = svc.get_requirement(requirement_id)
    if req is None:
        raise HTTPException(status_code=404, detail="Requirement not found")

    if body.text is not None:
        req.text = body.text
    if body.review_status is not None:
        req.review_status = ReviewStatus(body.review_status)

    svc.update_requirement(req)
    return _resp(req)
