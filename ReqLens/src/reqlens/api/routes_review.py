"""API routes – review queue, decisions, and pipeline execution."""

from __future__ import annotations

import structlog

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.domain.enums import ReviewStatus
from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import (
    ConflictFindingRepository,
    DocumentRepository,
    EvidenceAssessmentRepository,
    RequirementCandidateRepository,
    RequirementRepository,
    ReviewDecisionRepository,
    SourceSpanRepository,
    TraceLinkRepository,
)
from reqlens.services.review_service import ReviewService

logger = structlog.get_logger(__name__)
router = APIRouter()


class ReviewDecisionRequest(BaseModel):
    requirement_id: str
    decision: str
    reviewer: str = "human"
    comment: str = ""


class ReviewDecisionResponse(BaseModel):
    id: str
    requirement_id: str
    decision: str
    reviewer: str
    comment: str
    created_at: str


class PipelineRequest(BaseModel):
    steps: list[str] | None = None


class PipelineResponse(BaseModel):
    status: str
    steps_run: list[str]
    requirements_count: int = 0
    error: str | None = None


@router.post("/projects/{project_id}/pipeline", response_model=PipelineResponse)
async def run_pipeline(
    project_id: str,
    body: PipelineRequest | None = None,
    session: Session = Depends(get_db_session),
) -> PipelineResponse:
    from reqlens.agents.base import AgentContext
    from reqlens.agents.orchestrator import PipelineOrchestrator
    from reqlens.agents.extraction_agent import ExtractionAgent
    from reqlens.agents.evidence_agent import EvidenceAgent
    from reqlens.agents.classification_agent import ClassificationAgent
    from reqlens.agents.ambiguity_agent import AmbiguityAgent
    from reqlens.agents.dependency_agent import DependencyAgent
    from reqlens.agents.consistency_agent import ConsistencyAgent
    from reqlens.agents.traceability_agent import TraceabilityAgent
    from reqlens.agents.elicitation_agent import ElicitationAgent
    from reqlens.agents.composer_agent import ComposerAgent
    from reqlens.ingestion.span_index import SpanIndex
    from reqlens.llm.azure_client import AzureOpenAIClient
    from reqlens.storage.graph_store import GraphStore
    from reqlens.storage.vector_store import VectorStore
    from reqlens.storage.repositories import ProjectRepository

    project = ProjectRepository(session).get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")

    try:
        span_repo = SourceSpanRepository(session)
        doc_repo = DocumentRepository(session)
        spans = span_repo.list_by_project(project_id)
        req_repo = RequirementRepository(session)

        if not spans:
            from reqlens.ingestion.chunking import chunk_document
            from reqlens.ingestion.normalization import normalize_document_text
            docs = doc_repo.list_by_project(project_id)
            for doc in docs:
                cleaned = normalize_document_text(doc.content)
                doc_spans = chunk_document(text=cleaned, project_id=project_id, document_id=doc.id)
                span_repo.create_many(doc_spans)
                spans.extend(doc_spans)

        if not spans:
            return PipelineResponse(status="skipped", steps_run=[], error="No documents or spans found for project")

        llm_client = AzureOpenAIClient()
        vector_store = VectorStore()
        span_index = SpanIndex()
        span_index.add_spans(spans)
        graph_store = GraphStore(session, project_id)

        context = AgentContext(
            project_id=project_id,
            llm_client=llm_client,
            vector_store=vector_store,
            span_index=span_index,
            graph_store=graph_store,
        )

        orchestrator = PipelineOrchestrator(
            extraction_agent=ExtractionAgent(llm=llm_client, candidate_repo=RequirementCandidateRepository(session)),
            evidence_agent=EvidenceAgent(llm=llm_client, span_index=span_index),
            classification_agent=ClassificationAgent(llm=llm_client),
            ambiguity_agent=AmbiguityAgent(llm=llm_client),
            dependency_agent=DependencyAgent(llm=llm_client, vector_store=vector_store),
            consistency_agent=ConsistencyAgent(llm=llm_client),
            traceability_agent=TraceabilityAgent(llm=llm_client),
            elicitation_agent=ElicitationAgent(llm=llm_client),
            composer_agent=ComposerAgent(llm=llm_client),
            graph_store=graph_store,
            requirement_repo=req_repo,
            candidate_repo=RequirementCandidateRepository(session),
            conflict_repo=ConflictFindingRepository(session),
            trace_link_repo=TraceLinkRepository(session),
        )

        results = await orchestrator.run_full_pipeline(context, spans)
        reqs = RequirementRepository(session).list_by_project(project_id)
        return PipelineResponse(
            status="completed",
            steps_run=[r.agent_name for r in results] if results else [],
            requirements_count=len(reqs),
        )
    except Exception as exc:
        return PipelineResponse(status="failed", steps_run=[], error=f"{type(exc).__name__}: {exc}")


@router.get("/projects/{project_id}/review-queue")
async def get_review_queue(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> list[dict]:
    service = ReviewService(
        RequirementRepository(session),
        RequirementCandidateRepository(session),
        EvidenceAssessmentRepository(session),
        ReviewDecisionRepository(session),
    )
    return service.get_review_queue(project_id)


@router.post("/review-decisions", response_model=ReviewDecisionResponse)
async def submit_review_decision(
    body: ReviewDecisionRequest,
    session: Session = Depends(get_db_session),
) -> ReviewDecisionResponse:
    from reqlens.storage.graph_store import GraphStore

    service = ReviewService(
        RequirementRepository(session),
        RequirementCandidateRepository(session),
        EvidenceAssessmentRepository(session),
        ReviewDecisionRepository(session),
    )

    try:
        decision_status = ReviewStatus(body.decision)
    except ValueError:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid decision: {body.decision}. Must be one of: {[s.value for s in ReviewStatus]}",
        )

    # 1. Persist the review decision and update requirement status in SQL DB
    decision = service.submit_decision(body.requirement_id, decision_status, body.reviewer, body.comment)

    # 2. Sync the knowledge graph — same session, so the RequirementRow
    #    mutated by submit_decision is visible to GraphStore immediately.
    try:
        req = RequirementRepository(session).get(body.requirement_id)
        if req is None:
            logger.warning("review.graph_sync.requirement_not_found", requirement_id=body.requirement_id)
        else:
            graph_store = GraphStore(session, req.project_id)
            graph_store.sync_requirement_decision(
                requirement_id=body.requirement_id,
                decision=body.decision,
                requirement_text=req.text,
                kind=req.kind.value if hasattr(req.kind, "value") else str(req.kind),
            )
            session.commit()
            logger.info(
                "review.graph_sync.done",
                requirement_id=body.requirement_id,
                decision=body.decision,
                project_id=req.project_id,
            )
    except Exception as exc:
        logger.error(
            "review.graph_sync.failed",
            requirement_id=body.requirement_id,
            decision=body.decision,
            error=str(exc),
        )

    return ReviewDecisionResponse(
        id=decision.id,
        requirement_id=decision.requirement_id,
        decision=decision.decision.value,
        reviewer=decision.reviewer,
        comment=decision.comment,
        created_at=decision.created_at.isoformat(),
    )
