"""API routes – individual agent execution endpoints."""

from __future__ import annotations

import time

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.storage.db import get_db_session

logger = structlog.get_logger(__name__)
router = APIRouter()


# ---------------------------------------------------------------------------
# Response models
# ---------------------------------------------------------------------------

class AgentStepResponse(BaseModel):
    agent: str
    step: int
    status: str
    summary: str
    elapsed_s: float = 0.0
    created_ids: list[str] = []
    warnings: list[str] = []
    errors: list[str] = []


class PrepareResponse(BaseModel):
    spans_count: int
    status: str


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_project_or_404(session: Session, project_id: str):
    from reqlens.storage.repositories import ProjectRepository
    project = ProjectRepository(session).get(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail=f"Project '{project_id}' not found")
    return project


def _build_shared(session: Session, project_id: str):
    from reqlens.ingestion.span_index import SpanIndex
    from reqlens.llm.azure_client import AzureOpenAIClient
    from reqlens.storage.graph_store import GraphStore
    from reqlens.storage.repositories import SourceSpanRepository
    from reqlens.storage.vector_store import VectorStore

    spans        = SourceSpanRepository(session).list_by_project(project_id)
    llm          = AzureOpenAIClient()
    vector_store = VectorStore()
    span_index   = SpanIndex()
    span_index.add_spans(spans)
    graph_store  = GraphStore(session, project_id)
    return llm, vector_store, span_index, graph_store, spans


def _build_context(project_id, llm, vector_store, span_index, graph_store):
    from reqlens.agents.base import AgentContext
    return AgentContext(
        project_id=project_id,
        llm_client=llm,
        vector_store=vector_store,
        span_index=span_index,
        graph_store=graph_store,
    )


def _ok(agent, step, summary, elapsed, created_ids=None, warnings=None):
    return AgentStepResponse(
        agent=agent, step=step, status="completed",
        summary=summary, elapsed_s=round(elapsed, 3),
        created_ids=created_ids or [], warnings=warnings or [],
    )


def _fail(agent, step, exc, elapsed):
    return AgentStepResponse(
        agent=agent, step=step, status="failed",
        summary=f"{type(exc).__name__}: {exc}",
        elapsed_s=round(elapsed, 3),
        errors=[str(exc)],
    )


# ---------------------------------------------------------------------------
# PREPARE
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/prepare", response_model=PrepareResponse)
async def prepare_pipeline(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PrepareResponse:
    _get_project_or_404(session, project_id)

    from reqlens.storage.repositories import DocumentRepository, SourceSpanRepository

    span_repo = SourceSpanRepository(session)
    spans = span_repo.list_by_project(project_id)
    if spans:
        return PrepareResponse(spans_count=len(spans), status="skipped")

    from reqlens.ingestion.chunking import chunk_document
    from reqlens.ingestion.normalization import normalize_document_text

    docs = DocumentRepository(session).list_by_project(project_id)
    for doc in docs:
        doc_spans = chunk_document(
            text=normalize_document_text(doc.content),
            project_id=project_id,
            document_id=doc.id,
        )
        span_repo.create_many(doc_spans)
        spans.extend(doc_spans)

    return PrepareResponse(spans_count=len(spans), status="ready")


# ---------------------------------------------------------------------------
# STEP 1 – Extraction Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/extraction/run", response_model=AgentStepResponse)
async def run_extraction(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.extraction_agent import ExtractionAgent
    from reqlens.storage.repositories import RequirementCandidateRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, spans = _build_shared(session, project_id)
        if not spans:
            return AgentStepResponse(
                agent="Extraction Agent", step=1, status="skipped",
                summary="No source spans found — run /prepare first.",
            )

        ctx = _build_context(project_id, llm, vector_store, span_index, graph_store)
        candidate_repo = RequirementCandidateRepository(session)
        agent = ExtractionAgent(llm=llm, candidate_repo=candidate_repo)
        result = await agent.run(ctx, spans)
        persisted = candidate_repo.list_by_project(project_id)
        elapsed = time.perf_counter() - t0

        if result.status == "failed":
            return AgentStepResponse(
                agent="Extraction Agent", step=1, status="failed",
                summary=result.errors[0] if result.errors else "Extraction failed",
                elapsed_s=round(elapsed, 3), errors=result.errors,
            )

        return _ok("Extraction Agent", 1,
                   f"Extracted and persisted {len(persisted)} candidate(s) ({len(result.warnings)} warning(s))",
                   elapsed, created_ids=[c.id for c in persisted], warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="extraction", error=str(exc))
        return _fail("Extraction Agent", 1, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 2 – Evidence Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/evidence/run", response_model=AgentStepResponse)
async def run_evidence(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.evidence_agent import EvidenceAgent
    from reqlens.domain.enums import DependencyEdgeType, EvidenceStatus, RequirementStatus
    from reqlens.domain.ids import generate_id
    from reqlens.domain.models import Requirement
    from reqlens.storage.repositories import RequirementCandidateRepository, RequirementRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, spans = _build_shared(session, project_id)
        ctx        = _build_context(project_id, llm, vector_store, span_index, graph_store)
        candidates = RequirementCandidateRepository(session).list_by_project(project_id)

        if not candidates:
            return AgentStepResponse(
                agent="Evidence Agent", step=2, status="skipped",
                summary="No candidates in DB — run extraction first.",
            )

        agent       = EvidenceAgent(llm=llm, span_index=span_index)
        result      = await agent.run(ctx, candidates)
        assessments = agent.assess_candidates(ctx, candidates)

        req_repo     = RequirementRepository(session)
        requirements = []
        for cand, assessment in zip(candidates, assessments):
            if assessment.status == EvidenceStatus.entailed:
                req = Requirement(
                    id=generate_id("REQ"), project_id=project_id,
                    text=cand.text, kind=cand.requirement_kind, nfr_subtype=cand.nfr_subtype,
                    status=RequirementStatus.evidence_checked,
                    created_from_candidate_id=cand.id,
                    source_span_ids=assessment.supporting_span_ids or cand.source_span_ids,
                )
                requirements.append(req)
                graph_store.upsert_requirement_node(req.id, {"text": req.text, "kind": req.kind.value, "status": req.status.value})
                for span_id in req.source_span_ids:
                    graph_store.upsert_edge(req.id, span_id, DependencyEdgeType.derived_from.value, {"confidence": assessment.confidence})

        if requirements:
            req_repo.create_many(requirements)

        return _ok("Evidence Agent", 2,
                   f"{len(requirements)} of {len(candidates)} candidate(s) passed the evidence gate",
                   time.perf_counter() - t0,
                   created_ids=[r.id for r in requirements], warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="evidence", error=str(exc))
        return _fail("Evidence Agent", 2, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 3 – Classification Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/classification/run", response_model=AgentStepResponse)
async def run_classification(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.classification_agent import ClassificationAgent
    from reqlens.storage.repositories import RequirementRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, _ = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Classification Agent", step=3, status="skipped", summary="No requirements in DB.")

        result = await ClassificationAgent(llm=llm).run(ctx, requirements)
        return _ok("Classification Agent", 3, f"Classified {len(requirements)} requirement(s)",
                   time.perf_counter() - t0, warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="classification", error=str(exc))
        return _fail("Classification Agent", 3, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 4 – Ambiguity Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/ambiguity/run", response_model=AgentStepResponse)
async def run_ambiguity(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.ambiguity_agent import AmbiguityAgent
    from reqlens.storage.repositories import RequirementRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, _ = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Ambiguity Agent", step=4, status="skipped", summary="No requirements found.")

        result = await AmbiguityAgent(llm=llm).run(ctx, requirements)
        return _ok("Ambiguity Agent", 4,
                   f"Checked {len(requirements)} requirement(s) — {len(result.warnings)} flag(s)",
                   time.perf_counter() - t0, warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="ambiguity", error=str(exc))
        return _fail("Ambiguity Agent", 4, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 5 – Dependency Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/dependency/run", response_model=AgentStepResponse)
async def run_dependency(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.dependency_agent import DependencyAgent
    from reqlens.storage.repositories import RequirementRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, _ = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Dependency Agent", step=5, status="skipped", summary="No requirements found.")

        result = await DependencyAgent(llm=llm, vector_store=vector_store).run(ctx, requirements)
        return _ok("Dependency Agent", 5, f"Mapped dependencies for {len(requirements)} requirement(s)",
                   time.perf_counter() - t0, created_ids=result.created_ids, warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="dependency", error=str(exc))
        return _fail("Dependency Agent", 5, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 6 – Consistency Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/consistency/run", response_model=AgentStepResponse)
async def run_consistency(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.consistency_agent import ConsistencyAgent
    from reqlens.storage.repositories import RequirementRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, _ = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Consistency Agent", step=6, status="skipped", summary="No requirements found.")

        result = await ConsistencyAgent(llm=llm).run(ctx, requirements)
        return _ok("Consistency Agent", 6,
                   f"Checked {len(requirements)} requirement(s) — {len(result.warnings)} conflict(s)",
                   time.perf_counter() - t0, warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="consistency", error=str(exc))
        return _fail("Consistency Agent", 6, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 7 – Traceability Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/traceability/run", response_model=AgentStepResponse)
async def run_traceability(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.traceability_agent import TraceabilityAgent
    from reqlens.storage.repositories import RequirementRepository, TraceLinkRepository

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, spans = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Traceability Agent", step=7, status="skipped", summary="No requirements found.")

        agent = TraceabilityAgent(llm=llm)
        trace_links = agent.build_trace_links(ctx, requirements, spans)

        if trace_links:
            try:
                TraceLinkRepository(session).create_many(trace_links)
                logger.info("agent.traceability.persisted", project_id=project_id, count=len(trace_links))
            except Exception as persist_exc:
                logger.warning("agent.traceability.persist_failed", project_id=project_id, error=str(persist_exc))

        result = await agent.run(ctx, requirements, spans)
        return _ok("Traceability Agent", 7,
                   f"Created and persisted {len(trace_links)} traceability link(s)",
                   time.perf_counter() - t0,
                   created_ids=[l.id for l in trace_links], warnings=result.warnings)
    except Exception as exc:
        logger.error("agent.failed", step="traceability", error=str(exc))
        return _fail("Traceability Agent", 7, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 8 – Elicitation Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/elicitation/run", response_model=AgentStepResponse)
async def run_elicitation(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.ambiguity_agent import AmbiguityAgent
    from reqlens.agents.elicitation_agent import ElicitationAgent
    from reqlens.agents.extraction_agent import ExtractionAgent
    from reqlens.storage.repositories import RequirementRepository
    from reqlens.services.srs_cache import save_stakeholder_questions

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, spans = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements and not spans:
            return AgentStepResponse(agent="Elicitation Agent", step=8, status="skipped",
                                     summary="No requirements or source spans found.")

        # 1) Unresolved questions from the Extraction Agent.
        raw_questions: list[str] = []
        warnings: list[str] = []
        if spans:
            try:
                _, raw_questions = ExtractionAgent(llm)._extract_sync(ctx, spans)
            except Exception as exc:
                logger.warning("agent.elicitation.extraction_failed", error=str(exc))
                warnings.append(f"Could not re-derive extractor questions: {exc}")

        # 2) Ambiguity findings — give the elicitation agent more to ground on.
        quality_findings = []
        if requirements:
            try:
                quality_findings = AmbiguityAgent(llm=llm).analyse(ctx, requirements)
            except Exception as exc:
                logger.warning("agent.elicitation.ambiguity_failed", error=str(exc))
                warnings.append(f"Could not re-derive ambiguity findings: {exc}")

        # 3) Turn the above into clear, stakeholder-facing questions.
        stakeholder_questions = ElicitationAgent(llm=llm).generate_questions(
            ctx,
            open_questions=raw_questions,
            insufficient_requirements=None,
            quality_findings=quality_findings,
        )

        # 4) Cache so the Composer step picks them up without re-running the LLM.
        save_stakeholder_questions(project_id, stakeholder_questions)

        summary = (
            f"Generated {len(stakeholder_questions)} stakeholder question(s) "
            f"from {len(raw_questions)} unresolved + {len(quality_findings)} ambiguity finding(s)"
        )
        return _ok("Elicitation Agent", 8, summary, time.perf_counter() - t0, warnings=warnings)
    except Exception as exc:
        logger.error("agent.failed", step="elicitation", error=str(exc))
        return _fail("Elicitation Agent", 8, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 9 – Ingestion Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/ingestion/run", response_model=AgentStepResponse)
async def run_ingestion(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.domain.enums import RequirementStatus, ReviewStatus
    from reqlens.storage.repositories import RequirementRepository

    t0 = time.perf_counter()
    try:
        req_repo     = RequirementRepository(session)
        requirements = req_repo.list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Ingestion Agent", step=9, status="skipped", summary="No requirements to persist.")

        for req in requirements:
            req.review_status = ReviewStatus.accepted
            req.status        = RequirementStatus.accepted
            req_repo.update(req)

        return _ok("Ingestion Agent", 9, f"Marked {len(requirements)} requirement(s) as accepted",
                   time.perf_counter() - t0)
    except Exception as exc:
        logger.error("agent.failed", step="ingestion", error=str(exc))
        return _fail("Ingestion Agent", 9, exc, time.perf_counter() - t0)


# ---------------------------------------------------------------------------
# STEP 10 – Composer Agent
# ---------------------------------------------------------------------------

@router.post("/projects/{project_id}/agents/composer/run", response_model=AgentStepResponse)
async def run_composer(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> AgentStepResponse:
    _get_project_or_404(session, project_id)

    from reqlens.agents.composer_agent import ComposerAgent
    from reqlens.agents.consistency_agent import ConsistencyAgent
    from reqlens.agents.ambiguity_agent import AmbiguityAgent
    from reqlens.agents.elicitation_agent import ElicitationAgent
    from reqlens.agents.extraction_agent import ExtractionAgent
    from reqlens.storage.repositories import RequirementRepository
    from reqlens.services.srs_cache import save_srs_markdown, load_stakeholder_questions

    t0 = time.perf_counter()
    try:
        llm, vector_store, span_index, graph_store, spans = _build_shared(session, project_id)
        ctx          = _build_context(project_id, llm, vector_store, span_index, graph_store)
        requirements = RequirementRepository(session).list_by_project(project_id)

        if not requirements:
            return AgentStepResponse(agent="Composer Agent", step=10, status="skipped", summary="No requirements to compose.")

        # Prefer the Elicitation Agent's cached output (from step 8). If the
        # user ran the composer step standalone without elicitation, fall back
        # to re-running the same logic inline so the SRS still gets stakeholder
        # questions.
        open_questions: list[str] | None = load_stakeholder_questions(project_id)

        if open_questions is None:
            logger.info("composer.elicitation_cache_miss", project_id=project_id)

            # 1) Extractor's unresolved questions (raw, free-form).
            raw_questions: list[str] = []
            if spans:
                try:
                    _, raw_questions = ExtractionAgent(llm)._extract_sync(ctx, spans)
                except Exception as exc:
                    logger.warning("composer.open_questions_failed", error=str(exc))

            # 2) Ambiguity findings as additional context for elicitation.
            quality_findings = []
            try:
                quality_findings = AmbiguityAgent(llm=llm).analyse(ctx, requirements)
            except Exception as exc:
                logger.warning("composer.ambiguity_findings_failed", error=str(exc))

            # 3) Polish into stakeholder-facing questions.
            open_questions = raw_questions
            try:
                open_questions = ElicitationAgent(llm=llm).generate_questions(
                    ctx,
                    open_questions=raw_questions,
                    insufficient_requirements=None,
                    quality_findings=quality_findings,
                ) or raw_questions
            except Exception as exc:
                logger.warning("composer.elicitation_failed", error=str(exc))

        conflicts = ConsistencyAgent(llm=llm).detect_conflicts(ctx, requirements)
        composer  = ComposerAgent(llm=llm)
        srs       = composer.compose_srs(ctx, requirements, conflicts=conflicts, open_questions=open_questions)
        markdown  = composer.srs_to_markdown(srs)
        save_srs_markdown(project_id, markdown)

        return _ok("Composer Agent", 10,
                   f"SRS composed: {len(srs.sections)} section(s), {len(markdown)} chars — ready for export",
                   time.perf_counter() - t0)
    except Exception as exc:
        logger.error("agent.failed", step="composer", error=str(exc))
        return _fail("Composer Agent", 10, exc, time.perf_counter() - t0)
