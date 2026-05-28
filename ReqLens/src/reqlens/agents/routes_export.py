"""API routes export (SRS, traceability, graph)."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from fastapi.responses import PlainTextResponse

from reqlens.storage.db import get_session
from reqlens.storage.graph_store import GraphStore
from reqlens.storage.repositories import (
    ConflictFindingRepository,
    GraphEdgeRepository,
    RequirementRepository,
    SourceSpanRepository,
    TraceLinkRepository,
)
from reqlens.services.export_service import ExportService
from reqlens.agents.composer_agent import ComposerAgent
from reqlens.agents.extraction_agent import ExtractionAgent
from reqlens.agents.base import AgentContext
from reqlens.llm.azure_client import AzureOpenAIClient

router = APIRouter()

@router.get("/projects/{project_id}/export/srs/markdown")
async def export_srs_markdown(project_id: str) -> PlainTextResponse:
    """Return the composed SRS as Markdown.

    If the pipeline's Composer step has already run for this project,
    the cached markdown is returned instantly (no LLM call).  Otherwise
    we compose on demand as a fallback and cache the result so the next
    download is instant too.
    """
    from reqlens.services.srs_cache import load_srs_markdown, save_srs_markdown

    # Fast path: pipeline already produced this SRS.
    cached = load_srs_markdown(project_id)
    if cached is not None:
        return PlainTextResponse(cached, media_type="text/markdown")

    # Fallback: compose on demand (legacy behaviour).
    session = get_session()

    # Load accepted requirements from DB
    requirements = RequirementRepository(session).list_by_project(project_id)
    if not requirements:
        raise HTTPException(status_code=404, detail="No requirements found for this project.")


    edges = GraphEdgeRepository(session).list_by_project(project_id)
    conflicts = ConflictFindingRepository(session).list_by_project(project_id)

    # Recover open_questions by re-running extraction agent synchronously
    # on the persisted source spans (no LLM re-call needed for spans themselves)
    
    llm = AzureOpenAIClient()

    spans = SourceSpanRepository(session).list_by_project(project_id)
    context = AgentContext(project_id=project_id, run_id="export")

    open_questions: list[str] = []
    if spans:
        extraction_agent = ExtractionAgent(llm)
        _, open_questions = extraction_agent._extract_sync(context, spans)

    # Compose SRS and convert to Markdown using the previously unused method
    composer = ComposerAgent(llm)
    srs = composer.compose_srs(
        context=context,
        requirements=requirements,
        edges=edges,
        conflicts=conflicts,
        open_questions=open_questions,
    )

    markdown = composer.srs_to_markdown(srs)

    # Cache the result so subsequent downloads skip the LLM call.
    save_srs_markdown(project_id, markdown)

    return PlainTextResponse(markdown, media_type="text/markdown")


@router.get("/projects/{project_id}/export/srs/status")
async def export_srs_status(project_id: str) -> dict:
    """Tell the UI whether a composed SRS is already on disk for this project.

    The Export screen uses this to decide whether to show a direct
    download button immediately, or first prompt the user to compose.
    """
    from reqlens.services.srs_cache import has_srs_markdown
    return {"ready": has_srs_markdown(project_id)}

@router.get("/projects/{project_id}/export/graph/graphml")
async def export_graphml(project_id: str) -> PlainTextResponse:
    _graph_store = GraphStore(db_session=get_session(),project_id=project_id)
    service = ExportService(_graph_store)
    content = service.export_graph_graphml()
    return PlainTextResponse(content, media_type="application/xml")


@router.get("/projects/{project_id}/export/graph/json")
async def export_graph_json(project_id: str) -> PlainTextResponse:
    _graph_store = GraphStore(db_session=get_session(),project_id=project_id)
    service = ExportService(_graph_store)
    content = service.export_graph_json()
    return PlainTextResponse(content, media_type="application/json")


@router.get("/projects/{project_id}/export/traceability")
async def export_traceability(project_id: str) -> PlainTextResponse:
    session = get_session()
    _graph_store = GraphStore(db_session=session,project_id=project_id)
    reqs = RequirementRepository(session).list_by_project(project_id)
    links = TraceLinkRepository(session).list_by_project(project_id)
    service = ExportService(_graph_store)
    content = service.export_traceability_csv(reqs, links)
    return PlainTextResponse(content, media_type="text/csv")


@router.get("/projects/{project_id}/export/conflicts")
async def export_conflicts(project_id: str) -> PlainTextResponse:
    session = get_session()
    _graph_store = GraphStore(db_session=session,project_id=project_id)
    conflicts = ConflictFindingRepository(session).list_by_project(project_id)
    service = ExportService(_graph_store)
    content = service.export_conflicts_json(conflicts)
    return PlainTextResponse(content, media_type="application/json")
