"""API routes – export (SRS, traceability, graph)."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from fastapi.responses import PlainTextResponse
from sqlalchemy.orm import Session

from reqlens.storage.db import get_db_session
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
async def export_srs_markdown(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PlainTextResponse:
    from reqlens.services.srs_cache import load_srs_markdown, save_srs_markdown

    cached = load_srs_markdown(project_id)
    if cached is not None:
        return PlainTextResponse(cached, media_type="text/markdown")

    requirements = RequirementRepository(session).list_by_project(project_id)
    if not requirements:
        raise HTTPException(status_code=404, detail="No requirements found for this project.")

    edges = GraphEdgeRepository(session).list_by_project(project_id)
    conflicts = ConflictFindingRepository(session).list_by_project(project_id)

    llm = AzureOpenAIClient()
    spans = SourceSpanRepository(session).list_by_project(project_id)
    context = AgentContext(project_id=project_id, run_id="export")

    open_questions: list[str] = []
    if spans:
        _, open_questions = ExtractionAgent(llm)._extract_sync(context, spans)

    composer = ComposerAgent(llm)
    srs = composer.compose_srs(
        context=context,
        requirements=requirements,
        edges=edges,
        conflicts=conflicts,
        open_questions=open_questions,
    )
    markdown = composer.srs_to_markdown(srs)
    save_srs_markdown(project_id, markdown)
    return PlainTextResponse(markdown, media_type="text/markdown")


@router.get("/projects/{project_id}/export/graph/graphml")
async def export_graphml(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PlainTextResponse:
    service = ExportService(GraphStore(db_session=session, project_id=project_id))
    return PlainTextResponse(service.export_graph_graphml(), media_type="application/xml")


@router.get("/projects/{project_id}/export/graph/json")
async def export_graph_json(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PlainTextResponse:
    service = ExportService(GraphStore(db_session=session, project_id=project_id))
    return PlainTextResponse(service.export_graph_json(), media_type="application/json")


@router.get("/projects/{project_id}/export/traceability")
async def export_traceability(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PlainTextResponse:
    reqs = RequirementRepository(session).list_by_project(project_id)
    links = TraceLinkRepository(session).list_by_project(project_id)
    service = ExportService(GraphStore(db_session=session, project_id=project_id))
    return PlainTextResponse(service.export_traceability_csv(reqs, links), media_type="text/csv")


@router.get("/projects/{project_id}/export/conflicts")
async def export_conflicts(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> PlainTextResponse:
    conflicts = ConflictFindingRepository(session).list_by_project(project_id)
    service = ExportService(GraphStore(db_session=session, project_id=project_id))
    return PlainTextResponse(service.export_conflicts_json(conflicts), media_type="application/json")


@router.get("/projects/{project_id}/export/srs/status")
async def export_srs_status(project_id: str) -> dict:
    from reqlens.services.srs_cache import has_srs_markdown
    return {"ready": has_srs_markdown(project_id)}
