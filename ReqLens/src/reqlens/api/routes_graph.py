"""API routes – knowledge graph."""

from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.storage.db import get_db_session
from reqlens.storage.graph_store import GraphStore
from reqlens.storage.repositories import RequirementRepository

router = APIRouter()


class GraphResponse(BaseModel):
    nodes: list[dict]
    edges: list[dict]


class GraphStatsResponse(BaseModel):
    node_count: int
    edge_count: int


@router.get("/projects/{project_id}/graph", response_model=GraphResponse)
async def get_graph(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> GraphResponse:
    _graph_store = GraphStore(session, project_id)
    data = _graph_store.to_dict()
    return GraphResponse(nodes=data["nodes"], edges=data["edges"])


@router.get("/projects/{project_id}/graph/stats", response_model=GraphStatsResponse)
async def get_graph_stats(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> GraphStatsResponse:
    _graph_store = GraphStore(session, project_id)
    return GraphStatsResponse(
        node_count=_graph_store.node_count,
        edge_count=_graph_store.edge_count,
    )


@router.get("/projects/{project_id}/graph/neighborhood/{node_id}")
async def get_neighborhood(
    project_id: str,
    node_id: str,
    depth: int = 2,
    session: Session = Depends(get_db_session),
) -> GraphResponse:
    _graph_store = GraphStore(session, project_id)
    data = _graph_store.get_requirement_neighborhood(node_id, depth=depth)
    return GraphResponse(nodes=data["nodes"], edges=data["edges"])


class GraphSyncResponse(BaseModel):
    synced: int
    removed: int
    updated: int
    errors: int


@router.post("/projects/{project_id}/graph/sync-decisions", response_model=GraphSyncResponse)
async def sync_graph_with_decisions(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> GraphSyncResponse:
    """Bulk-sync the knowledge graph with the current review decisions.

    Iterates every requirement in the project and applies
    ``GraphStore.sync_requirement_decision`` for each one using the SAME
    session that loaded the requirements — so all reads and writes share
    one SQLAlchemy identity map and one transaction.  A single commit at
    the end atomically persists every status change.
    """
    import structlog
    _log = structlog.get_logger(__name__)

    # Both RequirementRepository and GraphStore share `session` — same
    # identity map, same transaction.  No split-session inconsistency.
    req_repo = RequirementRepository(session)
    requirements = req_repo.list_by_project(project_id)

    if not requirements:
        return GraphSyncResponse(synced=0, removed=0, updated=0, errors=0)

    graph_store = GraphStore(session, project_id)
    removed = updated = errors = 0

    for req in requirements:
        try:
            decision = (
                req.review_status.value
                if hasattr(req.review_status, "value")
                else str(req.review_status)
            )
            kind = req.kind.value if hasattr(req.kind, "value") else str(req.kind)
            graph_store.sync_requirement_decision(
                requirement_id=req.id,
                decision=decision,
                requirement_text=req.text,
                kind=kind,
            )
            if decision == "rejected":
                removed += 1
            else:
                updated += 1
        except Exception as exc:
            _log.error(
                "graph.bulk_sync.error",
                requirement_id=req.id,
                error=str(exc),
            )
            errors += 1

    try:
        session.commit()
    except Exception as exc:
        _log.error("graph.bulk_sync.commit_failed", error=str(exc))
        errors += 1

    total = removed + updated
    _log.info(
        "graph.bulk_sync.done",
        project_id=project_id,
        total=total,
        removed=removed,
        updated=updated,
        errors=errors,
    )
    return GraphSyncResponse(synced=total, removed=removed, updated=updated, errors=errors)
