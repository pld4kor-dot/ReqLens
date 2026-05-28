"""API routes – project versioning + change-request enforcement."""

from __future__ import annotations

import asyncio
from datetime import datetime

import structlog
from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.domain.ids import generate_id
from reqlens.domain.models import Document, GraphEdge, Project, Requirement, SourceSpan
from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import (
    DocumentRepository,
    GraphEdgeRepository,
    ProjectRepository,
    RequirementRepository,
    SourceSpanRepository,
)

logger = structlog.get_logger(__name__)
router = APIRouter()


class EnforceImpactRequest(BaseModel):
    change_request: str
    direct_ids:   list[str] = []
    indirect_ids: list[str] = []
    new_project_name: str | None = None


class EnforceImpactResponse(BaseModel):
    old_project_id:   str
    new_project_id:   str
    new_project_name: str
    impacted_count:   int
    rewritten_count:  int
    failed_ids:       list[str] = []


REWRITE_SYSTEM_PROMPT = """\
You are a requirements engineering assistant.  Your job is to REWRITE a single
existing requirement so that it incorporates a stated change request.

Rules:
- Output ONLY the new requirement text — no preamble, no explanation, no quotes.
- Preserve the requirement's original intent where it is not in conflict with
  the change request.
- Keep the requirement atomic, testable, and clearly worded.
- Preserve the same general style and terminology as the original.
- Do not add new requirements; rewrite the given one in place.
- Keep the rewritten text concise (typically 1-3 sentences).
"""


def _rewrite_one_requirement(llm, change_request: str, original_text: str,
                             req_kind: str, project_id: str) -> str:
    user_prompt = (
        f"Change request:\n{change_request}\n\n"
        f"Original requirement ({req_kind}):\n{original_text}\n\n"
        "Rewrite the requirement above so that it incorporates the change "
        "request.  Output only the new requirement text."
    )
    if hasattr(llm, "chat"):
        return llm.chat(
            system_prompt=REWRITE_SYSTEM_PROMPT,
            user_prompt=user_prompt,
            project_id=project_id,
            agent_name="impact_enforce",
        ).strip()

    from pydantic import BaseModel as _BM

    class _Rewritten(_BM):
        text: str

    result: _Rewritten = llm.structured_chat(
        system_prompt=REWRITE_SYSTEM_PROMPT,
        user_prompt=user_prompt + '\n\nRespond as JSON: {"text": "..."}',
        response_model=_Rewritten,
        project_id=project_id,
        agent_name="impact_enforce",
    )
    return result.text.strip()


def _clone_project_data(session: Session, source: Project, new_project: Project) -> dict[str, dict[str, str]]:
    doc_map:  dict[str, str] = {}
    span_map: dict[str, str] = {}

    docs = DocumentRepository(session).list_by_project(source.id)
    doc_repo = DocumentRepository(session)
    for d in docs:
        new_id = generate_id("DOC")
        doc_map[d.id] = new_id
        doc_repo.create(Document(
            id=new_id, project_id=new_project.id, filename=d.filename,
            document_type=d.document_type, content=d.content, created_at=datetime.utcnow(),
        ))

    spans = SourceSpanRepository(session).list_by_project(source.id)
    span_repo = SourceSpanRepository(session)
    cloned_spans: list[SourceSpan] = []
    for s in spans:
        new_id = generate_id("SPN")
        span_map[s.id] = new_id
        cloned_spans.append(SourceSpan(
            id=new_id, project_id=new_project.id,
            document_id=doc_map.get(s.document_id, s.document_id),
            span_index=s.span_index, text=s.text,
            char_start=s.char_start, char_end=s.char_end,
            speaker=s.speaker, section_title=getattr(s, "section_title", None),
            embedding=getattr(s, "embedding", None),
        ))
    if cloned_spans:
        span_repo.create_many(cloned_spans)

    return {"document": doc_map, "span": span_map}


def _clone_requirements(
    session: Session, source_project_id: str, new_project_id: str,
    span_map: dict[str, str], impacted_ids: set[str], rewrites: dict[str, str],
) -> tuple[int, dict[str, str]]:
    req_repo = RequirementRepository(session)
    source_reqs = req_repo.list_by_project(source_project_id)
    cloned: list[Requirement] = []
    req_id_map: dict[str, str] = {}

    for r in source_reqs:
        new_id = generate_id("REQ")
        req_id_map[r.id] = new_id
        new_text = rewrites[r.id] if (r.id in impacted_ids and r.id in rewrites) else r.text
        cloned.append(Requirement(
            id=new_id, project_id=new_project_id, text=new_text,
            kind=r.kind, nfr_subtype=r.nfr_subtype, status=r.status,
            review_status=r.review_status, quality_score=r.quality_score,
            created_from_candidate_id=None,
            source_span_ids=[span_map.get(sid, sid) for sid in (r.source_span_ids or [])],
        ))

    if cloned:
        req_repo.create_many(cloned)
    return len(cloned), req_id_map


def _clone_graph_edges(
    session: Session, source_project_id: str, new_project_id: str,
    span_map: dict[str, str], req_id_map: dict[str, str],
) -> int:
    edge_repo = GraphEdgeRepository(session)
    edges = edge_repo.list_by_project(source_project_id)
    id_map = {**span_map, **req_id_map}
    cloned = [
        GraphEdge(
            id=generate_id("GE"), project_id=new_project_id,
            source_node_id=id_map.get(e.source_node_id, e.source_node_id),
            target_node_id=id_map.get(e.target_node_id, e.target_node_id),
            edge_type=e.edge_type, confidence=e.confidence,
            created_by=e.created_by, review_status=e.review_status,
            explanation=getattr(e, "explanation", ""),
        )
        for e in edges
    ]
    if cloned:
        if hasattr(edge_repo, "create_many"):
            edge_repo.create_many(cloned)
        else:
            for ed in cloned:
                edge_repo.create(ed)
    return len(cloned)


@router.post("/projects/{project_id}/impact/enforce", response_model=EnforceImpactResponse)
async def enforce_impact(
    project_id: str,
    body: EnforceImpactRequest,
    session: Session = Depends(get_db_session),
) -> EnforceImpactResponse:
    project_repo = ProjectRepository(session)
    source = project_repo.get(project_id)
    if source is None:
        raise HTTPException(status_code=404, detail="Source project not found")

    impacted_ids = list(dict.fromkeys(body.direct_ids + body.indirect_ids))
    if not impacted_ids:
        raise HTTPException(status_code=400, detail="No requirement IDs to enforce.")
    if not body.change_request.strip():
        raise HTTPException(status_code=400, detail="change_request must not be empty.")

    req_repo = RequirementRepository(session)
    by_id = {r.id: r for r in req_repo.list_by_project(project_id)}
    targets = [by_id[i] for i in impacted_ids if i in by_id]
    if not targets:
        raise HTTPException(status_code=400, detail="None of the supplied requirement IDs exist on this project.")

    from reqlens.llm.azure_client import AzureOpenAIClient
    llm = AzureOpenAIClient()

    async def _rewrite(req: Requirement) -> tuple[str, str | None]:
        try:
            new_text = await asyncio.to_thread(
                _rewrite_one_requirement, llm, body.change_request, req.text,
                req.kind.value if hasattr(req.kind, "value") else str(req.kind), project_id,
            )
            return req.id, new_text
        except Exception as exc:
            logger.warning("rewrite.failed", req_id=req.id, error=str(exc))
            return req.id, None

    rewrite_results = await asyncio.gather(*[_rewrite(r) for r in targets])
    rewrites: dict[str, str] = {}
    failed_ids: list[str] = []
    for rid, new_text in rewrite_results:
        if new_text:
            rewrites[rid] = new_text
        else:
            failed_ids.append(rid)

    if body.new_project_name:
        new_name = body.new_project_name
    else:
        sibling_count = sum(
            1 for p in project_repo.list_all()
            if getattr(p, "parent_project_id", None) == source.id
        )
        new_name = f"{source.name} (rev {sibling_count + 1})"

    new_project = Project(
        id=generate_id("PRJ"), name=new_name,
        description=f"Forked from {source.id}. Change request: {body.change_request[:200]}",
        parent_project_id=source.id,
    )
    project_repo.create(new_project)

    id_maps = _clone_project_data(session, source, new_project)
    rewritten_count, req_id_map = _clone_requirements(
        session, source.id, new_project.id,
        id_maps["span"], set(rewrites.keys()), rewrites,
    )
    _clone_graph_edges(session, source.id, new_project.id, id_maps["span"], req_id_map)

    return EnforceImpactResponse(
        old_project_id=source.id, new_project_id=new_project.id,
        new_project_name=new_project.name, impacted_count=len(impacted_ids),
        rewritten_count=rewritten_count, failed_ids=failed_ids,
    )
