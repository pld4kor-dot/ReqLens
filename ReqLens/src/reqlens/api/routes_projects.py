"""API routes – projects."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.domain.models import Project
from reqlens.services.project_service import ProjectService
from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import ProjectRepository

router = APIRouter()


class CreateProjectRequest(BaseModel):
    name: str
    description: str = ""


class ProjectResponse(BaseModel):
    id: str
    name: str
    description: str
    created_at: str
    parent_project_id: str | None = None


class DeleteProjectResponse(BaseModel):
    project_id: str
    deleted: dict[str, int]
    total_rows: int


def _project_response(p: Project) -> ProjectResponse:
    return ProjectResponse(
        id=p.id,
        name=p.name,
        description=p.description,
        created_at=p.created_at.isoformat(),
        parent_project_id=getattr(p, "parent_project_id", None),
    )


@router.post("", response_model=ProjectResponse)
async def create_project(
    body: CreateProjectRequest,
    session: Session = Depends(get_db_session),
) -> ProjectResponse:
    service = ProjectService(ProjectRepository(session))
    project = service.create_project(body.name, body.description)
    return _project_response(project)


@router.get("", response_model=list[ProjectResponse])
async def list_projects(
    session: Session = Depends(get_db_session),
) -> list[ProjectResponse]:
    service = ProjectService(ProjectRepository(session))
    return [_project_response(p) for p in service.list_projects()]


@router.get("/{project_id}", response_model=ProjectResponse)
async def get_project(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> ProjectResponse:
    service = ProjectService(ProjectRepository(session))
    project = service.get_project(project_id)
    if project is None:
        raise HTTPException(status_code=404, detail="Project not found")
    return _project_response(project)


@router.get("/{project_id}/versions", response_model=list[ProjectResponse])
async def list_project_versions(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> list[ProjectResponse]:
    service = ProjectService(ProjectRepository(session))
    target = service.get_project(project_id)
    if target is None:
        raise HTTPException(status_code=404, detail="Project not found")

    root = target
    while getattr(root, "parent_project_id", None):
        parent = service.get_project(root.parent_project_id)
        if parent is None:
            break
        root = parent

    all_projects = service.list_projects()
    by_parent: dict[str | None, list[Project]] = {}
    for p in all_projects:
        by_parent.setdefault(getattr(p, "parent_project_id", None), []).append(p)

    chain: list[Project] = [root]
    queue: list[Project] = [root]
    while queue:
        cur = queue.pop(0)
        kids = by_parent.get(cur.id, [])
        kids.sort(key=lambda x: x.created_at)
        chain.extend(kids)
        queue.extend(kids)

    return [_project_response(p) for p in chain]


@router.delete("/{project_id}", response_model=DeleteProjectResponse)
async def delete_project(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> DeleteProjectResponse:
    service = ProjectService(ProjectRepository(session))
    try:
        deleted = service.delete_project(project_id)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))

    try:
        from reqlens.services.srs_cache import clear_srs_markdown
        clear_srs_markdown(project_id)
    except Exception:
        pass

    return DeleteProjectResponse(
        project_id=project_id,
        deleted=deleted,
        total_rows=sum(deleted.values()),
    )
