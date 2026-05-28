"""API routes – documents and source spans."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, UploadFile, File
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.domain.models import Document, SourceSpan
from reqlens.services.document_service import DocumentService
from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import DocumentRepository, SourceSpanRepository

router = APIRouter()


class DocumentResponse(BaseModel):
    id: str
    project_id: str
    filename: str
    document_type: str
    created_at: str


class SourceSpanResponse(BaseModel):
    id: str
    document_id: str
    span_index: int
    text: str
    char_start: int
    char_end: int
    speaker: str | None
    section_title: str | None


@router.post("/projects/{project_id}/documents", response_model=DocumentResponse)
async def upload_document(
    project_id: str,
    file: UploadFile = File(...),
    session: Session = Depends(get_db_session),
) -> DocumentResponse:
    service = DocumentService(
        DocumentRepository(session),
        SourceSpanRepository(session),
        llm=None,
    )
    content = await file.read()
    doc, spans = service.ingest_document(
        project_id,
        file.filename or "unnamed",
        content,
        compute_embeddings=False,
    )
    return DocumentResponse(
        id=doc.id,
        project_id=doc.project_id,
        filename=doc.filename,
        document_type=doc.document_type.value,
        created_at=doc.created_at.isoformat(),
    )


@router.get("/projects/{project_id}/documents", response_model=list[DocumentResponse])
async def list_documents(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> list[DocumentResponse]:
    service = DocumentService(
        DocumentRepository(session),
        SourceSpanRepository(session),
    )
    docs = service.list_documents(project_id)
    return [
        DocumentResponse(
            id=d.id, project_id=d.project_id, filename=d.filename,
            document_type=d.document_type.value, created_at=d.created_at.isoformat(),
        )
        for d in docs
    ]


@router.get("/projects/{project_id}/source-spans", response_model=list[SourceSpanResponse])
async def list_source_spans(
    project_id: str,
    session: Session = Depends(get_db_session),
) -> list[SourceSpanResponse]:
    spans = SourceSpanRepository(session).list_by_project(project_id)
    return [
        SourceSpanResponse(
            id=s.id, document_id=s.document_id, span_index=s.span_index,
            text=s.text, char_start=s.char_start, char_end=s.char_end,
            speaker=s.speaker, section_title=s.section_title,
        )
        for s in spans
    ]
