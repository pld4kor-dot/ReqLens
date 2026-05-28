"""Document service – ingest documents and create source spans."""

from __future__ import annotations

import structlog

from reqlens.domain.enums import DocumentType
from reqlens.domain.ids import generate_id
from reqlens.domain.models import Document, SourceSpan
from reqlens.ingestion.chunking import chunk_document
from reqlens.ingestion.loaders import detect_document_type, load_document
from reqlens.ingestion.normalization import normalize_document_text
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.storage.repositories import DocumentRepository, SourceSpanRepository

logger = structlog.get_logger(__name__)


class DocumentService:
    def __init__(
        self,
        doc_repo: DocumentRepository,
        span_repo: SourceSpanRepository,
        llm: AzureOpenAIClient | None = None,
    ) -> None:
        self.doc_repo = doc_repo
        self.span_repo = span_repo
        self.llm = llm

    def ingest_document(
        self,
        project_id: str,
        filename: str,
        content: bytes,
        *,
        document_type: DocumentType | None = None,
        compute_embeddings: bool = True,
    ) -> tuple[Document, list[SourceSpan]]:
        """Parse, chunk, and store a document with its source spans."""
        # 1. Load raw text
        raw_text = load_document(content, filename)
        text = normalize_document_text(raw_text)

        # 2. Create document record
        doc_type = document_type or detect_document_type(filename)
        doc = Document(
            project_id=project_id,
            filename=filename,
            document_type=doc_type,
            content=text,
        )
        self.doc_repo.create(doc)

        # 3. Chunk into source spans
        raw_spans = chunk_document(text)
        spans: list[SourceSpan] = []
        for rs in raw_spans:
            span = SourceSpan(
                project_id=project_id,
                document_id=doc.id,
                span_index=rs.span_index,
                text=rs.text,
                char_start=rs.char_start,
                char_end=rs.char_end,
                speaker=rs.speaker,
                section_title=rs.section_title,
            )
            spans.append(span)

        # 4. Compute embeddings
        if compute_embeddings and self.llm and spans:
            try:
                texts = [s.text for s in spans]
                embeddings = self.llm.embed_texts(texts, project_id=project_id, agent_name="ingestion")
                for span, emb in zip(spans, embeddings):
                    span.embedding = emb
            except Exception as exc:
                logger.error("document_service.embedding_error", error=str(exc))

        # 5. Persist
        self.span_repo.create_many(spans)

        logger.info(
            "document_service.ingested",
            document_id=doc.id,
            filename=filename,
            spans=len(spans),
        )
        return doc, spans

    def list_documents(self, project_id: str) -> list[Document]:
        return self.doc_repo.list_by_project(project_id)

    def list_spans(self, project_id: str) -> list[SourceSpan]:
        return self.span_repo.list_by_project(project_id)
