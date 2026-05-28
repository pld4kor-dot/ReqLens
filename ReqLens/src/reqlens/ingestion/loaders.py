"""Document loaders – parse uploaded files into raw text."""

from __future__ import annotations

import csv
import io
from pathlib import Path

import structlog

from reqlens.domain.enums import DocumentType

logger = structlog.get_logger(__name__)

# Map file suffixes to document types
_SUFFIX_TO_TYPE: dict[str, DocumentType] = {
    ".txt": DocumentType.transcript,
    ".md": DocumentType.srs,
    ".csv": DocumentType.user_story,
    ".pdf": DocumentType.srs,
}


def detect_document_type(filename: str) -> DocumentType:
    """Guess the document type from the filename/extension."""
    suffix = Path(filename).suffix.lower()
    return _SUFFIX_TO_TYPE.get(suffix, DocumentType.other)


def load_text(content: bytes, filename: str) -> str:
    """Load plain-text content (TXT, MD)."""
    return content.decode("utf-8", errors="replace")


def load_csv(content: bytes, filename: str) -> str:
    """Load CSV as a formatted text representation."""
    text_io = io.StringIO(content.decode("utf-8", errors="replace"))
    reader = csv.reader(text_io)
    rows = list(reader)
    if not rows:
        return ""

    header = rows[0]
    lines: list[str] = []
    for i, row in enumerate(rows[1:], start=1):
        parts = [f"{h}: {v}" for h, v in zip(header, row) if v.strip()]
        lines.append(f"[Row {i}] " + " | ".join(parts))

    return "\n".join(lines)


def load_pdf(content: bytes, filename: str) -> str:
    """Extract text from PDF using pypdf."""
    try:
        from pypdf import PdfReader

        reader = PdfReader(io.BytesIO(content))
        pages: list[str] = []
        for i, page in enumerate(reader.pages):
            text = page.extract_text() or ""
            if text.strip():
                pages.append(f"[Page {i + 1}]\n{text}")
        return "\n\n".join(pages)
    except ImportError:
        logger.error("pypdf not installed – cannot parse PDF")
        return ""
    except Exception as exc:
        logger.error("pdf_parse_error", filename=filename, error=str(exc))
        return ""


# Dispatcher
_LOADER_MAP: dict[str, callable] = {
    ".txt": load_text,
    ".md": load_text,
    ".csv": load_csv,
    ".pdf": load_pdf,
}


def load_document(content: bytes, filename: str) -> str:
    """Load a document into raw text regardless of format."""
    suffix = Path(filename).suffix.lower()
    loader = _LOADER_MAP.get(suffix, load_text)
    text = loader(content, filename)
    logger.info("loader.loaded", filename=filename, chars=len(text))
    return text
