"""Hierarchical chunker for PURE documents.

Two public functions:
- ``chunk_document_raw(text)`` — paragraph-based sliding-window over raw text.
- ``chunk_sections(sections)`` — splits each section independently so section
  context is carried into each chunk.

Both return a list of chunk dicts that are understood by ``requirement_extractor``.
"""

from __future__ import annotations

import re

import structlog

from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.pure.section_detector import Section

logger = structlog.get_logger(__name__)

_REQ_SIGNAL_RE = re.compile(r"\b(?:shall|must|will)\b", re.IGNORECASE)


def _count_signals(text: str) -> int:
    return len(_REQ_SIGNAL_RE.findall(text))


# ── Core splitting logic ───────────────────────────────────────────────────────

def _split_with_overlap(
    text: str,
    max_chars: int,
    overlap_chars: int,
) -> list[str]:
    """Split *text* into sliding-window chunks using paragraph boundaries.

    Strategy:
    1. Split by double-newline (paragraph boundary).
    2. Greedily accumulate paragraphs until the next paragraph would exceed
       ``max_chars``.
    3. When flushing, carry the last ``overlap_chars`` into the next chunk
       so requirements at boundaries are not missed.
    4. If a single paragraph exceeds ``max_chars``, split it at sentence or
       whitespace boundaries.
    """
    paragraphs = [p.strip() for p in re.split(r"\n{2,}", text) if p.strip()]
    if not paragraphs:
        return [text.strip()] if text.strip() else []

    chunks: list[str] = []
    current_parts: list[str] = []
    current_len: int = 0

    def flush() -> None:
        nonlocal current_parts, current_len
        if current_parts:
            body = "\n\n".join(current_parts)
            chunks.append(body)
        current_parts = []
        current_len = 0

    def _overlap_seed() -> list[str]:
        """Take the tail of the last chunk as seed for overlap."""
        if not chunks:
            return []
        tail = chunks[-1][-overlap_chars:]
        # Find a paragraph boundary so we don't start mid-sentence
        nl_pos = tail.find("\n\n")
        if nl_pos != -1:
            tail = tail[nl_pos + 2:]
        return [tail] if tail.strip() else []

    for para in paragraphs:
        if len(para) > max_chars:
            # Flush current accumulator first
            flush()
            # Split oversized paragraph at sentence boundaries
            sentences = re.split(r"(?<=[.!?])\s+", para)
            sub_buf = ""
            for sent in sentences:
                if len(sub_buf) + len(sent) + 1 <= max_chars:
                    sub_buf += (" " if sub_buf else "") + sent
                else:
                    if sub_buf:
                        chunks.append(sub_buf.strip())
                    sub_buf = sent
            if sub_buf:
                chunks.append(sub_buf.strip())
        elif current_len + len(para) + 2 > max_chars:
            flush()
            seed = _overlap_seed()
            current_parts = seed + [para]
            current_len = sum(len(p) for p in current_parts)
        else:
            current_parts.append(para)
            current_len += len(para) + 2  # +2 for the join separator

    flush()
    return chunks


# ── Public functions ──────────────────────────────────────────────────────────

def chunk_document_raw(text: str) -> list[dict]:
    """Chunk the full raw document text.

    Returns a list of dicts:
    ``{ chunk_id, text, req_signal_count, section_title, source_strategy }``
    """
    settings = get_settings()
    raw_chunks = _split_with_overlap(
        text,
        max_chars=settings.pure_raw_chunk_chars,
        overlap_chars=settings.pure_raw_chunk_overlap,
    )
    result = []
    for i, chunk_text in enumerate(raw_chunks):
        result.append(
            {
                "chunk_id": f"raw_chunk_{i + 1:04d}",
                "text": chunk_text,
                "req_signal_count": _count_signals(chunk_text),
                "section_title": None,
                "source_strategy": "raw_chunk",
            }
        )

    logger.info(
        "chunker.raw_done",
        total_chunks=len(result),
        chars=len(text),
    )
    return result


def chunk_sections(sections: list[Section]) -> list[dict]:
    """Chunk each section independently, carrying the section title as metadata.

    Only requirement-dense sections generate extraction-ready chunks.
    Context / other sections are passed through as a single oversized
    chunk so the global-context builder can use them without sending
    them for per-chunk extraction.
    """
    settings = get_settings()
    result = []
    global_idx = 0

    for section in sections:
        is_dense = section.is_requirement_dense
        sub_chunks = _split_with_overlap(
            section.text,
            max_chars=settings.pure_raw_chunk_chars,
            overlap_chars=settings.pure_raw_chunk_overlap,
        )
        for sub in sub_chunks:
            result.append(
                {
                    "chunk_id": f"sec_{global_idx:04d}_{section.title[:25].replace(' ', '_')}",
                    "text": sub,
                    "req_signal_count": _count_signals(sub),
                    "section_title": section.title,
                    "section_type": section.section_type,
                    "is_requirement_dense": is_dense,
                    "source_strategy": "section",
                }
            )
            global_idx += 1

    logger.info(
        "chunker.sections_done",
        total_chunks=len(result),
        dense=sum(1 for c in result if c.get("is_requirement_dense")),
    )
    return result
