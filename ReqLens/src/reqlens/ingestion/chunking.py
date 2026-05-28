"""Document chunking – split raw text into source spans.

Preserves character offsets so every span can be traced back
to its exact position in the original document.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import structlog

logger = structlog.get_logger(__name__)

# Default chunk parameters
DEFAULT_CHUNK_SIZE = 600  # target tokens (≈ 2 400 chars)
DEFAULT_CHUNK_OVERLAP = 100  # overlap tokens (≈ 400 chars)
_CHARS_PER_TOKEN = 4


@dataclass
class RawSpan:
    """Intermediate representation before domain SourceSpan creation."""

    text: str
    char_start: int
    char_end: int
    span_index: int
    speaker: str | None = None
    section_title: str | None = None


def _split_by_paragraphs(text: str) -> list[tuple[int, str]]:
    """Split text by double newlines, returning (char_start, paragraph)."""
    paragraphs: list[tuple[int, str]] = []
    for match in re.finditer(r"(?:^|\n\n)(.+?)(?=\n\n|$)", text, re.DOTALL):
        para = match.group(1).strip()
        if para:
            paragraphs.append((match.start(1), para))
    # Fallback: if regex yields nothing, treat entire text as one paragraph
    if not paragraphs and text.strip():
        paragraphs.append((0, text.strip()))
    return paragraphs


def _detect_speaker(paragraph: str) -> str | None:
    """Heuristic: detect speaker labels like 'Alice:' at start of paragraph."""
    m = re.match(r"^([A-Z][a-zA-Z\s]{1,30}):\s", paragraph)
    return m.group(1).strip() if m else None


def _detect_section(paragraph: str) -> str | None:
    """Heuristic: detect Markdown-style headers."""
    m = re.match(r"^(#{1,4})\s+(.+)$", paragraph, re.MULTILINE)
    return m.group(2).strip() if m else None


def chunk_document(
    text: str,
    *,
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[RawSpan]:
    """Split *text* into overlapping spans with character offsets.

    Strategy:
    1. Split by paragraphs first (natural boundaries).
    2. Merge small paragraphs up to ``chunk_size`` tokens.
    3. Split oversized paragraphs with overlap.
    """
    char_chunk = chunk_size * _CHARS_PER_TOKEN
    char_overlap = chunk_overlap * _CHARS_PER_TOKEN

    paragraphs = _split_by_paragraphs(text)
    spans: list[RawSpan] = []
    idx = 0

    current_text = ""
    current_start: int | None = None
    current_speaker: str | None = None
    current_section: str | None = None

    for para_start, para in paragraphs:
        # Detect metadata
        speaker = _detect_speaker(para) or current_speaker
        section = _detect_section(para) or current_section
        if _detect_section(para):
            current_section = section

        if len(para) > char_chunk:
            # Flush any accumulated text
            if current_text.strip():
                spans.append(
                    RawSpan(
                        text=current_text.strip(),
                        char_start=current_start or para_start,
                        char_end=(current_start or para_start) + len(current_text),
                        span_index=idx,
                        speaker=current_speaker,
                        section_title=current_section,
                    )
                )
                idx += 1
                current_text = ""
                current_start = None

            # Split oversized paragraph
            start = 0
            while start < len(para):
                end = min(start + char_chunk, len(para))
                chunk = para[start:end]
                spans.append(
                    RawSpan(
                        text=chunk.strip(),
                        char_start=para_start + start,
                        char_end=para_start + end,
                        span_index=idx,
                        speaker=speaker,
                        section_title=current_section,
                    )
                )
                idx += 1
                start = end - char_overlap if end < len(para) else end
        else:
            # Try to merge with current accumulator
            if current_start is None:
                current_start = para_start
                current_text = para
                current_speaker = speaker
            elif len(current_text) + len(para) + 2 <= char_chunk:
                current_text += "\n\n" + para
                current_speaker = speaker or current_speaker
            else:
                # Flush
                spans.append(
                    RawSpan(
                        text=current_text.strip(),
                        char_start=current_start,
                        char_end=current_start + len(current_text),
                        span_index=idx,
                        speaker=current_speaker,
                        section_title=current_section,
                    )
                )
                idx += 1
                current_text = para
                current_start = para_start
                current_speaker = speaker

    # Flush remaining
    if current_text.strip() and current_start is not None:
        spans.append(
            RawSpan(
                text=current_text.strip(),
                char_start=current_start,
                char_end=current_start + len(current_text),
                span_index=idx,
                speaker=current_speaker,
                section_title=current_section,
            )
        )

    logger.info("chunking.done", total_spans=len(spans), total_chars=len(text))
    return spans
