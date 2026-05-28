"""Span index – fast lookup of source spans by ID and embedding similarity."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import structlog

from reqlens.domain.models import SourceSpan

logger = structlog.get_logger(__name__)


class SpanIndex:
    """In-memory index for source spans with cosine-similarity search."""

    def __init__(self) -> None:
        self._spans: dict[str, SourceSpan] = {}
        self._embeddings: np.ndarray | None = None
        self._id_order: list[str] = []

    def add_spans(self, spans: Sequence[SourceSpan]) -> None:
        """Add spans to the index."""
        for span in spans:
            self._spans[span.id] = span

        # Rebuild embedding matrix
        self._id_order = [
            sid for sid, s in self._spans.items() if s.embedding is not None
        ]
        if self._id_order:
            vectors = [self._spans[sid].embedding for sid in self._id_order]  # type: ignore[arg-type]
            self._embeddings = np.array(vectors, dtype=np.float32)
            # L2-normalize for cosine similarity via dot product
            norms = np.linalg.norm(self._embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._embeddings = self._embeddings / norms

    def get(self, span_id: str) -> SourceSpan | None:
        return self._spans.get(span_id)

    def get_many(self, span_ids: list[str]) -> list[SourceSpan]:
        return [self._spans[sid] for sid in span_ids if sid in self._spans]

    def all_spans(self) -> list[SourceSpan]:
        return list(self._spans.values())

    def search_by_embedding(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 5,
    ) -> list[tuple[SourceSpan, float]]:
        """Return top-k spans by cosine similarity to *query_embedding*."""
        if self._embeddings is None or len(self._id_order) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self._embeddings @ q  # cosine similarities
        top_indices = np.argsort(scores)[::-1][:top_k]

        results: list[tuple[SourceSpan, float]] = []
        for idx in top_indices:
            span_id = self._id_order[idx]
            results.append((self._spans[span_id], float(scores[idx])))
        return results

    def search_by_text(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> list[SourceSpan]:
        """Simple substring search (fallback when no embeddings)."""
        query_lower = query.lower()
        scored: list[tuple[SourceSpan, int]] = []
        for span in self._spans.values():
            count = span.text.lower().count(query_lower)
            if count > 0:
                scored.append((span, count))
        scored.sort(key=lambda x: x[1], reverse=True)
        return [s for s, _ in scored[:top_k]]
