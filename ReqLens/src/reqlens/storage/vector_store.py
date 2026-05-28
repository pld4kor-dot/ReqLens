"""Vector store abstraction for source span and requirement embeddings."""

from __future__ import annotations

from typing import Sequence

import numpy as np
import structlog

logger = structlog.get_logger(__name__)


class VectorStore:
    """In-memory vector store (pgvector adapter can be added later).

    Stores id → embedding pairs and supports cosine-similarity search.
    """

    def __init__(self) -> None:
        self._vectors: dict[str, np.ndarray] = {}
        self._matrix: np.ndarray | None = None
        self._id_order: list[str] = []
        self._dirty = True

    def upsert(self, item_id: str, embedding: list[float]) -> None:
        self._vectors[item_id] = np.array(embedding, dtype=np.float32)
        self._dirty = True

    def upsert_many(self, items: Sequence[tuple[str, list[float]]]) -> None:
        for item_id, emb in items:
            self._vectors[item_id] = np.array(emb, dtype=np.float32)
        self._dirty = True

    def _rebuild_index(self) -> None:
        if not self._dirty:
            return
        self._id_order = list(self._vectors.keys())
        if self._id_order:
            mat = np.stack([self._vectors[k] for k in self._id_order])
            norms = np.linalg.norm(mat, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1, norms)
            self._matrix = mat / norms
        else:
            self._matrix = None
        self._dirty = False

    def search(
        self,
        query_embedding: list[float],
        *,
        top_k: int = 10,
        exclude_ids: set[str] | None = None,
    ) -> list[tuple[str, float]]:
        """Return (item_id, cosine_similarity) tuples."""
        self._rebuild_index()
        if self._matrix is None or len(self._id_order) == 0:
            return []

        q = np.array(query_embedding, dtype=np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm == 0:
            return []
        q = q / q_norm

        scores = self._matrix @ q
        order = np.argsort(scores)[::-1]

        results: list[tuple[str, float]] = []
        for idx in order:
            item_id = self._id_order[idx]
            if exclude_ids and item_id in exclude_ids:
                continue
            results.append((item_id, float(scores[idx])))
            if len(results) >= top_k:
                break
        return results

    def get(self, item_id: str) -> list[float] | None:
        vec = self._vectors.get(item_id)
        return vec.tolist() if vec is not None else None

    def __len__(self) -> int:
        return len(self._vectors)
