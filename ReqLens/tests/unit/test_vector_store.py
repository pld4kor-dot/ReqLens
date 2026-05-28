"""Unit tests for the vector store."""

import numpy as np
from reqinone2.storage.vector_store import VectorStore


class TestVectorStore:
    def test_upsert_and_search(self):
        vs = VectorStore()
        vs.upsert("item-1", [1.0, 0.0, 0.0])
        vs.upsert("item-2", [0.0, 1.0, 0.0])
        vs.upsert("item-3", [0.9, 0.1, 0.0])

        results = vs.search([1.0, 0.0, 0.0], top_k=2)
        assert len(results) == 2
        # item-1 should be most similar
        assert results[0][0] == "item-1"
        assert results[0][1] > 0.9

    def test_exclude_ids(self):
        vs = VectorStore()
        vs.upsert("item-1", [1.0, 0.0])
        vs.upsert("item-2", [0.9, 0.1])

        results = vs.search([1.0, 0.0], top_k=5, exclude_ids={"item-1"})
        assert all(r[0] != "item-1" for r in results)

    def test_empty_store(self):
        vs = VectorStore()
        results = vs.search([1.0, 0.0], top_k=5)
        assert results == []

    def test_upsert_many(self):
        vs = VectorStore()
        vs.upsert_many([
            ("a", [1.0, 0.0]),
            ("b", [0.0, 1.0]),
        ])
        assert len(vs) == 2

    def test_get(self):
        vs = VectorStore()
        vs.upsert("x", [0.5, 0.5])
        emb = vs.get("x")
        assert emb is not None
        assert len(emb) == 2
