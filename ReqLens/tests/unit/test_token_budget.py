"""Unit tests for token budget utilities."""

from reqinone2.llm.token_budget import (
    batch_texts_by_budget,
    estimate_tokens,
    trim_spans_to_budget,
)


class TestTokenBudget:
    def test_estimate_tokens(self):
        assert estimate_tokens("hello world") > 0

    def test_trim_within_budget(self):
        spans = ["short " * 10, "medium " * 50, "long " * 200]
        result = trim_spans_to_budget(spans, max_tokens=500)
        assert len(result) >= 1
        assert len(result) <= 3

    def test_trim_empty(self):
        assert trim_spans_to_budget([], max_tokens=1000) == []

    def test_batch_texts(self):
        texts = ["hello " * 100] * 10
        batches = batch_texts_by_budget(texts, max_tokens_per_batch=500)
        assert len(batches) >= 1
        for batch in batches:
            assert len(batch) >= 1
