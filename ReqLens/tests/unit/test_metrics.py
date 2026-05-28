"""Unit tests for evaluation metrics."""

from reqinone2.evaluation.metrics import (
    hallucination_rejection_rate,
    hits_at_k,
    macro_f1,
    mean_reciprocal_rank,
    per_class_metrics,
    precision_recall_f1,
    unsupported_acceptance_rate,
    weighted_f1,
)


class TestPrecisionRecallF1:
    def test_perfect_match(self):
        m = precision_recall_f1(["a", "b", "c"], ["a", "b", "c"])
        assert m["precision"] == 1.0
        assert m["recall"] == 1.0
        assert m["f1"] == 1.0

    def test_empty(self):
        m = precision_recall_f1([], [])
        assert m["f1"] == 0.0


class TestPerClassMetrics:
    def test_two_classes(self):
        gold = ["FR", "NFR", "FR", "NFR"]
        pred = ["FR", "FR", "FR", "NFR"]
        result = per_class_metrics(gold, pred)
        assert "FR" in result
        assert "NFR" in result
        assert result["FR"]["precision"] > 0


class TestHallucinationMetrics:
    def test_perfect_rejection(self):
        assert hallucination_rejection_rate(10, 10) == 1.0

    def test_no_hallucinations(self):
        assert hallucination_rejection_rate(0, 0) == 1.0

    def test_zero_acceptance(self):
        assert unsupported_acceptance_rate(10, 0) == 0.0

    def test_full_acceptance(self):
        assert unsupported_acceptance_rate(10, 10) == 1.0
