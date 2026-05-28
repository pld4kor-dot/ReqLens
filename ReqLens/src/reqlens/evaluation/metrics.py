"""Evaluation metrics – precision, recall, F1, and specialised RE metrics."""

from __future__ import annotations

from collections import Counter
from typing import Sequence


def precision_recall_f1(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, float]:
    """Compute micro precision, recall, F1 for label sequences."""
    gold_counter = Counter(gold)
    pred_counter = Counter(predicted)

    tp = sum((gold_counter & pred_counter).values())
    fp = sum(pred_counter.values()) - tp
    fn = sum(gold_counter.values()) - tp

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    return {"precision": precision, "recall": recall, "f1": f1}


def per_class_metrics(
    gold: Sequence[str],
    predicted: Sequence[str],
) -> dict[str, dict[str, float]]:
    """Per-class precision, recall, F1."""
    labels = sorted(set(gold) | set(predicted))
    result: dict[str, dict[str, float]] = {}

    for label in labels:
        tp = sum(1 for g, p in zip(gold, predicted) if g == label and p == label)
        fp = sum(1 for g, p in zip(gold, predicted) if g != label and p == label)
        fn = sum(1 for g, p in zip(gold, predicted) if g == label and p != label)

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        result[label] = {"precision": precision, "recall": recall, "f1": f1, "support": tp + fn}

    return result


def macro_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Macro-averaged F1 across all classes."""
    if not per_class:
        return 0.0
    return sum(m["f1"] for m in per_class.values()) / len(per_class)


def weighted_f1(per_class: dict[str, dict[str, float]]) -> float:
    """Weighted F1 (by support) across all classes."""
    total_support = sum(m["support"] for m in per_class.values())
    if total_support == 0:
        return 0.0
    return sum(m["f1"] * m["support"] for m in per_class.values()) / total_support


def hallucination_rejection_rate(
    total_hallucinations: int,
    rejected_hallucinations: int,
) -> float:
    """Fraction of injected hallucinations correctly rejected."""
    if total_hallucinations == 0:
        return 1.0
    return rejected_hallucinations / total_hallucinations


def unsupported_acceptance_rate(
    total_unsupported: int,
    accepted_unsupported: int,
) -> float:
    """Fraction of unsupported requirements incorrectly accepted.

    This should be as close to 0 as possible.
    """
    if total_unsupported == 0:
        return 0.0
    return accepted_unsupported / total_unsupported


def mean_reciprocal_rank(
    gold_edges: list[tuple[str, str]],
    ranked_predictions: list[list[tuple[str, str]]],
) -> float:
    """MRR for edge prediction."""
    gold_set = set(gold_edges)
    reciprocal_ranks: list[float] = []

    for ranking in ranked_predictions:
        for rank, pred in enumerate(ranking, start=1):
            if pred in gold_set:
                reciprocal_ranks.append(1.0 / rank)
                break
        else:
            reciprocal_ranks.append(0.0)

    return sum(reciprocal_ranks) / len(reciprocal_ranks) if reciprocal_ranks else 0.0


def hits_at_k(
    gold_edges: list[tuple[str, str]],
    ranked_predictions: list[list[tuple[str, str]]],
    k: int = 5,
) -> float:
    """Hits@K for edge prediction."""
    gold_set = set(gold_edges)
    hits = 0

    for ranking in ranked_predictions:
        top_k = ranking[:k]
        if any(pred in gold_set for pred in top_k):
            hits += 1

    return hits / len(ranked_predictions) if ranked_predictions else 0.0
