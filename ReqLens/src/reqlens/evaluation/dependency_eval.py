"""Dependency / edge prediction benchmark."""

from __future__ import annotations

import structlog

from reqlens.evaluation.metrics import (
    hits_at_k,
    mean_reciprocal_rank,
    per_class_metrics,
    macro_f1,
)

logger = structlog.get_logger(__name__)


def evaluate_dependency_prediction(
    gold_edges: list[tuple[str, str, str]],  # (src, tgt, edge_type)
    predicted_edges: list[tuple[str, str, str]],
) -> dict:
    """Evaluate dependency edge prediction quality."""
    # Edge-level precision/recall/F1
    gold_pairs = {(src, tgt) for src, tgt, _ in gold_edges}
    pred_pairs = {(src, tgt) for src, tgt, _ in predicted_edges}

    tp = len(gold_pairs & pred_pairs)
    fp = len(pred_pairs - gold_pairs)
    fn = len(gold_pairs - pred_pairs)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

    # Edge type classification
    gold_types = [t for _, _, t in gold_edges]
    pred_types = [t for _, _, t in predicted_edges][:len(gold_types)]
    type_metrics = per_class_metrics(gold_types, pred_types) if len(gold_types) == len(pred_types) else {}

    return {
        "edge_precision": precision,
        "edge_recall": recall,
        "edge_f1": f1,
        "edge_type_macro_f1": macro_f1(type_metrics) if type_metrics else 0.0,
        "edge_type_per_class": type_metrics,
    }
