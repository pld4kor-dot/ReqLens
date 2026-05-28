"""ReqFromSRS dataset evaluation."""

from __future__ import annotations

import structlog

from reqlens.evaluation.metrics import macro_f1, per_class_metrics, weighted_f1

logger = structlog.get_logger(__name__)


def evaluate_reqfromsrs(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    """Compute ReqFromSRS benchmark metrics."""
    per_class = per_class_metrics(gold_labels, predicted_labels)

    return {
        "per_class": per_class,
        "macro_f1": macro_f1(per_class),
        "weighted_f1": weighted_f1(per_class),
    }
