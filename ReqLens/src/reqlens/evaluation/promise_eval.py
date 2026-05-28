"""PROMISE dataset evaluation – FR/NFR classification benchmark."""

from __future__ import annotations

import csv
import io

import structlog

from reqlens.evaluation.metrics import macro_f1, per_class_metrics, weighted_f1

logger = structlog.get_logger(__name__)


def load_promise_dataset(csv_content: str) -> list[dict]:
    """Load PROMISE-style CSV: columns 'text' and 'label'."""
    reader = csv.DictReader(io.StringIO(csv_content))
    rows = []
    for row in reader:
        rows.append({
            "text": row.get("text", row.get("RequirementText", "")),
            "label": row.get("label", row.get("Class", "")),
        })
    return rows


def evaluate_promise(
    gold_labels: list[str],
    predicted_labels: list[str],
) -> dict:
    """Compute PROMISE benchmark metrics."""
    per_class = per_class_metrics(gold_labels, predicted_labels)

    # Binary FR/NFR
    gold_binary = ["FR" if g == "F" else "NFR" for g in gold_labels]
    pred_binary = ["FR" if p == "F" else "NFR" for p in predicted_labels]
    binary_metrics = per_class_metrics(gold_binary, pred_binary)

    return {
        "per_class": per_class,
        "macro_f1": macro_f1(per_class),
        "weighted_f1": weighted_f1(per_class),
        "binary_fr_nfr": binary_metrics,
        "binary_macro_f1": macro_f1(binary_metrics),
    }
