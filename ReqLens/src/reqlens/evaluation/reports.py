"""Benchmark report generation."""

from __future__ import annotations

import json
from datetime import datetime

import structlog

logger = structlog.get_logger(__name__)


def generate_benchmark_report(
    benchmark_type: str,
    dataset: str,
    metrics: dict,
    config: dict | None = None,
) -> dict:
    """Generate a structured benchmark report."""
    return {
        "benchmark_type": benchmark_type,
        "dataset": dataset,
        "timestamp": datetime.utcnow().isoformat(),
        "config": config or {},
        "metrics": metrics,
    }


def report_to_json(report: dict) -> str:
    """Serialize report to pretty JSON."""
    return json.dumps(report, indent=2, default=str)


def reports_to_summary_table(reports: list[dict]) -> str:
    """Generate a Markdown summary table from multiple benchmark reports."""
    if not reports:
        return "No benchmark reports available."

    lines = [
        "| Benchmark | Dataset | Key Metric | Value |",
        "|-----------|---------|------------|-------|",
    ]

    for report in reports:
        btype = report.get("benchmark_type", "?")
        dataset = report.get("dataset", "?")
        metrics = report.get("metrics", {})

        # Pick the most important metric
        key_metrics = [
            "macro_f1", "hallucination_rejection_rate",
            "unsupported_acceptance_rate", "edge_f1",
        ]
        for km in key_metrics:
            if km in metrics:
                val = metrics[km]
                lines.append(f"| {btype} | {dataset} | {km} | {val:.4f} |")
                break
        else:
            lines.append(f"| {btype} | {dataset} | - | - |")

    return "\n".join(lines)
