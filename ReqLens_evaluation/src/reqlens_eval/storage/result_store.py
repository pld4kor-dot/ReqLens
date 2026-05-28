"""Result store — persists EvalRunReport objects to disk.

Output layout:
    <eval_output_dir>/
        runs/
            <run_id>/
                report.json        — full EvalRunReport serialised to JSON
                track1_summary.json — flattened Track 1 metrics table
                track2_summary.json — flattened Track 2 metrics table
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from reqlens_eval.config import get_settings
from reqlens_eval.models.experiment import EvalRunReport

logger = structlog.get_logger(__name__)


def save_report(report: EvalRunReport, output_dir: Path | None = None) -> Path:
    """Persist an EvalRunReport to disk.

    Returns the path of the written directory.
    """
    settings = get_settings()
    base = (output_dir or settings.eval_output_path) / "runs" / report.run_id
    base.mkdir(parents=True, exist_ok=True)

    # Full report
    report_path = base / "report.json"
    report_path.write_text(report.model_dump_json(indent=2), encoding="utf-8")
    logger.info("result_store.saved_report", run_id=report.run_id, path=str(report_path))

    # Track 1 flat summary
    if report.track1_summaries:
        t1_rows = []
        for s in report.track1_summaries:
            t1_rows.append(
                {
                    "system_id": s.system_id,
                    "unit_count": s.unit_count,
                    "mean_uar": s.mean_uar,
                    "mean_hrr": s.mean_hrr,
                    "mean_grr": s.mean_grr,   # NEW
                }
            )
        t1_path = base / "track1_summary.json"
        t1_path.write_text(json.dumps(t1_rows, indent=2), encoding="utf-8")

    # Track 2 flat summary
    if report.track2_summaries:
        t2_rows = []
        for s in report.track2_summaries:
            t2_rows.append(
                {
                    "system_id": s.system_id,
                    "unit_count": s.unit_count,
                    "mean_dlr": s.mean_dlr,
                }
            )
        t2_path = base / "track2_summary.json"
        t2_path.write_text(json.dumps(t2_rows, indent=2), encoding="utf-8")

    return base


def load_report(run_id: str, output_dir: Path | None = None) -> EvalRunReport | None:
    """Load a previously saved EvalRunReport by run_id."""
    settings = get_settings()
    report_path = (output_dir or settings.eval_output_path) / "runs" / run_id / "report.json"
    if not report_path.exists():
        logger.error("result_store.report_not_found", run_id=run_id, path=str(report_path))
        return None
    try:
        data = json.loads(report_path.read_text(encoding="utf-8"))
        return EvalRunReport.model_validate(data)
    except Exception as exc:
        logger.error("result_store.load_failed", run_id=run_id, error=str(exc))
        return None


def list_runs(output_dir: Path | None = None) -> list[str]:
    """Return a list of run_ids that have been saved."""
    settings = get_settings()
    runs_dir = (output_dir or settings.eval_output_path) / "runs"
    if not runs_dir.exists():
        return []
    return sorted(d.name for d in runs_dir.iterdir() if d.is_dir())
