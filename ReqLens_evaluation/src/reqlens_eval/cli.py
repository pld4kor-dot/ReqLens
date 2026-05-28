"""CLI entry point for the reqlens-eval evaluation pipeline.

Commands
--------
  run     Run the full evaluation (all systems × all tracks by default).
  report  Display a saved evaluation report.
  list    List all saved evaluation runs.

Examples
--------
  # Run all systems, both tracks
  reqlens-eval run

  # Run only one system, one track, one unit
  reqlens-eval run --system baseline --track 1 --unit PROMISE_1

  # Display a saved run
  reqlens-eval report --run-id <uuid>

  # List all saved runs
  reqlens-eval list
"""

from __future__ import annotations

import sys

import click
import structlog

from reqlens_eval.adapters.factory import AVAILABLE_SYSTEMS
from reqlens_eval.config import get_settings


def _setup_logging() -> None:
    settings = get_settings()
    import logging

    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(message)s",
    )
    structlog.configure(
        wrapper_class=structlog.make_filtering_bound_logger(
            getattr(logging, settings.log_level.upper(), logging.INFO)
        ),
    )


@click.group()
def main() -> None:
    """reqlens-eval — evaluation pipeline for Baseline, ReqInOne v1, and ReqLens v2."""
    _setup_logging()


# ── run ──────────────────────────────────────────────────────────────────────

@main.command()
@click.option(
    "--system",
    "systems",
    multiple=True,
    type=click.Choice(AVAILABLE_SYSTEMS, case_sensitive=False),
    help="System(s) to evaluate (repeat flag for multiple). Default: all.",
)
@click.option(
    "--track",
    "tracks",
    multiple=True,
    type=click.Choice(["1", "2"], case_sensitive=False),
    help="Track(s) to evaluate (repeat flag for multiple). Default: both.",
)
@click.option("--unit", default=None, help="Evaluate only this unit ID (e.g. PROMISE_1).")
@click.option(
    "--variant",
    default=None,
    help="Use only this variant of poisoned artifacts (e.g. hallu_v1).",
)
@click.option(
    "--benchmark-dir",
    default=None,
    help="Override path to benchmark outputs directory.",
)
@click.option(
    "--no-save",
    is_flag=True,
    default=False,
    help="Skip saving results to disk (just print).",
)
def run(
    systems: tuple[str, ...],
    tracks: tuple[str, ...],
    unit: str | None,
    variant: str | None,
    benchmark_dir: str | None,
    no_save: bool,
) -> None:
    """Run the evaluation pipeline."""
    from pathlib import Path

    from reqlens_eval.execution.orchestrator import ExperimentOrchestrator
    from reqlens_eval.reporting.builder import print_report, save_markdown_report
    from reqlens_eval.storage.result_store import save_report

    orchestrator = ExperimentOrchestrator(
        benchmark_output_dir=Path(benchmark_dir) if benchmark_dir else None,
    )

    report = orchestrator.run(
        systems=list(systems) if systems else None,
        tracks=list(tracks) if tracks else None,
        unit_id=unit,
        variant_id=variant,
    )

    # Always print to console
    print_report(report)

    if not no_save:
        run_dir = save_report(report)
        md_path = save_markdown_report(report, run_dir)
        click.echo(f"\nResults saved to: {run_dir}")
        click.echo(f"Markdown report: {md_path}")


# ── report ────────────────────────────────────────────────────────────────────

@main.command()
@click.option("--run-id", required=True, help="Run ID to display (from 'reqlens-eval list').")
def report(run_id: str) -> None:
    """Display a saved evaluation report."""
    from reqlens_eval.reporting.builder import print_report
    from reqlens_eval.storage.result_store import load_report

    loaded = load_report(run_id)
    if loaded is None:
        click.echo(f"Run '{run_id}' not found.", err=True)
        sys.exit(1)
    print_report(loaded)


# ── list ─────────────────────────────────────────────────────────────────────

@main.command(name="list")
def list_runs_cmd() -> None:
    """List all saved evaluation runs."""
    from reqlens_eval.storage.result_store import list_runs

    runs = list_runs()
    if not runs:
        click.echo("No evaluation runs found.")
    else:
        click.echo(f"Found {len(runs)} run(s):")
        for r in runs:
            click.echo(f"  {r}")
