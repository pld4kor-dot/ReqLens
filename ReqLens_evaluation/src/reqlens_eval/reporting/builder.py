"""Report builder — formats EvalRunReport into human-readable output.

Produces:
  - A rich console table (when running interactively via CLI)
  - A Markdown summary table (written alongside JSON results)
"""

from __future__ import annotations

from pathlib import Path

from reqlens_eval.models.experiment import EvalRunReport, SystemEvalSummary


# ── Markdown ──────────────────────────────────────────────────────────────────

def build_markdown_report(report: EvalRunReport) -> str:
    """Generate a Markdown-formatted evaluation summary."""
    lines: list[str] = []
    lines.append(f"# Evaluation Report — Run `{report.run_id}`\n")
    lines.append(f"**Created:** {report.created_at}  ")
    lines.append(f"**Systems:** {', '.join(report.systems_evaluated)}  ")
    lines.append(f"**Tracks:** {', '.join(report.tracks_evaluated)}")
    lines.append("")

    if report.track1_summaries:
        lines.append("## Track 1 — Trustworthiness (UAR / HRR / GRR)\n")
        lines.append(
            "| System | Units | Mean UAR ↓ | Mean HRR ↑ | Mean GRR ↓ |"
        )
        lines.append("|--------|-------|-----------|-----------|-----------|")
        for s in report.track1_summaries:
            uar = f"{s.mean_uar:.4f}" if s.mean_uar is not None else "—"
            hrr = f"{s.mean_hrr:.4f}" if s.mean_hrr is not None else "—"
            grr = f"{s.mean_grr:.4f}" if s.mean_grr is not None else "—"
            lines.append(f"| `{s.system_id}` | {s.unit_count} | {uar} | {hrr} | {grr} |")
        lines.append("")

    if report.track2_summaries:
        lines.append("## Track 2 — Defect Detection (DLR)\n")
        lines.append("| System | Units | Mean DLR ↓ |")
        lines.append("|--------|-------|-----------|")
        for s in report.track2_summaries:
            dlr = f"{s.mean_dlr:.4f}" if s.mean_dlr is not None else "—"
            lines.append(f"| `{s.system_id}` | {s.unit_count} | {dlr} |")
        lines.append("")

    lines.append("---")
    lines.append(
        "> UAR (Unsupported Acceptance Rate): lower is better — "
        "fraction of hallucinations the system incorrectly accepted.\n"
        "> HRR (Hallucination Rejection Rate): higher is better — "
        "fraction of hallucinations the system correctly rejected.\n"
        "> GRR (Gold Rejection Rate): lower is better — "
        "fraction of legitimate gold requirements the system incorrectly rejected.\n"
        "> DLR (Defect Leakage Rate): lower is better — "
        "fraction of seeded defects that leaked into the system's extraction output."
    )
    return "\n".join(lines)


def save_markdown_report(report: EvalRunReport, run_dir: Path) -> Path:
    """Write the Markdown report to <run_dir>/report.md and return the path."""
    md_content = build_markdown_report(report)
    md_path = run_dir / "report.md"
    md_path.write_text(md_content, encoding="utf-8")
    return md_path


# ── Console (rich) rendering ──────────────────────────────────────────────────

def print_report(report: EvalRunReport) -> None:
    """Print an evaluation report to the console using rich tables."""
    try:
        from rich.console import Console
        from rich.table import Table
        from rich import box

        console = Console()
        console.print(f"\n[bold cyan]Evaluation Report — Run {report.run_id}[/bold cyan]")
        console.print(f"[dim]{report.created_at}[/dim]\n")

        if report.track1_summaries:
            t1 = Table(
                title="Track 1 — Trustworthiness (UAR / HRR / GRR)",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            t1.add_column("System", style="cyan")
            t1.add_column("Units", justify="right")
            t1.add_column("Mean UAR ↓", justify="right")
            t1.add_column("Mean HRR ↑", justify="right", style="green")
            t1.add_column("Mean GRR ↓", justify="right", style="yellow")
            for s in report.track1_summaries:
                uar = f"{s.mean_uar:.4f}" if s.mean_uar is not None else "—"
                hrr = f"{s.mean_hrr:.4f}" if s.mean_hrr is not None else "—"
                grr = f"{s.mean_grr:.4f}" if s.mean_grr is not None else "—"
                t1.add_row(s.system_id, str(s.unit_count), uar, hrr, grr)
            console.print(t1)
            console.print()

        if report.track2_summaries:
            t2 = Table(
                title="Track 2 — Defect Detection (DLR)",
                box=box.ROUNDED,
                show_header=True,
                header_style="bold magenta",
            )
            t2.add_column("System", style="cyan")
            t2.add_column("Units", justify="right")
            t2.add_column("Mean DLR ↓", justify="right", style="green")
            for s in report.track2_summaries:
                dlr = f"{s.mean_dlr:.4f}" if s.mean_dlr is not None else "—"
                t2.add_row(s.system_id, str(s.unit_count), dlr)
            console.print(t2)
            console.print()

    except ImportError:
        # Fallback: plain print
        print(build_markdown_report(report))
