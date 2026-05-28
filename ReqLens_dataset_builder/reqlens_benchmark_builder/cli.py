"""Command-line interface for the benchmark builder.

Usage:
    benchmark-builder promise        # PROMISE pipeline only
    benchmark-builder pure           # PURE pipeline only
    benchmark-builder both           # both pipelines
    benchmark-builder poison         # poisoning pipeline (reads existing outputs/)

Optional flags:
    --output-dir PATH   Override the output directory (default: outputs/)
    --log-level LEVEL   Logging level: DEBUG | INFO | WARNING (default: INFO)

Poison-specific flags (only used with 'poison' mode):
    --track {1,2,both}      Which track(s) to poison (default: both)
    --unit UNIT_ID          Poison only this unit (e.g. PROMISE_1); default: all
    --hallucinations N      Track 1: fake requirements per unit (default: 5)
    --contradictions N      Track 2: contradictions per unit (default: 2)
    --duplicates N          Track 2: duplicates per unit (default: 2)
"""

from __future__ import annotations

import argparse
import sys

import structlog

from reqlens_benchmark_builder.config import get_settings


def _configure_logging(level: str) -> None:
    import logging
    import os

    os.environ["LOG_LEVEL"] = level.upper()
    logging.basicConfig(
        format="%(message)s",
        level=getattr(logging, level.upper(), logging.INFO),
    )
    structlog.configure(
        processors=[
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.stdlib.add_log_level,
            structlog.dev.ConsoleRenderer(),
        ],
        wrapper_class=structlog.stdlib.BoundLogger,
        context_class=dict,
        logger_factory=structlog.stdlib.LoggerFactory(),
    )


def _print_summary(units: list, label: str) -> None:
    print(f"\n{'─' * 60}")
    print(f"{label} pipeline — {len(units)} unit(s) generated")
    for u in units:
        status = "PASS" if u.validation.passed else "FAIL"
        print(
            f"  [{status}] {u.id:40s}  "
            f"coverage={u.validation.coverage_rate:.0%}  "
            f"leaks={u.validation.unsupported_count}"
        )
    print(f"{'─' * 60}\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="benchmark-builder",
        description="ReqInOne benchmark / source-text generator",
    )
    parser.add_argument(
        "mode",
        choices=["promise", "pure", "both", "poison"],
        help="which pipeline(s) to run",
    )
    parser.add_argument(
        "--output-dir",
        default=None,
        help="override output directory (default: value of OUTPUT_DIR in .env)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="logging level (default: INFO)",
    )
    # Poison-specific arguments
    parser.add_argument(
        "--track",
        default="both",
        choices=["1", "2", "both"],
        help="(poison mode) which track(s) to poison",
    )
    parser.add_argument(
        "--unit",
        default=None,
        help="(poison mode) restrict to a single unit ID, e.g. PROMISE_1",
    )
    parser.add_argument(
        "--hallucinations",
        type=int,
        default=10,
        help="(poison mode, track 1) number of fake requirements per unit",
    )
    parser.add_argument(
        "--contradictions",
        type=int,
        default=5,
        help="(poison mode, track 2) number of contradictions per unit",
    )
    parser.add_argument(
        "--duplicates",
        type=int,
        default=5,
        help="(poison mode, track 2) number of duplicates per unit",
    )

    args = parser.parse_args()
    _configure_logging(args.log_level)

    # Apply override before the settings singleton is frozen
    if args.output_dir:
        import os
        os.environ["OUTPUT_DIR"] = args.output_dir

    settings = get_settings()
    print(f"Output directory : {settings.output_path.resolve()}")
    print(f"Chat deployment  : {settings.azure_openai_chat_deployment}")
    print(f"Mode             : {args.mode}\n")

    # ── Poison mode ───────────────────────────────────────────────────────────
    if args.mode == "poison":
        from reqlens_benchmark_builder.poisoning.pipeline import run_poison_pipeline
        counts = run_poison_pipeline(
            track=args.track,
            only_unit=args.unit,
            hallucination_count=args.hallucinations,
            contradiction_count=args.contradictions,
            duplicate_count=args.duplicates,
        )
        print(f"\n{'─' * 60}")
        print(f"Poisoning pipeline complete")
        print(f"  Processed  : {counts['processed']}")
        print(f"  T1 success : {counts['t1_success']}")
        print(f"  T2 success : {counts['t2_success']}")
        print(f"  Skipped    : {counts['skipped']}")
        print(f"  Failed     : {counts['failed']}")
        print(f"{'─' * 60}\n")
        sys.exit(0 if counts["failed"] == 0 else 1)

    # ── Generation modes ──────────────────────────────────────────────────────
    promise_units: list = []
    pure_units:    list = []

    if args.mode in {"promise", "both"}:
        from reqlens_benchmark_builder.pipelines.promise_pipeline import run_promise_pipeline
        promise_units = run_promise_pipeline()
        _print_summary(promise_units, "PROMISE")

    if args.mode in {"pure", "both"}:
        from reqlens_benchmark_builder.pipelines.pure_pipeline import run_pure_pipeline
        pure_units = run_pure_pipeline()
        _print_summary(pure_units, "PURE")

    total = len(promise_units) + len(pure_units)
    passed = sum(1 for u in promise_units + pure_units if u.validation.passed)
    print(f"Total: {total} unit(s) | Passed: {passed} | Failed: {total - passed}")

    sys.exit(0 if passed == total else 1)


if __name__ == "__main__":
    main()
