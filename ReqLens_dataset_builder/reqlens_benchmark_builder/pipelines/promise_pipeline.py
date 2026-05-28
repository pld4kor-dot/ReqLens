"""PROMISE pipeline.

End-to-end flow for each selected PROMISE project:

  1. Load gold requirements from CSV (grouped by ProjectID).
  2. Build a scenario brief.
  3. Generate a 3-artifact source bundle.
  4. Validate coverage + leakage.
  5. Repair if either threshold is not met (bounded by max_repair_rounds).
  6. Save benchmark unit (JSON + TXT).
"""

from __future__ import annotations

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.scenario_brief_builder import build_scenario_brief
from reqlens_benchmark_builder.generation.source_bundle_generator import generate_source_bundle
from reqlens_benchmark_builder.loaders.promise_loader import load_promise_projects
from reqlens_benchmark_builder.schemas.benchmark_models import (
    BenchmarkUnit,
    CoverageEntry,
    ValidationSummary,
)
from reqlens_benchmark_builder.validation.coverage_validator import validate_coverage
from reqlens_benchmark_builder.validation.repair import repair_source_bundle
from reqlens_benchmark_builder.validation.unsupported_validator import validate_unsupported
from reqlens_benchmark_builder.writers.io import write_benchmark_unit

logger = structlog.get_logger(__name__)


def _build_validation_summary(
    coverage_report: dict,
    unsupported_report: dict,
    repair_rounds: int,
    settings,
) -> ValidationSummary:
    rate  = coverage_report.get("coverage_rate", 0.0)
    missed = coverage_report.get("missing_req_ids", [])
    leaks  = unsupported_report.get("count", 0)
    passed = (
        rate >= settings.min_coverage_rate
        and leaks <= settings.max_unsupported_count
    )
    entries = [
        CoverageEntry(**e)
        for e in coverage_report.get("coverage", [])
        if isinstance(e, dict)
    ]
    return ValidationSummary(
        coverage_rate=rate,
        missing_req_ids=missed,
        unsupported_count=leaks,
        repair_rounds_used=repair_rounds,
        coverage_entries=entries,
        unsupported_implied=unsupported_report.get(
            "unsupported_implied_requirements", []
        ),
        passed=passed,
    )


def run_promise_pipeline() -> list[BenchmarkUnit]:
    """Run the full PROMISE benchmark-building pipeline.

    Returns a list of completed ``BenchmarkUnit`` objects (one per project).
    """
    settings = get_settings()
    llm      = AzureOpenAIClient()
    units: list[BenchmarkUnit] = []

    try:
        projects = load_promise_projects()
    except FileNotFoundError as exc:
        logger.error("promise_pipeline.load_failed", error=str(exc))
        return []

    for project in projects:
        unit_id          = f"PROMISE_{project.project_id}"
        gold_dicts       = [r.model_dump() for r in project.requirements]

        logger.info(
            "promise_pipeline.start",
            unit_id=unit_id,
            req_count=len(gold_dicts),
            label_dist=project.label_distribution,
        )

        # Step 2 – scenario brief
        brief = build_scenario_brief(llm, unit_id, gold_dicts)

        # Step 3 – source bundle
        source_texts = generate_source_bundle(llm, unit_id, brief, gold_dicts)
        source_dicts = [s.model_dump() for s in source_texts]

        # Step 4 – validate
        coverage_report    = validate_coverage(llm, unit_id, source_dicts, gold_dicts)
        unsupported_report = validate_unsupported(llm, unit_id, source_dicts, gold_dicts)

        # Step 5 – repair loop
        repair_rounds = 0
        while repair_rounds < settings.max_repair_rounds:
            rate  = coverage_report.get("coverage_rate", 0.0)
            leaks = unsupported_report.get("count", 0)
            if (
                rate >= settings.min_coverage_rate
                and leaks <= settings.max_unsupported_count
            ):
                break  # validation passed

            logger.info(
                "promise_pipeline.repair",
                unit_id=unit_id,
                round=repair_rounds + 1,
                coverage_rate=rate,
                unsupported_count=leaks,
            )
            source_texts = repair_source_bundle(
                llm=llm,
                unit_id=unit_id,
                brief=brief,
                source_texts=source_dicts,
                gold_requirements=gold_dicts,
                coverage_report=coverage_report,
                unsupported_report=unsupported_report,
            )
            source_dicts       = [s.model_dump() for s in source_texts]
            coverage_report    = validate_coverage(llm, unit_id, source_dicts, gold_dicts)
            unsupported_report = validate_unsupported(llm, unit_id, source_dicts, gold_dicts)
            repair_rounds += 1

        # Step 6 – assemble and save
        validation = _build_validation_summary(
            coverage_report, unsupported_report, repair_rounds, settings
        )
        unit = BenchmarkUnit(
            id=unit_id,
            origin="PROMISE",
            source_texts=source_texts,
            gold_requirements=project.requirements,
            validation=validation,
            brief=brief,
            metadata={
                "project_id":    project.project_id,
                "req_count":     len(gold_dicts),
                "label_dist":    project.label_distribution,
                "llm_usage":     llm.usage_summary(),
            },
        )

        out_dir = settings.output_path / "promise" / unit_id
        write_benchmark_unit(out_dir, unit)
        units.append(unit)

        logger.info(
            "promise_pipeline.unit_done",
            unit_id=unit_id,
            coverage_rate=validation.coverage_rate,
            passed=validation.passed,
        )

    logger.info(
        "promise_pipeline.complete",
        total_units=len(units),
        passed=sum(1 for u in units if u.validation.passed),
        llm_usage=llm.usage_summary(),
    )
    return units
