"""Poisoning pipeline orchestrator.

Scans the existing silver benchmark outputs and runs:
  - Track 1 poisoner  → hallucinated candidate pool artifacts
  - Track 2 poisoner  → defect-seeded source artifacts

Only units that pass the quality threshold (coverage_rate >= min_coverage_rate
and unsupported_count <= max_unsupported_count) are poisoned.

Usage (from benchmark_builder CLI):
    benchmark-builder poison --track 1
    benchmark-builder poison --track 2
    benchmark-builder poison --track both
    benchmark-builder poison --track both --unit PROMISE_1
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.poisoning.track1_poisoner import poison_track1
from reqlens_benchmark_builder.poisoning.track2_poisoner import poison_track2
from reqlens_benchmark_builder.poisoning.writer import (
    update_poison_manifest,
    write_track1_artifact,
    write_track2_artifact,
)

logger = structlog.get_logger(__name__)


def _load_unit(unit_json_path: Path) -> dict | None:
    """Load and return a unit.json as a dict, or None on failure."""
    try:
        return json.loads(unit_json_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.error("pipeline.load_unit_failed", path=str(unit_json_path), error=str(exc))
        return None


def _passes_quality_filter(unit: dict, settings) -> bool:
    """Return True if this benchmark unit meets the silver quality threshold."""
    validation = unit.get("validation", {})
    coverage_rate = validation.get("coverage_rate", 0.0)
    unsupported_count = validation.get("unsupported_count", 999)
    passed = (
        coverage_rate >= settings.min_coverage_rate
        and unsupported_count <= settings.max_unsupported_count
    )
    if not passed:
        logger.info(
            "pipeline.unit_filtered_out",
            unit_id=unit.get("id"),
            coverage_rate=coverage_rate,
            unsupported_count=unsupported_count,
        )
    return passed


def _iter_unit_paths(output_dir: Path, only_unit: str | None) -> list[Path]:
    """Yield all unit.json paths under output_dir, optionally filtered by unit ID."""
    paths: list[Path] = []
    for sub in ("pure", "promise"):
        sub_dir = output_dir / sub
        if not sub_dir.exists():
            continue
        for unit_dir in sorted(sub_dir.iterdir()):
            if not unit_dir.is_dir():
                continue
            unit_json = unit_dir / "unit.json"
            if not unit_json.exists():
                continue
            if only_unit and unit_dir.name != only_unit:
                continue
            paths.append(unit_json)
    return paths


def run_poison_pipeline(
    track: str = "both",
    only_unit: str | None = None,
    hallucination_count: int = 10,
    contradiction_count: int = 5,
    duplicate_count: int = 5,
    variant_id_t1: str = "hallu_v1",
    variant_id_t2: str = "defect_v1",
) -> dict[str, int]:
    """Run the poisoning pipeline over silver benchmark outputs.

    Args:
        track:               "1" | "2" | "both"
        only_unit:           If set, poison only this unit ID (e.g. "PROMISE_1").
        hallucination_count: Track 1 — number of fake requirements to inject.
        contradiction_count: Track 2 — number of contradictions to seed.
        duplicate_count:     Track 2 — number of duplicates to seed.
        variant_id_t1:       Variant tag for Track 1 artifacts.
        variant_id_t2:       Variant tag for Track 2 artifacts.

    Returns:
        Dict with counts: {"processed", "t1_success", "t2_success", "skipped", "failed"}
    """
    settings = get_settings()
    llm = AzureOpenAIClient()
    output_dir = settings.output_path

    unit_paths = _iter_unit_paths(output_dir, only_unit)
    if not unit_paths:
        logger.warning("pipeline.no_units_found", output_dir=str(output_dir))
        return {"processed": 0, "t1_success": 0, "t2_success": 0, "skipped": 0, "failed": 0}

    counts = {"processed": 0, "t1_success": 0, "t2_success": 0, "skipped": 0, "failed": 0}

    run_t1 = track in ("1", "both")
    run_t2 = track in ("2", "both")

    for unit_path in unit_paths:
        unit = _load_unit(unit_path)
        if unit is None:
            counts["failed"] += 1
            continue
        
        
        if not _passes_quality_filter(unit, settings):
            counts["skipped"] += 1
            continue


        unit_id = unit.get("id", unit_path.parent.name)
        origin = unit.get("origin", "UNKNOWN")

        counts["processed"] += 1

        source_texts = unit.get("source_texts", [])
        gold_requirements = unit.get("gold_requirements", [])
        brief = unit.get("brief", {})
        metadata = unit.get("metadata", {})

        if not source_texts or not gold_requirements:
            logger.warning("pipeline.unit_missing_data", unit_id=unit_id)
            counts["failed"] += 1
            continue

        logger.info("pipeline.processing_unit", unit_id=unit_id, track=track)

        # ── Track 1 ───────────────────────────────────────────────────────────
        if run_t1:
            # Scale hallucination count proportionally to gold pool size:
            # ~20% of golds, with floor of 5 and cap of 15.
            effective_hallu_count = max(5, min(15, len(gold_requirements) // 5))
            t1_artifact = poison_track1(
                llm=llm,
                unit_id=unit_id,
                origin=origin,
                source_texts=source_texts,
                gold_requirements=gold_requirements,
                brief=brief,
                metadata=metadata,
                hallucination_count=effective_hallu_count,
                variant_id=variant_id_t1,
            )
            if t1_artifact is not None:
                out_path = write_track1_artifact(output_dir, t1_artifact)
                update_poison_manifest(
                    output_dir,
                    {
                        "unit_id": unit_id,
                        "origin": origin,
                        "track_id": "track1",
                        "variant_id": variant_id_t1,
                        "artifact_id": t1_artifact.artifact_id,
                        "path": str(out_path),
                        "pool_size": len(t1_artifact.candidate_pool),
                        "fake_count": len(t1_artifact.seed_registry),
                    },
                )
                counts["t1_success"] += 1
            else:
                logger.error("pipeline.track1_failed", unit_id=unit_id)
                counts["failed"] += 1

        # ── Track 2 ───────────────────────────────────────────────────────────
        if run_t2:
            t2_artifact = poison_track2(
                llm=llm,
                unit_id=unit_id,
                origin=origin,
                source_texts=source_texts,
                gold_requirements=gold_requirements,
                brief=brief,
                metadata=metadata,
                contradiction_count=contradiction_count,
                duplicate_count=duplicate_count,
                variant_id=variant_id_t2,
            )
            if t2_artifact is not None:
                out_path = write_track2_artifact(output_dir, t2_artifact)
                contradiction_count_done = sum(
                    1 for s in t2_artifact.seed_registry if s.defect_type == "contradiction"
                )
                duplicate_count_done = sum(
                    1 for s in t2_artifact.seed_registry if s.defect_type == "duplicate"
                )
                update_poison_manifest(
                    output_dir,
                    {
                        "unit_id": unit_id,
                        "origin": origin,
                        "track_id": "track2",
                        "variant_id": variant_id_t2,
                        "artifact_id": t2_artifact.artifact_id,
                        "path": str(out_path),
                        "contradiction_count": contradiction_count_done,
                        "duplicate_count": duplicate_count_done,
                        "total_seeds": len(t2_artifact.seed_registry),
                    },
                )
                counts["t2_success"] += 1
            else:
                logger.error("pipeline.track2_failed", unit_id=unit_id)
                counts["failed"] += 1

    logger.info("pipeline.poison_complete", **counts)
    return counts
