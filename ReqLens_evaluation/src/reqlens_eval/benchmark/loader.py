"""Load poisoned benchmark artifacts from disk.

Expected directory layout (produced by reqlens_benchmark_builder):

    <benchmark_output_dir>/
        poisoned/
            manifest.json
            track1/
                <unit_id>/
                    poisoned_track1_<variant_id>.json
            track2/
                <unit_id>/
                    poisoned_track2_<variant_id>.json
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from reqlens_eval.models.artifacts import (
    PoisonedTrack1Artifact,
    PoisonedTrack2Artifact,
)

logger = structlog.get_logger(__name__)


def load_track1_artifacts(
    benchmark_output_dir: Path,
    unit_id: str | None = None,
    variant_id: str | None = None,
) -> list[PoisonedTrack1Artifact]:
    """Load Track 1 poisoned artifacts, optionally filtered by unit_id / variant_id."""
    track1_dir = benchmark_output_dir / "poisoned" / "track1"
    if not track1_dir.exists():
        logger.warning("loader.track1_dir_missing", path=str(track1_dir))
        return []

    artifacts: list[PoisonedTrack1Artifact] = []
    for unit_dir in sorted(track1_dir.iterdir()):
        if not unit_dir.is_dir():
            continue
        if unit_id and unit_dir.name != unit_id:
            continue
        for json_file in sorted(unit_dir.glob("poisoned_track1_*.json")):
            if variant_id and f"_{variant_id}.json" not in json_file.name:
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                artifact = PoisonedTrack1Artifact.model_validate(data)
                artifacts.append(artifact)
                logger.info(
                    "loader.track1_loaded",
                    unit_id=artifact.unit_id,
                    variant_id=artifact.variant_id,
                    pool_size=len(artifact.candidate_pool),
                    fake_count=len(artifact.seeded_fake_requirement_ids),
                )
            except Exception as exc:
                logger.error("loader.track1_load_failed", path=str(json_file), error=str(exc))

    return artifacts


def load_track2_artifacts(
    benchmark_output_dir: Path,
    unit_id: str | None = None,
    variant_id: str | None = None,
) -> list[PoisonedTrack2Artifact]:
    """Load Track 2 poisoned artifacts, optionally filtered by unit_id / variant_id."""
    track2_dir = benchmark_output_dir / "poisoned" / "track2"
    if not track2_dir.exists():
        logger.warning("loader.track2_dir_missing", path=str(track2_dir))
        return []

    artifacts: list[PoisonedTrack2Artifact] = []
    for unit_dir in sorted(track2_dir.iterdir()):
        if not unit_dir.is_dir():
            continue
        if unit_id and unit_dir.name != unit_id:
            continue
        for json_file in sorted(unit_dir.glob("poisoned_track2_*.json")):
            if variant_id and f"_{variant_id}.json" not in json_file.name:
                continue
            try:
                data = json.loads(json_file.read_text(encoding="utf-8"))
                artifact = PoisonedTrack2Artifact.model_validate(data)
                artifacts.append(artifact)
                logger.info(
                    "loader.track2_loaded",
                    unit_id=artifact.unit_id,
                    variant_id=artifact.variant_id,
                    seed_count=len(artifact.seed_registry),
                )
            except Exception as exc:
                logger.error("loader.track2_load_failed", path=str(json_file), error=str(exc))

    return artifacts


def load_manifest(benchmark_output_dir: Path) -> dict:
    """Load the poisoning manifest written by the benchmark builder."""
    manifest_path = benchmark_output_dir / "poisoned" / "manifest.json"
    if not manifest_path.exists():
        logger.warning("loader.manifest_missing", path=str(manifest_path))
        return {"artifacts": []}
    try:
        return json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.error("loader.manifest_load_failed", error=str(exc))
        return {"artifacts": []}
