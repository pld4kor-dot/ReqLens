"""Writers for poisoned benchmark artifacts.

Saves each artifact under:
  <output_dir>/poisoned/track1/<unit_id>/
      poisoned_track1_<variant_id>.json
  <output_dir>/poisoned/track2/<unit_id>/
      poisoned_track2_<variant_id>.json

A lightweight manifest file is also maintained at:
  <output_dir>/poisoned/manifest.json
listing all produced artifacts with quality summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from reqlens_benchmark_builder.poisoning.schemas import (
    PoisonedTrack1Artifact,
    PoisonedTrack2Artifact,
)

logger = structlog.get_logger(__name__)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_track1_artifact(
    base_output_dir: Path,
    artifact: PoisonedTrack1Artifact,
) -> Path:
    """Write a Track 1 poisoned artifact to disk.

    Returns the path of the written JSON file.
    """
    unit_dir = _ensure(
        base_output_dir / "poisoned" / "track1" / artifact.unit_id
    )
    out_path = unit_dir / f"poisoned_track1_{artifact.variant_id}.json"
    out_path.write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )
    logger.info(
        "writer.track1_artifact",
        unit_id=artifact.unit_id,
        path=str(out_path),
        pool_size=len(artifact.candidate_pool),
        fake_count=len(artifact.seed_registry),
    )
    return out_path


def write_track2_artifact(
    base_output_dir: Path,
    artifact: PoisonedTrack2Artifact,
) -> Path:
    """Write a Track 2 poisoned artifact to disk.

    Returns the path of the written JSON file.
    """
    unit_dir = _ensure(
        base_output_dir / "poisoned" / "track2" / artifact.unit_id
    )
    out_path = unit_dir / f"poisoned_track2_{artifact.variant_id}.json"
    out_path.write_text(
        artifact.model_dump_json(indent=2),
        encoding="utf-8",
    )

    contradiction_count = sum(
        1 for s in artifact.seed_registry if s.defect_type == "contradiction"
    )
    duplicate_count = sum(
        1 for s in artifact.seed_registry if s.defect_type == "duplicate"
    )

    logger.info(
        "writer.track2_artifact",
        unit_id=artifact.unit_id,
        path=str(out_path),
        contradictions=contradiction_count,
        duplicates=duplicate_count,
        total_seeds=len(artifact.seed_registry),
    )
    return out_path


def update_poison_manifest(
    base_output_dir: Path,
    entry: dict,
) -> None:
    """Append or update an entry in the poisoning manifest JSON.

    The manifest keeps a flat list of all produced poisoned artifacts
    so the evaluation framework can discover them without scanning directories.
    """
    manifest_path = _ensure(base_output_dir / "poisoned") / "manifest.json"

    if manifest_path.exists():
        try:
            existing = json.loads(manifest_path.read_text(encoding="utf-8"))
        except (json.JSONDecodeError, OSError):
            existing = {"artifacts": []}
    else:
        existing = {"artifacts": []}

    # Replace existing entry for same unit + track + variant, or append
    artifacts: list[dict] = existing.get("artifacts", [])
    updated = False
    for i, a in enumerate(artifacts):
        if (
            a.get("unit_id") == entry["unit_id"]
            and a.get("track_id") == entry["track_id"]
            and a.get("variant_id") == entry["variant_id"]
        ):
            artifacts[i] = entry
            updated = True
            break
    if not updated:
        artifacts.append(entry)

    existing["artifacts"] = artifacts
    manifest_path.write_text(
        json.dumps(existing, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )
