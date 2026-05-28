"""Output writers for benchmark units.

Saves each ``BenchmarkUnit`` as:
  <output_dir>/<origin>/<unit_id>/
      unit.json           — full benchmark unit (Pydantic JSON)
      validation.json     — validation summary only (quick reference)
      txt/
          01_interview_transcript.txt
          02_meeting_notes.txt
          03_email_thread.txt
"""

from __future__ import annotations

import json
from pathlib import Path

import structlog

from reqlens_benchmark_builder.schemas.benchmark_models import BenchmarkUnit

logger = structlog.get_logger(__name__)


def _ensure(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_benchmark_unit(base_dir: Path, unit: BenchmarkUnit) -> None:
    """Write all output files for a single benchmark unit.

    Args:
        base_dir: Root output directory for this unit.
                  Typically ``<output_dir>/promise/<unit_id>`` or
                  ``<output_dir>/pure/<unit_id>``.
        unit:     The fully populated ``BenchmarkUnit``.
    """
    _ensure(base_dir)

    # ── 1) Full JSON ──────────────────────────────────────────────────────────
    unit_json_path = base_dir / "unit.json"
    unit_json_path.write_text(
        unit.model_dump_json(indent=2), encoding="utf-8"
    )
    logger.info("writer.unit_json", path=str(unit_json_path))

    # ── 2) Validation summary (handy for quick inspection) ────────────────────
    val_json_path = base_dir / "validation.json"
    val_json_path.write_text(
        json.dumps(unit.validation.model_dump(), indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    # ── 3) TXT artifacts ─────────────────────────────────────────────────────
    txt_dir = _ensure(base_dir / "txt")
    for i, artifact in enumerate(unit.source_texts, start=1):
        safe_type = artifact.type.replace("/", "_").replace(" ", "_")
        filename  = f"{i:02d}_{safe_type}.txt"
        content   = (
            f"TYPE  : {artifact.type}\n"
            f"TITLE : {artifact.title}\n"
            f"UNIT  : {unit.id}\n"
            + "=" * 60 + "\n\n"
            + artifact.text
        )
        (txt_dir / filename).write_text(content, encoding="utf-8")

    logger.info(
        "writer.done",
        unit_id=unit.id,
        output_dir=str(base_dir),
        artifacts=len(unit.source_texts),
        gold_reqs=len(unit.gold_requirements),
        coverage_rate=unit.validation.coverage_rate,
        passed=unit.validation.passed,
    )
