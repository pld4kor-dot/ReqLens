"""PROMISE CSV loader.

Reads a PROMISE-style requirements CSV and groups rows by project ID into
``PromiseProject`` objects, each carrying normalized ``GoldRequirement`` entries.

Supports both headered and positional (no-header) CSVs.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import pandas as pd
import structlog

from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.schemas.benchmark_models import GoldRequirement

logger = structlog.get_logger(__name__)

# ── Label mapping ──────────────────────────────────────────────────────────────
# Maps raw PROMISE class labels → (requirement_kind, nfr_subtype)
# Raw label is always preserved in GoldRequirement.raw_label.
_LABEL_MAP: dict[str, tuple[str, str]] = {
    "F":  ("functional",     "not_applicable"),   # Functional
    "PE": ("non_functional", "performance"),       # Performance
    "SE": ("non_functional", "security"),          # Security
    "US": ("non_functional", "usability"),         # Usability
    "LF": ("non_functional", "usability"),         # Look & Feel → usability
    "A":  ("non_functional", "availability"),      # Availability
    "FT": ("non_functional", "reliability"),       # Fault Tolerance → reliability
    "SC": ("non_functional", "scalability"),       # Scalability
    "MN": ("non_functional", "maintainability"),   # Maintainability
    "L":  ("constraint",     "compliance"),        # Legal
    "O":  ("constraint",     "other"),             # Operational / Other
    "PO": ("constraint",     "other"),             # Process/Operational (alt token)
    "OP": ("constraint",     "other"),
}


def _normalize_label(raw: str) -> str:
    """Strip whitespace and upper-case for consistent lookup."""
    return re.sub(r"\s+", "", raw).upper()


def _map_label(raw: str) -> tuple[str, str]:
    return _LABEL_MAP.get(_normalize_label(raw), ("functional", "not_applicable"))


def _clean_text(text: str) -> str:
    """Light normalization: collapse whitespace, strip leading/trailing."""
    return re.sub(r"\s+", " ", str(text)).strip()


# ── Domain object ──────────────────────────────────────────────────────────────

@dataclass
class PromiseProject:
    project_id: str
    requirements: list[GoldRequirement] = field(default_factory=list)

    @property
    def label_distribution(self) -> dict[str, int]:
        dist: dict[str, int] = {}
        for r in self.requirements:
            dist[r.raw_label or "?"] = dist.get(r.raw_label or "?", 0) + 1
        return dist


# ── Loader ────────────────────────────────────────────────────────────────────

def load_promise_projects() -> list[PromiseProject]:
    """Load and group PROMISE requirements from CSV per configuration.

    Returns a list of ``PromiseProject`` objects, each containing a deduplicated
    set of gold requirements.  At most ``settings.promise_max_projects`` projects
    are returned, each capped at ``settings.promise_max_reqs_per_project`` rows.
    """
    settings = get_settings()
    path = settings.promise_input_path

    if not path.exists():
        raise FileNotFoundError(
            f"PROMISE CSV not found at '{path}'. "
            "Place the file there or update PROMISE_INPUT in .env"
        )

    # ── Load ─────────────────────────────────────────────────────────────────
    if settings.promise_has_header:
        df = pd.read_csv(path, encoding="utf-8", encoding_errors="replace")
        project_col = settings.promise_project_col
        text_col    = settings.promise_text_col
        label_col   = settings.promise_label_col
    else:
        df = pd.read_csv(
            path, header=None, encoding="utf-8", encoding_errors="replace"
        )
        if df.shape[1] < 3:
            raise ValueError(
                "Header-less PROMISE CSV must have at least 3 columns "
                "(ProjectID, RequirementText, class1)."
            )
        df.columns = [
            "ProjectID", "RequirementText", "class1",
            *[f"col_{i}" for i in range(3, df.shape[1])],
        ]
        project_col = "ProjectID"
        text_col    = "RequirementText"
        label_col   = "class1"

    # ── Validate columns ──────────────────────────────────────────────────────
    missing = {project_col, text_col, label_col} - set(df.columns)
    if missing:
        raise ValueError(
            f"Required columns missing from PROMISE CSV: {missing}. "
            f"Available columns: {list(df.columns)}"
        )

    # ── Drop blank text rows ──────────────────────────────────────────────────
    df = df.dropna(subset=[text_col])
    df[text_col] = df[text_col].apply(_clean_text)
    df = df[df[text_col].str.len() > 0]

    # ── Group by project ──────────────────────────────────────────────────────
    projects: list[PromiseProject] = []

    for project_id, group in df.groupby(project_col):
        if len(projects) >= settings.promise_max_projects:
            break

        group = (
            group
            .drop_duplicates(subset=[text_col])
            .head(settings.promise_max_reqs_per_project)
            .reset_index(drop=True)
        )

        reqs: list[GoldRequirement] = []
        for idx, row in group.iterrows():
            text      = _clean_text(row[text_col])
            raw_label = _clean_text(str(row[label_col]))
            kind, nfr = _map_label(raw_label)
            reqs.append(
                GoldRequirement(
                    id=f"PROMISE_{project_id}_R{int(idx) + 1:03d}",
                    text=text,
                    raw_label=raw_label,
                    requirement_kind=kind,
                    nfr_subtype=nfr,
                )
            )

        projects.append(
            PromiseProject(project_id=str(project_id), requirements=reqs)
        )
        logger.info(
            "promise_loader.project_loaded",
            project_id=project_id,
            req_count=len(reqs),
            label_dist=projects[-1].label_distribution,
        )

    logger.info("promise_loader.done", total_projects=len(projects))
    return projects
