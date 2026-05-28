"""Typed ID generation utilities."""

from __future__ import annotations

import uuid
from typing import Literal

# Prefix → entity mapping for human-readable IDs
_PREFIX_MAP = {
    "PRJ": "project",
    "DOC": "document",
    "SPN": "source_span",
    "CAND": "candidate",
    "REQ": "requirement",
    "EVD": "evidence",
    "CLF": "classification",
    "QF": "quality_finding",
    "GE": "graph_edge",
    "CF": "conflict",
    "TL": "trace_link",
    "RD": "review_decision",
    "RUN": "agent_run",
    "BR": "benchmark_run",
    "LC": "llm_call",
}

IDPrefix = Literal[
    "PRJ", "DOC", "SPN", "CAND", "REQ", "EVD", "CLF",
    "QF", "GE", "CF", "TL", "RD", "RUN", "BR", "LC",
]


def generate_id(prefix: IDPrefix) -> str:
    """Generate a prefixed UUID-based ID, e.g. ``REQ-a1b2c3d4``."""
    short = uuid.uuid4().hex[:8]
    return f"{prefix}-{short}"


def validate_id_prefix(id_value: str, expected_prefix: IDPrefix) -> bool:
    """Check that an ID string starts with the expected prefix."""
    return id_value.startswith(f"{expected_prefix}-")
