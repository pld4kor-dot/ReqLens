"""Pydantic models for the benchmark unit schema.

These are kept intentionally minimal for v1.  The schema is designed so that
it can later be lightly transformed into ReqInOne SourceSpan / Requirement
objects without breaking changes.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

from pydantic import BaseModel, Field


# ── Vocabulary aligned with ReqInOne enums ──────────────────────────────────────

REQUIREMENT_KINDS = {
    "functional",
    "non_functional",
    "constraint",
    "domain_assumption",
    "business_rule",
}

NFR_SUBTYPES = {
    "security",
    "privacy",
    "usability",
    "reliability",
    "availability",
    "performance",
    "maintainability",
    "portability",
    "scalability",
    "compliance",
    "other",
    "not_applicable",
}

# ── Sub-models ───────────────────────────────────────────────────────────────────


class GoldRequirement(BaseModel):
    """One atomic gold-standard requirement entry."""

    id: str = Field(description="Unique ID within the benchmark unit, e.g. PROMISE_6_R001")
    text: str = Field(description="Normalized requirement text")
    raw_label: str | None = Field(
        default=None,
        description="Original dataset label (e.g. 'PE' from PROMISE)",
    )
    requirement_kind: str = Field(
        default="functional",
        description="One of: functional | non_functional | constraint | domain_assumption | business_rule",
    )
    nfr_subtype: str = Field(
        default="not_applicable",
        description="NFR sub-category when requirement_kind == non_functional",
    )
    # Provenance – where this requirement came from in the source document (PURE only)
    source_strategy: str | None = Field(
        default=None,
        description="'section', 'raw_chunk', 'both', or None for PROMISE",
    )
    source_region: str | None = Field(
        default=None,
        description="Section title or chunk ID from which this was extracted",
    )


class SourceArtifact(BaseModel):
    """One generated raw source text artifact."""

    type: str = Field(
        description="Artifact genre: interview_transcript | meeting_notes | email_thread"
    )
    title: str = Field(description="Short human-readable title for the artifact")
    text: str = Field(description="The full generated text of the artifact")


class CoverageEntry(BaseModel):
    """Per-requirement coverage verdict from the validator."""

    req_id: str
    supported: bool
    evidence_snippets: list[str] = Field(default_factory=list)
    reason: str = ""


class ValidationSummary(BaseModel):
    """Aggregated validation results for a benchmark unit."""

    coverage_rate: float = 0.0
    missing_req_ids: list[str] = Field(default_factory=list)
    unsupported_count: int = 0
    repair_rounds_used: int = 0
    # Detailed per-requirement verdicts
    coverage_entries: list[CoverageEntry] = Field(default_factory=list)
    unsupported_implied: list[dict[str, Any]] = Field(default_factory=list)
    passed: bool = False


class DocumentProfile(BaseModel):
    """Lightweight structural profile of a PURE document."""

    char_count: int = 0
    heading_density: int = 0
    shall_count: int = 0
    must_count: int = 0
    will_count: int = 0
    use_case_count: int = 0
    stakeholder_count: int = 0
    requirement_keyword_count: int = 0
    structure_strength: str = "unknown"  # strong | mixed | weak | unknown


class BenchmarkUnit(BaseModel):
    """Top-level output artifact for one benchmark entry."""

    id: str
    origin: str = Field(description="Dataset origin: PROMISE | PURE")
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    source_texts: list[SourceArtifact]
    gold_requirements: list[GoldRequirement]
    validation: ValidationSummary
    # Scenario brief used to generate source texts
    brief: dict[str, Any] = Field(default_factory=dict)
    # Pipeline metadata
    metadata: dict[str, Any] = Field(default_factory=dict)
