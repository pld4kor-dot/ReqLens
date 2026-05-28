"""Models for experiment runs, system outputs, and evaluation results."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Per-candidate decision (Track 1) ─────────────────────────────────────────

class CandidateDecision(BaseModel):
    """Decision for a single candidate requirement in Track 1."""

    candidate_id: str
    status: Literal["accepted", "rejected", "uncertain"]
    confidence: float = 0.0
    # Where the decision came from
    signal_source: Literal[
        "system_metadata", "system_output", "llm_judge", "heuristic"
    ] = "system_output"
    explanation: str = ""


# ── System outputs ────────────────────────────────────────────────────────────

class Track1SystemOutput(BaseModel):
    """Raw output from a system adapter for a Track 1 artifact."""

    unit_id: str
    artifact_id: str
    system_id: str
    decisions: list[CandidateDecision]
    execution_time_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


class ExtractedRequirement(BaseModel):
    """A single requirement extracted by a system (Track 2)."""

    id: str
    text: str
    requirement_kind: str = "functional"
    nfr_subtype: str = "not_applicable"
    metadata: dict[str, Any] = Field(default_factory=dict)


class Track2SystemOutput(BaseModel):
    """Raw output from a system adapter for a Track 2 artifact."""

    unit_id: str
    artifact_id: str
    system_id: str
    extracted_requirements: list[ExtractedRequirement]
    execution_time_s: float = 0.0
    metadata: dict[str, Any] = Field(default_factory=dict)


# ── Judge outputs ─────────────────────────────────────────────────────────────

class JudgeVerdict(BaseModel):
    """LLM judge verdict for one seed item or candidate."""

    target_id: str  # seed_item_id (Track 2) or candidate_id (Track 1)
    verdict: Literal["accepted", "rejected", "leaked", "detected", "uncertain"]
    confidence: float = 0.0
    reasoning: str = ""
    prompt_family: str = ""  # A | B | C | D


# ── Per-unit evaluation results ───────────────────────────────────────────────

class Track1UnitResult(BaseModel):
    """Evaluation result for one Track 1 artifact × one system."""

    unit_id: str
    artifact_id: str
    system_id: str
    total_candidates: int
    gold_count: int
    fake_count: int
    final_decisions: list[CandidateDecision]
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)
    hallucinations_accepted: int = 0
    hallucinations_rejected: int = 0
    # NEW: gold requirement tracking
    golds_rejected: int = 0   # gold reqs incorrectly rejected (over-aggressive filter)
    golds_accepted: int = 0   # gold reqs correctly accepted
    uar: float = 0.0   # Unsupported Acceptance Rate  (lower = better)
    hrr: float = 0.0   # Hallucination Rejection Rate (higher = better)
    grr: float = 0.0   # Gold Rejection Rate          (lower = better)


class Track2UnitResult(BaseModel):
    """Evaluation result for one Track 2 artifact × one system."""

    unit_id: str
    artifact_id: str
    system_id: str
    total_seeds: int
    seeds_leaked: int = 0
    seeds_detected: int = 0
    defect_leakage_rate: float = 0.0  # lower = better
    judge_verdicts: list[JudgeVerdict] = Field(default_factory=list)


# ── Aggregate summaries ───────────────────────────────────────────────────────

class SystemEvalSummary(BaseModel):
    """Aggregate metrics for one system across all evaluated units."""

    system_id: str
    track_id: str
    unit_count: int = 0
    # Track 1
    mean_uar: float | None = None
    mean_hrr: float | None = None
    mean_grr: float | None = None   # NEW: Gold Rejection Rate
    # Track 2
    mean_dlr: float | None = None
    unit_results: list[dict[str, Any]] = Field(default_factory=list)


class EvalRunReport(BaseModel):
    """Full evaluation run report, serialised to outputs/."""

    run_id: str
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )
    systems_evaluated: list[str]
    tracks_evaluated: list[str]
    track1_summaries: list[SystemEvalSummary] = Field(default_factory=list)
    track2_summaries: list[SystemEvalSummary] = Field(default_factory=list)
    metadata: dict[str, Any] = Field(default_factory=dict)
