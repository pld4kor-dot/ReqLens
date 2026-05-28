"""Pydantic models for poisoned benchmark artifacts.

Two artifact types are produced:
- PoisonedTrack1Artifact  — candidate pool (gold + hallucinated fakes) for UAR/HRR evaluation
- PoisonedTrack2Artifact  — source_texts modified with seeded defects for defect leakage evaluation

Both share a common SeedItem registry that the evaluation framework consumes
to know exactly what was injected and what the correct system behaviour should be.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Seed item: Track 1 (hallucinated requirement) ────────────────────────────

class HallucinationSeedItem(BaseModel):
    """Registry entry for one injected hallucinated candidate requirement."""

    seed_item_id: str
    track_id: Literal["track1"] = "track1"
    seed_type: Literal["hallucinated_requirement"] = "hallucinated_requirement"

    # The fake requirement itself
    requirement_id: str          # e.g. PROMISE_1_FAKE_001
    requirement_text: str
    requirement_kind: str        # functional | non_functional | constraint | ...
    nfr_subtype: str = "not_applicable"

    # Why it is unsupported
    unsupported_reason: str = ""  # brief LLM-generated note on why it has no evidence

    # What metric(s) use this seed
    target_metrics: list[str] = Field(default_factory=lambda: ["UAR", "HRR"])
    judge_task_type: Literal["support_check", "hallucination_fate"] = "hallucination_fate"

    # What a correct system should do
    expected_behavior: str = "reject_or_mark_insufficient_evidence"
    acceptable_statuses: list[str] = Field(
        default_factory=lambda: ["rejected", "insufficient_evidence", "dropped", "absent"]
    )
    should_be_supported: bool = False


# ── Seed item: Track 2 (contradiction or duplicate) ──────────────────────────

class DefectSeedItem(BaseModel):
    """Registry entry for one seeded defect (contradiction or duplicate)."""

    seed_item_id: str
    track_id: Literal["track2"] = "track2"
    seed_type: Literal["defect_seed"] = "defect_seed"
    defect_type: Literal["contradiction", "duplicate"]
    defect_subtype: str  # value_conflict | semantic_conflict | exact_duplicate | paraphrase_duplicate

    # Which original gold requirement(s) are involved
    original_req_ids: list[str]
    original_req_texts: list[str]

    # The injected conflicting / duplicate text
    injected_text: str

    # Which source artifact it was injected into (type field of SourceArtifact)
    injected_into_artifact_type: str   # e.g. "meeting_notes" | "email_thread"

    # Description for judge prompt context
    defect_description: str

    # Judge task
    judge_task_type: Literal["contradiction_check", "duplicate_check"]
    target_metrics: list[str] = Field(default_factory=lambda: ["DEFECT_LEAKAGE_RATE"])

    # What a correct system should do
    expected_behavior: str = "detect_and_resolve"
    should_leak_to_final_output: bool = False


# ── Candidate pool item (Track 1) ─────────────────────────────────────────────

class CandidatePoolItem(BaseModel):
    """One item in the Track 1 candidate pool (gold or fake)."""

    id: str
    text: str
    requirement_kind: str
    nfr_subtype: str = "not_applicable"
    origin: Literal["gold", "seeded_fake"]
    seed_item_id: str | None = None   # only set when origin == seeded_fake


# ── Top-level poisoned artifacts ─────────────────────────────────────────────

class PoisonedTrack1Artifact(BaseModel):
    """Complete Track 1 poisoned benchmark artifact.

    source_texts are unmodified (same as the clean unit).
    candidate_pool merges gold requirements with hallucinated fakes.
    gold_requirement_ids lets the evaluator identify which candidates are real.
    """

    artifact_id: str
    unit_id: str
    origin: str                   # PROMISE | PURE
    track_id: Literal["track1"] = "track1"
    variant_id: str               # e.g. "hallu_v1"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Unmodified source evidence bundle
    source_texts: list[dict[str, Any]]

    # Merged pool: gold reqs + injected fakes (shuffled)
    candidate_pool: list[CandidatePoolItem]

    # For evaluator convenience
    gold_requirement_ids: list[str]
    seeded_fake_requirement_ids: list[str]

    # Full seed registry
    seed_registry: list[HallucinationSeedItem]

    # Pass-through from original unit
    brief: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PoisonedTrack2Artifact(BaseModel):
    """Complete Track 2 poisoned benchmark artifact.

    source_texts are modified — defects are injected into the raw text.
    gold_requirements are unchanged (the clean set).
    seed_registry describes exactly what was injected and where.
    """

    artifact_id: str
    unit_id: str
    origin: str                   # PROMISE | PURE
    track_id: Literal["track2"] = "track2"
    variant_id: str               # e.g. "defect_v1"
    created_at: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat()
    )

    # Modified source texts (defects already embedded)
    source_texts: list[dict[str, Any]]

    # Gold requirements (unchanged, for reference and judge context)
    gold_requirements: list[dict[str, Any]]

    # Full seed registry
    seed_registry: list[DefectSeedItem]

    # Pass-through from original unit
    brief: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
