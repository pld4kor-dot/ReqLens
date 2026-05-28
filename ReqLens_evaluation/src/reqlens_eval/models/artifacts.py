"""Standalone Pydantic models mirroring the poisoned artifact JSON schemas.

These are defined independently from reqlens_benchmark_builder to avoid
coupling the evaluation pipeline to the builder's internal modules.
"""

from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


# ── Candidate pool (Track 1) ──────────────────────────────────────────────────

class CandidatePoolItem(BaseModel):
    """One item in the Track 1 candidate pool (gold or seeded fake)."""

    id: str
    text: str
    requirement_kind: str
    nfr_subtype: str = "not_applicable"
    origin: Literal["gold", "seeded_fake"]
    seed_item_id: str | None = None  # only set when origin == "seeded_fake"


# ── Seed registries ───────────────────────────────────────────────────────────

class HallucinationSeedItem(BaseModel):
    """Registry entry for one injected hallucinated requirement (Track 1)."""

    seed_item_id: str
    track_id: Literal["track1"] = "track1"
    seed_type: Literal["hallucinated_requirement"] = "hallucinated_requirement"
    requirement_id: str
    requirement_text: str
    requirement_kind: str
    nfr_subtype: str = "not_applicable"
    unsupported_reason: str = ""
    target_metrics: list[str] = Field(default_factory=lambda: ["UAR", "HRR"])
    judge_task_type: Literal["support_check", "hallucination_fate"] = "hallucination_fate"
    expected_behavior: str = "reject_or_mark_insufficient_evidence"
    acceptable_statuses: list[str] = Field(
        default_factory=lambda: ["rejected", "insufficient_evidence", "dropped", "absent"]
    )
    should_be_supported: bool = False


class DefectSeedItem(BaseModel):
    """Registry entry for one seeded defect (Track 2)."""

    seed_item_id: str
    track_id: Literal["track2"] = "track2"
    seed_type: Literal["defect_seed"] = "defect_seed"
    defect_type: Literal["contradiction", "duplicate"]
    defect_subtype: str
    original_req_ids: list[str]
    original_req_texts: list[str]
    injected_text: str
    injected_into_artifact_type: str
    defect_description: str
    judge_task_type: Literal["contradiction_check", "duplicate_check"]
    target_metrics: list[str] = Field(default_factory=lambda: ["DEFECT_LEAKAGE_RATE"])
    expected_behavior: str = "detect_and_resolve"
    should_leak_to_final_output: bool = False


# ── Top-level poisoned artifacts ──────────────────────────────────────────────

class PoisonedTrack1Artifact(BaseModel):
    """Complete Track 1 poisoned benchmark artifact loaded from disk.

    source_texts are unmodified; candidate_pool merges gold + injected fakes.
    """

    artifact_id: str
    unit_id: str
    origin: str
    track_id: Literal["track1"] = "track1"
    variant_id: str
    created_at: str = ""

    source_texts: list[dict[str, Any]]
    candidate_pool: list[CandidatePoolItem]
    gold_requirement_ids: list[str]
    seeded_fake_requirement_ids: list[str]
    seed_registry: list[HallucinationSeedItem]

    brief: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)


class PoisonedTrack2Artifact(BaseModel):
    """Complete Track 2 poisoned benchmark artifact loaded from disk.

    source_texts have seeded defects (contradictions / duplicates) injected.
    gold_requirements are the clean reference set.
    """

    artifact_id: str
    unit_id: str
    origin: str
    track_id: Literal["track2"] = "track2"
    variant_id: str
    created_at: str = ""

    source_texts: list[dict[str, Any]]
    gold_requirements: list[dict[str, Any]]
    seed_registry: list[DefectSeedItem]

    brief: dict[str, Any] = Field(default_factory=dict)
    metadata: dict[str, Any] = Field(default_factory=dict)
