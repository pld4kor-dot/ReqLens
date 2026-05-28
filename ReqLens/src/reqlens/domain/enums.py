"""Domain enumerations used across the entire system."""

from __future__ import annotations

from enum import Enum


# ── Requirement kind ────────────────────────────────────────────────
class RequirementKind(str, Enum):
    functional = "functional"
    non_functional = "non_functional"
    constraint = "constraint"
    domain_assumption = "domain_assumption"
    business_rule = "business_rule"


class NFRSubtype(str, Enum):
    security = "security"
    privacy = "privacy"
    usability = "usability"
    reliability = "reliability"
    availability = "availability"
    performance = "performance"
    maintainability = "maintainability"
    portability = "portability"
    scalability = "scalability"
    compliance = "compliance"
    other = "other"
    not_applicable = "not_applicable"


# ── Requirement lifecycle ───────────────────────────────────────────
class RequirementStatus(str, Enum):
    candidate = "candidate"
    evidence_checked = "evidence_checked"
    classified = "classified"
    graph_linked = "graph_linked"
    review_required = "review_required"
    accepted = "accepted"
    auto_accepted = "auto_accepted"
    rejected = "rejected"
    included_in_srs = "included_in_srs"


class ReviewStatus(str, Enum):
    pending = "pending"
    accepted = "accepted"
    rejected = "rejected"
    needs_revision = "needs_revision"
    deferred = "deferred"


# ── Evidence ────────────────────────────────────────────────────────
class EvidenceStatus(str, Enum):
    entailed = "entailed"
    contradicted = "contradicted"
    insufficient_evidence = "insufficient_evidence"


# ── Graph edges ─────────────────────────────────────────────────────
class DependencyEdgeType(str, Enum):
    derived_from = "derived_from"
    refines = "refines"
    requires = "requires"
    conflicts_with = "conflicts_with"
    duplicates = "duplicates"
    constrains = "constrains"
    tested_by = "tested_by"
    regulated_by = "regulated_by"
    affected_by = "affected_by"
    realized_by = "realized_by"
    owned_by = "owned_by"


# ── Quality / ambiguity ────────────────────────────────────────────
class QualityIssueType(str, Enum):
    vague_term = "vague_term"
    missing_measurable_criterion = "missing_measurable_criterion"
    non_atomic = "non_atomic"
    weak_modality = "weak_modality"
    incomplete = "incomplete"
    untestable = "untestable"


class Severity(str, Enum):
    low = "low"
    medium = "medium"
    high = "high"
    critical = "critical"


# ── Conflict ────────────────────────────────────────────────────────
class ConflictType(str, Enum):
    contradiction = "contradiction"
    duplicate = "duplicate"
    numeric_inconsistency = "numeric_inconsistency"
    temporal_inconsistency = "temporal_inconsistency"
    scope_overlap = "scope_overlap"


# ── Trace link type ─────────────────────────────────────────────────
class TraceLinkType(str, Enum):
    source_to_requirement = "source_to_requirement"
    requirement_to_test = "requirement_to_test"
    requirement_to_design = "requirement_to_design"
    requirement_to_regulation = "requirement_to_regulation"
    requirement_to_goal = "requirement_to_goal"
    requirement_to_use_case = "requirement_to_use_case"


# ── Document type ───────────────────────────────────────────────────
class DocumentType(str, Enum):
    transcript = "transcript"
    srs = "srs"
    user_story = "user_story"
    test_case = "test_case"
    policy = "policy"
    change_request = "change_request"
    design = "design"
    regulation = "regulation"
    other = "other"


# ── Agent run status ────────────────────────────────────────────────
class AgentRunStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"
    partial = "partial"
