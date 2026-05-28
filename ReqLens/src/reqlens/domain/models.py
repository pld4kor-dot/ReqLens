"""Core domain models Pydantic objects shared across layers."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from pydantic import BaseModel, Field

from reqlens.domain.enums import (
    AgentRunStatus,
    ConflictType,
    DependencyEdgeType,
    DocumentType,
    EvidenceStatus,
    NFRSubtype,
    QualityIssueType,
    RequirementKind,
    RequirementStatus,
    ReviewStatus,
    Severity,
    TraceLinkType,
)
from reqlens.domain.ids import generate_id


# -- Project ---------------------------------------------------------
class Project(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("PRJ"))
    name: str
    description: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)
    # Set to the parent project ID for forked (versioned) projects; None for originals.
    parent_project_id: str | None = None


# -- Document --------------------------------------------------------
class Document(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("DOC"))
    project_id: str
    filename: str
    document_type: DocumentType = DocumentType.other
    content: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Source Span -----------------------------------------------------
class SourceSpan(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("SPN"))
    project_id: str
    document_id: str
    span_index: int
    text: str
    char_start: int
    char_end: int
    speaker: str | None = None
    section_title: str | None = None
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Requirement Candidate -------------------------------------------
class RequirementCandidate(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("CAND"))
    project_id: str
    text: str
    requirement_kind: RequirementKind
    nfr_subtype: NFRSubtype = NFRSubtype.not_applicable
    source_span_ids: list[str] = Field(default_factory=list)
    stakeholders: list[str] = Field(default_factory=list)
    rationale: str = ""
    confidence: float = 0.0
    status: RequirementStatus = RequirementStatus.candidate
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Requirement (promoted from candidate) ---------------------------
class Requirement(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("REQ"))
    project_id: str
    text: str
    kind: RequirementKind
    nfr_subtype: NFRSubtype = NFRSubtype.not_applicable
    status: RequirementStatus = RequirementStatus.evidence_checked
    review_status: ReviewStatus = ReviewStatus.pending
    quality_score: float | None = None
    created_from_candidate_id: str | None = None
    source_span_ids: list[str] = Field(default_factory=list)
    embedding: list[float] | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)
    updated_at: datetime = Field(default_factory=datetime.utcnow)


# -- Evidence Assessment ---------------------------------------------
class EvidenceAssessment(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("EVD"))
    project_id: str
    requirement_candidate_id: str
    status: EvidenceStatus
    supporting_span_ids: list[str] = Field(default_factory=list)
    contradicting_span_ids: list[str] = Field(default_factory=list)
    explanation: str = ""
    confidence: float = 0.0
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Classification Result -------------------------------------------
class ClassificationResult(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("CLF"))
    requirement_id: str
    kind: RequirementKind
    nfr_subtype: NFRSubtype = NFRSubtype.not_applicable
    confidence: float = 0.0
    rationale: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Quality Finding -------------------------------------------------
class QualityFinding(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("QF"))
    requirement_id: str
    issue_type: QualityIssueType
    severity: Severity
    explanation: str
    suggested_rewrite: str | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Graph Edge ------------------------------------------------------
class GraphEdge(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("GE"))
    project_id: str
    source_node_id: str
    target_node_id: str
    edge_type: DependencyEdgeType
    confidence: float = 0.0
    created_by: str = ""  # agent name
    review_status: ReviewStatus = ReviewStatus.pending
    explanation: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Conflict Finding -----------------------------------------------
class ConflictFinding(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("CF"))
    project_id: str
    conflict_type: ConflictType
    involved_requirement_ids: list[str]
    severity: Severity
    explanation: str
    suggested_resolution: str | None = None
    resolved: bool = False
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Trace Link ------------------------------------------------------
class TraceLink(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("TL"))
    project_id: str
    source_id: str
    target_id: str
    link_type: TraceLinkType
    confidence: float = 0.0
    review_status: ReviewStatus = ReviewStatus.pending
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Review Decision -------------------------------------------------
class ReviewDecision(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("RD"))
    requirement_id: str
    decision: ReviewStatus
    reviewer: str = "human"
    comment: str = ""
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Agent Run -------------------------------------------------------
class AgentRun(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("RUN"))
    project_id: str
    agent_name: str
    status: AgentRunStatus = AgentRunStatus.pending
    created_ids: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    errors: list[str] = Field(default_factory=list)
    started_at: datetime | None = None
    finished_at: datetime | None = None
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- LLM Call Log ----------------------------------------------------
class LLMCallLog(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("LC"))
    project_id: str | None = None
    agent_name: str = ""
    model: str = ""
    prompt_hash: str = ""
    input_tokens: int = 0
    output_tokens: int = 0
    latency_ms: int = 0
    cost_estimate: float = 0.0
    status: str = "ok"
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Benchmark Run ---------------------------------------------------
class BenchmarkRun(BaseModel):
    id: str = Field(default_factory=lambda: generate_id("BR"))
    benchmark_type: str
    dataset: str = ""
    metrics: dict[str, Any] = Field(default_factory=dict)
    config: dict[str, Any] = Field(default_factory=dict)
    created_at: datetime = Field(default_factory=datetime.utcnow)


# -- Acceptance Policy -----------------------------------------------
#
#  "LLMs propose. Evidence gates. Graph validates. Humans approve."
#
#  A requirement can enter the final SRS only when:
#    1. It has at least one source span.
#    2. Evidence agent says ENTAILED or HUMAN_APPROVED.
#    3. It has no unresolved severe conflict.
#    4. It passes minimum quality checks.
#    5. It has review status ACCEPTED or AUTO_ACCEPTED.


def requirement_eligible_for_srs(
    req: Requirement,
    evidence: EvidenceAssessment | None,
    conflicts: list[ConflictFinding],
    quality_findings: list[QualityFinding],
) -> tuple[bool, list[str]]:
    """Return (is_eligible, list_of_blocking_reasons)."""
    reasons: list[str] = []

    # Rule 1: must have source spans
    if not req.source_span_ids:
        reasons.append("No source spans linked.")

    # Rule 2: evidence must be entailed or human-approved
    if evidence is None:
        reasons.append("No evidence assessment found.")
    elif evidence.status not in (EvidenceStatus.entailed,):
        if req.review_status != ReviewStatus.accepted:
            reasons.append(
                f"Evidence status is {evidence.status.value} and not human-approved."
            )

    # Rule 3: no unresolved severe conflicts
    severe_unresolved = [
        c for c in conflicts
        if c.severity in (Severity.high, Severity.critical) and not c.resolved
    ]
    if severe_unresolved:
        reasons.append(
            f"{len(severe_unresolved)} unresolved severe conflict(s)."
        )

    # Rule 4: quality checks
    critical_findings = [
        f for f in quality_findings if f.severity == Severity.critical
    ]
    if critical_findings:
        reasons.append(
            f"{len(critical_findings)} critical quality finding(s)."
        )

    # Rule 5: review status
    if req.review_status not in (ReviewStatus.accepted,):
        # auto-accept if all other checks pass and gate is disabled
        if reasons:  # other issues exist
            reasons.append(f"Review status is {req.review_status.value}.")

    return (len(reasons) == 0, reasons)
