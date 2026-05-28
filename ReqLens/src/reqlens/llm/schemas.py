"""Pydantic schemas for LLM structured outputs.

These models are passed as ``response_format`` to
``client.beta.chat.completions.parse(...)`` so Azure OpenAI returns
machine-readable, validated JSON.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from reqlens.domain.enums import (
    ConflictType,
    DependencyEdgeType,
    EvidenceStatus,
    NFRSubtype,
    QualityIssueType,
    RequirementKind,
    Severity,
)


# ── Extraction ──────────────────────────────────────────────────────

class CandidateRequirementLLM(BaseModel):
    """One atomic requirement proposed by the extraction agent."""

    temp_id: str = Field(description="Temporary candidate ID, e.g. CAND-001")
    text: str = Field(description="Atomic requirement statement")
    requirement_kind: RequirementKind
    nfr_subtype: NFRSubtype = NFRSubtype.not_applicable
    source_span_ids: list[str] = Field(
        default_factory=list,
        description="IDs of source spans that support this requirement",
    )
    stakeholders: list[str] = Field(default_factory=list)
    rationale: str = Field(description="Why this requirement was extracted")
    confidence: float = Field(ge=0.0, le=1.0)


class RequirementExtractionOutput(BaseModel):
    """Structured output of the extraction agent."""

    candidates: list[CandidateRequirementLLM]
    unresolved_questions: list[str] = Field(
        default_factory=list,
        description="Questions the extraction agent wants a stakeholder to answer",
    )


# ── Evidence Assessment ─────────────────────────────────────────────

class EvidenceAssessmentLLM(BaseModel):
    """Structured output of the evidence verification agent."""

    requirement_temp_id: str
    status: EvidenceStatus
    supporting_span_ids: list[str] = Field(default_factory=list)
    contradicting_span_ids: list[str] = Field(default_factory=list)
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class EvidenceBatchOutput(BaseModel):
    """Batch of evidence assessments."""

    assessments: list[EvidenceAssessmentLLM]


# ── Classification ──────────────────────────────────────────────────

class ClassificationLLM(BaseModel):
    """Structured output of the classification agent."""

    requirement_id: str
    kind: RequirementKind
    nfr_subtype: NFRSubtype = NFRSubtype.not_applicable
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str


class ClassificationBatchOutput(BaseModel):
    """Batch of classification results."""

    classifications: list[ClassificationLLM]


# ── Ambiguity / Quality ────────────────────────────────────────────

class QualityFindingLLM(BaseModel):
    """A single quality issue found by the ambiguity agent."""

    requirement_id: str
    issue_type: QualityIssueType
    severity: Severity
    explanation: str
    suggested_rewrite: str | None = None


class AmbiguityAnalysisOutput(BaseModel):
    """Structured output of the ambiguity agent."""

    findings: list[QualityFindingLLM]


# ── Dependency ──────────────────────────────────────────────────────

class DependencyEdgeCandidateLLM(BaseModel):
    """A proposed typed edge between two requirements."""

    source_requirement_id: str
    target_requirement_id: str
    edge_type: DependencyEdgeType
    explanation: str
    confidence: float = Field(ge=0.0, le=1.0)


class DependencyAnalysisOutput(BaseModel):
    """Structured output of the dependency agent."""

    edges: list[DependencyEdgeCandidateLLM]


# ── Consistency / Conflict ──────────────────────────────────────────

class ConflictFindingLLM(BaseModel):
    """A conflict between requirements."""

    conflict_type: ConflictType
    involved_requirement_ids: list[str]
    severity: Severity
    explanation: str
    suggested_resolution: str | None = None


class ConsistencyAnalysisOutput(BaseModel):
    """Structured output of the consistency agent."""

    conflicts: list[ConflictFindingLLM]


# ── Traceability ────────────────────────────────────────────────────

class TraceLinkLLM(BaseModel):
    """A proposed trace link."""

    source_id: str
    target_id: str
    link_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    explanation: str = ""


class TraceabilityOutput(BaseModel):
    """Structured output of the traceability agent."""

    links: list[TraceLinkLLM]


# ── Impact Analysis ─────────────────────────────────────────────────

class ImpactedNode(BaseModel):
    """A node affected by a change request."""

    node_id: str
    node_type: str
    impact_level: str = Field(description="direct or indirect")
    explanation: str


class ImpactAnalysisOutput(BaseModel):
    """Structured output of the impact agent."""

    change_summary: str
    directly_affected: list[ImpactedNode]
    indirectly_affected: list[ImpactedNode]
    suggested_review_tasks: list[str] = Field(default_factory=list)


# ── Composer ────────────────────────────────────────────────────────

class SRSSection(BaseModel):
    """One section of the generated SRS."""

    title: str
    content: str
    requirement_ids: list[str] = Field(
        default_factory=list,
        description="IDs of requirements referenced in this section",
    )


class SRSOutput(BaseModel):
    """Structured output of the composer agent."""

    sections: list[SRSSection]
    open_questions: list[str] = Field(default_factory=list)
    conflict_summary: str = ""
