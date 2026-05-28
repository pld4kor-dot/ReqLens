"""Repository layer CRUD operations for domain objects """

from __future__ import annotations

from datetime import datetime
from typing import Sequence

import structlog
from sqlalchemy.orm import Session

from reqlens.domain.models import (
    AgentRun,
    BenchmarkRun,
    ClassificationResult,
    ConflictFinding,
    Document,
    EvidenceAssessment,
    GraphEdge,
    LLMCallLog,
    Project,
    QualityFinding,
    Requirement,
    RequirementCandidate,
    ReviewDecision,
    SourceSpan,
    TraceLink,
)
from reqlens.storage.db import (
    AgentRunRow,
    BenchmarkRunRow,
    ClassificationResultRow,
    ConflictFindingRow,
    DocumentRow,
    EvidenceAssessmentRow,
    GraphEdgeRow,
    LLMCallLogRow,
    ProjectRow,
    QualityFindingRow,
    RequirementCandidateRow,
    RequirementRow,
    ReviewDecisionRow,
    SourceSpanRow,
    TraceLinkRow,
)

logger = structlog.get_logger(__name__)


# -- Generic helpers -------------------------------------------------

def _to_dict(model) -> dict:
    """Convert Pydantic model to dict, handling datetime serialization."""
    data = model.model_dump()
    for k, v in data.items():
        if isinstance(v, datetime):
            data[k] = v
    return data


# -- Project ---------------------------------------------------------

class ProjectRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, project: Project) -> Project:
        row = ProjectRow(**_to_dict(project))
        self.session.add(row)
        self.session.commit()
        return project

    def get(self, project_id: str) -> Project | None:
        row = self.session.get(ProjectRow, project_id)
        if row is None:
            return None
        return Project(
            id=row.id,
            name=row.name,
            description=row.description or "",
            created_at=row.created_at,
            parent_project_id=row.parent_project_id,
        )

    def list_all(self) -> list[Project]:
        rows = self.session.query(ProjectRow).order_by(ProjectRow.created_at.desc()).all()
        return [
            Project(
                id=r.id,
                name=r.name,
                description=r.description or "",
                created_at=r.created_at,
                parent_project_id=r.parent_project_id,
            )
            for r in rows
        ]


# -- Document --------------------------------------------------------

class DocumentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, doc: Document) -> Document:
        row = DocumentRow(**_to_dict(doc))
        self.session.add(row)
        self.session.commit()
        return doc

    def list_by_project(self, project_id: str) -> list[Document]:
        rows = self.session.query(DocumentRow).filter_by(project_id=project_id).all()
        return [
            Document(
                id=r.id, project_id=r.project_id, filename=r.filename,
                document_type=r.document_type, content=r.content or "",
                created_at=r.created_at,
            )
            for r in rows
        ]


# -- Source Span -----------------------------------------------------

class SourceSpanRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, spans: Sequence[SourceSpan]) -> None:
        for span in spans:
            row = SourceSpanRow(**_to_dict(span))
            self.session.add(row)
        self.session.commit()

    def list_by_project(self, project_id: str) -> list[SourceSpan]:
        rows = self.session.query(SourceSpanRow).filter_by(project_id=project_id).order_by(SourceSpanRow.span_index).all()
        return [
            SourceSpan(
                id=r.id, project_id=r.project_id, document_id=r.document_id,
                span_index=r.span_index, text=r.text, char_start=r.char_start,
                char_end=r.char_end, speaker=r.speaker,
                section_title=r.section_title, embedding=r.embedding,
                created_at=r.created_at,
            )
            for r in rows
        ]

    def list_by_document(self, document_id: str) -> list[SourceSpan]:
        rows = self.session.query(SourceSpanRow).filter_by(document_id=document_id).order_by(SourceSpanRow.span_index).all()
        return [
            SourceSpan(
                id=r.id, project_id=r.project_id, document_id=r.document_id,
                span_index=r.span_index, text=r.text, char_start=r.char_start,
                char_end=r.char_end, speaker=r.speaker,
                section_title=r.section_title, embedding=r.embedding,
                created_at=r.created_at,
            )
            for r in rows
        ]

    def get_many(self, span_ids: list[str]) -> list[SourceSpan]:
        rows = self.session.query(SourceSpanRow).filter(SourceSpanRow.id.in_(span_ids)).all()
        return [
            SourceSpan(
                id=r.id, project_id=r.project_id, document_id=r.document_id,
                span_index=r.span_index, text=r.text, char_start=r.char_start,
                char_end=r.char_end, speaker=r.speaker,
                section_title=r.section_title, embedding=r.embedding,
                created_at=r.created_at,
            )
            for r in rows
        ]


# -- Requirement Candidate ------------------------------------------

class RequirementCandidateRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, candidates: Sequence[RequirementCandidate]) -> None:
        for c in candidates:
            row = RequirementCandidateRow(**_to_dict(c))
            self.session.add(row)
        self.session.commit()

    def list_by_project(self, project_id: str) -> list[RequirementCandidate]:
        rows = self.session.query(RequirementCandidateRow).filter_by(project_id=project_id).all()
        return [
            RequirementCandidate(
                id=r.id, project_id=r.project_id, text=r.text,
                requirement_kind=r.requirement_kind, nfr_subtype=r.nfr_subtype,
                source_span_ids=r.source_span_ids or [], stakeholders=r.stakeholders or [],
                rationale=r.rationale or "", confidence=r.confidence or 0.0,
                status=r.status, created_at=r.created_at,
            )
            for r in rows
        ]

    def update_status(self, candidate_id: str, status: str) -> None:
        row = self.session.get(RequirementCandidateRow, candidate_id)
        if row:
            row.status = status
            self.session.commit()


# -- Requirement -----------------------------------------------------

class RequirementRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, req: Requirement) -> Requirement:
        row = RequirementRow(**_to_dict(req))
        self.session.add(row)
        self.session.commit()
        return req

    def create_many(self, reqs: Sequence[Requirement]) -> None:
        for r in reqs:
            row = RequirementRow(**_to_dict(r))
            self.session.add(row)
        self.session.commit()

    def get(self, requirement_id: str) -> Requirement | None:
        row = self.session.get(RequirementRow, requirement_id)
        if row is None:
            return None
        return Requirement(
            id=row.id, project_id=row.project_id, text=row.text,
            kind=row.kind, nfr_subtype=row.nfr_subtype, status=row.status,
            review_status=row.review_status, quality_score=row.quality_score,
            created_from_candidate_id=row.created_from_candidate_id,
            source_span_ids=row.source_span_ids or [],
            embedding=row.embedding, created_at=row.created_at,
            updated_at=row.updated_at,
        )

    def list_by_project(self, project_id: str) -> list[Requirement]:
        rows = self.session.query(RequirementRow).filter_by(project_id=project_id).all()
        return [
            Requirement(
                id=r.id, project_id=r.project_id, text=r.text,
                kind=r.kind, nfr_subtype=r.nfr_subtype, status=r.status,
                review_status=r.review_status, quality_score=r.quality_score,
                created_from_candidate_id=r.created_from_candidate_id,
                source_span_ids=r.source_span_ids or [],
                embedding=r.embedding, created_at=r.created_at,
                updated_at=r.updated_at,
            )
            for r in rows
        ]

    def update(self, req: Requirement) -> None:
        row = self.session.get(RequirementRow, req.id)
        if row:
            for field in ("text", "kind", "nfr_subtype", "status", "review_status",
                          "quality_score", "source_span_ids", "embedding"):
                setattr(row, field, getattr(req, field))
            row.updated_at = datetime.utcnow()
            self.session.commit()


# -- Evidence Assessment ---------------------------------------------

class EvidenceAssessmentRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, assessment: EvidenceAssessment) -> EvidenceAssessment:
        row = EvidenceAssessmentRow(**_to_dict(assessment))
        self.session.add(row)
        self.session.commit()
        return assessment

    def create_many(self, assessments: Sequence[EvidenceAssessment]) -> None:
        for a in assessments:
            row = EvidenceAssessmentRow(**_to_dict(a))
            self.session.add(row)
        self.session.commit()

    def get_by_candidate(self, candidate_id: str) -> EvidenceAssessment | None:
        row = self.session.query(EvidenceAssessmentRow).filter_by(
            requirement_candidate_id=candidate_id
        ).first()
        if row is None:
            return None
        return EvidenceAssessment(
            id=row.id, project_id=row.project_id,
            requirement_candidate_id=row.requirement_candidate_id,
            status=row.status, supporting_span_ids=row.supporting_span_ids or [],
            contradicting_span_ids=row.contradicting_span_ids or [],
            explanation=row.explanation or "", confidence=row.confidence or 0.0,
            created_at=row.created_at,
        )

    def list_by_project(self, project_id: str) -> list[EvidenceAssessment]:
        rows = self.session.query(EvidenceAssessmentRow).filter_by(project_id=project_id).all()
        return [
            EvidenceAssessment(
                id=r.id, project_id=r.project_id,
                requirement_candidate_id=r.requirement_candidate_id,
                status=r.status, supporting_span_ids=r.supporting_span_ids or [],
                contradicting_span_ids=r.contradicting_span_ids or [],
                explanation=r.explanation or "", confidence=r.confidence or 0.0,
                created_at=r.created_at,
            )
            for r in rows
        ]


# -- Classification Result ------------------------------------------

class ClassificationResultRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, results: Sequence[ClassificationResult]) -> None:
        for r in results:
            row = ClassificationResultRow(**_to_dict(r))
            self.session.add(row)
        self.session.commit()

    def get_by_requirement(self, requirement_id: str) -> ClassificationResult | None:
        row = self.session.query(ClassificationResultRow).filter_by(
            requirement_id=requirement_id
        ).first()
        if row is None:
            return None
        return ClassificationResult(
            id=row.id, requirement_id=row.requirement_id, kind=row.kind,
            nfr_subtype=row.nfr_subtype, confidence=row.confidence or 0.0,
            rationale=row.rationale or "", created_at=row.created_at,
        )


# -- Quality Finding -------------------------------------------------

class QualityFindingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, findings: Sequence[QualityFinding]) -> None:
        for f in findings:
            row = QualityFindingRow(**_to_dict(f))
            self.session.add(row)
        self.session.commit()

    def list_by_requirement(self, requirement_id: str) -> list[QualityFinding]:
        rows = self.session.query(QualityFindingRow).filter_by(requirement_id=requirement_id).all()
        return [
            QualityFinding(
                id=r.id, requirement_id=r.requirement_id, issue_type=r.issue_type,
                severity=r.severity, explanation=r.explanation,
                suggested_rewrite=r.suggested_rewrite, created_at=r.created_at,
            )
            for r in rows
        ]

    def list_by_project(self, project_id: str, requirement_ids: list[str]) -> list[QualityFinding]:
        if not requirement_ids:
            return []
        rows = self.session.query(QualityFindingRow).filter(
            QualityFindingRow.requirement_id.in_(requirement_ids)
        ).all()
        return [
            QualityFinding(
                id=r.id, requirement_id=r.requirement_id, issue_type=r.issue_type,
                severity=r.severity, explanation=r.explanation,
                suggested_rewrite=r.suggested_rewrite, created_at=r.created_at,
            )
            for r in rows
        ]


# -- Graph Edge ------------------------------------------------------

class GraphEdgeRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, edges: Sequence[GraphEdge]) -> None:
        for e in edges:
            row = GraphEdgeRow(**_to_dict(e))
            self.session.add(row)
        self.session.commit()

    def list_by_project(self, project_id: str) -> list[GraphEdge]:
        rows = self.session.query(GraphEdgeRow).filter_by(project_id=project_id).all()
        return [
            GraphEdge(
                id=r.id, project_id=r.project_id,
                source_node_id=r.source_node_id, target_node_id=r.target_node_id,
                edge_type=r.edge_type, confidence=r.confidence or 0.0,
                created_by=r.created_by or "", review_status=r.review_status or "pending",
                explanation=r.explanation or "", created_at=r.created_at,
            )
            for r in rows
        ]


# -- Conflict Finding -----------------------------------------------

class ConflictFindingRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, findings: Sequence[ConflictFinding]) -> None:
        for f in findings:
            row = ConflictFindingRow(**_to_dict(f))
            self.session.add(row)
        self.session.commit()

    def list_by_project(self, project_id: str) -> list[ConflictFinding]:
        rows = self.session.query(ConflictFindingRow).filter_by(project_id=project_id).all()
        return [
            ConflictFinding(
                id=r.id, project_id=r.project_id, conflict_type=r.conflict_type,
                involved_requirement_ids=r.involved_requirement_ids or [],
                severity=r.severity, explanation=r.explanation,
                suggested_resolution=r.suggested_resolution, resolved=r.resolved or False,
                created_at=r.created_at,
            )
            for r in rows
        ]


# -- Trace Link ------------------------------------------------------

class TraceLinkRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, links: Sequence[TraceLink]) -> None:
        for link in links:
            row = TraceLinkRow(**_to_dict(link))
            self.session.add(row)
        self.session.commit()

    def list_by_project(self, project_id: str) -> list[TraceLink]:
        rows = self.session.query(TraceLinkRow).filter_by(project_id=project_id).all()
        return [
            TraceLink(
                id=r.id, project_id=r.project_id,
                source_id=r.source_id, target_id=r.target_id,
                link_type=r.link_type, confidence=r.confidence or 0.0,
                review_status=r.review_status or "pending", created_at=r.created_at,
            )
            for r in rows
        ]


# -- Review Decision -------------------------------------------------

class ReviewDecisionRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, decision: ReviewDecision) -> ReviewDecision:
        row = ReviewDecisionRow(**_to_dict(decision))
        self.session.add(row)
        self.session.commit()
        return decision


# -- Agent Run -------------------------------------------------------

class AgentRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run: AgentRun) -> AgentRun:
        row = AgentRunRow(**_to_dict(run))
        self.session.add(row)
        self.session.commit()
        return run

    def update(self, run: AgentRun) -> None:
        row = self.session.get(AgentRunRow, run.id)
        if row:
            row.status = run.status.value if hasattr(run.status, "value") else run.status
            row.created_ids = run.created_ids
            row.warnings = run.warnings
            row.errors = run.errors
            row.finished_at = run.finished_at
            self.session.commit()


# -- LLM Call Log ----------------------------------------------------

class LLMCallLogRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create_many(self, logs: Sequence[LLMCallLog]) -> None:
        for log in logs:
            row = LLMCallLogRow(**_to_dict(log))
            self.session.add(row)
        self.session.commit()


# -- Benchmark Run ---------------------------------------------------

class BenchmarkRunRepository:
    def __init__(self, session: Session) -> None:
        self.session = session

    def create(self, run: BenchmarkRun) -> BenchmarkRun:
        row = BenchmarkRunRow(**_to_dict(run))
        self.session.add(row)
        self.session.commit()
        return run

    def get(self, run_id: str) -> BenchmarkRun | None:
        row = self.session.get(BenchmarkRunRow, run_id)
        if row is None:
            return None
        return BenchmarkRun(
            id=row.id, benchmark_type=row.benchmark_type,
            dataset=row.dataset or "", metrics=row.metrics or {},
            config=row.config or {}, created_at=row.created_at,
        )