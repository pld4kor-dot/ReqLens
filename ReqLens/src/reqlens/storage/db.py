"""SQLAlchemy database engine and session management."""

from __future__ import annotations

from sqlalchemy import (
    JSON,
    Boolean,
    Column,
    DateTime,
    Float,
    Integer,
    String,
    Text,
    create_engine,
    func,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from reqlens.config.settings import get_settings


class Base(DeclarativeBase):
    """Declarative base for all ORM models."""
    pass


# ── ORM Table Models ───────────────────────────────────────────────

class ProjectRow(Base):
    __tablename__ = "projects"

    id                = Column(String, primary_key=True)
    name              = Column(String, nullable=False)
    description       = Column(Text, default="")
    created_at        = Column(DateTime, server_default=func.now())
    # Set to the ID of the original project that this version was forked
    # from.  NULL for the root (original) project.
    parent_project_id = Column(String, nullable=True, index=True)


class DocumentRow(Base):
    __tablename__ = "documents"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    filename = Column(String, nullable=False)
    document_type = Column(String, default="other")
    content = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class SourceSpanRow(Base):
    __tablename__ = "source_spans"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    document_id = Column(String, nullable=False, index=True)
    span_index = Column(Integer, nullable=False)
    text = Column(Text, nullable=False)
    char_start = Column(Integer, nullable=False)
    char_end = Column(Integer, nullable=False)
    speaker = Column(String, nullable=True)
    section_title = Column(String, nullable=True)
    embedding = Column(JSON, nullable=True)  # list[float] stored as JSON
    created_at = Column(DateTime, server_default=func.now())


class RequirementCandidateRow(Base):
    __tablename__ = "requirement_candidates"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    requirement_kind = Column(String, nullable=False)
    nfr_subtype = Column(String, default="not_applicable")
    source_span_ids = Column(JSON, default=list)
    stakeholders = Column(JSON, default=list)
    rationale = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    status = Column(String, default="candidate")
    created_at = Column(DateTime, server_default=func.now())


class RequirementRow(Base):
    __tablename__ = "requirements"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    text = Column(Text, nullable=False)
    kind = Column(String, nullable=False)
    nfr_subtype = Column(String, default="not_applicable")
    status = Column(String, default="evidence_checked")
    review_status = Column(String, default="pending")
    quality_score = Column(Float, nullable=True)
    created_from_candidate_id = Column(String, nullable=True)
    source_span_ids = Column(JSON, default=list)
    embedding = Column(JSON, nullable=True)
    created_at = Column(DateTime, server_default=func.now())
    updated_at = Column(DateTime, server_default=func.now(), onupdate=func.now())


class EvidenceAssessmentRow(Base):
    __tablename__ = "evidence_assessments"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    requirement_candidate_id = Column(String, nullable=False, index=True)
    status = Column(String, nullable=False)
    supporting_span_ids = Column(JSON, default=list)
    contradicting_span_ids = Column(JSON, default=list)
    explanation = Column(Text, default="")
    confidence = Column(Float, default=0.0)
    created_at = Column(DateTime, server_default=func.now())


class ClassificationResultRow(Base):
    __tablename__ = "classification_results"

    id = Column(String, primary_key=True)
    requirement_id = Column(String, nullable=False, index=True)
    kind = Column(String, nullable=False)
    nfr_subtype = Column(String, default="not_applicable")
    confidence = Column(Float, default=0.0)
    rationale = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class QualityFindingRow(Base):
    __tablename__ = "quality_findings"

    id = Column(String, primary_key=True)
    requirement_id = Column(String, nullable=False, index=True)
    issue_type = Column(String, nullable=False)
    severity = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    suggested_rewrite = Column(Text, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class GraphEdgeRow(Base):
    __tablename__ = "graph_edges"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    source_node_id = Column(String, nullable=False, index=True)
    target_node_id = Column(String, nullable=False, index=True)
    edge_type = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    created_by = Column(String, default="")
    review_status = Column(String, default="pending")
    explanation = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class ConflictFindingRow(Base):
    __tablename__ = "conflict_findings"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    conflict_type = Column(String, nullable=False)
    involved_requirement_ids = Column(JSON, nullable=False)
    severity = Column(String, nullable=False)
    explanation = Column(Text, nullable=False)
    suggested_resolution = Column(Text, nullable=True)
    resolved = Column(Boolean, default=False)
    created_at = Column(DateTime, server_default=func.now())


class TraceLinkRow(Base):
    __tablename__ = "trace_links"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    source_id = Column(String, nullable=False, index=True)
    target_id = Column(String, nullable=False, index=True)
    link_type = Column(String, nullable=False)
    confidence = Column(Float, default=0.0)
    review_status = Column(String, default="pending")
    created_at = Column(DateTime, server_default=func.now())


class ReviewDecisionRow(Base):
    __tablename__ = "review_decisions"

    id = Column(String, primary_key=True)
    requirement_id = Column(String, nullable=False, index=True)
    decision = Column(String, nullable=False)
    reviewer = Column(String, default="human")
    comment = Column(Text, default="")
    created_at = Column(DateTime, server_default=func.now())


class AgentRunRow(Base):
    __tablename__ = "agent_runs"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=False, index=True)
    agent_name = Column(String, nullable=False)
    status = Column(String, default="pending")
    created_ids = Column(JSON, default=list)
    warnings = Column(JSON, default=list)
    errors = Column(JSON, default=list)
    started_at = Column(DateTime, nullable=True)
    finished_at = Column(DateTime, nullable=True)
    created_at = Column(DateTime, server_default=func.now())


class LLMCallLogRow(Base):
    __tablename__ = "llm_calls"

    id = Column(String, primary_key=True)
    project_id = Column(String, nullable=True, index=True)
    agent_name = Column(String, default="")
    model = Column(String, default="")
    prompt_hash = Column(String, default="", index=True)
    input_tokens = Column(Integer, default=0)
    output_tokens = Column(Integer, default=0)
    latency_ms = Column(Integer, default=0)
    cost_estimate = Column(Float, default=0.0)
    status = Column(String, default="ok")
    created_at = Column(DateTime, server_default=func.now())


class BenchmarkRunRow(Base):
    __tablename__ = "benchmark_runs"

    id = Column(String, primary_key=True)
    benchmark_type = Column(String, nullable=False)
    dataset = Column(String, default="")
    metrics = Column(JSON, default=dict)
    config = Column(JSON, default=dict)
    created_at = Column(DateTime, server_default=func.now())


# ── Engine / Session factory ────────────────────────────────────────

_engine = None
_SessionLocal = None


def get_engine():
    global _engine
    if _engine is None:
        settings = get_settings()
        _engine = create_engine(settings.database_url, echo=settings.is_dev)
    return _engine


def get_session_factory() -> sessionmaker[Session]:
    global _SessionLocal
    if _SessionLocal is None:
        _SessionLocal = sessionmaker(bind=get_engine())
    return _SessionLocal


def get_session() -> Session:
    factory = get_session_factory()
    return factory()


from typing import Generator

def get_db_session() -> Generator[Session, None, None]:
    """FastAPI dependency that yields a single SQLAlchemy session for the
    entire request and closes it when the request ends.

    Use via Depends(get_db_session) so every repository and store
    created inside one request handler shares the same session — and
    therefore the same transaction and identity map.  This prevents the
    split-session bug where one call to get_session() reads rows that a
    second independent call cannot see because they are in separate
    SQLAlchemy identity maps.
    """
    factory = get_session_factory()
    session = factory()
    try:
        yield session
    finally:
        session.close()


def _ensure_parent_project_id_column() -> None:
    """Best-effort migration: add parent_project_id column to existing
    `projects` table on startup.  Safe to call against an already-up-to-date
    schema (it does nothing then).
    """
    try:
        from sqlalchemy import inspect, text
        engine = get_engine()
        inspector = inspect(engine)
        if "projects" not in inspector.get_table_names():
            return
        cols = {c["name"] for c in inspector.get_columns("projects")}
        if "parent_project_id" in cols:
            return
        with engine.begin() as conn:
            conn.execute(text(
                "ALTER TABLE projects ADD COLUMN parent_project_id VARCHAR"
            ))
    except Exception:
        # Migration is best-effort; do not crash startup
        pass


def create_all_tables() -> None:
    """Create all tables (for dev / testing – use Alembic in production)."""
    Base.metadata.create_all(get_engine())
    _ensure_parent_project_id_column()
