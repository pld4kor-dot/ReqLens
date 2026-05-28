"""Project service – CRUD + cascade delete for projects.

Added: ``ProjectService.delete_project(project_id)`` does a full cascade
delete across every table that holds rows scoped to this project.

The schema does not declare foreign keys with ON DELETE CASCADE, so the
cascade is implemented here.  Indirect rows that reference only
``requirement_id`` (classification_results, quality_findings,
review_decisions) are located via the project's requirement IDs *before*
the requirements are deleted so nothing is orphaned.
"""

from __future__ import annotations

import structlog
from sqlalchemy.orm import Session

from reqlens.domain.models import Project
from reqlens.storage.repositories import ProjectRepository

logger = structlog.get_logger(__name__)


class ProjectService:
    def __init__(self, repo: ProjectRepository) -> None:
        self.repo = repo

    # ── CRUD ────────────────────────────────────────────────────────────

    def create_project(self, name: str, description: str = "") -> Project:
        project = Project(name=name, description=description)
        return self.repo.create(project)

    def get_project(self, project_id: str) -> Project | None:
        return self.repo.get(project_id)

    def list_projects(self) -> list[Project]:
        return self.repo.list_all()

    # ── Cascade delete ──────────────────────────────────────────────────

    def delete_project(self, project_id: str) -> dict[str, int]:
        """Delete a project and every row related to it.

        Returns a per-table count of deleted rows for confirmation.
        Raises ValueError if the project does not exist.
        """
        session: Session = self.repo.session

        project = self.repo.get(project_id)
        if project is None:
            raise ValueError(f"Project '{project_id}' not found")

        from reqlens.storage.db import (
            AgentRunRow,
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

        deleted: dict[str, int] = {}

        # 1) Capture requirement IDs FIRST — needed for indirect cascade
        req_ids: list[str] = [
            r[0]
            for r in session.query(RequirementRow.id)
            .filter(RequirementRow.project_id == project_id)
            .all()
        ]

        # 2) Indirect (requirement-scoped) rows
        if req_ids:
            deleted["classification_results"] = (
                session.query(ClassificationResultRow)
                .filter(ClassificationResultRow.requirement_id.in_(req_ids))
                .delete(synchronize_session=False)
            )
            deleted["quality_findings"] = (
                session.query(QualityFindingRow)
                .filter(QualityFindingRow.requirement_id.in_(req_ids))
                .delete(synchronize_session=False)
            )
            deleted["review_decisions"] = (
                session.query(ReviewDecisionRow)
                .filter(ReviewDecisionRow.requirement_id.in_(req_ids))
                .delete(synchronize_session=False)
            )
        else:
            deleted["classification_results"] = 0
            deleted["quality_findings"] = 0
            deleted["review_decisions"] = 0

        # 3) Direct (project-scoped) rows
        direct_tables: list[tuple[str, type]] = [
            ("evidence_assessments",   EvidenceAssessmentRow),
            ("requirement_candidates", RequirementCandidateRow),
            ("requirements",           RequirementRow),
            ("trace_links",            TraceLinkRow),
            ("conflict_findings",      ConflictFindingRow),
            ("graph_edges",            GraphEdgeRow),
            ("source_spans",           SourceSpanRow),
            ("documents",              DocumentRow),
            ("agent_runs",             AgentRunRow),
            ("llm_calls",              LLMCallLogRow),
        ]
        for table_name, row_class in direct_tables:
            deleted[table_name] = (
                session.query(row_class)
                .filter(row_class.project_id == project_id)
                .delete(synchronize_session=False)
            )

        # 4) The project row itself
        deleted["projects"] = (
            session.query(ProjectRow)
            .filter(ProjectRow.id == project_id)
            .delete(synchronize_session=False)
        )

        session.commit()
        total = sum(deleted.values())
        logger.info(
            "project.deleted",
            project_id=project_id,
            project_name=project.name,
            total_rows=total,
            **deleted,
        )
        return deleted
