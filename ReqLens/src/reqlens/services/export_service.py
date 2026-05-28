"""Export service – generate SRS, traceability matrix, graph exports."""

from __future__ import annotations

import csv
import io
import json

import structlog

from reqlens.domain.models import ConflictFinding, GraphEdge, Requirement, TraceLink
from reqlens.llm.schemas import SRSOutput
from reqlens.storage.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class ExportService:
    def __init__(self, graph_store: GraphStore) -> None:
        self.graph_store = graph_store

    def export_srs_markdown(self, srs: SRSOutput) -> str:
        """Export SRS as Markdown."""
        lines = ["# Software Requirements Specification\n"]
        for section in srs.sections:
            lines.append(f"## {section.title}\n")
            lines.append(section.content + "\n")
            if section.requirement_ids:
                lines.append(f"*Referenced requirements: {', '.join(section.requirement_ids)}*\n")

        if srs.open_questions:
            lines.append("## Open Questions\n")
            for q in srs.open_questions:
                lines.append(f"- {q}")
            lines.append("")

        if srs.conflict_summary:
            lines.append("## Conflict Report\n")
            lines.append(srs.conflict_summary + "\n")

        return "\n".join(lines)

    def export_traceability_csv(
        self,
        requirements: list[Requirement],
        trace_links: list[TraceLink],
    ) -> str:
        """Export traceability matrix as CSV."""
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow([
            "requirement_id", "requirement_text", "kind",
            "linked_to", "link_type", "confidence",
        ])
        link_map: dict[str, list[TraceLink]] = {}
        for tl in trace_links:
            link_map.setdefault(tl.target_id, []).append(tl)
            link_map.setdefault(tl.source_id, []).append(tl)

        for req in requirements:
            links = link_map.get(req.id, [])
            if not links:
                writer.writerow([req.id, req.text, req.kind.value, "", "", ""])
            else:
                for link in links:
                    other_id = link.source_id if link.target_id == req.id else link.target_id
                    writer.writerow([
                        req.id, req.text, req.kind.value,
                        other_id, link.link_type.value, f"{link.confidence:.2f}",
                    ])

        return buf.getvalue()

    def export_graph_graphml(self) -> str:
        """Export knowledge graph as GraphML."""
        return self.graph_store.to_graphml()

    def export_graph_json(self) -> str:
        """Export knowledge graph as JSON."""
        return json.dumps(self.graph_store.to_dict(), indent=2, default=str)

    def export_conflicts_json(self, conflicts: list[ConflictFinding]) -> str:
        """Export conflicts as JSON."""
        data = [c.model_dump(mode="json") for c in conflicts]
        return json.dumps(data, indent=2, default=str)

    def export_benchmark_json(self, metrics: dict) -> str:
        """Export benchmark results as JSON."""
        return json.dumps(metrics, indent=2, default=str)
