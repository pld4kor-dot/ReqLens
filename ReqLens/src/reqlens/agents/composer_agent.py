"""Composer Agent – generate SRS from accepted requirements only.

STRICT RULE: The Composer Agent must NOT invent requirements.
It may only use accepted requirements, approved graph edges,
source-backed glossary terms, and accepted review decisions.
"""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.enums import ReviewStatus
from reqlens.domain.models import ConflictFinding, GraphEdge, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import COMPOSER_SYSTEM_V1
from reqlens.llm.schemas import SRSOutput

logger = structlog.get_logger(__name__)


class ComposerAgent(BaseAgent):
    name = "composer"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
        edges: list[GraphEdge] | None = None,
        conflicts: list[ConflictFinding] | None = None,
        open_questions: list[str] | None = None,
    ) -> AgentResult:
        srs = self.compose_srs(context, requirements, edges, conflicts, open_questions)
        return AgentResult(
            agent_name=self.name,
            status="completed",
        )

    def compose_srs(
        self,
        context: AgentContext,
        requirements: list[Requirement],
        edges: list[GraphEdge] | None = None,
        conflicts: list[ConflictFinding] | None = None,
        open_questions: list[str] | None = None,
    ) -> SRSOutput:
        """Compose an SRS from accepted requirements only."""
        # Filter to accepted requirements only
        accepted = [
            r for r in requirements
            if r.review_status in (ReviewStatus.accepted,)
        ]

        if not accepted:
            logger.warning("composer.no_accepted_requirements")
            return SRSOutput(
                sections=[],
                open_questions=open_questions or [],
                conflict_summary="No accepted requirements to compose SRS from.",
            )

        # Build prompts
        req_block = "\n\n".join(
            f"[{r.id}] Kind={r.kind.value} NFR_subtype={r.nfr_subtype.value}\n{r.text}"
            for r in accepted
        )

        edge_block = ""
        if edges:
            edge_block = "\n".join(
                f"  {e.source_node_id} --[{e.edge_type.value}]--> {e.target_node_id}: {e.explanation}"
                for e in edges[:100]
            )

        conflict_block = ""
        if conflicts:
            conflict_block = "\n".join(
                f"  [{c.conflict_type.value}] {', '.join(c.involved_requirement_ids)}: {c.explanation}"
                for c in conflicts
            )

        questions_block = ""
        if open_questions:
            questions_block = "\n".join(f"  - {q}" for q in open_questions)

        user_prompt = (
            f"Accepted requirements ({len(accepted)}):\n\n{req_block}\n\n"
            f"Dependency edges:\n{edge_block or '  (none)'}\n\n"
            f"Conflicts:\n{conflict_block or '  (none)'}\n\n"
            f"Open questions:\n{questions_block or '  (none)'}\n\n"
            "Generate the SRS."
        )

        try:
            output: SRSOutput = self.llm.structured_chat(
                system_prompt=COMPOSER_SYSTEM_V1,
                user_prompt=user_prompt,
                response_model=SRSOutput,
                project_id=context.project_id,
                agent_name=self.name,
            )
        except Exception as exc:
            logger.error("composer.llm_error", error=str(exc))
            output = SRSOutput(
                sections=[],
                open_questions=open_questions or [],
                conflict_summary=f"LLM error: {exc}",
            )

        # Ensure pipeline-provided open_questions and conflicts make it into the
        # rendered SRS. The LLM may either embed them inside ``sections`` (where
        # ``srs_to_markdown`` filters them out to avoid duplication) or leave the
        # parallel structured fields empty/short, both of which would otherwise
        # cause the Open Questions and Conflict Report sections to disappear
        # from the generated document.
        if open_questions:
            # Trust the inputs the pipeline gave us — they came from the
            # Elicitation Agent (or its fallback to raw extractor questions).
            output.open_questions = list(open_questions)

        if conflicts and not output.conflict_summary.strip():
            output.conflict_summary = self._format_conflict_summary(conflicts)

        logger.info(
            "composer.done",
            sections=len(output.sections),
            open_questions=len(output.open_questions),
            conflict_summary_chars=len(output.conflict_summary),
        )
        return output

    @staticmethod
    def _format_conflict_summary(conflicts: list[ConflictFinding]) -> str:
        """Build a deterministic markdown summary from conflict findings.

        Used as a fallback when the LLM-rendered ``conflict_summary`` is empty
        so that the conflicts surfaced by the Consistency Agent always show up
        in the final SRS.
        """
        lines: list[str] = [f"{len(conflicts)} conflict(s) detected:\n"]
        for c in conflicts:
            ctype = getattr(c.conflict_type, "value", str(c.conflict_type))
            involved = ", ".join(c.involved_requirement_ids) or "—"
            lines.append(f"- **[{ctype}]** {involved}: {c.explanation}")
        return "\n".join(lines)

    def srs_to_markdown(self, srs: SRSOutput) -> str:
        """Convert SRS output to Markdown string.

        Note on de-duplication
        ----------------------
        The composer system prompt asks the LLM to produce 10 sections, the
        last two of which are "Open Questions" and "Conflict Report". The LLM
        therefore returns them inside ``srs.sections``. At the same time,
        ``SRSOutput`` exposes them as dedicated structured fields
        (``open_questions: list[str]`` and ``conflict_summary: str``), which
        this method also renders at the bottom of the document.

        Without filtering, both would appear in the markdown — the user sees
        an "## Open Questions" block from ``srs.sections`` and then a second
        identical "## Open Questions" block from ``srs.open_questions`` a few
        lines later (likewise for the Conflict Report).

        To keep the structured fields authoritative — they're the ones the
        pipeline feeds explicitly via ``open_questions=stakeholder_questions``
        and the consistency-derived conflict list — we drop any section whose
        title matches those two reserved titles when walking ``srs.sections``.
        """
        _RESERVED_TITLES = {"open questions", "conflict report"}

        lines: list[str] = ["# Software Requirements Specification\n"]

        for section in srs.sections:
            if section.title.strip().lower() in _RESERVED_TITLES:
                continue
            lines.append(f"## {section.title}\n")
            lines.append(section.content)
            if section.requirement_ids:
                lines.append(f"\n*Requirements: {', '.join(section.requirement_ids)}*")
            lines.append("")

        if srs.open_questions:
            lines.append("## Open Questions\n")
            for q in srs.open_questions:
                lines.append(f"- {q}")
            lines.append("")

        if srs.conflict_summary:
            lines.append("## Conflict Report\n")
            lines.append(srs.conflict_summary)
            lines.append("")

        return "\n".join(lines)
