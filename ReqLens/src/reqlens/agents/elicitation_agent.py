"""Elicitation Agent – generate stakeholder questions for gaps and ambiguities."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.models import QualityFinding, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import ELICITATION_SYSTEM_V1

logger = structlog.get_logger(__name__)


class ElicitationAgent(BaseAgent):
    name = "elicitation"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        open_questions: list[str],
        insufficient_requirements: list[Requirement] | None = None,
        quality_findings: list[QualityFinding] | None = None,
    ) -> AgentResult:
        questions = self.generate_questions(
            context, open_questions, insufficient_requirements, quality_findings
        )
        return AgentResult(
            agent_name=self.name,
            status="completed",
            warnings=[f"Generated {len(questions)} stakeholder question(s)."],
        )

    def generate_questions(
        self,
        context: AgentContext,
        open_questions: list[str],
        insufficient_requirements: list[Requirement] | None = None,
        quality_findings: list[QualityFinding] | None = None,
    ) -> list[str]:
        """Generate clear, specific stakeholder questions."""
        parts: list[str] = []

        if open_questions:
            parts.append("Open questions from extraction:\n" + "\n".join(f"- {q}" for q in open_questions))

        if insufficient_requirements:
            block = "\n".join(f"- [{r.id}] {r.text}" for r in insufficient_requirements)
            parts.append(f"Requirements with insufficient evidence:\n{block}")

        if quality_findings:
            block = "\n".join(
                f"- [{f.requirement_id}] {f.issue_type.value}: {f.explanation}"
                for f in quality_findings
            )
            parts.append(f"Quality findings needing clarification:\n{block}")

        if not parts:
            return []

        user_prompt = "\n\n".join(parts) + "\n\nGenerate stakeholder questions."

        try:
            response = self.llm.response_text(
                instructions=ELICITATION_SYSTEM_V1,
                input_text=user_prompt,
                project_id=context.project_id,
                agent_name=self.name,
            )
            # Parse response as list of questions (one per line)
            questions = [
                line.strip().lstrip("- ").lstrip("0123456789.)")
                for line in response.strip().split("\n")
                if line.strip() and not line.strip().startswith("#")
            ]
            return [q for q in questions if q]
        except Exception as exc:
            logger.error("elicitation.llm_error", error=str(exc))
            return open_questions  # fallback: return original questions
