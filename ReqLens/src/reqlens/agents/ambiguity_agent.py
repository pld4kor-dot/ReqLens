"""Ambiguity Agent – detect poor-quality requirement wording."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.ids import generate_id
from reqlens.domain.models import QualityFinding, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import AMBIGUITY_SYSTEM_V1
from reqlens.llm.schemas import AmbiguityAnalysisOutput

logger = structlog.get_logger(__name__)


class AmbiguityAgent(BaseAgent):
    name = "ambiguity"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> AgentResult:
        findings = self.analyse(context, requirements)
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[f.id for f in findings],
        )

    def analyse(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> list[QualityFinding]:
        """Detect ambiguity and quality issues in requirements."""
        if not requirements:
            return []

        req_block = "\n\n".join(
            f"[{r.id}] {r.text}" for r in requirements
        )
        user_prompt = (
            f"Requirements to analyse for quality issues:\n\n{req_block}\n\n"
            "Identify any ambiguity, vagueness, non-atomicity, or other quality issues."
        )

        try:
            output: AmbiguityAnalysisOutput = self.llm.structured_chat(
                system_prompt=AMBIGUITY_SYSTEM_V1,
                user_prompt=user_prompt,
                response_model=AmbiguityAnalysisOutput,
                project_id=context.project_id,
                agent_name=self.name,
            )
        except Exception as exc:
            logger.error("ambiguity.llm_error", error=str(exc))
            return []

        findings: list[QualityFinding] = []
        for f in output.findings:
            finding = QualityFinding(
                id=generate_id("QF"),
                requirement_id=f.requirement_id,
                issue_type=f.issue_type,
                severity=f.severity,
                explanation=f.explanation,
                suggested_rewrite=f.suggested_rewrite,
            )
            findings.append(finding)

        logger.info("ambiguity.done", total=len(findings))
        return findings
