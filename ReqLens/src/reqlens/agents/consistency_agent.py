"""Consistency Agent – detect contradictions and duplicates among requirements."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.ids import generate_id
from reqlens.domain.models import ConflictFinding, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import CONSISTENCY_SYSTEM_V1
from reqlens.llm.schemas import ConsistencyAnalysisOutput

logger = structlog.get_logger(__name__)


class ConsistencyAgent(BaseAgent):
    name = "consistency"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> AgentResult:
        findings = self.detect_conflicts(context, requirements)
        status = "completed"
        warnings: list[str] = []
        if findings:
            warnings.append(f"{len(findings)} conflict(s) found.")
        return AgentResult(
            agent_name=self.name,
            status=status,
            created_ids=[f.id for f in findings],
            warnings=warnings,
        )

    def detect_conflicts(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> list[ConflictFinding]:
        """Detect contradictions and duplicates in the requirement set."""
        if len(requirements) < 2:
            return []

        req_block = "\n\n".join(
            f"[{r.id}] ({r.kind.value}) {r.text}" for r in requirements
        )
        user_prompt = (
            f"Requirements:\n\n{req_block}\n\n"
            "Detect contradictions, duplicates, and inconsistencies."
        )

        try:
            output: ConsistencyAnalysisOutput = self.llm.structured_chat(
                system_prompt=CONSISTENCY_SYSTEM_V1,
                user_prompt=user_prompt,
                response_model=ConsistencyAnalysisOutput,
                project_id=context.project_id,
                agent_name=self.name,
            )
        except Exception as exc:
            logger.error("consistency.llm_error", error=str(exc))
            return []

        findings: list[ConflictFinding] = []
        for cf in output.conflicts:
            finding = ConflictFinding(
                id=generate_id("CF"),
                project_id=context.project_id,
                conflict_type=cf.conflict_type,
                involved_requirement_ids=cf.involved_requirement_ids,
                severity=cf.severity,
                explanation=cf.explanation,
                suggested_resolution=cf.suggested_resolution,
            )
            findings.append(finding)

        logger.info("consistency.done", conflicts=len(findings))
        return findings
