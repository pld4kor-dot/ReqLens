"""Classification Agent – classify requirements as FR/NFR with subtypes."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.ids import generate_id
from reqlens.domain.models import ClassificationResult, Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import CLASSIFICATION_SYSTEM_V1
from reqlens.llm.schemas import ClassificationBatchOutput

logger = structlog.get_logger(__name__)


class ClassificationAgent(BaseAgent):
    name = "classification"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> AgentResult:
        results = self.classify(context, requirements)
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[r.id for r in results],
        )

    def classify(
        self,
        context: AgentContext,
        requirements: list[Requirement],
    ) -> list[ClassificationResult]:
        """Classify a batch of requirements."""
        if not requirements:
            return []

        req_block = "\n\n".join(
            f"[{r.id}] {r.text}" for r in requirements
        )
        user_prompt = (
            f"Requirements to classify:\n\n{req_block}\n\n"
            "Classify each requirement."
        )

        try:
            output: ClassificationBatchOutput = self.llm.structured_chat(
                system_prompt=CLASSIFICATION_SYSTEM_V1,
                user_prompt=user_prompt,
                response_model=ClassificationBatchOutput,
                project_id=context.project_id,
                agent_name=self.name,
            )
        except Exception as exc:
            logger.error("classification.llm_error", error=str(exc))
            return []

        results: list[ClassificationResult] = []
        for clf in output.classifications:
            result = ClassificationResult(
                id=generate_id("CLF"),
                requirement_id=clf.requirement_id,
                kind=clf.kind,
                nfr_subtype=clf.nfr_subtype,
                confidence=clf.confidence,
                rationale=clf.rationale,
            )
            results.append(result)

        logger.info("classification.done", total=len(results))
        return results
