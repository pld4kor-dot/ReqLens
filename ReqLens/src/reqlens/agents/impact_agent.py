"""Impact Agent – change impact analysis on the requirement graph."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.models import Requirement
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import IMPACT_SYSTEM_V1
from reqlens.llm.schemas import ImpactAnalysisOutput
from reqlens.storage.graph_store import GraphStore

logger = structlog.get_logger(__name__)


class ImpactAgent(BaseAgent):
    name = "impact"

    def __init__(self, llm: AzureOpenAIClient, graph_store: GraphStore) -> None:
        self.llm = llm
        self.graph_store = graph_store

    async def run(
        self,
        context: AgentContext,
        change_request: str,
        requirements: list[Requirement],
    ) -> AgentResult:
        output = self.analyse_impact(context, change_request, requirements)
        affected_ids = [n.node_id for n in output.directly_affected + output.indirectly_affected]
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=affected_ids,
        )

    def analyse_impact(
        self,
        context: AgentContext,
        change_request: str,
        requirements: list[Requirement],
    ) -> ImpactAnalysisOutput:
        """Analyse the impact of a change request on the requirement graph."""
        # Build graph context
        graph_data = self.graph_store.to_dict()
        req_block = "\n\n".join(f"[{r.id}] ({r.kind.value}) {r.text}" for r in requirements)

        # Summarise graph edges
        edge_summary = "\n".join(
            f"  {e['source']} --[{e.get('edge_type', '?')}]--> {e['target']}"
            for e in graph_data.get("edges", [])[:100]  # limit for token budget
        )

        user_prompt = (
            f"Change request:\n{change_request}\n\n"
            f"Requirements:\n{req_block}\n\n"
            f"Graph edges:\n{edge_summary}\n\n"
            "Identify all affected requirements and artifacts."
        )

        try:
            output: ImpactAnalysisOutput = self.llm.structured_chat(
                system_prompt=IMPACT_SYSTEM_V1,
                user_prompt=user_prompt,
                response_model=ImpactAnalysisOutput,
                project_id=context.project_id,
                agent_name=self.name,
            )
        except Exception as exc:
            logger.error("impact.llm_error", error=str(exc))
            output = ImpactAnalysisOutput(
                change_summary=change_request,
                directly_affected=[],
                indirectly_affected=[],
            )

        logger.info(
            "impact.done",
            direct=len(output.directly_affected),
            indirect=len(output.indirectly_affected),
        )
        return output
