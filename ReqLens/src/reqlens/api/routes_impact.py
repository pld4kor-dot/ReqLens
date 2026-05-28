"""API routes – impact analysis."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.storage.db import get_db_session
from reqlens.storage.graph_store import GraphStore

router = APIRouter()


class ImpactRequest(BaseModel):
    change_request: str


class ImpactedNodeResponse(BaseModel):
    node_id: str
    node_type: str
    impact_level: str
    explanation: str


class ImpactResponse(BaseModel):
    change_summary: str
    directly_affected: list[ImpactedNodeResponse]
    indirectly_affected: list[ImpactedNodeResponse]
    suggested_review_tasks: list[str]


@router.post("/projects/{project_id}/impact", response_model=ImpactResponse)
async def analyse_impact(
    project_id: str,
    body: ImpactRequest,
    session: Session = Depends(get_db_session),
) -> ImpactResponse:
    from reqlens.agents.impact_agent import ImpactAgent
    from reqlens.agents.base import AgentContext
    from reqlens.llm.azure_client import AzureOpenAIClient
    from reqlens.storage.repositories import RequirementRepository

    requirements = RequirementRepository(session).list_by_project(project_id)
    if not requirements:
        raise HTTPException(status_code=404, detail="No requirements found for this project.")

    llm = AzureOpenAIClient()
    graph_store = GraphStore(session, project_id)
    context = AgentContext(project_id=project_id, run_id="impact")

    agent = ImpactAgent(llm=llm, graph_store=graph_store)
    output = agent.analyse_impact(context, body.change_request, requirements)

    return ImpactResponse(
        change_summary=output.change_summary,
        directly_affected=[
            ImpactedNodeResponse(
                node_id=n.node_id, node_type=n.node_type,
                impact_level=n.impact_level, explanation=n.explanation,
            )
            for n in output.directly_affected
        ],
        indirectly_affected=[
            ImpactedNodeResponse(
                node_id=n.node_id, node_type=n.node_type,
                impact_level=n.impact_level, explanation=n.explanation,
            )
            for n in output.indirectly_affected
        ],
        suggested_review_tasks=output.suggested_review_tasks,
    )
