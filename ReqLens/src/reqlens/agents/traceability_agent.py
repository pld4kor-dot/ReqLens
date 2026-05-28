"""Traceability Agent – build trace links between requirements and artifacts."""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.enums import TraceLinkType
from reqlens.domain.ids import generate_id
from reqlens.domain.models import Requirement, SourceSpan, TraceLink
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import TRACEABILITY_SYSTEM_V1
from reqlens.llm.schemas import TraceabilityOutput

logger = structlog.get_logger(__name__)


class TraceabilityAgent(BaseAgent):
    name = "traceability"

    def __init__(self, llm: AzureOpenAIClient) -> None:
        self.llm = llm

    async def run(
        self,
        context: AgentContext,
        requirements: list[Requirement],
        spans: list[SourceSpan] | None = None,
        test_artifacts: list[dict] | None = None,
    ) -> AgentResult:
        # Build links once and surface the ids via AgentResult so callers
        # (both the orchestrator and the standalone API route) can persist them
        # without a second LLM round-trip.
        links = self.build_trace_links(context, requirements, spans, test_artifacts)
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[link.id for link in links],
        )

    def build_trace_links(
        self,
        context: AgentContext,
        requirements: list[Requirement],
        spans: list[SourceSpan] | None = None,
        test_artifacts: list[dict] | None = None,
    ) -> list[TraceLink]:
        """Build trace links from requirements to source spans and tests."""
        links: list[TraceLink] = []

        # 1. Source-to-requirement links — ask the LLM to score each
        #    (span_id, req_id) pair that is already known to be linked via
        #    source_span_ids.  Fall back to confidence=1.0 for any pair the
        #    LLM does not return (e.g. on error or omission).
        span_req_pairs: list[tuple[str, str]] = [
            (span_id, req.id)
            for req in requirements
            for span_id in req.source_span_ids
        ]

        if span_req_pairs:
            # Build a lookup: span_id → span text (for the prompt)
            span_text: dict[str, str] = {}
            if spans:
                for s in spans:
                    span_text[s.id] = s.text

            # Confidence map keyed by (source_id, target_id) from LLM response
            llm_confidence: dict[tuple[str, str], float] = {}
            try:
                span_block = "\n\n".join(
                    f"[{sid}] {span_text.get(sid, '(text unavailable)')}"
                    for sid, _ in span_req_pairs
                )
                req_block = "\n\n".join(
                    f"[{req.id}] {req.text}" for req in requirements
                )
                user_prompt = (
                    f"Requirements:\n{req_block}\n\n"
                    f"Source spans:\n{span_block}\n\n"
                    "Score the confidence of each source_to_requirement link "
                    "listed below. Only return links for the pairs given.\n"
                    + "\n".join(
                        f"- source_id={sid} target_id={rid}"
                        for sid, rid in span_req_pairs
                    )
                )
                output: TraceabilityOutput = self.llm.structured_chat(
                    system_prompt=TRACEABILITY_SYSTEM_V1,
                    user_prompt=user_prompt,
                    response_model=TraceabilityOutput,
                    project_id=context.project_id,
                    agent_name=self.name,
                )
                for tl in output.links:
                    llm_confidence[(tl.source_id, tl.target_id)] = tl.confidence
            except Exception as exc:
                logger.warning(
                    "traceability.source_confidence_llm_error",
                    error=str(exc),
                    fallback="1.0",
                )

            for span_id, req_id in span_req_pairs:
                confidence = llm_confidence.get((span_id, req_id), 1.0)
                links.append(TraceLink(
                    id=generate_id("TL"),
                    project_id=context.project_id,
                    source_id=span_id,
                    target_id=req_id,
                    link_type=TraceLinkType.source_to_requirement,
                    confidence=confidence,
                ))

        # 2. If test artifacts provided, use LLM for requirement-to-test links
        if test_artifacts and requirements:
            artifact_block = "\n\n".join(
                f"[{a.get('id', 'TC-???')}] {a.get('text', '')}"
                for a in test_artifacts
            )
            req_block = "\n\n".join(
                f"[{r.id}] {r.text}" for r in requirements
            )
            user_prompt = (
                f"Requirements:\n{req_block}\n\n"
                f"Test artifacts:\n{artifact_block}\n\n"
                "Propose trace links between requirements and test artifacts."
            )

            try:
                output: TraceabilityOutput = self.llm.structured_chat(
                    system_prompt=TRACEABILITY_SYSTEM_V1,
                    user_prompt=user_prompt,
                    response_model=TraceabilityOutput,
                    project_id=context.project_id,
                    agent_name=self.name,
                )
                for tl in output.links:
                    link = TraceLink(
                        id=generate_id("TL"),
                        project_id=context.project_id,
                        source_id=tl.source_id,
                        target_id=tl.target_id,
                        link_type=TraceLinkType.requirement_to_test,
                        confidence=tl.confidence,
                    )
                    links.append(link)
            except Exception as exc:
                logger.error("traceability.llm_error", error=str(exc))

        logger.info("traceability.done", total_links=len(links))
        return links
