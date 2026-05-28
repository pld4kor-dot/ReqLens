"""Evidence Agent – verify candidate requirements against source evidence.

Core anti-hallucination gate: a requirement can only be promoted if the
evidence agent says ENTAILED or a human overrides.
"""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.enums import EvidenceStatus, RequirementStatus
from reqlens.domain.ids import generate_id
from reqlens.domain.models import EvidenceAssessment, RequirementCandidate, SourceSpan
from reqlens.ingestion.span_index import SpanIndex
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import EVIDENCE_SYSTEM_V1
from reqlens.llm.schemas import EvidenceAssessmentLLM

logger = structlog.get_logger(__name__)


class EvidenceAgent(BaseAgent):
    name = "evidence"

    def __init__(self, llm: AzureOpenAIClient, span_index: SpanIndex) -> None:
        self.llm = llm
        self.span_index = span_index

    async def run(
        self,
        context: AgentContext,
        candidates: list[RequirementCandidate],
    ) -> AgentResult:
        """Assess each candidate against source evidence."""
        assessments = self.assess_candidates(context, candidates)
        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[a.id for a in assessments],
        )

    def assess_candidates(
        self,
        context: AgentContext,
        candidates: list[RequirementCandidate],
    ) -> list[EvidenceAssessment]:
        """For each candidate, retrieve top-k spans and judge entailment."""
        assessments: list[EvidenceAssessment] = []

        for candidate in candidates:
            # Retrieve relevant spans
            spans = self._retrieve_evidence_spans(candidate)

            if not spans:
                # No evidence found at all
                assessment = EvidenceAssessment(
                    id=generate_id("EVD"),
                    project_id=context.project_id,
                    requirement_candidate_id=candidate.id,
                    status=EvidenceStatus.insufficient_evidence,
                    explanation="No source spans found for this candidate.",
                    confidence=0.0,
                )
                assessments.append(assessment)
                continue

            # Build prompt
            span_block = "\n\n".join(
                f"[{s.id}] {s.text}" for s in spans
            )
            user_prompt = (
                f"Candidate requirement ({candidate.id}):\n"
                f"{candidate.text}\n\n"
                f"Source spans:\n{span_block}"
            )

            try:
                llm_result: EvidenceAssessmentLLM = self.llm.structured_chat(
                    system_prompt=EVIDENCE_SYSTEM_V1,
                    user_prompt=user_prompt,
                    response_model=EvidenceAssessmentLLM,
                    project_id=context.project_id,
                    agent_name=self.name,
                )

                assessment = EvidenceAssessment(
                    id=generate_id("EVD"),
                    project_id=context.project_id,
                    requirement_candidate_id=candidate.id,
                    status=llm_result.status,
                    supporting_span_ids=llm_result.supporting_span_ids,
                    contradicting_span_ids=llm_result.contradicting_span_ids,
                    explanation=llm_result.explanation,
                    confidence=llm_result.confidence,
                )
            except Exception as exc:
                logger.error("evidence.llm_error", candidate=candidate.id, error=str(exc))
                assessment = EvidenceAssessment(
                    id=generate_id("EVD"),
                    project_id=context.project_id,
                    requirement_candidate_id=candidate.id,
                    status=EvidenceStatus.insufficient_evidence,
                    explanation=f"LLM error: {exc}",
                    confidence=0.0,
                )

            assessments.append(assessment)

        entailed = sum(1 for a in assessments if a.status == EvidenceStatus.entailed)
        contradicted = sum(1 for a in assessments if a.status == EvidenceStatus.contradicted)
        insufficient = sum(1 for a in assessments if a.status == EvidenceStatus.insufficient_evidence)

        logger.info(
            "evidence.done",
            total=len(assessments),
            entailed=entailed,
            contradicted=contradicted,
            insufficient=insufficient,
        )

        return assessments

    def _retrieve_evidence_spans(
        self,
        candidate: RequirementCandidate,
        top_k: int = 5,
    ) -> list[SourceSpan]:
        """Retrieve source spans relevant to a candidate.

        Strategy: use the candidate's linked span IDs first, then
        fall back to embedding search.
        """
        # Direct link
        if candidate.source_span_ids:
            spans = self.span_index.get_many(candidate.source_span_ids)
            if spans:
                return spans

        # Embedding search fallback
        return self.span_index.search_by_text(candidate.text, top_k=top_k)
