"""Extraction Agent – extract atomic candidate requirements from source spans.

Bug fixes in this version
--------------------------
1. ExtractionAgent.run() now accepts an optional ``candidate_repo``
   (RequirementCandidateRepository) and persists the extracted candidates
   immediately after deduplication.  Previously run() discarded the
   candidates it extracted — they were never written to the DB — so every
   downstream agent saw an empty list.

2. _extract_sync() is no longer called from run().  Both methods share the
   same internal _do_extraction() helper so the LLM is called exactly once.

3. The orchestrator / routes_agents no longer need to call _extract_sync()
   separately; they can read persisted candidates from the DB via
   RequirementCandidateRepository.list_by_project().
"""

from __future__ import annotations

import structlog

from reqlens.agents.base import AgentContext, AgentResult, BaseAgent
from reqlens.domain.enums import RequirementStatus
from reqlens.domain.ids import generate_id
from reqlens.domain.models import RequirementCandidate, SourceSpan
from reqlens.llm.azure_client import AzureOpenAIClient
from reqlens.llm.prompts import EXTRACTION_SYSTEM_V1
from reqlens.llm.schemas import RequirementExtractionOutput
from reqlens.llm.token_budget import batch_texts_by_budget

logger = structlog.get_logger(__name__)


class ExtractionAgent(BaseAgent):
    name = "extraction"

    def __init__(
        self,
        llm: AzureOpenAIClient,
        candidate_repo=None,   # RequirementCandidateRepository | None
    ) -> None:
        self.llm = llm
        self.candidate_repo = candidate_repo   # set by routes_agents / orchestrator

    # ------------------------------------------------------------------
    # Public: async run (called by orchestrator and routes_agents)
    # ------------------------------------------------------------------

    async def run(
        self,
        context: AgentContext,
        spans: list[SourceSpan],
    ) -> AgentResult:
        """Extract candidates, deduplicate, persist to DB, return AgentResult.

        If ``self.candidate_repo`` is set, candidates are written to the
        database so downstream agents can read them.  If it is None the
        candidates are returned only in ``AgentResult.created_ids`` (useful
        for unit tests that don't need DB persistence).
        """
        if not spans:
            return AgentResult(
                agent_name=self.name,
                status="completed",
                warnings=["No spans provided — nothing extracted."],
            )

        candidates, questions = self._do_extraction(context, spans)

        if not candidates:
            return AgentResult(
                agent_name=self.name,
                status="completed",
                warnings=["LLM returned zero candidates for all batches."],
            )

        # ── Persist to DB so downstream agents can read from the DB ───
        if self.candidate_repo is not None:
            try:
                self.candidate_repo.create_many(candidates)
                logger.info(
                    "extraction.persisted",
                    count=len(candidates),
                    project_id=context.project_id,
                )
            except Exception as exc:
                logger.error("extraction.persist_failed", error=str(exc))
                return AgentResult(
                    agent_name=self.name,
                    status="failed",
                    errors=[f"DB persist failed: {exc}"],
                )
        else:
            logger.warning(
                "extraction.no_repo",
                note="candidate_repo not set; candidates not persisted to DB",
            )

        logger.info(
            "extraction.done",
            total=len(candidates),
            questions=len(questions),
            project_id=context.project_id,
        )

        return AgentResult(
            agent_name=self.name,
            status="completed",
            created_ids=[c.id for c in candidates],
        )

    # ------------------------------------------------------------------
    # Public: synchronous helper (kept for backward-compat with orchestrator)
    # ------------------------------------------------------------------

    def get_candidates(
        self,
        context: AgentContext,
        spans: list[SourceSpan],
    ) -> tuple[list[RequirementCandidate], list[str]]:
        """Synchronous wrapper — returns (candidates, questions).

        NOTE: this does NOT persist to the DB.  Callers that need persistence
        should call run() instead.
        """
        return self._do_extraction(context, spans)

    # kept for backward compat — same as get_candidates
    def _extract_sync(
        self,
        context: AgentContext,
        spans: list[SourceSpan],
    ) -> tuple[list[RequirementCandidate], list[str]]:
        return self._do_extraction(context, spans)

    # ------------------------------------------------------------------
    # Internal: shared extraction logic (called by both run and _extract_sync)
    # ------------------------------------------------------------------

    def _do_extraction(
        self,
        context: AgentContext,
        spans: list[SourceSpan],
    ) -> tuple[list[RequirementCandidate], list[str]]:
        """Run the LLM extraction loop and return (deduped_candidates, questions).

        Does NOT touch the database.  Persistence is the caller's responsibility.
        """
        if not spans:
            return [], []

        span_texts = [f"[{s.id}] {s.text}" for s in spans]
        batches = batch_texts_by_budget(span_texts, max_tokens_per_batch=80_000)

        all_candidates: list[RequirementCandidate] = []
        all_questions: list[str] = []

        for batch_idx, batch in enumerate(batches):
            user_prompt = (
                "Source spans:\n\n"
                + "\n\n---\n\n".join(batch)
                + "\n\nExtract all atomic requirements from the above source spans."
            )
            try:
                output: RequirementExtractionOutput = self.llm.structured_chat(
                    system_prompt=EXTRACTION_SYSTEM_V1,
                    user_prompt=user_prompt,
                    response_model=RequirementExtractionOutput,
                    project_id=context.project_id,
                    agent_name=self.name,
                )
            except Exception as exc:
                logger.error(
                    "extraction.llm_error",
                    batch=batch_idx,
                    error=str(exc),
                )
                continue

            for llm_cand in output.candidates:
                candidate = RequirementCandidate(
                    id=generate_id("CAND"),
                    project_id=context.project_id,
                    text=llm_cand.text,
                    requirement_kind=llm_cand.requirement_kind,
                    nfr_subtype=llm_cand.nfr_subtype,
                    source_span_ids=llm_cand.source_span_ids,
                    stakeholders=llm_cand.stakeholders,
                    rationale=llm_cand.rationale,
                    confidence=llm_cand.confidence,
                    status=RequirementStatus.candidate,
                )
                all_candidates.append(candidate)

            all_questions.extend(output.unresolved_questions)

        deduped = self._deduplicate(all_candidates)
        return deduped, all_questions

    # ------------------------------------------------------------------
    # Internal: deduplication
    # ------------------------------------------------------------------

    @staticmethod
    def _deduplicate(
        candidates: list[RequirementCandidate],
    ) -> list[RequirementCandidate]:
        """Remove near-identical candidates based on normalised text."""
        seen: set[str] = set()
        result: list[RequirementCandidate] = []
        for c in candidates:
            key = c.text.strip().lower()
            if key not in seen:
                seen.add(key)
                result.append(c)
        return result
