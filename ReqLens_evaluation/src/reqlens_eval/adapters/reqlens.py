"""ReqLens v2 (ReqLens) system adapter.

ReqLens v2 is the full multi-agent RE pipeline in ReqLens/.
This adapter calls its agents directly (headless, no FastAPI / DB required)
by adding the ReqLens/src directory to sys.path at runtime.

Track 1 – evaluate_candidates():
  Uses ReqLens v2's EvidenceAgent (the anti-hallucination gate) to assess
  each candidate in the pool against the source texts.  Source texts are
  chunked into SourceSpan objects; a SpanIndex is built with text-search
  fallback (embeddings are optional — computed when the embedding deployment
  is configured, else skipped gracefully).

  Decision mapping:
    EvidenceStatus.entailed              → accepted  (system_metadata)
    EvidenceStatus.contradicted          → rejected  (system_metadata), or
                                           uncertain if confidence <= 0.75 —
                                           low-confidence contradictions from a
                                           partial span set are re-evaluated by judge
    EvidenceStatus.insufficient_evidence → rejected  (system_metadata), or
                                           uncertain if confidence < 0.90 —
                                           borderline cases are re-evaluated by
                                           the shared family-A judge
    exception / no signal                → uncertain

Track 2 – extract_requirements():
  Uses ReqLens v2's ExtractionAgent to extract requirements from the
  (poisoned) source texts, then runs ConsistencyAgent to detect
  contradictions and duplicates in the extracted set.

  Post-extraction filtering:
    duplicate     → keep only the first occurrence (later copies dropped)
    contradiction → remove ALL requirements in the conflicting pair/group
                    (a seeded contradiction will appear as two conflicting
                    requirements; dropping both surfaces the defect)

  The agent is called without DB persistence (candidate_repo=None) so no
  database setup is needed.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import structlog

from reqlens_eval.adapters.base import SystemAdapter
from reqlens_eval.config import get_settings
from reqlens_eval.models.artifacts import (
    PoisonedTrack1Artifact,
    PoisonedTrack2Artifact,
)
from reqlens_eval.models.experiment import (
    CandidateDecision,
    ExtractedRequirement,
    Track1SystemOutput,
    Track2SystemOutput,
)

logger = structlog.get_logger(__name__)

# Sentinel to detect whether the ReqLens import was successful
_REQLENS_AVAILABLE: bool = False
_IMPORT_ERROR: str = ""


def _ensure_reqlens_on_path() -> bool:
    """Add ReqLens/src to sys.path so reqlens can be imported."""
    global _REQLENS_AVAILABLE, _IMPORT_ERROR
    if _REQLENS_AVAILABLE:
        return True

    settings = get_settings()
    src_path = Path(settings.reqlens_src_path).resolve()
    if not src_path.exists():
        _IMPORT_ERROR = f"ReqLens src path not found: {src_path}"
        logger.error("reqlens.src_path_missing", path=str(src_path))
        return False

    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))

    try:
        import reqlens  # noqa: F401  — just verify import works
        _REQLENS_AVAILABLE = True
        return True
    except ImportError as exc:
        _IMPORT_ERROR = str(exc)
        logger.error("reqlens.import_failed", error=str(exc))
        return False


def _make_span_id(doc_idx: int, chunk_idx: int) -> str:
    return f"SPN_{doc_idx:02d}_{chunk_idx:03d}"


def _chunk_text(text: str, chunk_size: int = 1500, overlap: int = 150) -> list[str]:
    """Split text into overlapping character-level chunks."""
    if len(text) <= chunk_size:
        return [text]
    chunks = []
    start = 0
    while start < len(text):
        end = min(start + chunk_size, len(text))
        chunks.append(text[start:end])
        if end == len(text):
            break
        start += chunk_size - overlap
    return chunks


def _build_spans_from_source_texts(
    source_texts: list[dict[str, Any]],
    project_id: str,
) -> list[Any]:
    """Convert source_texts dicts to SourceSpan objects for ReqLens agents."""
    from reqlens.domain.models import SourceSpan  # type: ignore[import]

    spans = []
    for doc_idx, st in enumerate(source_texts):
        full_text = st.get("text", "")
        doc_type = st.get("type", "document")
        title = st.get("title", "")
        chunks = _chunk_text(full_text)
        for chunk_idx, chunk_text in enumerate(chunks):
            span = SourceSpan(
                id=_make_span_id(doc_idx, chunk_idx),
                project_id=project_id,
                document_id=f"DOC_{doc_idx:02d}",
                span_index=chunk_idx,
                text=chunk_text,
                char_start=0,
                char_end=len(chunk_text),
                section_title=title,
                speaker=doc_type,
            )
            spans.append(span)
    return spans


def _build_candidates_from_pool(
    candidate_pool: list[Any],
    project_id: str,
) -> list[Any]:
    """Convert CandidatePoolItem objects to RequirementCandidate for EvidenceAgent."""
    from reqlens.domain.enums import NFRSubtype, RequirementKind, RequirementStatus  # type: ignore[import]
    from reqlens.domain.models import RequirementCandidate  # type: ignore[import]

    # Map string → enum with safe fallback
    def _kind(k: str) -> RequirementKind:
        try:
            return RequirementKind(k)
        except ValueError:
            return RequirementKind.functional

    def _nfr(n: str) -> NFRSubtype:
        try:
            return NFRSubtype(n)
        except ValueError:
            return NFRSubtype.not_applicable

    candidates = []
    for item in candidate_pool:
        cand = RequirementCandidate(
            id=item.id,
            project_id=project_id,
            text=item.text,
            requirement_kind=_kind(item.requirement_kind),
            nfr_subtype=_nfr(item.nfr_subtype),
            status=RequirementStatus.candidate,
        )
        candidates.append(cand)
    return candidates


class ReqLensV2Adapter(SystemAdapter):
    """ReqLens v2 (ReqLens) adapter — headless agent execution."""

    @property
    def system_id(self) -> str:
        return "reqlens_v2"

    # ── Track 1 ──────────────────────────────────────────────────────────────

    def evaluate_candidates(
        self,
        artifact: PoisonedTrack1Artifact,
    ) -> Track1SystemOutput:
        if not _ensure_reqlens_on_path():
            logger.error(
                "reqlens.track1_unavailable",
                reason=_IMPORT_ERROR,
                unit_id=artifact.unit_id,
            )
            return self._fallback_track1(artifact, reason=_IMPORT_ERROR)

        from reqlens.agents.base import AgentContext  # type: ignore[import]
        from reqlens.agents.evidence_agent import EvidenceAgent  # type: ignore[import]
        from reqlens.domain.enums import EvidenceStatus  # type: ignore[import]
        from reqlens.ingestion.span_index import SpanIndex  # type: ignore[import]
        from reqlens.llm.azure_client import AzureOpenAIClient  # type: ignore[import]

        project_id = f"EVAL_{artifact.unit_id}"
        t0 = time.perf_counter()

        # Build spans and index
        spans = _build_spans_from_source_texts(artifact.source_texts, project_id)
        span_index = SpanIndex()
        span_index.add_spans(spans)

        # Create headless LLM client
        llm = AzureOpenAIClient()

        # Optionally embed spans for vector search (graceful skip if unavailable)
        self._try_embed_spans(llm, spans, span_index)

        # Build candidates
        candidates = _build_candidates_from_pool(artifact.candidate_pool, project_id)

        # Pre-link each candidate to its most relevant spans via embedding similarity.
        # EvidenceAgent._retrieve_evidence_spans() checks source_span_ids first and
        # only falls back to substring text search — which always fails for synthesised
        # requirement text.  Pre-linking gives the agent real evidence spans to use.
        self._link_candidates_to_spans(llm, candidates, span_index)

        # Create evidence agent
        try:
            agent = EvidenceAgent(llm=llm, span_index=span_index)
            context = AgentContext(project_id=project_id)
            assessments = agent.assess_candidates(context, candidates)
        except Exception as exc:
            logger.error(
                "reqlens.evidence_agent_failed",
                unit_id=artifact.unit_id,
                error=str(exc),
            )
            return self._fallback_track1(artifact, reason=str(exc))

        # Build per-candidate evidence map: each candidate's judge sees only its
        # pre-linked retrieved spans (not the full raw source).  This bounds the
        # Family-A judge's evidence to what reqlens_v2 actually retrieved, making
        # Track 1 architecturally honest.  Candidates without linked spans
        # (e.g. when embeddings are unavailable) fall back to full source_texts
        # inside normalize_track1.
        span_dict = {s.id: s for s in spans}
        candidate_evidence: dict[str, list[dict]] = {}
        for cand in candidates:
            if cand.source_span_ids:
                linked_spans = [
                    span_dict[sid]
                    for sid in cand.source_span_ids
                    if sid in span_dict
                ]
                if linked_spans:
                    candidate_evidence[cand.id] = [
                        {
                            "type": "retrieved_span",
                            "title": s.section_title or "",
                            "text": s.text,
                        }
                        for s in linked_spans
                    ]

        # Map assessments → CandidateDecision
        assessment_map = {a.requirement_candidate_id: a for a in assessments}
        decisions: list[CandidateDecision] = []
        for item in artifact.candidate_pool:
            assessment = assessment_map.get(item.id)
            if assessment is None:
                decisions.append(
                    CandidateDecision(
                        candidate_id=item.id,
                        status="uncertain",
                        confidence=0.0,
                        signal_source="system_metadata",
                        explanation="No assessment produced by evidence agent.",
                    )
                )
                continue

            if assessment.status == EvidenceStatus.entailed:
                status = "accepted"
            else:
                # EvidenceAgent acts as evidence curator, not gatekeeper.
                # Positive signal (entailed) is reliable — the agent found
                # supporting spans.  Negative signals (contradicted /
                # insufficient_evidence) are unreliable because the agent
                # only sees top-k retrieved spans out of the full source.
                # All non-entailed candidates are escalated to the Family-A
                # judge, which receives the agent's retrieved spans as
                # curated evidence.  This prevents false gold rejections
                # while preserving the architectural advantage of span-based
                # evidence retrieval.
                status = "uncertain"

            decisions.append(
                CandidateDecision(
                    candidate_id=item.id,
                    status=status,
                    confidence=assessment.confidence,
                    signal_source="system_metadata",
                    explanation=assessment.explanation or "",
                )
            )

        logger.info(
            "reqlens.track1_done",
            unit_id=artifact.unit_id,
            total=len(decisions),
            elapsed=round(time.perf_counter() - t0, 2),
        )
        return Track1SystemOutput(
            unit_id=artifact.unit_id,
            artifact_id=artifact.artifact_id,
            system_id=self.system_id,
            decisions=decisions,
            execution_time_s=time.perf_counter() - t0,
            metadata={
                "span_count": len(spans),
                "candidate_evidence": candidate_evidence,
            },
        )

    # ── Track 2 ──────────────────────────────────────────────────────────────

    def extract_requirements(
        self,
        artifact: PoisonedTrack2Artifact,
    ) -> Track2SystemOutput:
        if not _ensure_reqlens_on_path():
            logger.error(
                "reqlens.track2_unavailable",
                reason=_IMPORT_ERROR,
                unit_id=artifact.unit_id,
            )
            return self._fallback_track2(artifact, reason=_IMPORT_ERROR)

        from reqlens.agents.base import AgentContext  # type: ignore[import]
        from reqlens.agents.consistency_agent import ConsistencyAgent  # type: ignore[import]
        from reqlens.agents.extraction_agent import ExtractionAgent  # type: ignore[import]
        from reqlens.domain.enums import ConflictType  # type: ignore[import]
        from reqlens.domain.models import Requirement  # type: ignore[import]
        from reqlens.llm.azure_client import AzureOpenAIClient  # type: ignore[import]

        project_id = f"EVAL_{artifact.unit_id}"
        t0 = time.perf_counter()

        spans = _build_spans_from_source_texts(artifact.source_texts, project_id)

        # ── Step 1: Extract candidates (synchronous, no DB writes) ───────────
        try:
            llm = AzureOpenAIClient()
            extraction_agent = ExtractionAgent(llm=llm, candidate_repo=None)
            context = AgentContext(project_id=project_id)
            candidates, _ = extraction_agent.get_candidates(context, spans)
        except Exception as exc:
            logger.error(
                "reqlens.extraction_agent_failed",
                unit_id=artifact.unit_id,
                error=str(exc),
            )
            return self._fallback_track2(artifact, reason=str(exc))

        logger.info(
            "reqlens.track2_extracted_raw",
            unit_id=artifact.unit_id,
            raw_count=len(candidates),
        )

        # ── Step 2: Run ConsistencyAgent over the full extracted set ─────────
        # ConsistencyAgent expects list[Requirement] (promoted domain object).
        # We promote each RequirementCandidate → Requirement; no DB write occurs.
        requirements_for_consistency: list[Requirement] = [
            Requirement(
                id=cand.id,
                project_id=cand.project_id,
                text=cand.text,
                kind=cand.requirement_kind,   # candidate.requirement_kind → Requirement.kind
                nfr_subtype=cand.nfr_subtype,
                source_span_ids=cand.source_span_ids,
            )
            for cand in candidates
        ]

        conflict_findings = []
        try:
            consistency_agent = ConsistencyAgent(llm=llm)
            conflict_findings = consistency_agent.detect_conflicts(
                context, requirements_for_consistency
            )
            logger.info(
                "reqlens.track2_consistency_done",
                unit_id=artifact.unit_id,
                conflicts_found=len(conflict_findings),
                contradictions=sum(
                    1 for f in conflict_findings
                    if f.conflict_type == ConflictType.contradiction
                ),
                duplicates=sum(
                    1 for f in conflict_findings
                    if f.conflict_type == ConflictType.duplicate
                ),
            )
            for finding in conflict_findings:
                logger.info(
                    "reqlens.track2_conflict",
                    unit_id=artifact.unit_id,
                    conflict_type=finding.conflict_type.value,
                    severity=finding.severity.value,
                    involved_ids=finding.involved_requirement_ids,
                    explanation=finding.explanation,
                    suggested_resolution=finding.suggested_resolution,
                )
        except Exception as exc:
            logger.warning(
                "reqlens.track2_consistency_skipped",
                unit_id=artifact.unit_id,
                error=str(exc),
            )

        # ── Step 3: Filter conflicting requirements ───────────────────────────
        # duplicate:    keep the first occurrence; drop later copies.
        # contradiction: drop ALL involved IDs — a seeded contradiction produces
        #               two conflicting requirements; dropping both flags the defect.
        ids_to_drop: set[str] = set()
        for finding in conflict_findings:
            if finding.conflict_type == ConflictType.duplicate:
                involved_ordered = [
                    c.id for c in candidates
                    if c.id in finding.involved_requirement_ids
                ]
                ids_to_drop.update(involved_ordered[1:])  # keep [0], drop the rest
            elif finding.conflict_type == ConflictType.contradiction:
                ids_to_drop.update(finding.involved_requirement_ids)

        if ids_to_drop:
            logger.info(
                "reqlens.track2_filtered",
                unit_id=artifact.unit_id,
                dropped_count=len(ids_to_drop),
                dropped_ids=sorted(ids_to_drop),
            )

        # ── Step 4: Build final ExtractedRequirement list ────────────────────
        extracted: list[ExtractedRequirement] = [
            ExtractedRequirement(
                id=cand.id,
                text=cand.text,
                requirement_kind=cand.requirement_kind.value,
                nfr_subtype=cand.nfr_subtype.value,
            )
            for cand in candidates
            if cand.id not in ids_to_drop
        ]

        logger.info(
            "reqlens.track2_done",
            unit_id=artifact.unit_id,
            raw_extracted=len(candidates),
            after_consistency_filter=len(extracted),
            conflicts_resolved=len(ids_to_drop),
            elapsed=round(time.perf_counter() - t0, 2),
        )
        return Track2SystemOutput(
            unit_id=artifact.unit_id,
            artifact_id=artifact.artifact_id,
            system_id=self.system_id,
            extracted_requirements=extracted,
            execution_time_s=time.perf_counter() - t0,
            metadata={
                "span_count": len(spans),
                "raw_extracted": len(candidates),
                "conflicts_found": len(conflict_findings),
                "ids_filtered": sorted(ids_to_drop),
            },
        )

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _try_embed_spans(self, llm: Any, spans: list[Any], span_index: Any) -> None:
        """Attempt to embed spans for vector search; silently skip on failure."""
        try:
            texts = [s.text for s in spans]
            embeddings = llm.embed_texts(texts)
            for span, emb in zip(spans, embeddings):
                span.embedding = emb
            # Rebuild index with embeddings
            span_index.add_spans(spans)
        except Exception as exc:
            logger.warning(
                "reqlens.embedding_skipped",
                reason=str(exc),
            )

    def _link_candidates_to_spans(
        self, llm: Any, candidates: list[Any], span_index: Any, top_k: int = 8
    ) -> None:
        """Embed each candidate and pre-assign source_span_ids via embedding similarity.

        EvidenceAgent._retrieve_evidence_spans() uses source_span_ids first (direct
        link), then falls back to substring text search which fails for synthesised
        requirement sentences.  Pre-linking via embeddings gives the agent real source
        evidence to judge against.

        Silently skipped when the span index has no embeddings (e.g. embedding
        deployment not configured) — in that case source_span_ids stays empty and
        EvidenceAgent falls back to text search as before.
        """
        try:
            texts = [c.text for c in candidates]
            embeddings = llm.embed_texts(texts)
            linked = 0
            for candidate, emb in zip(candidates, embeddings):
                results = span_index.search_by_embedding(emb, top_k=top_k)
                if results:
                    candidate.source_span_ids = [s.id for s, _ in results]
                    linked += 1
            logger.info(
                "reqlens.candidates_linked",
                total=len(candidates),
                linked=linked,
                top_k=top_k,
            )
        except Exception as exc:
            logger.warning("reqlens.candidate_linking_skipped", reason=str(exc))

    def _fallback_track1(
        self,
        artifact: PoisonedTrack1Artifact,
        reason: str = "",
    ) -> Track1SystemOutput:
        """Return all-uncertain output when ReqLens v2 is unavailable."""
        decisions = [
            CandidateDecision(
                candidate_id=item.id,
                status="uncertain",
                confidence=0.0,
                signal_source="system_metadata",
                explanation=f"ReqLens v2 unavailable: {reason}",
            )
            for item in artifact.candidate_pool
        ]
        return Track1SystemOutput(
            unit_id=artifact.unit_id,
            artifact_id=artifact.artifact_id,
            system_id=self.system_id,
            decisions=decisions,
            metadata={"error": reason},
        )

    def _fallback_track2(
        self,
        artifact: PoisonedTrack2Artifact,
        reason: str = "",
    ) -> Track2SystemOutput:
        """Return empty extraction output when ReqLens v2 is unavailable."""
        return Track2SystemOutput(
            unit_id=artifact.unit_id,
            artifact_id=artifact.artifact_id,
            system_id=self.system_id,
            extracted_requirements=[],
            metadata={"error": reason},
        )
