"""Baseline system adapter – direct LLM calls, no RE pipeline.

This represents the "no-pipeline" baseline: a raw GPT call that is given
the source documents and asked to assess / extract requirements directly.

Track 1 – evaluate_candidates():
    For each candidate in the pool, ask the LLM whether it is supported
    by the source documents.  One call per candidate (sequential).

Track 2 – extract_requirements():
    Ask the LLM to extract all requirements from the (poisoned) source
    documents in a single call and return them as a JSON array.
"""

from __future__ import annotations

import json
import time
from typing import Any

import structlog
from openai import OpenAI

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


# ── Static prompts ────────────────────────────────────────────────────────────

_SUPPORT_CHECK_SYSTEM = """\
You are a requirements engineering expert performing evidence grounding.

You will receive:
1. Source evidence documents (stakeholder interview transcript, meeting notes, email thread).
2. A single candidate software requirement.

Your task: determine whether the candidate requirement is SUPPORTED by the source documents.
A requirement is supported if the source documents explicitly state or clearly imply this need.

Respond with a strict JSON object — no extra text:
{
  "status": "accepted" | "rejected",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence>"
}"""

_EXTRACTION_SYSTEM = """\
You are a requirements engineering expert.

Extract ALL functional and non-functional requirements from the provided source documents.
- Include only requirements that are explicitly stated or clearly implied.
- Do NOT fabricate requirements not grounded in the source text.
- Each requirement must be a single, precise, atomic statement.

Respond with a strict JSON object — no extra text:
{
  "requirements": [
    {
      "id": "REQ_001",
      "text": "<requirement statement>",
      "requirement_kind": "functional" | "non_functional" | "constraint",
      "nfr_subtype": "<performance|security|usability|reliability|not_applicable>"
    }
  ]
}"""


def _build_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.azure_openai_api_key,
        base_url=settings.azure_openai_base_url,
    )


def _source_block(source_texts: list[dict[str, Any]]) -> str:
    """Format source_texts into a single readable evidence block."""
    parts = []
    for st in source_texts:
        doc_type = st.get("type", "document").replace("_", " ").upper()
        title = st.get("title", "")
        text = st.get("text", "")
        parts.append(f"=== {doc_type}: {title} ===\n{text}")
    return "\n\n".join(parts)


class BaselineAdapter(SystemAdapter):
    """Direct LLM adapter — no requirement engineering pipeline."""

    @property
    def system_id(self) -> str:
        return "baseline"

    # ── Track 1 ──────────────────────────────────────────────────────────────

    def evaluate_candidates(
        self,
        artifact: PoisonedTrack1Artifact,
    ) -> Track1SystemOutput:
        settings = get_settings()
        client = _build_client()
        source_block = _source_block(artifact.source_texts)
        decisions: list[CandidateDecision] = []
        t0 = time.perf_counter()

        for item in artifact.candidate_pool:
            user_prompt = (
                f"SOURCE DOCUMENTS:\n{source_block}\n\n"
                f"CANDIDATE REQUIREMENT (id={item.id}):\n{item.text}\n\n"
                "Is this candidate requirement supported by the source documents?"
            )
            try:
                resp = client.chat.completions.create(
                    model=settings.azure_openai_chat_deployment,
                    messages=[
                        {"role": "system", "content": _SUPPORT_CHECK_SYSTEM},
                        {"role": "user", "content": user_prompt},
                    ],
                    response_format={"type": "json_object"},
                    max_completion_tokens=16000
                )
                raw = resp.choices[0].message.content or "{}"
                parsed = json.loads(raw)
                status = parsed.get("status", "uncertain")
                if status not in ("accepted", "rejected"):
                    status = "uncertain"
                conf = parsed.get("confidence", 0.5)
                decisions.append(
                    CandidateDecision(
                        candidate_id=item.id,
                        status=status,
                        confidence=float(conf) if conf is not None else 0.5,
                        signal_source="system_output",
                        explanation=parsed.get("explanation", ""),
                    )
                )
            except Exception as exc:
                logger.error(
                    "baseline.support_check_error",
                    candidate_id=item.id,
                    error=str(exc),
                )
                decisions.append(
                    CandidateDecision(
                        candidate_id=item.id,
                        status="uncertain",
                        confidence=0.0,
                        signal_source="system_output",
                        explanation=f"LLM error: {exc}",
                    )
                )

        logger.info(
            "baseline.track1_done",
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
        )

    # ── Track 2 ──────────────────────────────────────────────────────────────

    def extract_requirements(
        self,
        artifact: PoisonedTrack2Artifact,
    ) -> Track2SystemOutput:
        settings = get_settings()
        client = _build_client()
        source_block = _source_block(artifact.source_texts)
        t0 = time.perf_counter()

        user_prompt = (
            "Extract all requirements from the following source documents:\n\n"
            f"{source_block}"
        )
        extracted: list[ExtractedRequirement] = []

        try:
            resp = client.chat.completions.create(
                model=settings.azure_openai_chat_deployment,
                messages=[
                    {"role": "system", "content": _EXTRACTION_SYSTEM},
                    {"role": "user", "content": user_prompt},
                ],
                response_format={"type": "json_object"},
                max_completion_tokens=16000
            )
            raw = resp.choices[0].message.content or "{}"
            parsed = json.loads(raw)
            items = parsed.get("requirements", parsed.get("items", []))
            if not isinstance(items, list):
                items = []
            for i, item in enumerate(items):
                extracted.append(
                    ExtractedRequirement(
                        id=item.get("id", f"BASELINE_REQ_{i + 1:03d}"),
                        text=item.get("text", ""),
                        requirement_kind=item.get("requirement_kind", "functional"),
                        nfr_subtype=item.get("nfr_subtype", "not_applicable"),
                    )
                )
        except Exception as exc:
            logger.error(
                "baseline.extraction_error",
                unit_id=artifact.unit_id,
                error=str(exc),
            )

        logger.info(
            "baseline.track2_done",
            unit_id=artifact.unit_id,
            extracted=len(extracted),
            elapsed=round(time.perf_counter() - t0, 2),
        )
        return Track2SystemOutput(
            unit_id=artifact.unit_id,
            artifact_id=artifact.artifact_id,
            system_id=self.system_id,
            extracted_requirements=extracted,
            execution_time_s=time.perf_counter() - t0,
        )
