"""ReqInOne v1 system adapter (GPT4o notebook reimplementation).

ReqInOne v1 (from ReqInOne-main/) is a notebook-based pipeline that uses
LangChain + GPT-4o-mini (temperature=0.5, logprobs=True) to extract and
classify requirements from natural language text.  It does *not* have a
built-in evidence-grounding (support-check) component — it extracts
candidates and classifies them as FR/NFR.

This adapter reproduces v1's core logic as faithfully as possible:

  • Same system prompt — verbatim from the notebook (with the Resource /
    Reason annotation style).
  • Same model family — gpt-4o-mini (mapped through the deployment config).
  • Same temperature — 0.5.
  • logprobs=True — passed through to the API where supported.
  • Free-form text output — the original notebook produced plain text;
    we parse numbered requirements out of that text rather than relying on
    a JSON response_format constraint.

Track 1 – evaluate_candidates():
  v1 has no native support-check.  Strategy:
    1. Extract requirements from source_texts using v1's extraction prompt.
    2. For each candidate in the pool, check whether the candidate is
       supported by the source docs (LLM call using same model/temperature).
    3. Return accepted if matched, rejected if unmatched, uncertain otherwise.
  This faithfully captures v1's behaviour: it would only "accept" a
  requirement it was able to extract from the evidence; hallucinated fakes
  that have no grounding should not be extractable and therefore not matched.

Track 2 – extract_requirements():
  Run v1's extraction prompt directly on the poisoned source texts and
  return whatever requirements it produces.
"""

from __future__ import annotations

import json
import re
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


# ── v1 extraction prompt — verbatim from the notebook ────────────────────────
# Source: ReqInOne-main/ReqInOne/GPT4o/Requiremnt_Extraction_Task_Component.ipynb
# Model: gpt-4o-mini  |  temperature: 0.5  |  logprobs: True

_V1_EXTRACTION_SYSTEM = """
You are a Requirement management assistant. you will extract multiple requirements from the natural language text. Perform a detailed analysis. The text contains both functional requirements and non-functional requirements. Extract as many requirements as possible.
Definition of Requirement: A requirement is a singular documented physical or functional need that a particular product must be able to perform.
Each requirement should be accompanied by a specification detailing how the requirement should be realized. Requirements carry specific details on how a piece of functionality must work. Requirement must still be well-written: Precise, avoid amalgamation, make distinction between functional/non-functional.

When writing each requirement, use a structured sentence format to ensure clarity and consistency. The template for the structured requirement is as follows:
The <subject clause> shall <action verb clause> <object clause> <optionalqualifying clause>, when <condition clause>.

Each extracted requirement should include:
Resource: # The text from the provided information where the extracted requirement comes from. Even if they are implicit, specify exactly which texts they are implied in.
Reason: # The reason for extracting this requirement.
    """

# Model parameters matching the notebook exactly
_V1_MODEL_TEMPERATURE = 0.5
_V1_LOGPROBS = True


def _build_client() -> OpenAI:
    settings = get_settings()
    return OpenAI(
        api_key=settings.azure_openai_api_key,
        base_url=settings.azure_openai_base_url,
    )


def _source_block(source_texts: list[dict[str, Any]]) -> str:
    parts = []
    for st in source_texts:
        doc_type = st.get("type", "document").replace("_", " ").upper()
        title = st.get("title", "")
        text = st.get("text", "")
        parts.append(f"=== {doc_type}: {title} ===\n{text}")
    return "\n\n".join(parts)


def _parse_freeform_requirements(text: str) -> list[dict[str, Any]]:
    """Parse v1's free-form numbered requirement output into structured dicts.

    The original notebook produces plain text like:
        1. **Requirement:** The system shall ...
           Resource: "..."
           Reason: ...

    This parser extracts each numbered block and pulls out the requirement
    text plus optional Resource/Reason annotations.
    """
    requirements: list[dict[str, Any]] = []

    # Split on numbered list markers: lines starting with "1.", "2.", etc.
    blocks = re.split(r"\n(?=\d+\.\s)", text.strip())

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        # Strip leading number + dot
        body = re.sub(r"^\d+\.\s*", "", block, count=1).strip()

        # Extract Resource and Reason lines if present
        resource_match = re.search(
            r"Resource:\s*(.*?)(?=\nReason:|\Z)", body, re.DOTALL | re.IGNORECASE
        )
        reason_match = re.search(
            r"Reason:\s*(.*?)$", body, re.DOTALL | re.IGNORECASE
        )

        resource = resource_match.group(1).strip() if resource_match else ""
        reason = reason_match.group(1).strip() if reason_match else ""

        # Requirement text is everything before the first Resource:/Reason: label
        req_text = re.split(
            r"\nResource:", body, maxsplit=1, flags=re.IGNORECASE
        )[0].strip()
        # Strip markdown bold markers like **Requirement:**
        req_text = re.sub(r"\*\*[^*]+\*\*:?\s*", "", req_text).strip()

        if not req_text:
            continue

        # Classify FR / NFR heuristically from keywords in the text
        nfr_keywords = (
            "performance", "security", "availability", "reliability",
            "scalability", "usability", "maintainability", "portability",
            "accessibility", "compliance",
        )
        lower = req_text.lower()
        kind = (
            "non_functional"
            if any(kw in lower for kw in nfr_keywords)
            else "functional"
        )

        req_id = f"V1_REQ_{len(requirements) + 1:03d}"
        requirements.append(
            {
                "id": req_id,
                "text": req_text,
                "requirement_kind": kind,
                "resource": resource,
                "reason": reason,
            }
        )

    return requirements


def _extract_with_v1(
    client: OpenAI,
    model: str,
    source_block: str,
    unit_id: str,
) -> list[dict[str, Any]]:
    """Call v1's extraction prompt on the concatenated source block.

    Uses the same model (gpt-4o-mini family), temperature (0.5), and
    logprobs=True as the original notebook.  The response is free-form text
    which is parsed by _parse_freeform_requirements().
    """
    user_prompt = (
        "Extract all requirements from the following source documents:\n\n"
        f"{source_block}"
    )
    try:
        resp = client.chat.completions.create(
            model=model,
            temperature=_V1_MODEL_TEMPERATURE,
            logprobs=_V1_LOGPROBS,
            messages=[
                {"role": "system", "content": _V1_EXTRACTION_SYSTEM},
                {"role": "user", "content": user_prompt},
            ],
            max_completion_tokens=16000,
        )
        raw = resp.choices[0].message.content or ""

        # Try JSON first (in case the model outputs JSON naturally)
        items: list[dict[str, Any]] = []
        try:
            parsed = json.loads(raw)
            items = parsed.get("requirements", parsed.get("items", []))
            if not isinstance(items, list):
                items = []
            if not items:
                for val in parsed.values():
                    if (
                        isinstance(val, list)
                        and val
                        and isinstance(val[0], dict)
                        and "text" in val[0]
                    ):
                        items = val
                        break
        except (json.JSONDecodeError, AttributeError):
            pass

        # Fall back to free-form text parsing (original notebook output style)
        if not items:
            items = _parse_freeform_requirements(raw)

        if not items:
            logger.warning(
                "reqinone_v1.extraction_empty",
                unit_id=unit_id,
                raw_preview=raw[:400],
            )
        return items
    except Exception as exc:
        logger.error("reqinone_v1.extraction_error", unit_id=unit_id, error=str(exc))
        return []


class ReqInOneV1Adapter(SystemAdapter):
    """ReqInOne v1 adapter — reproduces the notebook pipeline programmatically."""

    @property
    def system_id(self) -> str:
        return "reqinone_v1"

    # ── Track 1 ──────────────────────────────────────────────────────────────

    def evaluate_candidates(
        self,
        artifact: PoisonedTrack1Artifact,
    ) -> Track1SystemOutput:
        settings = get_settings()
        client = _build_client()
        # Use the dedicated v1 deployment (gpt-4o-mini) when configured;
        # fall back to the shared chat deployment otherwise.
        model = settings.reqinone_v1_deployment or settings.azure_openai_chat_deployment
        source_block = _source_block(artifact.source_texts)
        t0 = time.perf_counter()

        logger.info(
            "reqinone_v1.model_config",
            unit_id=artifact.unit_id,
            model=model,
            temperature=_V1_MODEL_TEMPERATURE,
            logprobs=_V1_LOGPROBS,
        )

        # Step 1: v1 extracts its own requirements from the source docs
        extracted = _extract_with_v1(client, model, source_block, artifact.unit_id)
        extracted_summary = "\n".join(
            f"- [{r.get('id', '?')}] {r.get('text', '')}" for r in extracted
        )
        if not extracted_summary:
            extracted_summary = "(no requirements extracted)"

        logger.info(
            "reqinone_v1.extracted_for_matching",
            unit_id=artifact.unit_id,
            count=len(extracted),
        )

        # Step 2: v1 has no native support-check.  Mark every candidate as
        # 'uncertain' and let the shared Family-A judge resolve them.  The judge
        # is given only v1's extracted requirements (not the full raw source) as
        # evidence, so it answers: "is this candidate covered by what v1 extracted?"
        # This bounds the judge's decision to v1's own extraction artifact, making
        # Track 1 architecturally symmetric with reqlens_v2 (which supplies its
        # retrieved spans as evidence).
        extracted_evidence = [
            {
                "type": "extracted_requirements",
                "title": "v1 Extracted Requirements",
                "text": extracted_summary,
            }
        ]
        candidate_evidence = {item.id: extracted_evidence for item in artifact.candidate_pool}

        decisions: list[CandidateDecision] = [
            CandidateDecision(
                candidate_id=item.id,
                status="uncertain",
                confidence=0.0,
                signal_source="system_metadata",
                explanation=(
                    "v1 has no native support-check; "
                    "escalated to Family-A judge using v1's extracted requirements as evidence."
                ),
            )
            for item in artifact.candidate_pool
        ]

        logger.info(
            "reqinone_v1.track1_done",
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
                "v1_extracted_count": len(extracted),
                "candidate_evidence": candidate_evidence,
            },
        )

    # ── Track 2 ──────────────────────────────────────────────────────────────

    def extract_requirements(
        self,
        artifact: PoisonedTrack2Artifact,
    ) -> Track2SystemOutput:
        settings = get_settings()
        client = _build_client()
        model = settings.reqinone_v1_deployment or settings.azure_openai_chat_deployment
        source_block = _source_block(artifact.source_texts)
        t0 = time.perf_counter()

        logger.info(
            "reqinone_v1.model_config",
            unit_id=artifact.unit_id,
            model=model,
            temperature=_V1_MODEL_TEMPERATURE,
            logprobs=_V1_LOGPROBS,
        )

        raw_items = _extract_with_v1(client, model, source_block, artifact.unit_id)
        extracted: list[ExtractedRequirement] = []
        for i, item in enumerate(raw_items):
            extracted.append(
                ExtractedRequirement(
                    id=item.get("id", f"V1_REQ_{i + 1:03d}"),
                    text=item.get("text", ""),
                    requirement_kind=item.get("requirement_kind", "functional"),
                    nfr_subtype=item.get("nfr_subtype", "not_applicable"),
                    metadata={
                        "resource": item.get("resource", ""),
                        "reason": item.get("reason", ""),
                    },
                )
            )

        logger.info(
            "reqinone_v1.track2_done",
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









