"""Requirement merger for PURE documents.

Takes raw candidate requirements from both extraction paths (section + raw),
performs text-level deduplication first (cheap), then sends batches to the
LLM for semantic deduplication and paraphrase merging.

Returns a final, clean list of ``GoldRequirement`` objects.
"""

from __future__ import annotations

import re

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_merge_requirements_prompt,
)
from reqlens_benchmark_builder.schemas.benchmark_models import GoldRequirement

logger = structlog.get_logger(__name__)

# Maximum number of candidates per LLM merge call.
# Large sets are batched to avoid exceeding context limits.
_MERGE_BATCH_SIZE = 80


def _normalize_for_dedup(text: str) -> str:
    """Lower-case, collapse whitespace, strip punctuation for exact dedup."""
    text = text.lower().strip()
    text = re.sub(r"[^a-z0-9\s]", "", text)
    return re.sub(r"\s+", " ", text)


def _text_dedup(candidates: list[dict]) -> list[dict]:
    """Remove exact duplicates (after normalization) cheaply before the LLM call."""
    seen: set[str] = set()
    out: list[dict] = []
    for c in candidates:
        key = _normalize_for_dedup(c.get("text", ""))
        if key and key not in seen:
            seen.add(key)
            out.append(c)
    logger.info(
        "merger.text_dedup",
        before=len(candidates),
        after=len(out),
        dropped=len(candidates) - len(out),
    )
    return out


def _merge_batch(
    llm: AzureOpenAIClient,
    doc_id: str,
    batch: list[dict],
    batch_no: int,
) -> list[dict]:
    """Send one batch to the LLM for semantic merge and return raw merged list."""
    settings = get_settings()
    prompt = build_merge_requirements_prompt(doc_id=doc_id, candidates=batch)
    try:
        result = llm.chat_json(
            system_prompt=EXTRACTION_SYSTEM_PROMPT,
            user_prompt=prompt,
            temperature=settings.temp_generation,
            model=settings.azure_openai_reasoning_deployment,
            max_tokens=settings.max_tokens_merge,
        )
        return result.get("gold_requirements", [])
    except Exception as exc:
        logger.error(
            "merger.llm_error",
            doc_id=doc_id,
            batch=batch_no,
            error=str(exc)[:200],
        )
        # On LLM failure, return the input batch as-is (better than losing data)
        return batch


def merge_requirements(
    llm: AzureOpenAIClient,
    doc_id: str,
    candidates: list[dict],
) -> list[GoldRequirement]:
    """Deduplicate and normalize candidate requirements into final gold requirements.

    Steps:
    1. Text-level exact dedup (free).
    2. Split into batches of ``_MERGE_BATCH_SIZE`` and send each to the LLM.
    3. If multiple batches, do a second-pass merge across batch outputs.
    4. Convert to ``GoldRequirement`` objects.
    """
    if not candidates:
        logger.warning("merger.no_candidates", doc_id=doc_id)
        return []

    # Step 1: cheap text dedup
    deduped = _text_dedup(candidates)

    # Step 2: LLM batch merge
    batches = [
        deduped[i : i + _MERGE_BATCH_SIZE]
        for i in range(0, len(deduped), _MERGE_BATCH_SIZE)
    ]
    batch_results: list[dict] = []
    for i, batch in enumerate(batches):
        merged = _merge_batch(llm, doc_id, batch, batch_no=i + 1)
        batch_results.extend(merged)

    # Step 3: if more than one batch was used, do a second-pass merge
    if len(batches) > 1 and batch_results:
        logger.info(
            "merger.second_pass",
            doc_id=doc_id,
            candidates=len(batch_results),
        )
        # Text dedup again before second-pass LLM call
        batch_results = _text_dedup(batch_results)
        if len(batch_results) > _MERGE_BATCH_SIZE:
            # Still too large — just proceed; second-pass gets top candidates
            batch_results = batch_results[:_MERGE_BATCH_SIZE]
        final_raw = _merge_batch(llm, doc_id, batch_results, batch_no=0)
    else:
        final_raw = batch_results

    # Step 4: convert to GoldRequirement objects
    out: list[GoldRequirement] = []
    seen_texts: set[str] = set()
    for i, raw in enumerate(final_raw):
        text = (raw.get("text") or "").strip()
        if not text or len(text) < 10:
            continue
        norm_key = _normalize_for_dedup(text)
        if norm_key in seen_texts:
            continue
        seen_texts.add(norm_key)
        out.append(
            GoldRequirement(
                id=f"PURE_{doc_id}_R{i + 1:03d}",
                text=text,
                raw_label=raw.get("raw_label"),
                requirement_kind=raw.get("requirement_kind", "functional"),
                nfr_subtype=raw.get("nfr_subtype", "not_applicable"),
                source_strategy=raw.get("source_strategy"),
                source_region=raw.get("source_region"),
            )
        )

    logger.info(
        "merger.done",
        doc_id=doc_id,
        gold_count=len(out),
    )
    return out
