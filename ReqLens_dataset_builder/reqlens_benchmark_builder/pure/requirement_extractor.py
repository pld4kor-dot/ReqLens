"""Requirement extractor for PURE documents.

Sends requirement-dense chunks to the LLM and collects raw candidate
requirements.  Two strategies are supported:
- ``"section"`` — chunks produced by the section-aware path
- ``"raw_chunk"`` — chunks from the full-document sliding-window path

Filtering:
- Chunks with fewer than ``settings.pure_req_signal_min`` requirement signals
  (shall/must/will) are skipped before the LLM is called.
- At most ``settings.pure_max_section_chunks`` / ``pure_max_raw_chunks``
  are processed to bound API cost on very large documents.
"""

from __future__ import annotations

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    EXTRACTION_SYSTEM_PROMPT,
    build_extract_from_chunk_prompt,
)

logger = structlog.get_logger(__name__)


def _determine_chunk_limit(strategy: str) -> int:
    settings = get_settings()
    return (
        settings.pure_max_section_chunks
        if strategy == "section"
        else settings.pure_max_raw_chunks
    )


def extract_requirements_from_chunks(
    llm: AzureOpenAIClient,
    doc_id: str,
    chunks: list[dict],
    *,
    strategy: str,
) -> list[dict]:
    """Extract candidate requirements from a list of chunks.

    Args:
        llm:      Shared AzureOpenAIClient.
        doc_id:   Document identifier (used for IDs and logging).
        chunks:   List of chunk dicts from the chunker.
        strategy: ``"section"`` or ``"raw_chunk"``.

    Returns:
        A flat list of raw candidate dicts with keys:
        ``text``, ``requirement_kind``, ``nfr_subtype``,
        ``raw_label``, ``source_strategy``, ``source_region``.
    """
    settings = get_settings()
    min_signals = settings.pure_req_signal_min
    max_chunks  = _determine_chunk_limit(strategy)

    all_candidates: list[dict] = []
    processed = 0
    skipped_signal = 0

    for chunk in chunks:
        if processed >= max_chunks:
            logger.info(
                "extractor.chunk_limit_reached",
                doc_id=doc_id,
                strategy=strategy,
                limit=max_chunks,
            )
            break

        # Skip chunks with too few requirement signals
        if chunk.get("req_signal_count", 0) < min_signals:
            # For section strategy, also skip non-dense sections
            if strategy == "section" and not chunk.get("is_requirement_dense", True):
                skipped_signal += 1
                continue
            # For raw_chunk, skip unless there are at least min_signals
            if strategy == "raw_chunk":
                skipped_signal += 1
                continue

        prompt = build_extract_from_chunk_prompt(
            doc_id=doc_id,
            chunk_id=chunk["chunk_id"],
            chunk_text=chunk["text"],
            strategy=strategy,
            section_title=chunk.get("section_title"),
        )

        try:
            result = llm.chat_json(
                system_prompt=EXTRACTION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=settings.temp_generation,
                model=settings.azure_openai_extraction_deployment,
                max_tokens=settings.max_tokens_extraction,
            )
        except Exception as exc:
            logger.error(
                "extractor.llm_error",
                chunk_id=chunk["chunk_id"],
                error=str(exc)[:200],
            )
            processed += 1
            continue

        for cand in result.get("requirements", []):
            text = (cand.get("text") or "").strip()
            if not text or len(text) < 10:
                continue
            cand["source_strategy"] = strategy
            cand["source_region"]   = chunk.get("section_title") or chunk["chunk_id"]
            all_candidates.append(cand)

        processed += 1

    logger.info(
        "extractor.done",
        doc_id=doc_id,
        strategy=strategy,
        chunks_processed=processed,
        chunks_skipped=skipped_signal,
        candidates_found=len(all_candidates),
    )
    return all_candidates
