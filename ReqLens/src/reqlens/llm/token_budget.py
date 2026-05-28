"""Token budget estimation and trimming utilities.

Ensures we stay within Azure OpenAI context-window limits
before submitting prompts.
"""

from __future__ import annotations

from typing import Sequence

import structlog

logger = structlog.get_logger(__name__)

# Approximate tokens-per-character ratio for English text
_CHARS_PER_TOKEN = 4


def estimate_tokens(text: str) -> int:
    """Fast heuristic token estimate (≈ 1 token per 4 chars)."""
    return max(1, len(text) // _CHARS_PER_TOKEN)


def estimate_tokens_precise(text: str, model: str = "gpt-4.1") -> int:
    """Use tiktoken for precise token count (falls back to heuristic)."""
    try:
        import tiktoken

        enc = tiktoken.encoding_for_model(model)
        return len(enc.encode(text))
    except Exception:
        return estimate_tokens(text)


def trim_spans_to_budget(
    spans: Sequence[str],
    *,
    max_tokens: int,
    system_prompt_tokens: int = 0,
    output_reserve_tokens: int = 4_000,
) -> list[str]:
    """Select as many spans as fit within the token budget.

    Spans are added in order. Each span that would bust the limit is
    skipped (greedy packing).
    """
    available = max_tokens - system_prompt_tokens - output_reserve_tokens
    if available <= 0:
        logger.warning("token_budget.no_room", max_tokens=max_tokens)
        return []

    selected: list[str] = []
    used = 0
    for span in spans:
        t = estimate_tokens(span)
        if used + t <= available:
            selected.append(span)
            used += t
        else:
            break  # spans are ordered; stop at first overflow

    logger.debug(
        "token_budget.trim",
        total_spans=len(spans),
        selected_spans=len(selected),
        estimated_tokens=used,
        budget=available,
    )
    return selected


def batch_texts_by_budget(
    texts: Sequence[str],
    *,
    max_tokens_per_batch: int,
) -> list[list[str]]:
    """Split a list of texts into batches that each fit within
    *max_tokens_per_batch*.
    """
    batches: list[list[str]] = []
    current_batch: list[str] = []
    current_tokens = 0

    for text in texts:
        t = estimate_tokens(text)
        if current_tokens + t > max_tokens_per_batch and current_batch:
            batches.append(current_batch)
            current_batch = []
            current_tokens = 0
        current_batch.append(text)
        current_tokens += t

    if current_batch:
        batches.append(current_batch)

    return batches
