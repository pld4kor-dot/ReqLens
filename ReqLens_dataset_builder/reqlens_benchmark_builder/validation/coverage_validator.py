"""Coverage validator.

Checks whether every gold requirement is traceable to (inferable from) the
generated source bundle.  Returns a structured report used by the pipeline
to decide whether a repair pass is needed.

Batching strategy
-----------------
Sending all requirements in one LLM call risks:
  - Hitting the output-token limit mid-JSON (truncation → invalid JSON)
  - Exponential prompt growth for large PURE gold sets (60+ reqs)

Solution: split gold requirements into batches of VALIDATION_BATCH_SIZE,
issue one chat_json call per batch with a compact prompt, then merge results.
Each batch call targets ~10 requirements so response JSON stays small and safe.
"""

from __future__ import annotations

import math
from typing import Any

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.prompts import (
    VALIDATION_SYSTEM_PROMPT,
    build_coverage_prompt_batch,
)
from reqlens_benchmark_builder.schemas.benchmark_models import CoverageEntry

logger = structlog.get_logger(__name__)

# Requirements per LLM call — small enough so JSON output stays well within limits
VALIDATION_BATCH_SIZE = 10


def validate_coverage(
    llm: AzureOpenAIClient,
    unit_id: str,
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
) -> dict[str, Any]:
    """Check that every gold requirement is supported by the source bundle.

    Splits requirements into batches of up to VALIDATION_BATCH_SIZE to avoid
    output-token truncation on large PURE gold sets.

    Returns a dict with keys:
        coverage        : list of per-requirement verdicts (CoverageEntry dicts)
        coverage_rate   : float in [0, 1]
        missing_req_ids : list of IDs that are not supported
    """
    settings = get_settings()

    if not gold_requirements:
        return {"coverage": [], "coverage_rate": 1.0, "missing_req_ids": []}

    # Partition into small batches
    total_reqs = len(gold_requirements)
    total_batches = math.ceil(total_reqs / VALIDATION_BATCH_SIZE)
    batches = [
        gold_requirements[i : i + VALIDATION_BATCH_SIZE]
        for i in range(0, total_reqs, VALIDATION_BATCH_SIZE)
    ]

    logger.info(
        "coverage_validator.batched",
        unit_id=unit_id,
        total_reqs=total_reqs,
        batches=total_batches,
        batch_size=VALIDATION_BATCH_SIZE,
    )

    all_entries: list[dict[str, Any]] = []
    supported_count = 0
    missing_ids: list[str] = []

    for batch_idx, batch in enumerate(batches):
        prompt = build_coverage_prompt_batch(
            unit_id=unit_id,
            source_texts=source_texts,
            gold_requirements=batch,
            batch_index=batch_idx,
            total_batches=total_batches,
        )

        try:
            raw = llm.chat_json(
                system_prompt=VALIDATION_SYSTEM_PROMPT,
                user_prompt=prompt,
                temperature=settings.temp_validation,
                model=settings.azure_openai_reasoning_deployment,
                max_tokens=settings.max_tokens_validation,
            )
        except Exception as exc:
            logger.error(
                "coverage_validator.batch_llm_error",
                unit_id=unit_id,
                batch=batch_idx,
                error=str(exc)[:200],
            )
            # Pessimistic fallback for this batch only
            for r in batch:
                rid = r.get("id", "")
                missing_ids.append(rid)
                all_entries.append(
                    CoverageEntry(
                        req_id=rid,
                        supported=False,
                        reason="LLM validation failed for this batch",
                    ).model_dump()
                )
            continue

        coverage_raw = raw.get("coverage", [])
        batch_req_ids = {r.get("id", "") for r in batch}
        seen_in_batch: set[str] = set()

        for item in coverage_raw:
            req_id   = str(item.get("req_id", ""))
            sup      = bool(item.get("supported", False))
            snippets = item.get("evidence_snippets") or []
            reason   = str(item.get("reason", ""))
            seen_in_batch.add(req_id)

            if sup:
                supported_count += 1
            else:
                missing_ids.append(req_id)

            all_entries.append(
                CoverageEntry(
                    req_id=req_id,
                    supported=sup,
                    evidence_snippets=snippets if isinstance(snippets, list) else [],
                    reason=reason,
                ).model_dump()
            )

        # Any requirement the LLM silently omitted → treat as unsupported
        for rid in batch_req_ids - seen_in_batch:
            missing_ids.append(rid)
            all_entries.append(
                CoverageEntry(
                    req_id=rid,
                    supported=False,
                    reason="Not mentioned in LLM batch coverage report",
                ).model_dump()
            )

        logger.info(
            "coverage_validator.batch_done",
            unit_id=unit_id,
            batch=batch_idx,
            batch_size=len(batch),
            supported=sum(1 for e in coverage_raw if e.get("supported")),
        )

    total = len(gold_requirements)
    rate  = round(supported_count / total, 4) if total else 1.0

    logger.info(
        "coverage_validator.done",
        unit_id=unit_id,
        coverage_rate=rate,
        missing=len(missing_ids),
        total_entries=len(all_entries),
    )

    return {
        "coverage": all_entries,
        "coverage_rate": rate,
        "missing_req_ids": missing_ids,
    }
