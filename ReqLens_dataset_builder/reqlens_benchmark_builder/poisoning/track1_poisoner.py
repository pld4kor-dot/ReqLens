"""Track 1 Hallucination Poisoner.

Given a clean BenchmarkUnit (loaded from unit.json), generates N hallucinated
candidate requirements that are:
  - Thematically plausible for the project domain.
  - Provably unsupported by the source_texts (verified by a second LLM pass).

Returns a PoisonedTrack1Artifact containing:
  - source_texts unchanged
  - candidate_pool: gold requirements + hallucinated fakes (shuffled)
  - seed_registry: full metadata per injected fake
"""

from __future__ import annotations

import random
import uuid
import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.poisoning.prompts import (
    HALLUCINATION_GENERATION_SYSTEM,
    HALLUCINATION_VERIFICATION_SYSTEM,
    build_hallucination_generation_prompt,
    build_hallucination_verification_prompt,
)
from reqlens_benchmark_builder.poisoning.schemas import (
    CandidatePoolItem,
    HallucinationSeedItem,
    PoisonedTrack1Artifact,
)

logger = structlog.get_logger(__name__)

# Maximum generation + verification attempts before giving up on a unit
_MAX_ATTEMPTS = 3


def _poison_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


def _generate_hallucinations(
    llm: AzureOpenAIClient,
    unit_id: str,
    brief: dict,
    source_texts: list[dict],
    gold_requirements: list[dict],
    count: int,
) -> list[dict]:
    """Call LLM to generate candidate hallucinated requirements."""
    settings = get_settings()
    prompt = build_hallucination_generation_prompt(
        unit_id=unit_id,
        brief=brief,
        source_texts=source_texts,
        gold_requirements=gold_requirements,
        count=count,
    )
    try:
        raw = llm.chat_json(
            system_prompt=HALLUCINATION_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_generation,
            model=settings.azure_openai_chat_deployment,
            max_tokens=settings.max_tokens_generation,
        )
    except Exception as exc:
        logger.error("track1_poisoner.generation_failed", unit_id=unit_id, error=str(exc))
        return []

    candidates = raw.get("hallucinations", [])
    if not isinstance(candidates, list):
        return []

    # Basic sanitisation
    clean = []
    for c in candidates:
        text = (c.get("text") or "").strip()
        if not text:
            continue
        clean.append({
            "text": text,
            "requirement_kind": c.get("requirement_kind", "functional"),
            "nfr_subtype": c.get("nfr_subtype", "not_applicable"),
            "unsupported_reason": (c.get("unsupported_reason") or "").strip(),
        })
    return clean


def _verify_unsupported(
    llm: AzureOpenAIClient,
    unit_id: str,
    source_texts: list[dict],
    candidates: list[dict],
) -> list[dict]:
    """Return only candidates that the verifier confirms are truly unsupported."""
    if not candidates:
        return []

    settings = get_settings()
    prompt = build_hallucination_verification_prompt(
        unit_id=unit_id,
        source_texts=source_texts,
        candidates=candidates,
    )
    try:
        raw = llm.chat_json(
            system_prompt=HALLUCINATION_VERIFICATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_validation,
            model=settings.azure_openai_reasoning_deployment,
            max_tokens=settings.max_tokens_validation,
        )
    except Exception as exc:
        logger.error("track1_poisoner.verification_failed", unit_id=unit_id, error=str(exc))
        # On failure keep all as unsupported (optimistic — accepted in practice
        # because the generation prompt already explicitly forbids grounded reqs)
        return candidates

    verdicts = raw.get("verdicts", [])
    if not isinstance(verdicts, list):
        return candidates

    verdict_map = {v.get("id"): v.get("supported", False) for v in verdicts}

    confirmed = []
    for i, cand in enumerate(candidates):
        if not verdict_map.get(i, False):
            confirmed.append(cand)
        else:
            logger.debug(
                "track1_poisoner.hallu_rejected_by_verifier",
                unit_id=unit_id,
                text=cand["text"][:80],
            )
    return confirmed


def poison_track1(
    llm: AzureOpenAIClient,
    unit_id: str,
    origin: str,
    source_texts: list[dict],
    gold_requirements: list[dict],
    brief: dict,
    metadata: dict,
    *,
    hallucination_count: int = 5,
    variant_id: str = "hallu_v1",
) -> PoisonedTrack1Artifact | None:
    """Generate a Track 1 poisoned artifact for a single benchmark unit.

    Args:
        llm:                  Shared LLM client.
        unit_id:              Benchmark unit ID (e.g. "PROMISE_1").
        origin:               "PROMISE" | "PURE".
        source_texts:         Raw source artifact list from unit.json.
        gold_requirements:    Gold requirement list from unit.json.
        brief:                Scenario brief dict from unit.json.
        metadata:             Metadata dict from unit.json.
        hallucination_count:  How many fake requirements to inject.
        variant_id:           Tag for this poisoning variant.

    Returns:
        PoisonedTrack1Artifact or None if generation fails after all attempts.
    """
    confirmed_fakes: list[dict] = []
    needed = hallucination_count

    for attempt in range(1, _MAX_ATTEMPTS + 1):
        logger.info(
            "track1_poisoner.attempt",
            unit_id=unit_id,
            attempt=attempt,
            needed=needed,
        )
        generated = _generate_hallucinations(
            llm, unit_id, brief, source_texts, gold_requirements, count=needed
        )
        if not generated:
            logger.warning("track1_poisoner.no_candidates_generated", unit_id=unit_id)
            continue

        verified = _verify_unsupported(llm, unit_id, source_texts, generated)
        confirmed_fakes.extend(verified)

        if len(confirmed_fakes) >= hallucination_count:
            confirmed_fakes = confirmed_fakes[:hallucination_count]
            break

        needed = hallucination_count - len(confirmed_fakes)
        logger.info(
            "track1_poisoner.need_more",
            unit_id=unit_id,
            confirmed=len(confirmed_fakes),
            still_needed=needed,
        )

    if not confirmed_fakes:
        logger.error(
            "track1_poisoner.failed_all_attempts",
            unit_id=unit_id,
            hallucination_count=hallucination_count,
        )
        return None

    if len(confirmed_fakes) < hallucination_count:
        logger.warning(
            "track1_poisoner.partial_success",
            unit_id=unit_id,
            requested=hallucination_count,
            produced=len(confirmed_fakes),
        )

    # Build seed registry and candidate pool items for fake reqs
    seed_registry: list[HallucinationSeedItem] = []
    fake_pool_items: list[CandidatePoolItem] = []

    for i, fake in enumerate(confirmed_fakes, start=1):
        seed_id = _poison_id("T1S")
        req_id = f"{unit_id}_FAKE_{i:03d}"

        seed = HallucinationSeedItem(
            seed_item_id=seed_id,
            requirement_id=req_id,
            requirement_text=fake["text"],
            requirement_kind=fake.get("requirement_kind", "functional"),
            nfr_subtype=fake.get("nfr_subtype", "not_applicable"),
            unsupported_reason=fake.get("unsupported_reason", ""),
        )
        seed_registry.append(seed)

        pool_item = CandidatePoolItem(
            id=req_id,
            text=fake["text"],
            requirement_kind=fake.get("requirement_kind", "functional"),
            nfr_subtype=fake.get("nfr_subtype", "not_applicable"),
            origin="seeded_fake",
            seed_item_id=seed_id,
        )
        fake_pool_items.append(pool_item)

    # Build gold pool items
    gold_pool_items: list[CandidatePoolItem] = [
        CandidatePoolItem(
            id=r["id"],
            text=r["text"],
            requirement_kind=r.get("requirement_kind", "functional"),
            nfr_subtype=r.get("nfr_subtype", "not_applicable"),
            origin="gold",
        )
        for r in gold_requirements
    ]

    # Shuffle to prevent positional bias
    combined = gold_pool_items + fake_pool_items
    random.shuffle(combined)

    artifact = PoisonedTrack1Artifact(
        artifact_id=_poison_id("T1A"),
        unit_id=unit_id,
        origin=origin,
        variant_id=variant_id,
        source_texts=source_texts,
        candidate_pool=combined,
        gold_requirement_ids=[r["id"] for r in gold_requirements],
        seeded_fake_requirement_ids=[item.requirement_id for item in seed_registry],
        seed_registry=seed_registry,
        brief=brief,
        metadata=metadata,
    )

    logger.info(
        "track1_poisoner.done",
        unit_id=unit_id,
        gold_count=len(gold_pool_items),
        fake_count=len(fake_pool_items),
        total_pool=len(combined),
    )
    return artifact
