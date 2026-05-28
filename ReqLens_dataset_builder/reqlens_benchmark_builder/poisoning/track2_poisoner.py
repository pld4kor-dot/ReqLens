"""Track 2 Defect Poisoner.

Given a clean BenchmarkUnit (loaded from unit.json), seeds M contradictions
and M duplicates into the source_texts to produce a PoisonedTrack2Artifact.

Strategy:
- Contradictions: pick gold requirements with concrete, testable attributes
  (numeric values, boolean constraints, access rules), generate a conflicting
  statement from a different stakeholder, inject into a different source artifact.
- Duplicates: pick gold requirements and paraphrase them in informal language,
  inject into a different source artifact than where the requirement was
  originally implied.

The seed_registry records exactly what was injected so the evaluation framework
can direct LLM judge prompts to the right defect type.
"""

from __future__ import annotations

import copy
import random
import re
import uuid
import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.poisoning.prompts import (
    CONTRADICTION_GENERATION_SYSTEM,
    CONTRADICTION_VERIFICATION_SYSTEM,
    DUPLICATE_GENERATION_SYSTEM,
    DUPLICATE_VERIFICATION_SYSTEM,
    build_contradiction_generation_prompt,
    build_contradiction_verification_prompt,
    build_duplicate_generation_prompt,
    build_duplicate_verification_prompt,
)
from reqlens_benchmark_builder.poisoning.schemas import (
    DefectSeedItem,
    PoisonedTrack2Artifact,
)

logger = structlog.get_logger(__name__)

# Artifact types available for injection rotation
_ARTIFACT_TYPES = ["interview_transcript", "meeting_notes", "email_thread"]

# Requirement kinds most likely to have concrete contradictable attributes
_PREFERRED_KINDS_FOR_CONTRADICTION = {
    "non_functional",
    "constraint",
    "functional",
}


def _poison_id(prefix: str) -> str:
    return f"{prefix}_{uuid.uuid4().hex[:8].upper()}"


def _pick_injection_artifact(
    source_texts: list[dict],
    avoid_types: list[str],
) -> dict | None:
    """Pick a source artifact to inject into, preferring types not in avoid_types."""
    available = [a for a in source_texts if a["type"] not in avoid_types]
    if not available:
        available = source_texts  # fallback: any artifact
    return random.choice(available) if available else None


# Lead-in phrases per artifact type — makes injected text blend in naturally
_LEAD_INS: dict[str, list[str]] = {
    "interview_transcript": [
        "Actually, one more thing I wanted to bring up — ",
        "Oh wait, before we move on, ",
        "Sorry, going back to something I mentioned earlier — ",
        "I just remembered, there's another point I should mention: ",
    ],
    "meeting_notes": [
        "- Side note from a stakeholder: ",
        "- Follow-up point: ",
        "- Additional input: ",
        "- Raised during discussion: ",
    ],
    "email_thread": [
        "On another note, ",
        "One more thing — ",
        "Quick follow-up: ",
        "P.S. ",
    ],
}


def _find_paragraph_break_positions(text: str) -> list[int]:
    """Return character positions of paragraph breaks (blank-line boundaries).

    Only returns positions in the middle 60% of the document to avoid
    placing injected text too close to the start or the very end.
    """
    min_pos = int(len(text) * 0.20)
    max_pos = int(len(text) * 0.80)
    positions: list[int] = []
    for m in re.finditer(r"\n\s*\n", text):
        pos = m.end()
        if min_pos <= pos <= max_pos:
            positions.append(pos)
    return positions


def _inject_text_into_artifact(artifact: dict, injected_text: str) -> dict:
    """Return a new artifact dict with the injected text placed mid-document.

    Strategy:
    1. Find paragraph-break positions in the middle 60% of the document.
    2. Pick one at random and insert the text there with a natural lead-in.
    3. Fall back to appending at the end if no suitable position is found.
    """
    artifact = copy.deepcopy(artifact)
    text = artifact["text"]
    artifact_type = artifact.get("type", "email_thread")

    # Pick a contextual lead-in
    lead_ins = _LEAD_INS.get(artifact_type, _LEAD_INS["email_thread"])
    lead_in = random.choice(lead_ins)
    full_injection = lead_in + injected_text

    positions = _find_paragraph_break_positions(text)
    if positions:
        insert_at = random.choice(positions)
        artifact["text"] = (
            text[:insert_at].rstrip()
            + "\n\n"
            + full_injection
            + "\n\n"
            + text[insert_at:].lstrip()
        )
    else:
        # Fallback: append at the end
        artifact["text"] = text.rstrip() + "\n\n" + full_injection
    return artifact


def _generate_contradiction(
    llm: AzureOpenAIClient,
    unit_id: str,
    target_req: dict,
    injection_artifact_type: str,
    other_artifact_types: list[str],
) -> dict | None:
    """Generate one contradiction for target_req and return raw LLM dict or None."""
    settings = get_settings()
    prompt = build_contradiction_generation_prompt(
        unit_id=unit_id,
        target_requirement=target_req,
        target_artifact_type=injection_artifact_type,
        other_source_types=other_artifact_types,
    )
    try:
        raw = llm.chat_json(
            system_prompt=CONTRADICTION_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_generation,
            model=settings.azure_openai_chat_deployment,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.error("track2_poisoner.contradiction_failed", unit_id=unit_id, error=str(exc))
        return None

    text = (raw.get("injected_text") or "").strip()
    if not text:
        return None
    return {
        "injected_text": text,
        "defect_subtype": raw.get("defect_subtype", "value_conflict"),
        "defect_description": (raw.get("defect_description") or "").strip(),
    }


def _verify_contradiction(
    llm: AzureOpenAIClient,
    unit_id: str,
    candidate: dict,
    target_req: dict,
) -> bool:
    """Verify a generated contradiction is a real, clear conflict.

    Returns True if the verifier confirms a genuine conflict.
    """
    settings = get_settings()
    prompt = build_contradiction_verification_prompt(
        unit_id=unit_id,
        target_requirement=target_req,
        injected_text=candidate["injected_text"],
    )
    try:
        raw = llm.chat_json(
            system_prompt=CONTRADICTION_VERIFICATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_validation,
            model=settings.azure_openai_reasoning_deployment,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning(
            "track2_poisoner.contradiction_verify_error",
            unit_id=unit_id, error=str(exc),
        )
        # Optimistic: accept on verification failure
        return True

    is_conflict = raw.get("is_genuine_conflict", True)
    if not is_conflict:
        logger.debug(
            "track2_poisoner.contradiction_rejected_by_verifier",
            unit_id=unit_id,
            req_id=target_req["id"],
            reason=raw.get("reason", ""),
        )
    return bool(is_conflict)


def _verify_duplicate(
    llm: AzureOpenAIClient,
    unit_id: str,
    candidate: dict,
    target_req: dict,
) -> bool:
    """Verify a generated duplicate is semantically equivalent but lexically different.

    Returns True if the verifier confirms a valid paraphrase duplicate.
    """
    settings = get_settings()
    prompt = build_duplicate_verification_prompt(
        unit_id=unit_id,
        target_requirement=target_req,
        injected_text=candidate["injected_text"],
    )
    try:
        raw = llm.chat_json(
            system_prompt=DUPLICATE_VERIFICATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_validation,
            model=settings.azure_openai_reasoning_deployment,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.warning(
            "track2_poisoner.duplicate_verify_error",
            unit_id=unit_id, error=str(exc),
        )
        return True

    is_valid = raw.get("is_valid_duplicate", True)
    if not is_valid:
        logger.debug(
            "track2_poisoner.duplicate_rejected_by_verifier",
            unit_id=unit_id,
            req_id=target_req["id"],
            reason=raw.get("reason", ""),
        )
    return bool(is_valid)


def _generate_duplicate(
    llm: AzureOpenAIClient,
    unit_id: str,
    target_req: dict,
    injection_artifact_type: str,
) -> dict | None:
    """Generate one paraphrase duplicate for target_req and return raw LLM dict or None."""
    settings = get_settings()
    prompt = build_duplicate_generation_prompt(
        unit_id=unit_id,
        target_requirement=target_req,
        target_artifact_type=injection_artifact_type,
    )
    try:
        raw = llm.chat_json(
            system_prompt=DUPLICATE_GENERATION_SYSTEM,
            user_prompt=prompt,
            temperature=settings.temp_generation,
            model=settings.azure_openai_chat_deployment,
            max_tokens=2048,
        )
    except Exception as exc:
        logger.error("track2_poisoner.duplicate_failed", unit_id=unit_id, error=str(exc))
        return None

    text = (raw.get("injected_text") or "").strip()
    if not text:
        return None
    return {
        "injected_text": text,
        "defect_subtype": raw.get("defect_subtype", "paraphrase_duplicate"),
        "defect_description": (raw.get("defect_description") or "").strip(),
    }


def poison_track2(
    llm: AzureOpenAIClient,
    unit_id: str,
    origin: str,
    source_texts: list[dict],
    gold_requirements: list[dict],
    brief: dict,
    metadata: dict,
    *,
    contradiction_count: int = 2,
    duplicate_count: int = 2,
    variant_id: str = "defect_v1",
) -> PoisonedTrack2Artifact | None:
    """Generate a Track 2 poisoned artifact for a single benchmark unit.

    Args:
        llm:                  Shared LLM client.
        unit_id:              Benchmark unit ID (e.g. "PROMISE_1").
        origin:               "PROMISE" | "PURE".
        source_texts:         Raw source artifact list from unit.json.
        gold_requirements:    Gold requirement list from unit.json.
        brief:                Scenario brief dict from unit.json.
        metadata:             Metadata dict from unit.json.
        contradiction_count:  How many contradictions to seed.
        duplicate_count:      How many duplicates to seed.
        variant_id:           Tag for this poisoning variant.

    Returns:
        PoisonedTrack2Artifact or None if no defects could be generated.
    """
    # Work on a deep copy of source_texts — modifications are applied cumulatively
    modified_source_texts: list[dict] = copy.deepcopy(source_texts)
    seed_registry: list[DefectSeedItem] = []
    artifact_types = [a["type"] for a in source_texts]

    # --- Contradictions -------------------------------------------------
    # Prefer requirements with concrete, testable attributes
    preferred = [
        r for r in gold_requirements
        if r.get("requirement_kind") in _PREFERRED_KINDS_FOR_CONTRADICTION
    ]
    contradiction_pool = preferred if len(preferred) >= contradiction_count else gold_requirements

    # Shuffle to pick diverse requirements
    contradiction_pool_shuffled = list(contradiction_pool)
    random.shuffle(contradiction_pool_shuffled)

    used_req_ids_for_contradiction: set[str] = set()
    contradictions_added = 0

    for target_req in contradiction_pool_shuffled:
        if contradictions_added >= contradiction_count:
            break
        if target_req["id"] in used_req_ids_for_contradiction:
            continue

        # Pick an artifact type different from where most requirements live
        # (rotate through available types)
        injection_type = _ARTIFACT_TYPES[contradictions_added % len(_ARTIFACT_TYPES)]
        other_types = [t for t in artifact_types if t != injection_type]

        result = _generate_contradiction(
            llm=llm,
            unit_id=unit_id,
            target_req=target_req,
            injection_artifact_type=injection_type,
            other_artifact_types=other_types,
        )
        if not result:
            logger.warning(
                "track2_poisoner.contradiction_skipped",
                unit_id=unit_id,
                req_id=target_req["id"],
            )
            continue

        # Verify the contradiction is a genuine conflict
        if not _verify_contradiction(llm, unit_id, result, target_req):
            logger.info(
                "track2_poisoner.contradiction_failed_verification",
                unit_id=unit_id,
                req_id=target_req["id"],
            )
            continue

        # Find the artifact to inject into
        target_artifact_idx = next(
            (i for i, a in enumerate(modified_source_texts) if a["type"] == injection_type),
            None,
        )
        if target_artifact_idx is None:
            # Fall back to last artifact
            target_artifact_idx = len(modified_source_texts) - 1

        modified_source_texts[target_artifact_idx] = _inject_text_into_artifact(
            modified_source_texts[target_artifact_idx],
            result["injected_text"],
        )

        seed_item = DefectSeedItem(
            seed_item_id=_poison_id("T2S"),
            defect_type="contradiction",
            defect_subtype=result.get("defect_subtype", "value_conflict"),
            original_req_ids=[target_req["id"]],
            original_req_texts=[target_req["text"]],
            injected_text=result["injected_text"],
            injected_into_artifact_type=modified_source_texts[target_artifact_idx]["type"],
            defect_description=result.get("defect_description", ""),
            judge_task_type="contradiction_check",
        )
        seed_registry.append(seed_item)
        used_req_ids_for_contradiction.add(target_req["id"])
        contradictions_added += 1
        logger.debug(
            "track2_poisoner.contradiction_seeded",
            unit_id=unit_id,
            req_id=target_req["id"],
            artifact_type=injection_type,
        )

    # --- Duplicates -----------------------------------------------------
    duplicate_pool = list(gold_requirements)
    random.shuffle(duplicate_pool)

    used_req_ids_for_duplicate: set[str] = set()
    duplicates_added = 0

    for target_req in duplicate_pool:
        if duplicates_added >= duplicate_count:
            break
        if target_req["id"] in used_req_ids_for_duplicate:
            continue

        injection_type = _ARTIFACT_TYPES[(duplicates_added + 1) % len(_ARTIFACT_TYPES)]

        result = _generate_duplicate(
            llm=llm,
            unit_id=unit_id,
            target_req=target_req,
            injection_artifact_type=injection_type,
        )
        if not result:
            logger.warning(
                "track2_poisoner.duplicate_skipped",
                unit_id=unit_id,
                req_id=target_req["id"],
            )
            continue

        # Verify the duplicate is semantically equivalent but lexically different
        if not _verify_duplicate(llm, unit_id, result, target_req):
            logger.info(
                "track2_poisoner.duplicate_failed_verification",
                unit_id=unit_id,
                req_id=target_req["id"],
            )
            continue

        target_artifact_idx = next(
            (i for i, a in enumerate(modified_source_texts) if a["type"] == injection_type),
            None,
        )
        if target_artifact_idx is None:
            target_artifact_idx = len(modified_source_texts) - 1

        modified_source_texts[target_artifact_idx] = _inject_text_into_artifact(
            modified_source_texts[target_artifact_idx],
            result["injected_text"],
        )

        seed_item = DefectSeedItem(
            seed_item_id=_poison_id("T2S"),
            defect_type="duplicate",
            defect_subtype=result.get("defect_subtype", "paraphrase_duplicate"),
            original_req_ids=[target_req["id"]],
            original_req_texts=[target_req["text"]],
            injected_text=result["injected_text"],
            injected_into_artifact_type=modified_source_texts[target_artifact_idx]["type"],
            defect_description=result.get("defect_description", ""),
            judge_task_type="duplicate_check",
        )
        seed_registry.append(seed_item)
        used_req_ids_for_duplicate.add(target_req["id"])
        duplicates_added += 1
        logger.debug(
            "track2_poisoner.duplicate_seeded",
            unit_id=unit_id,
            req_id=target_req["id"],
            artifact_type=injection_type,
        )

    if not seed_registry:
        logger.error("track2_poisoner.no_defects_seeded", unit_id=unit_id)
        return None

    artifact = PoisonedTrack2Artifact(
        artifact_id=_poison_id("T2A"),
        unit_id=unit_id,
        origin=origin,
        variant_id=variant_id,
        source_texts=modified_source_texts,
        gold_requirements=gold_requirements,
        seed_registry=seed_registry,
        brief=brief,
        metadata=metadata,
    )

    logger.info(
        "track2_poisoner.done",
        unit_id=unit_id,
        contradictions=contradictions_added,
        duplicates=duplicates_added,
        total_seeds=len(seed_registry),
    )
    return artifact
