"""LLM judge prompt families for Track 1 and Track 2 evaluation.

Prompt families:
  A — support_check     : Is this candidate requirement supported by the source?
                           (Used when normalizing uncertain Track 1 decisions)
  B — hallucination_fate: For a known hallucinated candidate, was it correctly
                           rejected by the system?  (secondary HRR signal)
  C — contradiction_check: For a seeded contradiction, did it leak to the extraction?
  D — duplicate_check   : For a seeded duplicate, did it leak to the extraction?
  E — pairwise_similarity: Utility — are two requirement texts semantically equivalent?
"""

from __future__ import annotations

# ── Family A: Support Check (Track 1 normalisation) ───────────────────────────

FAMILY_A_SYSTEM = """\
You are a requirements engineering expert performing evidence grounding.

You will receive:
1. Source evidence documents (stakeholder interview, meeting notes, email thread).
2. A candidate software requirement.

Task: decide if the candidate is SUPPORTED by the source documents.
# OLD: A candidate is supported when:
# OLD:   - The source documents explicitly state or clearly imply it, OR
# OLD:   - It is a reasonable, closely-linked elaboration of an explicitly stated need
# OLD:     (e.g. adding one logically-implied field, annotation type, or detail that
# OLD:     is consistent with and directly derivable from the source).
# OLD: A candidate is NOT supported when it adds constraints, values, or functionality
# OLD: that go clearly beyond what is stated or reasonably derivable from the sources.
A candidate is supported when:
  - The source documents explicitly state or clearly imply it, OR
  - It is a reasonable, closely-linked elaboration of an explicitly stated need
    (e.g. adding one logically-implied field, annotation type, or detail that
    is consistent with and directly derivable from the source), OR
  - It adds standard requirements-metadata that is directly implied by a stated
    need (e.g. unique identifier, rationale, traceability link, verification method)
    when the source already requires that requirements be documented and traceable, OR
  - It describes integrating, deploying, or providing access to software or data
    that the source explicitly calls for (even if the integration mechanism itself
    is not spelled out).

A candidate is NOT supported when it adds concrete constraints, numeric values,
specific protocols, or new functionality that go clearly beyond what is stated
or reasonably derivable from the sources.

When the evidence is ambiguous — the candidate is a plausible requirement that
a competent analyst could reasonably derive from the source, but the exact
wording or values are not explicitly stated — lean toward SUPPORTED.  Reserve
"rejected" for candidates that introduce clearly unsupported capabilities or
contradict the source evidence.

Respond with a strict JSON object — no extra text:
{
  "status": "accepted" | "rejected",
  "confidence": <float 0.0-1.0>,
  "explanation": "<one sentence>"
}"""

FAMILY_A_USER_TEMPLATE = """\
SOURCE DOCUMENTS:
{source_block}

CANDIDATE REQUIREMENT (id={candidate_id}):
{candidate_text}

Is this candidate requirement SUPPORTED by the source documents?"""


# ── Family B: Hallucination Fate (Track 1 HRR secondary signal) ───────────────

FAMILY_B_SYSTEM = """\
You are a requirements engineering expert auditing a requirement extraction system.

You will receive:
1. The full list of requirements outputted by a system.
2. A HALLUCINATED (fabricated, not grounded) requirement that was injected into the evaluation.

Task: determine whether the system REJECTED this hallucinated requirement
(i.e., it does NOT appear — with the same meaning — in the system's output).

A hallucination is "rejected" (correctly filtered) if:
  - It does not appear in the system output, OR
  - It appears only with clear "unsupported / rejected" labels.
A hallucination is "accepted" (incorrectly passed) if it appears in the system output
as a valid / supported requirement.

Respond with a strict JSON object — no extra text:
{
  "verdict": "rejected" | "accepted",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}"""

FAMILY_B_USER_TEMPLATE = """\
SYSTEM OUTPUT (all extracted requirements):
{extraction_text}

HALLUCINATED REQUIREMENT (id={seed_item_id}):
{hallucinated_text}

Reason it is unsupported: {unsupported_reason}

Was this hallucinated requirement correctly REJECTED by the system?"""


# ── Family C: Contradiction Check (Track 2) ────────────────────────────────────

FAMILY_C_SYSTEM = """\
You are a requirements engineering expert auditing a contradictions detection system.

You will receive:
1. The extraction output of a system (list of requirements it produced).
2. A SEEDED CONTRADICTION — a deliberately injected conflicting requirement.
   It conflicts with one or more original requirements.

Task: determine whether the contradiction LEAKED to the system's output.
A contradiction "leaked" if:
  - The seeded conflicting requirement (or one semantically equivalent to it)
    appears in the system output as a valid requirement, AND
  - No resolution note, conflict flag, or deletion is evident.
  - Even if the exact wording differs, if the CONFLICTING VALUE or CONDITION
    from the injected statement appears in any extracted requirement, it has leaked.

A contradiction was "detected" (correctly handled) if:
  - The injected conflicting statement does NOT appear in the output, OR
  - The original requirement appears but the contradicting value/constraint
    has been corrected or excluded, OR
  - Both the original and the contradiction were removed entirely (indicating
    the system flagged the conflict and dropped both), OR
  - The conflict was explicitly resolved or merged into a single requirement.

Respond with a strict JSON object — no extra text:
{
  "verdict": "leaked" | "detected",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one to two sentences>"
}"""

# OLD FAMILY_C_USER_TEMPLATE (extraction text shown first — LLM had to read 50+
# requirements before knowing what to search for, causing 0.5 confidence defaults):
# FAMILY_C_USER_TEMPLATE = """\
# SYSTEM EXTRACTION OUTPUT:
# {extraction_text}
#
# SEEDED CONTRADICTION (seed_id={seed_item_id}):
# Injected text: {injected_text}
# Conflicts with: {original_req_texts}
# Description: {defect_description}
#
# Did this seeded contradiction LEAK to the system's extracted requirements?"""

# NEW: seed description comes first so the judge knows what to search for.
FAMILY_C_USER_TEMPLATE = """\
SEEDED CONTRADICTION (seed_id={seed_item_id}):
Injected conflicting statement: {injected_text}
Conflicts with original requirement: {original_req_texts}
Description: {defect_description}

Your task: scan every numbered requirement in the EXTRACTION OUTPUT below and
decide whether the injected statement's meaning — specifically the CONFLICTING
value, constraint, or condition — appears as a valid extracted requirement.

The extraction contains {extraction_count} requirements.

SYSTEM EXTRACTION OUTPUT:
{extraction_text}

Did this seeded contradiction LEAK to the system's extracted requirements?"""


# ── Family D: Duplicate Check (Track 2) ────────────────────────────────────────

FAMILY_D_SYSTEM = """\
You are a requirements engineering expert auditing a duplicate detection system.

You will receive:
1. The extraction output of a system (list of requirements it produced).
2. A SEEDED DUPLICATE — a deliberately injected near-identical or paraphrase requirement.
   It expresses the same need as an existing original requirement.

Task: determine whether the duplicate LEAKED to the system's output.
A duplicate "leaked" if:
  - Both the original requirement AND the seeded duplicate (or a near-paraphrase)
    appear separately in the system output, essentially expressing the same need twice.
  - Even if the wording differs, if TWO separate extracted requirements express
    the same underlying capability or constraint, the duplicate has leaked.

A duplicate was "detected" (merged/removed) if:
  - Only one form of the requirement appears in the output, OR
  - The system clearly merged both into a single unified requirement, OR
  - Neither the original nor the duplicate appears (both were removed).

IMPORTANT: Count carefully. If the same need appears once in the output (regardless
of which version's wording was used), the duplicate was DETECTED. Only if it appears
TWICE (or more) has it LEAKED.

Respond with a strict JSON object — no extra text:
{
  "verdict": "leaked" | "detected",
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one to two sentences>"
}"""

# OLD FAMILY_D_USER_TEMPLATE (extraction text shown first; leaked/detected
# distinction was ambiguous for paraphrase duplicates):
# FAMILY_D_USER_TEMPLATE = """\
# SYSTEM EXTRACTION OUTPUT:
# {extraction_text}
#
# SEEDED DUPLICATE (seed_id={seed_item_id}):
# Injected text: {injected_text}
# Original requirement(s): {original_req_texts}
# Description: {defect_description}
#
# Did this seeded duplicate LEAK to the system's extracted requirements?"""

# NEW: seed description comes first; leaked/detected criteria spelled out explicitly.
FAMILY_D_USER_TEMPLATE = """\
SEEDED DUPLICATE (seed_id={seed_item_id}):
Injected near-duplicate text: {injected_text}
Original requirement it paraphrases: {original_req_texts}
Description: {defect_description}

Your task: scan every numbered requirement in the EXTRACTION OUTPUT below.
A duplicate HAS LEAKED if the extraction contains TWO (or more) separate
requirements that express the same underlying need — the original AND an
additional paraphrase.
A duplicate was DETECTED (merged/removed) if the extraction contains only ONE
version of that need, regardless of which wording was kept.

The extraction contains {extraction_count} requirements.

SYSTEM EXTRACTION OUTPUT:
{extraction_text}

Did this seeded duplicate LEAK (appear separately alongside the original)?"""


# ── Family E: Pairwise Similarity (utility) ─────────────────────────────────────

FAMILY_E_SYSTEM = """\
You are a requirements engineering expert.

Determine whether two requirement statements express the SAME functional or non-functional need.
They may be worded differently (e.g., paraphrase, restructured sentence) but semantically equivalent.

Respond with a strict JSON object — no extra text:
{
  "equivalent": true | false,
  "confidence": <float 0.0-1.0>,
  "reasoning": "<one sentence>"
}"""

FAMILY_E_USER_TEMPLATE = """\
REQUIREMENT A: {req_a}
REQUIREMENT B: {req_b}

Are these two requirements semantically EQUIVALENT (same need, even if worded differently)?"""
