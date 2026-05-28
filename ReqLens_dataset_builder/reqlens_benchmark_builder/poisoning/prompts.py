"""Prompt templates for the poisoning pipeline.

System prompts are module-level constants.
User prompts are built by functions that accept typed parameters.

Three logical roles:
- TRACK1  — hallucination generation + support-verification
- TRACK2  — contradiction / duplicate generation
"""

from __future__ import annotations

import json
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# System prompts
# ══════════════════════════════════════════════════════════════════════════════

HALLUCINATION_GENERATION_SYSTEM = """\
You are an expert requirements engineer tasked with generating realistic but
UNSUPPORTED requirement statements for a benchmark evaluation dataset.

Your role is to create plausible-sounding requirements that:
1. Match the domain, tone, and style of the project (same "shall" language, same kind/subtype).
2. Are NOT inferable from, mentioned in, or implied by the provided source texts in any way.
3. Introduce system capabilities that are entirely absent from the source documents.
4. Are specific enough to be testable and atomic.

Core rules:
- Do NOT paraphrase or slightly modify existing gold requirements.
- Do NOT introduce capabilities that are even loosely hinted at in the source.
- Each generated requirement should address a clearly distinct feature or constraint.
- Output ONLY valid JSON matching the schema in the user message.
""".strip()

HALLUCINATION_VERIFICATION_SYSTEM = """\
You are a strict benchmark quality auditor.

Your task is to verify that each candidate requirement has NO support whatsoever
in the provided source texts.

Rules:
1. A requirement is "unsupported" only if the source texts contain no mention,
   implication, or indirect reference to that capability or constraint.
2. If the source texts even loosely hint at the requirement, mark it as "supported".
3. Be strict — when in doubt, mark as "supported" so we can regenerate.
4. Judge ONLY from the provided texts; use no external knowledge.
5. Output ONLY valid JSON matching the schema in the user message.
""".strip()

CONTRADICTION_GENERATION_SYSTEM = """\
You are an expert requirements engineer tasked with generating realistic
contradictions for a controlled benchmark evaluation dataset.

Your role is to create a contradicting requirement statement that:
1. Directly conflicts with a specified gold requirement on a concrete, testable attribute
   (e.g., a numerical value, a time constraint, a boolean behaviour, an access rule).
2. Sounds perfectly natural — as if a different stakeholder made a conflicting request
   during a meeting, interview, or email exchange.
3. Is specific and concrete (not vague or subjective).
4. Embeds the conflict naturally in 2-4 sentences of surrounding conversational context
   so it reads as part of a longer discussion — not as a standalone assertion.

Core rules:
- The contradiction must be clear and unambiguous — not a matter of interpretation.
- Do NOT simply negate the requirement — change a specific value or condition instead.
- Write in informal stakeholder language, not in SRS "shall" format.
- Include enough surrounding context (2-4 sentences) that the injected text reads
  naturally as part of a conversation, not as a tacked-on statement.
- The conflicting detail MUST be embedded within the natural flow of the text, not
  stated in isolation.
- Output ONLY valid JSON matching the schema in the user message.
""".strip()

CONTRADICTION_VERIFICATION_SYSTEM = """\
You are a strict benchmark quality auditor verifying seeded contradictions.

You will receive:
1. An original gold requirement.
2. A generated contradicting statement.

Your task is to verify that:
1. The contradiction creates a GENUINE, unambiguous conflict with the original
   requirement on a concrete, testable attribute.
2. The conflict is clear enough that a careful requirements analyst would flag it.
3. The contradiction is NOT just a vague restatement or minor rewording.

Rules:
- A valid contradiction changes a specific value, constraint, or condition.
- An invalid contradiction uses vague language that could be interpreted as
  compatible with the original, or contradicts on a trivial/immaterial aspect.
- When in doubt, mark as invalid so we can regenerate.
- Output ONLY valid JSON matching the schema in the user message.
""".strip()

DUPLICATE_GENERATION_SYSTEM = """\
You are an expert requirements engineer tasked with generating realistic
duplicate (paraphrase) requirements for a controlled benchmark evaluation dataset.

Your role is to create a paraphrase of an existing gold requirement that:
1. Expresses the same semantic intent as the original.
2. Uses different wording, sentence structure, or phrasing — as if a different
   stakeholder restated the same need in their own words.
3. Sounds natural in informal stakeholder communication.
4. Is similar enough that a careful system should detect it as a duplicate,
   but different enough that naive text matching would miss it.
5. Embeds the restated need naturally in 2-4 sentences of surrounding conversational
   context so it reads as part of a longer discussion.

Core rules:
- Do NOT copy the original wording verbatim.
- The paraphrase must be semantically equivalent — same capability, same constraint.
- Write in informal stakeholder language, not in SRS "shall" format.
- Include enough surrounding context (2-4 sentences) that the injected text reads
  naturally as part of a conversation, not as a tacked-on statement.
- Output ONLY valid JSON matching the schema in the user message.
""".strip()

DUPLICATE_VERIFICATION_SYSTEM = """\
You are a strict benchmark quality auditor verifying seeded duplicates.

You will receive:
1. An original gold requirement.
2. A generated paraphrase intended to be a duplicate.

Your task is to verify that:
1. The paraphrase expresses the SAME semantic need as the original requirement.
2. The wording is sufficiently DIFFERENT that naive text matching would not detect it.
3. A careful requirements analyst reviewing both statements would flag them as
   expressing the same underlying need.

Rules:
- A valid duplicate has the same semantic intent with different wording.
- An invalid duplicate either changes the meaning (adds/removes constraints) or
  is too similar in wording (trivial word substitution).
- When in doubt, mark as invalid so we can regenerate.
- Output ONLY valid JSON matching the schema in the user message.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# User prompt builders — Track 1 (hallucinations)
# ══════════════════════════════════════════════════════════════════════════════

def build_hallucination_generation_prompt(
    unit_id: str,
    brief: dict[str, Any],
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
    count: int,
) -> str:
    source_block = "\n\n".join(
        f"[{a['type'].upper()}]\n{a['text']}" for a in source_texts
    )
    gold_block = "\n".join(
        f"  - [{r.get('requirement_kind','?')}] {r['text']}"
        for r in gold_requirements
    )
    brief_json = json.dumps(brief, indent=2, ensure_ascii=False)

    return f"""\
Unit ID: {unit_id}

Project brief (for domain / style grounding):
{brief_json}

Source texts (the evidence bundle — hallucinations must NOT appear here):
<source_texts>
{source_block}
</source_texts>

Existing gold requirements (hallucinations must NOT overlap with these):
{gold_block}

Generate exactly {count} hallucinated requirement(s) in JSON using this schema:
{{
  "hallucinations": [
    {{
      "text": "The system shall ...",
      "requirement_kind": "functional | non_functional | constraint | domain_assumption | business_rule",
      "nfr_subtype": "security | privacy | usability | reliability | availability | performance | maintainability | portability | scalability | compliance | other | not_applicable",
      "unsupported_reason": "brief explanation of why this has zero evidence in the source texts"
    }}
  ]
}}

Each hallucination must:
- Be a plausible system requirement for this domain and project.
- Be entirely absent from the source texts above.
- Introduce a DISTINCT new capability not covered by any gold requirement.
- Use formal "The system shall ..." language.
- Have nfr_subtype set to "not_applicable" unless requirement_kind is "non_functional".
""".strip()


def build_hallucination_verification_prompt(
    unit_id: str,
    source_texts: list[dict[str, Any]],
    candidates: list[dict[str, Any]],
) -> str:
    source_block = "\n\n".join(
        f"[{a['type'].upper()}]\n{a['text']}" for a in source_texts
    )
    cand_json = json.dumps(
        [{"id": i, "text": c["text"]} for i, c in enumerate(candidates)],
        indent=2,
        ensure_ascii=False,
    )

    return f"""\
Unit ID: {unit_id}

Source texts:
<source_texts>
{source_block}
</source_texts>

Candidate hallucinated requirements to verify:
{cand_json}

For each candidate, determine whether it is supported (even loosely) by the source texts.

Return JSON using this schema:
{{
  "verdicts": [
    {{
      "id": 0,
      "supported": false,
      "reason": "brief explanation"
    }}
  ]
}}

Mark supported=true if the source texts mention, hint at, or imply this capability in any way.
Mark supported=false only if the source texts are entirely silent about this capability.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# User prompt builders — Track 2 (contradictions and duplicates)
# ══════════════════════════════════════════════════════════════════════════════

def build_contradiction_generation_prompt(
    unit_id: str,
    target_requirement: dict[str, Any],
    target_artifact_type: str,
    other_source_types: list[str],
) -> str:
    return f"""\
Unit ID: {unit_id}

Target gold requirement (this is the one to contradict):
  ID   : {target_requirement['id']}
  Text : {target_requirement['text']}
  Kind : {target_requirement.get('requirement_kind', 'functional')}

This requirement was implied in the source artifact of type: {other_source_types[0] if other_source_types else 'interview_transcript'}

Generate a contradiction to be injected into a different artifact (type: {target_artifact_type}).
The contradiction must conflict on a specific, concrete attribute of the requirement above.

Return JSON using this schema:
{{
  "injected_text": "2-4 sentences of natural stakeholder conversation that embed the conflicting detail naturally within the discussion flow",
  "defect_subtype": "value_conflict | semantic_conflict",
  "defect_description": "one sentence describing exactly what conflicts with what"
}}

The injected_text should:
- Sound like natural stakeholder conversation (part of meeting notes, interview, or email).
- NOT be a standalone formal assertion — embed the conflict in surrounding discussion context.
- Be 2-4 sentences long, with the conflicting detail woven into the natural flow.
- NOT use SRS "shall" format.
""".strip()


def build_contradiction_verification_prompt(
    unit_id: str,
    target_requirement: dict[str, Any],
    injected_text: str,
) -> str:
    return f"""\
Unit ID: {unit_id}

Original gold requirement:
  ID   : {target_requirement['id']}
  Text : {target_requirement['text']}

Generated contradicting statement:
  {injected_text}

Does the generated statement create a genuine, unambiguous conflict with the
original requirement on a concrete, testable attribute?

Return JSON using this schema:
{{
  "is_genuine_conflict": true | false,
  "reason": "brief explanation of why this is or is not a valid contradiction"
}}

A genuine conflict means: a careful requirements analyst reviewing both statements
side by side would immediately flag them as contradictory (different numeric values,
opposite access rules, mutually exclusive conditions, etc.).
""".strip()


def build_duplicate_generation_prompt(
    unit_id: str,
    target_requirement: dict[str, Any],
    target_artifact_type: str,
) -> str:
    return f"""\
Unit ID: {unit_id}

Target gold requirement (this is the one to paraphrase as a duplicate):
  ID   : {target_requirement['id']}
  Text : {target_requirement['text']}
  Kind : {target_requirement.get('requirement_kind', 'functional')}

Generate a paraphrase of the above requirement to be injected into a source artifact
of type: {target_artifact_type}

The paraphrase must express the same semantic intent using different wording.

Return JSON using this schema:
{{
  "injected_text": "2-4 sentences of natural stakeholder conversation that embed the restated requirement naturally within the discussion flow",
  "defect_subtype": "exact_duplicate | paraphrase_duplicate",
  "defect_description": "one sentence describing that this duplicates requirement {target_requirement['id']}"
}}

The injected_text should:
- Sound like natural stakeholder conversation (part of meeting notes, interview, or email).
- NOT be a standalone formal assertion — embed the restated need in surrounding discussion context.
- Be 2-4 sentences long, with the duplicate detail woven into the natural flow.
- NOT use SRS "shall" format.
- Express the SAME functional or non-functional need using DIFFERENT words.
""".strip()


def build_duplicate_verification_prompt(
    unit_id: str,
    target_requirement: dict[str, Any],
    injected_text: str,
) -> str:
    return f"""\
Unit ID: {unit_id}

Original gold requirement:
  ID   : {target_requirement['id']}
  Text : {target_requirement['text']}

Generated paraphrase:
  {injected_text}

Verify this paraphrase by answering two questions:
1. Does it express the SAME semantic need as the original? (same capability, same constraint)
2. Is the wording sufficiently DIFFERENT from the original? (not just trivial word substitution)

Return JSON using this schema:
{{
  "is_valid_duplicate": true | false,
  "reason": "brief explanation"
}}

A valid duplicate: same meaning, different words. A competent analyst should recognise
both as expressing the same underlying requirement.
""".strip()
