"""All prompt templates for the benchmark builder.

System prompts are constants.  User prompts are built by functions that accept
strongly-typed parameters and return a formatted string.

Three logical roles:
- GENERATION  — for scenario brief and source-bundle generation
- EXTRACTION  — for PURE requirement extraction and merging
- VALIDATION  — for coverage checking and unsupported-requirement detection
"""

from __future__ import annotations

import json
from typing import Any


# ══════════════════════════════════════════════════════════════════════════════
# System prompts
# ══════════════════════════════════════════════════════════════════════════════

GENERATION_SYSTEM_PROMPT = """\
You are an expert requirements-engineering benchmark generator.

Your role is to reverse-engineer realistic, messy upstream stakeholder source
text from a set of structured gold requirements or a project scenario brief.
The output should convincingly simulate how real stakeholders communicate BEFORE
a polished SRS is ever written.

Core rules:
1. Every gold requirement must be inferable from at least one artifact in the
   source bundle – but stated in informal stakeholder language, not SRS language.
2. Do NOT copy requirement text verbatim.  Paraphrase, split across speakers,
   imply, or reference indirectly.
3. Do NOT introduce new system capabilities that are not covered by the gold
   requirements.  Conversational noise, opinions, and off-topic small-talk are
   fine; ungrounded technical features are not.
4. Vary how requirements appear: some stated explicitly, some implied by context,
   some revealed through questions/objections.
5. Output ONLY valid JSON matching the schema in the user message.
""".strip()


EXTRACTION_SYSTEM_PROMPT = """\
You are an expert requirements engineer assigned to extract structured
requirements from raw SRS / project document text.

Core rules:
1. Extract only requirements that are explicitly or clearly implicitly SUPPORTED
   by the chunk text.  Do not invent requirements.
2. Keep each requirement concise, atomic, and testable where possible.
3. Rewrite passive or verbose phrasing into active system-capability statements
   ("The system shall …").
4. Assign requirement_kind from:
     functional | non_functional | constraint | domain_assumption | business_rule
5. Assign nfr_subtype (only when kind == non_functional) from:
     security | privacy | usability | reliability | availability | performance |
     maintainability | portability | scalability | compliance | other | not_applicable
6. Avoid duplicating requirements already obviously covered by a previous
   statement in the same chunk.
7. Output ONLY valid JSON matching the schema in the user message.
""".strip()


VALIDATION_SYSTEM_PROMPT = """\
You are a strict benchmark quality auditor.

Your task is to verify whether a generated source bundle is a valid benchmark
artifact by checking two properties:
a) COVERAGE – every gold requirement is traceable to (inferable from) the source.
b) LEAKAGE  – the source does not strongly imply additional system capabilities
               that are NOT in the gold set.

Core rules:
1. Judge ONLY from the provided texts; use no external knowledge.
2. For coverage: a requirement is "supported" if a reader could reasonably
   derive it from the source bundle, even if stated informally or indirectly.
   Exact wording match is NOT required.
3. For leakage: flag only significant ungrounded features, not minor conversational
   details or implementation-level implementation specifics.
4. Be decisive – every requirement gets a clear supported/unsupported verdict.
5. Output ONLY valid JSON matching the schema in the user message.
""".strip()


# ══════════════════════════════════════════════════════════════════════════════
# User prompt builders
# ══════════════════════════════════════════════════════════════════════════════

# ── 1) Scenario brief ─────────────────────────────────────────────────────────

def build_scenario_brief_prompt(
    unit_id: str,
    gold_requirements: list[dict[str, Any]],
    global_context: str | None = None,
) -> str:
    ctx_block = (
        f"\nDocument context (for domain / stakeholder grounding):\n"
        f"<context>\n{global_context[:8000]}\n</context>\n"
        if global_context
        else ""
    )

    # Group requirements by kind for a cleaner prompt
    by_kind: dict[str, list[str]] = {}
    for r in gold_requirements:
        kind = r.get("requirement_kind", "functional")
        by_kind.setdefault(kind, []).append(r.get("text", ""))

    req_table = "\n".join(
        f"[{kind.upper()}]\n" + "\n".join(f"  - {t}" for t in texts)
        for kind, texts in by_kind.items()
    )

    return f"""\
Unit ID: {unit_id}
{ctx_block}
Gold requirements ({len(gold_requirements)} total):
{req_table}

Generate a concise project scenario brief in JSON using this schema:
{{
  "project_name": "string — short evocative name",
  "domain": "string — application domain (e.g. healthcare, e-commerce, …)",
  "users": ["string — user role 1", "string — user role 2"],
  "stakeholders": ["string — stakeholder/actor 1"],
  "business_goals": ["string — high-level business objective"],
  "core_features": ["string — main feature or capability"],
  "quality_concerns": ["string — NFR concern (performance, security, …)"],
  "constraints": ["string — technical/legal/operational constraint"],
  "terminology": ["string — key domain-specific term"]
}}

The brief must reflect ALL requirement kinds above.  Keep each list concise \
(3–7 items).  Do not echo the requirement text verbatim.
""".strip()


# ── 2) Source bundle generation ───────────────────────────────────────────────

def _scale_targets(req_count: int) -> dict[str, int]:
    """Return minimum word targets for each artifact, scaled to requirement count.

    Small projects (≤15 reqs)  → compact artifacts — fast to read and validate.
    Medium projects (16-35 reqs) → medium artifacts — richer context.
    Large projects  (>35 reqs)  → long artifacts — necessary to cover all reqs.
    """
    if req_count <= 15:
        return {"transcript": 1200, "notes": 700, "email": 600}
    elif req_count <= 35:
        return {"transcript": 2200, "notes": 1200, "email": 900}
    else:
        return {"transcript": 3500, "notes": 2000, "email": 1400}


def _speaker_count_guidance(req_count: int) -> str:
    if req_count <= 15:
        return "18–26 speaker turns, 3–4 distinct speaker roles"
    elif req_count <= 35:
        return "28–40 speaker turns, 4–5 distinct speaker roles"
    else:
        return "45–65 speaker turns, 5–6 distinct speaker roles"


def build_source_bundle_prompt(
    unit_id: str,
    brief: dict[str, Any],
    gold_requirements: list[dict[str, Any]],
) -> str:
    req_count = len(gold_requirements)
    targets   = _scale_targets(req_count)
    speakers  = _speaker_count_guidance(req_count)

    gold_json = json.dumps(
        [{"id": r["id"], "text": r["text"], "kind": r.get("requirement_kind")}
         for r in gold_requirements],
        indent=2,
        ensure_ascii=False,
    )

    # For large sets, also hint the model to distribute reqs across artifacts
    distribution_note = (
        f"With {req_count} requirements to cover, deliberately spread them "
        "across all three artifacts.  Aim for roughly equal coverage in each — "
        "no single artifact should carry most of the requirements."
        if req_count > 20
        else "Distribute requirements across all three artifacts."
    )

    return f"""\
Unit ID: {unit_id}

Project brief:
{json.dumps(brief, indent=2, ensure_ascii=False)}

Gold requirements to cover ({req_count} items):
{gold_json}

Generate exactly 3 raw source artifacts in JSON using this schema:
{{
  "source_texts": [
    {{
      "type": "interview_transcript",
      "title": "string — descriptive title",
      "text": "string — full artifact text (≥ {targets['transcript']} words)"
    }},
    {{
      "type": "meeting_notes",
      "title": "string — descriptive title",
      "text": "string — full artifact text (≥ {targets['notes']} words)"
    }},
    {{
      "type": "email_thread",
      "title": "string — descriptive title",
      "text": "string — full artifact text (≥ {targets['email']} words)"
    }}
  ]
}}

=== REALISM MANDATE ===
These artifacts must look like RAW, UNPOLISHED real-world stakeholder communication —
NOT synthesized or cleaned-up text.  Apply ALL of the following chaos patterns:

INTERVIEW TRANSCRIPT  ({speakers}):
  - Label turns as "FirstName [Role]: text"
  - Include: false starts ("Actually, wait—"), self-corrections, unfinished thoughts
  - Speakers interrupt, talk over each other, change topic mid-sentence
  - Real confusion: a stakeholder misunderstands a technical term, gets corrected
  - Off-topic tangent (scheduling, related product complaint) that gets redirected
  - Disagreement between two speakers that gets partially resolved
  - Requirements slip out indirectly ("yeah and obviously you'd want it to notify you...")
  - Some speakers use precise numbers/metrics; others are vague ("it needs to be fast")
  - At least one speaker expresses doubt or doesn't know the answer
  - Mix of technical jargon and plain language in the same conversation

MEETING NOTES  (raw notes taken during meeting, not polished minutes):
  - Header: date (2025), room/call link, attendees with roles
  - Notes are incomplete — bullet fragments, abbreviations, partial sentences
  - Gaps marked as "[?]" or "[TBD]" or "check with X"
  - Action items mixed in awkwardly with discussion points
  - Some bullets are clearly contradictory or unclear — realistic meeting chaos
  - One subsection titled "Open Issues" or "Parking Lot" with unresolved items
  - At least one crossed-out idea or marked "NOT decided yet"
  - Numbers/quantities hand-noted without unit context in some cases

EMAIL THREAD  (5–7 emails, chronological):
  - From / To / CC / Subject / body for each email
  - Informal tone: casual greetings, colloquialisms, typos, "sent from mobile"
  - Reply chains quote previous emails partially (not fully)
  - Subject line evolves (RE: / FWD: / subject drift across emails)
  - At least one email is a long rambling explanation with multiple topics
  - At least one email introduces new constraints or pushback on previous agreement
  - At least one cc'd person replies unexpectedly with a new concern
  - Thread ends with partial resolution and at least one open action item

{distribution_note}

Coverage rule: every gold requirement (by ID) must be traceable to at least \
one of the three artifacts.  Requirements may appear as explicit statements, \
indirect implications, or conversational references — NOT as verbatim copies.
""".strip()


# ── 3) Chunk-level requirement extraction (PURE) ─────────────────────────────

def build_extract_from_chunk_prompt(
    doc_id: str,
    chunk_id: str,
    chunk_text: str,
    strategy: str,
    section_title: str | None,
) -> str:
    section_line = f"Section: {section_title}" if section_title else "Section: (unspecified)"

    return f"""\
Document ID : {doc_id}
Chunk ID    : {chunk_id}
Strategy    : {strategy}
{section_line}

Chunk text:
\"\"\"
{chunk_text}
\"\"\"

Extract all distinct atomic requirements supported by the chunk above.

Return JSON using this schema:
{{
  "requirements": [
    {{
      "text": "string — one concise active-voice requirement statement",
      "requirement_kind": "functional | non_functional | constraint | domain_assumption | business_rule",
      "nfr_subtype": "performance | security | usability | reliability | availability | \
scalability | maintainability | portability | compliance | privacy | other | not_applicable",
      "raw_label": null
    }}
  ]
}}

Return an empty list if no clear requirements are present: {{"requirements": []}}
""".strip()


# ── 4) Requirement merge (PURE) ──────────────────────────────────────────────

def build_merge_requirements_prompt(
    doc_id: str,
    candidates: list[dict[str, Any]],
) -> str:
    candidates_json = json.dumps(
        [
            {
                "text": c.get("text", ""),
                "requirement_kind": c.get("requirement_kind", "functional"),
                "nfr_subtype": c.get("nfr_subtype", "not_applicable"),
                "source_strategy": c.get("source_strategy"),
                "source_region": c.get("source_region"),
            }
            for c in candidates
        ],
        indent=2,
        ensure_ascii=False,
    )

    return f"""\
Document ID: {doc_id}
Total candidates: {len(candidates)}

Candidate requirements:
{candidates_json}

Task: produce a clean, deduplicated list of gold requirements.

Rules:
- Merge exact duplicates and near-paraphrases into the single best statement.
- Keep distinct requirements separate even if they overlap in topic.
- Preserve the clearest and most specific phrasing of merged candidates.
- Use the majority vote for requirement_kind / nfr_subtype when merging.
- Set source_strategy to "both" when merged from section + raw_chunk candidates.
- If a candidate's source_region is not meaningful after merging, use the most
  descriptive one.

Return JSON:
{{
  "gold_requirements": [
    {{
      "text": "string",
      "requirement_kind": "functional | non_functional | constraint | domain_assumption | business_rule",
      "nfr_subtype": "performance | security | usability | … | not_applicable",
      "raw_label": null,
      "source_strategy": "section | raw_chunk | both",
      "source_region": "string | null"
    }}
  ]
}}
""".strip()


# ── 5) Coverage validation ────────────────────────────────────────────────────

def build_coverage_prompt(
    unit_id: str,
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
) -> str:
    """Full (non-batched) coverage prompt — kept for compatibility."""
    sources_json = json.dumps(
        [{"type": s["type"], "text": s["text"]} for s in source_texts],
        indent=2,
        ensure_ascii=False,
    )
    reqs_json = json.dumps(
        [{"id": r["id"], "text": r["text"]} for r in gold_requirements],
        indent=2,
        ensure_ascii=False,
    )

    return f"""\
Unit ID: {unit_id}

Source bundle ({len(source_texts)} artifacts):
{sources_json}

Gold requirements ({len(gold_requirements)} items):
{reqs_json}

For each gold requirement decide whether it is "supported" by the source bundle.
"Supported" = a reader could reasonably derive or infer this requirement from the
source, even if phrased differently or split across statements.

Return compact JSON — keep evidence_snippets ≤ 50 chars, reason ≤ 20 words:
{{
  "coverage": [
    {{
      "req_id": "string",
      "supported": true,
      "evidence_snippets": ["≤50-char quote"],
      "reason": "brief reason"
    }}
  ],
  "coverage_rate": 0.0,
  "missing_req_ids": ["ids of unsupported requirements"]
}}

coverage_rate = supported_count / total_count (0.0 – 1.0).
""".strip()


def build_coverage_prompt_batch(
    unit_id: str,
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
    batch_index: int,
    total_batches: int,
) -> str:
    """Coverage prompt for a subset (batch) of gold requirements.

    Sends the full source bundle but only the requirements in this batch,
    keeping response sizes predictable regardless of total requirement count.
    """
    # Compact source: truncate each artifact text to 6000 chars to stay within limits
    compact_sources = []
    for s in source_texts:
        t = s.get("text", "")
        compact_sources.append({
            "type": s.get("type", ""),
            "text": t[:6000] + (" …[truncated]" if len(t) > 6000 else ""),
        })
    sources_json = json.dumps(compact_sources, indent=2, ensure_ascii=False)
    reqs_json = json.dumps(
        [{"id": r["id"], "text": r["text"]} for r in gold_requirements],
        indent=2,
        ensure_ascii=False,
    )

    return f"""\
Unit ID: {unit_id}  |  Batch {batch_index + 1} of {total_batches}

Source bundle ({len(source_texts)} artifacts — texts may be truncated for context):
{sources_json}

Gold requirements in this batch ({len(gold_requirements)} items):
{reqs_json}

For EACH requirement listed above decide whether it is "supported" by the source bundle.
"Supported" = a reader could reasonably derive or infer this requirement from the
source, even if phrased differently or split across statements.

Return compact JSON.  Keep evidence_snippets ≤ 50 chars.  Keep reason ≤ 15 words.
{{
  "coverage": [
    {{
      "req_id": "string",
      "supported": true,
      "evidence_snippets": ["≤50-char quote"],
      "reason": "brief reason"
    }}
  ]
}}

You MUST include one entry per requirement in the batch.  Do not skip any.
""".strip()


# ── 6) Unsupported-requirement detection ─────────────────────────────────────

def build_unsupported_prompt(
    unit_id: str,
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
) -> str:
    sources_json = json.dumps(
        [{"type": s["type"], "text": s["text"]} for s in source_texts],
        indent=2,
        ensure_ascii=False,
    )
    gold_texts = [r.get("text", "") for r in gold_requirements]
    gold_summary = json.dumps(gold_texts, ensure_ascii=False)

    return f"""\
Unit ID: {unit_id}

Source bundle ({len(source_texts)} artifacts):
{sources_json}

Gold requirement texts (for comparison):
{gold_summary}

Identify any SIGNIFICANT system-capability requirements implied by the source bundle
that are NOT present in the gold set above.

"Significant" means: a clear statement that the system should have a feature,
behavior, or quality attribute not covered by any gold requirement.
Do NOT flag: conversational filler, opinions, procedural logistics, or minor
implementation details.

Return JSON:
{{
  "unsupported_implied_requirements": [
    {{
      "text": "string — inferred implied requirement",
      "source_snippet": "short quote from the source artifact",
      "source_type": "interview_transcript | meeting_notes | email_thread",
      "reason": "why this is considered a new requirement"
    }}
  ],
  "count": 0
}}

Return {{"unsupported_implied_requirements": [], "count": 0}} if none found.
""".strip()


# ── 7) Repair ─────────────────────────────────────────────────────────────────

def build_repair_prompt(
    unit_id: str,
    brief: dict[str, Any],
    source_texts: list[dict[str, Any]],
    gold_requirements: list[dict[str, Any]],
    coverage_report: dict[str, Any],
    unsupported_report: dict[str, Any],
) -> str:
    missing_ids  = coverage_report.get("missing_req_ids", [])
    missing_reqs = [r for r in gold_requirements if r.get("id") in missing_ids]
    leaks        = unsupported_report.get("unsupported_implied_requirements", [])

    missing_json     = json.dumps(missing_reqs, indent=2, ensure_ascii=False)
    leaks_json       = json.dumps(leaks, indent=2, ensure_ascii=False)
    current_json     = json.dumps(
        [{"type": s["type"], "title": s.get("title", ""), "text": s["text"]}
         for s in source_texts],
        indent=2,
        ensure_ascii=False,
    )

    return f"""\
Unit ID: {unit_id}

Project brief:
{json.dumps(brief, indent=2, ensure_ascii=False)}

Current source bundle:
{current_json}

Validation failures to fix:

MISSING (not covered by current source bundle):
{missing_json if missing_reqs else "None"}

LEAKAGE (implied by source but NOT in gold set):
{leaks_json if leaks else "None"}

Repair the source bundle so that:
1. Every missing gold requirement is now inferable from at least one artifact.
2. Passages that strongly imply the leaking capabilities are softened or removed.
3. All other requirements stay covered.
4. The 3 artifact types and the overall realistic tone are preserved.

Return JSON using the same schema as the original generation (source_texts list \
with type / title / text).
""".strip()
