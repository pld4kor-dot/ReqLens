"""Track 2 normalizer — converts extraction outputs into a flat text representation.

The normalizer merges all extracted requirements from a ``Track2SystemOutput``
into a plain-text block (one requirement per line) that the LLM judge can
reason over when deciding whether a seeded defect was "leaked".

No LLM calls happen here — this module produces the judge-ready payload.
"""

from __future__ import annotations

from reqlens_eval.models.experiment import ExtractedRequirement, Track2SystemOutput


def build_extraction_text(system_output: Track2SystemOutput) -> str:
    """Serialise extracted requirements to a numbered plain-text block."""
    if not system_output.extracted_requirements:
        return "(no requirements extracted)"
    lines = []
    for i, req in enumerate(system_output.extracted_requirements, start=1):
        kind = req.requirement_kind.upper()
        nfr = (
            f" [{req.nfr_subtype}]"
            if req.nfr_subtype and req.nfr_subtype != "not_applicable"
            else ""
        )
        lines.append(f"{i}. [{kind}{nfr}] {req.text}")
    return "\n".join(lines)


def get_extracted_texts(system_output: Track2SystemOutput) -> list[str]:
    """Return the text of every extracted requirement as a plain list."""
    return [req.text for req in system_output.extracted_requirements]
