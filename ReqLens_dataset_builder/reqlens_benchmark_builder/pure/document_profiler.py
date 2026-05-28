"""Lightweight structural profiler for PURE documents.

The profiler scans raw text and assembles a ``DocumentProfile`` with signal
counts, heading density, and an overall structure-strength estimate.  This
guides downstream modules on how much to trust section-detection vs. raw-text
chunking.
"""

from __future__ import annotations

import re

from reqlens_benchmark_builder.schemas.benchmark_models import DocumentProfile

# ── Heading patterns ──────────────────────────────────────────────────────────
# Multiple heading conventions seen in SRS/IEEEE-830-style documents:
#   1)  "4.1.2 System Requirements"   (numbered)
#   2)  "# Introduction"              (Markdown ATX)
#   3)  "INTRODUCTION"                (ALL-CAPS short line)
#   4)  Lines ending in underlining   (RST-style — rare in SRS)

_NUMBERED_HEADING_RE = re.compile(
    r"^\s*(\d+(\.\d+)*)\s+[A-Z].{2,80}$", re.MULTILINE
)
_MARKDOWN_HEADING_RE = re.compile(r"^#{1,4}\s+.+$", re.MULTILINE)
_ALLCAPS_HEADING_RE  = re.compile(r"^[A-Z][A-Z\s\-/]{4,60}$", re.MULTILINE)

# Requirement signal words
_SHALL_RE = re.compile(r"\bshall\b", re.IGNORECASE)
_MUST_RE  = re.compile(r"\bmust\b",  re.IGNORECASE)
_WILL_RE  = re.compile(r"\bwill\b",  re.IGNORECASE)

# Domain contextual signals
_USE_CASE_RE    = re.compile(r"\buse[\s-]case\b",    re.IGNORECASE)
_STAKEHOLDER_RE = re.compile(r"\bstakeholder\b",     re.IGNORECASE)
_REQUIREMENT_RE = re.compile(r"\brequirements?\b",   re.IGNORECASE)


def profile_document(text: str) -> DocumentProfile:
    """Compute a lightweight structural profile of *text*.

    ``structure_strength`` is set to:
    - ``"strong"``  — well-structured document (good headings + rich req signals)
    - ``"mixed"``   — partial structure (some headings or moderate req signals)
    - ``"weak"``    — nearly unstructured prose
    """
    numbered_headings = _NUMBERED_HEADING_RE.findall(text)
    markdown_headings = _MARKDOWN_HEADING_RE.findall(text)
    allcaps_headings  = _ALLCAPS_HEADING_RE.findall(text)

    heading_density = (
        len(numbered_headings)
        + len(markdown_headings)
        + len(allcaps_headings)
    )

    shall_count        = len(_SHALL_RE.findall(text))
    must_count         = len(_MUST_RE.findall(text))
    will_count         = len(_WILL_RE.findall(text))
    use_case_count     = len(_USE_CASE_RE.findall(text))
    stakeholder_count  = len(_STAKEHOLDER_RE.findall(text))
    req_keyword_count  = len(_REQUIREMENT_RE.findall(text))

    req_signal_total = shall_count + must_count + will_count

    # Classify structure strength
    if heading_density >= 8 and req_signal_total >= 10:
        strength = "strong"
    elif heading_density >= 4 or req_signal_total >= 5:
        strength = "mixed"
    else:
        strength = "weak"

    return DocumentProfile(
        char_count=len(text),
        heading_density=heading_density,
        shall_count=shall_count,
        must_count=must_count,
        will_count=will_count,
        use_case_count=use_case_count,
        stakeholder_count=stakeholder_count,
        requirement_keyword_count=req_keyword_count,
        structure_strength=strength,
    )
