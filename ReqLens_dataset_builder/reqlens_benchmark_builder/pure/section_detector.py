"""Heuristic section detector for PURE SRS documents.

Design goals:
- Detect section boundaries from multiple heading conventions (numbered,
  Markdown ATX, ALL-CAPS, bracket-style page markers).
- Classify each section by content type: context/requirements/nfr/usecase/other.
- Flag requirement-dense sections so the extractor can prioritise them.
- Never FAIL — if detection finds nothing meaningful, return the whole text
  as a single section.

This is intentionally heuristic; correctness of every section boundary is
not required.  The raw-text extraction path always runs in parallel as backup.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

import structlog

logger = structlog.get_logger(__name__)

# ── Section-type keyword vocabularies ─────────────────────────────────────────

_CONTEXT_KEYWORDS = [
    "introduction", "background", "overview", "purpose", "scope", "motivation",
    "objective", "goals", "project", "problem statement", "problem", "about",
    "description", "document", "organization", "intended audience", "audience",
    "glossary", "abbreviation", "acronym", "reference", "stakeholder",
    "actor", "user", "environment",
]

_REQUIREMENT_KEYWORDS = [
    "functional requirement", "functional", "feature", "shall", "must", "will",
    "requirement", "use case", "scenario", "system capability", "capability",
    "traceability", "rtm", "specification", "behavior", "behaviour",
]

_NFR_KEYWORDS = [
    "non-functional", "non functional", "quality", "supplementary",
    "performance", "security", "reliability", "usability", "availability",
    "scalability", "maintainability", "portability", "compliance",
    "constraint", "assumption", "interface", "operational", "legal",
]

_USECASE_KEYWORDS = [
    "use case", "use-case", "actor", "precondition", "postcondition",
    "main flow", "alternate flow", "exception flow", "scenario",
]

# ── Heading patterns (ordered by specificity) ─────────────────────────────────

_HEADING_PATTERNS = [
    # Numbered: "4.1.2 Section Title" or "4.1.2. Section Title"
    re.compile(
        r"^\s*(\d+(?:\.\d+)*\.?)\s+([A-Z].{2,120})\s*$",
        re.MULTILINE,
    ),
    # Markdown ATX: "## Section Title"
    re.compile(r"^(#{1,4})\s+(.{2,120})\s*$", re.MULTILINE),
    # ALL CAPS heading (5–60 chars, may contain spaces and basic punctuation)
    re.compile(r"^([A-Z][A-Z\s\-/:]{4,59}[A-Z])\s*$", re.MULTILINE),
    # Page markers from PDF extractors: "[Page N]"
    re.compile(r"^(\[Page\s+\d+\])\s*$", re.MULTILINE),
]


# ── Data class ────────────────────────────────────────────────────────────────

@dataclass
class Section:
    title: str
    text: str
    section_type: str = "other"       # context | requirements | nfr | usecase | other
    is_requirement_dense: bool = False
    matched_keywords: list[str] = field(default_factory=list)
    # Approximate requirement signal count in this section
    req_signal_count: int = 0


# ── Helpers ───────────────────────────────────────────────────────────────────

def _count_req_signals(text: str) -> int:
    return len(re.findall(r"\b(?:shall|must|will)\b", text, flags=re.IGNORECASE))


def _classify_section(title: str, text: str) -> tuple[str, list[str], bool]:
    """Return (section_type, matched_keywords, is_requirement_dense)."""
    combined = f"{title} {text}".lower()

    nfr_hits = [kw for kw in _NFR_KEYWORDS if kw in combined]
    req_hits  = [kw for kw in _REQUIREMENT_KEYWORDS if kw in combined]
    ctx_hits  = [kw for kw in _CONTEXT_KEYWORDS if kw in combined]
    uc_hits   = [kw for kw in _USECASE_KEYWORDS if kw in combined]

    req_signals = _count_req_signals(text)

    # Priority: nfr > requirements > usecase > context > other
    if len(nfr_hits) >= 2 or (len(nfr_hits) >= 1 and req_signals >= 3):
        stype = "nfr"
        kws = nfr_hits[:5]
    elif len(req_hits) >= 2 or req_signals >= 5:
        stype = "requirements"
        kws = req_hits[:5]
    elif len(uc_hits) >= 2:
        stype = "usecase"
        kws = uc_hits[:5]
    elif len(ctx_hits) >= 2:
        stype = "context"
        kws = ctx_hits[:5]
    else:
        stype = "other"
        kws = []

    is_dense = req_signals >= 3 or stype in ("requirements", "nfr")

    return stype, kws, is_dense


def _find_heading_positions(text: str) -> list[tuple[int, str]]:
    """Return sorted list of (char_offset, heading_title) for all headings."""
    hits: list[tuple[int, str]] = []

    for pattern in _HEADING_PATTERNS:
        for m in pattern.finditer(text):
            # Extract a clean title from whichever group has content
            groups = [g for g in m.groups() if g and g.strip()]
            title = " ".join(groups).strip() if groups else m.group(0).strip()
            hits.append((m.start(), title))

    # Sort by position and deduplicate overlapping matches
    hits.sort(key=lambda x: x[0])
    deduped: list[tuple[int, str]] = []
    prev_pos = -1
    for pos, title in hits:
        if pos > prev_pos + 5:  # ignore headings detected within 5 chars of each other
            deduped.append((pos, title))
            prev_pos = pos

    return deduped


# ── Public API ────────────────────────────────────────────────────────────────

def detect_sections(text: str) -> list[Section]:
    """Split *text* into heuristic sections with classification.

    Falls back to a single FULL_DOCUMENT section when no headings are found.
    Sections with fewer than 50 characters of body text are merged with the
    following section (avoids dozens of empty heading stubs).
    """
    heading_positions = _find_heading_positions(text)

    if not heading_positions:
        logger.info("section_detector.no_headings_found", fallback="FULL_DOCUMENT")
        rs = _count_req_signals(text)
        stype, kws, is_dense = _classify_section("FULL_DOCUMENT", text)
        return [
            Section(
                title="FULL_DOCUMENT",
                text=text,
                section_type=stype,
                is_requirement_dense=is_dense,
                matched_keywords=kws,
                req_signal_count=rs,
            )
        ]

    # Build section bodies between heading positions
    raw_sections: list[tuple[str, str]] = []
    for i, (start, title) in enumerate(heading_positions):
        # Section body = text from just after the heading line to the next heading
        body_start = text.find("\n", start)
        body_start = body_start + 1 if body_start != -1 else start + len(title)
        body_end   = heading_positions[i + 1][0] if i + 1 < len(heading_positions) else len(text)
        body       = text[body_start:body_end].strip()
        if body or i == 0:
            raw_sections.append((title, body))

    # Classify and build Section objects; merge stubs
    sections: list[Section] = []
    for title, body in raw_sections:
        if len(body) < 50 and sections:
            # Append stub body to previous section
            sections[-1].text += "\n\n" + body
        else:
            rs = _count_req_signals(body)
            stype, kws, is_dense = _classify_section(title, body)
            sections.append(
                Section(
                    title=title,
                    text=body,
                    section_type=stype,
                    is_requirement_dense=is_dense,
                    matched_keywords=kws,
                    req_signal_count=rs,
                )
            )

    logger.info(
        "section_detector.done",
        total=len(sections),
        requirement_dense=sum(1 for s in sections if s.is_requirement_dense),
    )
    return sections
