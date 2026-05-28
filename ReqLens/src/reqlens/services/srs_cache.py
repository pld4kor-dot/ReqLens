"""Disk-backed cache for composed SRS markdown, keyed by project_id.

Why this exists
---------------
The Composer Agent is expensive: it makes an LLM call against every
accepted requirement.  Before this cache, running the pipeline composed
the SRS but threw the rendered markdown away, so the user had to click
"Generate SRS Document" on the Export screen, which fired the composer
a second time.

This module gives the pipeline (specifically the per-agent composer
endpoint) a place to stash the rendered markdown so the export endpoint
can serve it back instantly.  When the pipeline re-runs, the new
composer output simply overwrites the cached file.

Storage layout
--------------
~/.reqlens_cache/srs/<project_id>.md

The directory is created on demand.  We deliberately avoid the DB so
this change requires no schema migration.
"""

from __future__ import annotations

from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)


def _cache_dir() -> Path:
    d = Path.home() / ".reqlens_cache" / "srs"
    d.mkdir(parents=True, exist_ok=True)
    return d


def _cache_path(project_id: str) -> Path:
    # project_id is generated server-side, but be defensive about
    # characters that aren't safe as filenames.
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)
    return _cache_dir() / f"{safe}.md"


def save_srs_markdown(project_id: str, markdown: str) -> None:
    """Persist the composed SRS markdown for this project.

    Overwrites any previous SRS for the same project.  Silent on I/O
    failure — caching is best-effort; the export endpoint will fall
    back to re-composing if the file is missing.
    """
    try:
        path = _cache_path(project_id)
        path.write_text(markdown, encoding="utf-8")
        logger.info("srs_cache.saved", project_id=project_id, bytes=len(markdown))
    except Exception as exc:
        logger.warning("srs_cache.save_failed", project_id=project_id, error=str(exc))


def load_srs_markdown(project_id: str) -> str | None:
    """Return the cached SRS markdown, or None if no cache exists."""
    path = _cache_path(project_id)
    if not path.is_file():
        return None
    try:
        return path.read_text(encoding="utf-8")
    except Exception as exc:
        logger.warning("srs_cache.load_failed", project_id=project_id, error=str(exc))
        return None


def has_srs_markdown(project_id: str) -> bool:
    """Quick existence check used by the UI to decide whether to show
    the download button immediately."""
    return _cache_path(project_id).is_file()


def clear_srs_markdown(project_id: str) -> None:
    """Remove the cached SRS for this project."""
    try:
        _cache_path(project_id).unlink(missing_ok=True)
    except Exception as exc:
        logger.warning("srs_cache.clear_failed", project_id=project_id, error=str(exc))


# ─────────────────────────────────────────────────────────────────────────────
# Stakeholder-questions sidecar cache
# ─────────────────────────────────────────────────────────────────────────────
# Stored next to the SRS markdown as JSON: <project_id>.questions.json
# Lets the Elicitation Agent run once (its own pipeline step) and the Composer
# pick up its output without re-invoking the LLM.

import json


def _questions_cache_path(project_id: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_" else "_" for c in project_id)
    return _cache_dir() / f"{safe}.questions.json"


def save_stakeholder_questions(project_id: str, questions: list[str]) -> None:
    """Persist the elicitation agent's stakeholder questions for this project."""
    try:
        path = _questions_cache_path(project_id)
        path.write_text(json.dumps(list(questions), ensure_ascii=False, indent=2), encoding="utf-8")
        logger.info("questions_cache.saved", project_id=project_id, count=len(questions))
    except Exception as exc:
        logger.warning("questions_cache.save_failed", project_id=project_id, error=str(exc))


def load_stakeholder_questions(project_id: str) -> list[str] | None:
    """Return the cached stakeholder questions, or None if no cache exists."""
    path = _questions_cache_path(project_id)
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
        if isinstance(data, list):
            return [str(q) for q in data]
        return None
    except Exception as exc:
        logger.warning("questions_cache.load_failed", project_id=project_id, error=str(exc))
        return None
