"""Text normalization utilities for ingested documents."""

from __future__ import annotations

import re
import unicodedata


def normalize_whitespace(text: str) -> str:
    """Collapse runs of whitespace, strip leading/trailing."""
    return re.sub(r"[ \t]+", " ", text).strip()


def normalize_unicode(text: str) -> str:
    """NFC-normalize Unicode and replace common oddities."""
    text = unicodedata.normalize("NFC", text)
    # Replace smart quotes with straight quotes
    text = text.replace("\u2018", "'").replace("\u2019", "'")
    text = text.replace("\u201c", '"').replace("\u201d", '"')
    # Replace en/em dash with hyphen
    text = text.replace("\u2013", "-").replace("\u2014", "-")
    return text


def redact_pii(text: str) -> str:
    """Basic PII redaction – emails, phone numbers, tokens."""
    # Email addresses
    text = re.sub(
        r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}",
        "[REDACTED_EMAIL]",
        text,
    )
    # Phone numbers (simple patterns)
    text = re.sub(
        r"\b\+?\d{1,3}[-.\s]?\(?\d{1,4}\)?[-.\s]?\d{1,4}[-.\s]?\d{1,9}\b",
        "[REDACTED_PHONE]",
        text,
    )
    # API keys / tokens (long hex or base64 strings)
    text = re.sub(
        r"\b[A-Za-z0-9+/]{32,}={0,2}\b",
        "[REDACTED_TOKEN]",
        text,
    )
    return text


def normalize_document_text(text: str, *, redact: bool = False) -> str:
    """Full normalization pipeline for ingested document text."""
    text = normalize_unicode(text)
    text = normalize_whitespace(text)
    if redact:
        text = redact_pii(text)
    return text
