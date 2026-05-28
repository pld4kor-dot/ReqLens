"""Unit tests for normalization."""

from reqinone2.ingestion.normalization import (
    normalize_document_text,
    normalize_unicode,
    normalize_whitespace,
    redact_pii,
)


class TestNormalization:
    def test_whitespace(self):
        assert normalize_whitespace("  hello   world  ") == "hello world"

    def test_unicode_quotes(self):
        assert normalize_unicode("\u201chello\u201d") == '"hello"'

    def test_redact_email(self):
        assert "[REDACTED_EMAIL]" in redact_pii("Contact alice@example.com today")

    def test_full_pipeline(self):
        result = normalize_document_text("  \u201chello\u201d   world  ", redact=False)
        assert result == '"hello" world'
