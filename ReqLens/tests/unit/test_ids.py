"""Unit tests for domain ID generation."""

from reqinone2.domain.ids import generate_id, validate_id_prefix


class TestIds:
    def test_generate_id_format(self):
        rid = generate_id("REQ")
        assert rid.startswith("REQ-")
        assert len(rid) == 12  # "REQ-" + 8 hex chars

    def test_unique(self):
        ids = {generate_id("REQ") for _ in range(100)}
        assert len(ids) == 100

    def test_validate_prefix(self):
        rid = generate_id("REQ")
        assert validate_id_prefix(rid, "REQ") is True
        assert validate_id_prefix(rid, "SPN") is False
