"""Unit tests for acceptance rules."""

from reqinone2.domain.enums import (
    EvidenceStatus,
    RequirementKind,
    RequirementStatus,
    ReviewStatus,
    Severity,
)
from reqinone2.domain.models import (
    ConflictFinding,
    EvidenceAssessment,
    QualityFinding,
    Requirement,
    requirement_eligible_for_srs,
)


def _make_req(**overrides) -> Requirement:
    defaults = dict(
        project_id="PRJ-001",
        text="Test requirement",
        kind=RequirementKind.functional,
        status=RequirementStatus.accepted,
        review_status=ReviewStatus.accepted,
        source_span_ids=["SPN-001"],
    )
    defaults.update(overrides)
    return Requirement(**defaults)


def _make_evidence(candidate_id: str, status: EvidenceStatus) -> EvidenceAssessment:
    return EvidenceAssessment(
        project_id="PRJ-001",
        requirement_candidate_id=candidate_id,
        status=status,
        explanation="test",
    )


class TestAcceptanceRules:
    def test_fully_eligible(self):
        req = _make_req()
        evidence = _make_evidence("CAND-001", EvidenceStatus.entailed)
        eligible, reasons = requirement_eligible_for_srs(req, evidence, [], [])
        assert eligible is True
        assert reasons == []

    def test_no_source_spans(self):
        req = _make_req(source_span_ids=[])
        evidence = _make_evidence("CAND-001", EvidenceStatus.entailed)
        eligible, reasons = requirement_eligible_for_srs(req, evidence, [], [])
        assert eligible is False
        assert any("source span" in r.lower() for r in reasons)

    def test_no_evidence(self):
        req = _make_req()
        eligible, reasons = requirement_eligible_for_srs(req, None, [], [])
        assert eligible is False
        assert any("evidence" in r.lower() for r in reasons)

    def test_contradicted_evidence(self):
        req = _make_req(review_status=ReviewStatus.pending)
        evidence = _make_evidence("CAND-001", EvidenceStatus.contradicted)
        eligible, reasons = requirement_eligible_for_srs(req, evidence, [], [])
        assert eligible is False

    def test_severe_conflict_blocks(self):
        req = _make_req()
        evidence = _make_evidence("CAND-001", EvidenceStatus.entailed)
        conflict = ConflictFinding(
            project_id="PRJ-001",
            conflict_type="contradiction",
            involved_requirement_ids=[req.id],
            severity=Severity.critical,
            explanation="Contradicts another req.",
        )
        eligible, reasons = requirement_eligible_for_srs(req, evidence, [conflict], [])
        assert eligible is False
        assert any("conflict" in r.lower() for r in reasons)

    def test_critical_quality_finding_blocks(self):
        req = _make_req()
        evidence = _make_evidence("CAND-001", EvidenceStatus.entailed)
        finding = QualityFinding(
            requirement_id=req.id,
            issue_type="vague_term",
            severity=Severity.critical,
            explanation="Uses vague term.",
        )
        eligible, reasons = requirement_eligible_for_srs(req, evidence, [], [finding])
        assert eligible is False
