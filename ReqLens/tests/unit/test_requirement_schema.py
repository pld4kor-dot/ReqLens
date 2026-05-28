"""Unit tests for Pydantic requirement schemas."""

import pytest
from pydantic import ValidationError

from reqinone2.llm.schemas import (
    CandidateRequirementLLM,
    EvidenceAssessmentLLM,
    RequirementExtractionOutput,
)
from reqinone2.domain.enums import EvidenceStatus, NFRSubtype, RequirementKind


class TestCandidateRequirementSchema:
    def test_valid_candidate(self):
        c = CandidateRequirementLLM(
            temp_id="CAND-001",
            text="The system shall allow event registration.",
            requirement_kind=RequirementKind.functional,
            rationale="Stated by Alice in interview.",
            confidence=0.9,
            source_span_ids=["SPN-abc"],
        )
        assert c.temp_id == "CAND-001"
        assert c.nfr_subtype == NFRSubtype.not_applicable

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            CandidateRequirementLLM(
                temp_id="CAND-002",
                text="Bad confidence",
                requirement_kind=RequirementKind.functional,
                rationale="test",
                confidence=1.5,
            )

    def test_empty_source_spans_allowed(self):
        c = CandidateRequirementLLM(
            temp_id="CAND-003",
            text="Hypothetical requirement.",
            requirement_kind=RequirementKind.functional,
            rationale="Inferred",
            confidence=0.3,
        )
        assert c.source_span_ids == []


class TestExtractionOutput:
    def test_valid_output(self):
        output = RequirementExtractionOutput(
            candidates=[
                CandidateRequirementLLM(
                    temp_id="CAND-001",
                    text="Req 1",
                    requirement_kind=RequirementKind.functional,
                    rationale="test",
                    confidence=0.8,
                ),
            ],
            unresolved_questions=["What about mobile support?"],
        )
        assert len(output.candidates) == 1
        assert len(output.unresolved_questions) == 1

    def test_empty_output(self):
        output = RequirementExtractionOutput(candidates=[])
        assert len(output.candidates) == 0


class TestEvidenceAssessmentSchema:
    def test_entailed(self):
        a = EvidenceAssessmentLLM(
            requirement_temp_id="CAND-001",
            status=EvidenceStatus.entailed,
            supporting_span_ids=["SPN-001"],
            explanation="Directly stated by Alice.",
            confidence=0.95,
        )
        assert a.status == EvidenceStatus.entailed

    def test_insufficient_evidence(self):
        a = EvidenceAssessmentLLM(
            requirement_temp_id="CAND-002",
            status=EvidenceStatus.insufficient_evidence,
            explanation="Not found in any source span.",
            confidence=0.1,
        )
        assert a.contradicting_span_ids == []
