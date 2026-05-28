"""Abstract base class for all system adapters."""

from __future__ import annotations

from abc import ABC, abstractmethod

from reqlens_eval.models.artifacts import (
    PoisonedTrack1Artifact,
    PoisonedTrack2Artifact,
)
from reqlens_eval.models.experiment import (
    Track1SystemOutput,
    Track2SystemOutput,
)


class SystemAdapter(ABC):
    """Contract every system adapter must implement.

    One adapter = one system under evaluation (Baseline, ReqInOne v1, ReqLens v2).
    The adapter is stateless and callable for any artifact independently.
    """

    @property
    @abstractmethod
    def system_id(self) -> str:
        """Unique stable identifier, e.g. 'baseline' | 'reqinone_v1' | 'reqlens_v2'."""
        ...

    @abstractmethod
    def evaluate_candidates(
        self,
        artifact: PoisonedTrack1Artifact,
    ) -> Track1SystemOutput:
        """Track 1: decide which candidates in the pool are supported by the source texts.

        The system receives:
          - artifact.source_texts  — unmodified stakeholder evidence bundle
          - artifact.candidate_pool — gold requirements + injected hallucinated fakes (shuffled)

        It must return a ``Track1SystemOutput`` with one ``CandidateDecision``
        per candidate (status: accepted | rejected | uncertain).
        'uncertain' decisions are later resolved by the LLM judge.
        """
        ...

    @abstractmethod
    def extract_requirements(
        self,
        artifact: PoisonedTrack2Artifact,
    ) -> Track2SystemOutput:
        """Track 2: run full extraction on the poisoned source texts.

        The system receives:
          - artifact.source_texts — source texts with seeded defects embedded

        It must return a ``Track2SystemOutput`` with whatever requirements
        it extracts. The evaluation judge will then check whether seeded defects
        (contradictions / duplicates) survived or were resolved.
        """
        ...
