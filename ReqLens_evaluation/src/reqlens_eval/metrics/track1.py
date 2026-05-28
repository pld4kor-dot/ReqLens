"""Track 1 metrics: UAR, HRR, and GRR.

Definitions
-----------
Given:
  - H = total number of seeded hallucinated (fake) requirements
  - H_accepted = hallucinations accepted by the system (incorrectly treated as valid)
  - H_rejected = hallucinations rejected by the system (correctly filtered out)
  - G = total number of gold (legitimate) requirements
  - G_rejected = gold requirements incorrectly rejected by the system
  - G_accepted = gold requirements correctly accepted

  UAR = H_accepted / H          lower is better   [0.0 = perfect, 1.0 = worst]
  HRR = H_rejected / H          higher is better  [1.0 = perfect, 0.0 = worst]
  HRR = 1 - UAR  (they are complementary)

  GRR = G_rejected / G          lower is better   [0.0 = perfect, 1.0 = worst]
  GRR captures the opposite failure mode from UAR: a system that is overly
  aggressive will have low UAR (rejects fakes) but high GRR (also rejects golds).
"""

from __future__ import annotations

from reqlens_eval.models.artifacts import PoisonedTrack1Artifact
from reqlens_eval.models.experiment import (
    CandidateDecision,
    JudgeVerdict,
    Track1UnitResult,
)


def compute_track1_unit_result(
    artifact: PoisonedTrack1Artifact,
    resolved_decisions: list[CandidateDecision],
    judge_verdicts: list[JudgeVerdict],
    system_id: str,
) -> Track1UnitResult:
    """Compute UAR + HRR for a single unit × system pair.

    Args:
        artifact:           The Track 1 poisoned artifact.
        resolved_decisions: Fully resolved (no 'uncertain') candidate decisions.
        judge_verdicts:     LLM judge verdicts for any supplementary checks.
        system_id:          Identifier of the system under evaluation.

    Returns:
        Track1UnitResult with hallucination_accepted, hallucination_rejected, UAR, HRR.
    """
    fake_ids = set(artifact.seeded_fake_requirement_ids)
    gold_ids = set(artifact.gold_requirement_ids)

    # Build a lookup from candidate_id → final status
    decision_map = {d.candidate_id: d for d in resolved_decisions}

    hallucinations_accepted = 0
    hallucinations_rejected = 0

    for fake_id in fake_ids:
        decision = decision_map.get(fake_id)
        if decision is None:
            # Missing decision — treat conservatively as accepted (penalises system)
            hallucinations_accepted += 1
        elif decision.status == "accepted":
            hallucinations_accepted += 1
        else:
            hallucinations_rejected += 1

    # GRR: count gold requirements that were incorrectly rejected
    golds_rejected = 0
    golds_accepted = 0
    for gold_id in gold_ids:
        decision = decision_map.get(gold_id)
        if decision is None:
            # Missing decision for a gold — treat conservatively as rejected
            golds_rejected += 1
        elif decision.status == "rejected":
            golds_rejected += 1
        else:
            golds_accepted += 1

    total_fakes = len(fake_ids)
    total_golds = len(gold_ids)
    uar = hallucinations_accepted / total_fakes if total_fakes > 0 else 0.0
    hrr = hallucinations_rejected / total_fakes if total_fakes > 0 else 0.0
    grr = golds_rejected / total_golds if total_golds > 0 else 0.0

    return Track1UnitResult(
        unit_id=artifact.unit_id,
        artifact_id=artifact.artifact_id,
        system_id=system_id,
        total_candidates=len(artifact.candidate_pool),
        gold_count=len(gold_ids),
        fake_count=total_fakes,
        final_decisions=resolved_decisions,
        judge_verdicts=judge_verdicts,
        hallucinations_accepted=hallucinations_accepted,
        hallucinations_rejected=hallucinations_rejected,
        golds_rejected=golds_rejected,
        golds_accepted=golds_accepted,
        uar=round(uar, 4),
        hrr=round(hrr, 4),
        grr=round(grr, 4),
    )


def aggregate_track1(unit_results: list[Track1UnitResult]) -> dict[str, float]:
    """Compute mean UAR, HRR, and GRR across all evaluated units.

    Returns:
        {"mean_uar": float, "mean_hrr": float, "mean_grr": float, "unit_count": int}
    """
    if not unit_results:
        return {"mean_uar": 0.0, "mean_hrr": 0.0, "mean_grr": 0.0, "unit_count": 0}

    mean_uar = sum(r.uar for r in unit_results) / len(unit_results)
    mean_hrr = sum(r.hrr for r in unit_results) / len(unit_results)
    mean_grr = sum(r.grr for r in unit_results) / len(unit_results)
    return {
        "mean_uar": round(mean_uar, 4),
        "mean_hrr": round(mean_hrr, 4),
        "mean_grr": round(mean_grr, 4),
        "unit_count": len(unit_results),
    }
