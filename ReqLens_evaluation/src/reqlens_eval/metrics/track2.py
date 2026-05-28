"""Track 2 metrics: Defect Leakage Rate (DLR).

Definition
----------
Given:
  - D = total number of seeded defects (contradictions + duplicates)
  - D_leaked   = defects that appeared in the system's extracted requirements
  - D_detected = defects that were resolved / removed during extraction

  DLR = D_leaked / D          lower is better   [0.0 = perfect, 1.0 = worst]

A defect is "leaked" when the LLM judge gives verdict = 'leaked'.
A defect is "detected" when verdict = 'detected'.
"""

from __future__ import annotations

from reqlens_eval.models.artifacts import PoisonedTrack2Artifact
from reqlens_eval.models.experiment import JudgeVerdict, Track2UnitResult


def compute_track2_unit_result(
    artifact: PoisonedTrack2Artifact,
    judge_verdicts: list[JudgeVerdict],
    system_id: str,
) -> Track2UnitResult:
    """Compute DLR for a single unit × system pair.

    Args:
        artifact:       The Track 2 poisoned artifact.
        judge_verdicts: One verdict per seeded defect (from JudgeRouter).
        system_id:      Identifier of the system under evaluation.

    Returns:
        Track2UnitResult with seeds_leaked, seeds_detected, defect_leakage_rate.
    """
    total_seeds = len(artifact.seed_registry)

    seeds_leaked = sum(1 for v in judge_verdicts if v.verdict == "leaked")
    seeds_detected = sum(1 for v in judge_verdicts if v.verdict == "detected")
    # Uncertain verdicts (e.g. judge error) count as leaked — conservative
    unresolved = total_seeds - seeds_leaked - seeds_detected
    seeds_leaked += unresolved

    dlr = seeds_leaked / total_seeds if total_seeds > 0 else 0.0

    return Track2UnitResult(
        unit_id=artifact.unit_id,
        artifact_id=artifact.artifact_id,
        system_id=system_id,
        total_seeds=total_seeds,
        seeds_leaked=seeds_leaked,
        seeds_detected=seeds_detected,
        defect_leakage_rate=round(dlr, 4),
        judge_verdicts=judge_verdicts,
    )


def aggregate_track2(unit_results: list[Track2UnitResult]) -> dict[str, float]:
    """Compute mean DLR across all evaluated units.

    Returns:
        {"mean_dlr": float, "unit_count": int}
    """
    if not unit_results:
        return {"mean_dlr": 0.0, "unit_count": 0}

    mean_dlr = sum(r.defect_leakage_rate for r in unit_results) / len(unit_results)
    return {
        "mean_dlr": round(mean_dlr, 4),
        "unit_count": len(unit_results),
    }
