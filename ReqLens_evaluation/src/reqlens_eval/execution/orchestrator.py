"""Experiment orchestrator — runs the full evaluation pipeline end-to-end.

Flow per system × track:

Track 1 (Trustworthiness):
  1. Load Track 1 poisoned artifacts from disk.
  2. For each artifact:
     a. Call system.evaluate_candidates(artifact) → Track1SystemOutput
     b. Normalize uncertain decisions (LLM judge family A) → resolved decisions
     c. Compute UAR + HRR → Track1UnitResult
  3. Aggregate across units → SystemEvalSummary

Track 2 (Defect Detection):
  1. Load Track 2 poisoned artifacts from disk.
  2. For each artifact:
     a. Call system.extract_requirements(artifact) → Track2SystemOutput
     b. Build extraction text (normalised plain-text block)
     c. Run LLM judge for each seed (family C / D) → JudgeVerdict list
     d. Compute DLR → Track2UnitResult
  3. Aggregate across units → SystemEvalSummary
"""

from __future__ import annotations

import uuid
from pathlib import Path

import structlog

from reqlens_eval.adapters.base import SystemAdapter
from reqlens_eval.adapters.factory import AVAILABLE_SYSTEMS, get_adapter
from reqlens_eval.benchmark.loader import load_track1_artifacts, load_track2_artifacts
from reqlens_eval.config import get_settings
from reqlens_eval.judging.router import JudgeRouter
from reqlens_eval.metrics.track1 import aggregate_track1, compute_track1_unit_result
from reqlens_eval.metrics.track2 import aggregate_track2, compute_track2_unit_result
from reqlens_eval.models.experiment import (
    EvalRunReport,
    SystemEvalSummary,
    Track1UnitResult,
    Track2UnitResult,
)
from reqlens_eval.normalization.track1 import normalize_track1
from reqlens_eval.normalization.track2 import build_extraction_text

logger = structlog.get_logger(__name__)


class ExperimentOrchestrator:
    """Coordinates all evaluation tasks across systems and tracks."""

    def __init__(
        self,
        benchmark_output_dir: Path | None = None,
        judge_router: JudgeRouter | None = None,
    ) -> None:
        settings = get_settings()
        self._benchmark_dir = benchmark_output_dir or settings.benchmark_path
        self._judge = judge_router or JudgeRouter()

    # ── Public entry point ────────────────────────────────────────────────────

    def run(
        self,
        systems: list[str] | None = None,
        tracks: list[str] | None = None,
        unit_id: str | None = None,
        variant_id: str | None = None,
    ) -> EvalRunReport:
        """Run the full evaluation and return a complete EvalRunReport.

        Args:
            systems:    System IDs to evaluate (default: all registered).
                        Choices: 'baseline', 'reqinone_v1', 'reqlens_v2'.
            tracks:     Tracks to evaluate: ['1'], ['2'], or ['1', '2'] (default).
            unit_id:    If set, evaluate only this unit (e.g. 'PROMISE_1').
            variant_id: If set, use only this variant of poisoned artifact.
        """
        run_id = str(uuid.uuid4())[:8]
        # systems = systems or AVAILABLE_SYSTEMS
        systems = systems or ["reqinone_v1", "reqlens_v2"]
        tracks = tracks or ["1", "2"]

        logger.info(
            "orchestrator.run_start",
            run_id=run_id,
            systems=systems,
            tracks=tracks,
            unit_id=unit_id,
        )

        track1_summaries: list[SystemEvalSummary] = []
        track2_summaries: list[SystemEvalSummary] = []

        for system_id in systems:
            try:
                adapter = get_adapter(system_id)
            except ValueError as exc:
                logger.error("orchestrator.unknown_system", system_id=system_id, error=str(exc))
                continue

            if "1" in tracks:
                summary = self._run_track1(adapter, unit_id=unit_id, variant_id=variant_id)
                track1_summaries.append(summary)

            if "2" in tracks:
                summary = self._run_track2(adapter, unit_id=unit_id, variant_id=variant_id)
                track2_summaries.append(summary)

        report = EvalRunReport(
            run_id=run_id,
            systems_evaluated=systems,
            tracks_evaluated=tracks,
            track1_summaries=track1_summaries,
            track2_summaries=track2_summaries,
            metadata={
                "benchmark_dir": str(self._benchmark_dir),
                "unit_filter": unit_id,
                "variant_filter": variant_id,
            },
        )
        logger.info("orchestrator.run_done", run_id=run_id)
        return report

    # ── Track 1 ──────────────────────────────────────────────────────────────

    def _run_track1(
        self,
        adapter: SystemAdapter,
        unit_id: str | None,
        variant_id: str | None,
    ) -> SystemEvalSummary:
        artifacts = load_track1_artifacts(
            self._benchmark_dir,
            unit_id=unit_id,
            variant_id=variant_id,
        )
        if not artifacts:
            logger.warning(
                "orchestrator.track1_no_artifacts",
                system_id=adapter.system_id,
                benchmark_dir=str(self._benchmark_dir),
            )
            return SystemEvalSummary(
                system_id=adapter.system_id,
                track_id="track1",
                unit_count=0,
            )

        unit_results: list[Track1UnitResult] = []

        for artifact in artifacts:
            logger.info(
                "orchestrator.track1_artifact",
                system_id=adapter.system_id,
                unit_id=artifact.unit_id,
            )
            # Build a quick lookup map: candidate_id → text (for judge prompts)
            candidate_text_map = {
                item.id: item.text for item in artifact.candidate_pool
            }

            # Step a: system evaluates candidates
            system_output = adapter.evaluate_candidates(artifact)

            # Step b: resolve any 'uncertain' decisions via judge (family A)
            # Per-candidate evidence overrides the global source_texts when the
            # adapter supplied it (reqlens_v2 → retrieved spans; reqinone_v1 →
            # extracted requirements list).  Falls back to full source_texts for
            # candidates with no entry or when the map is absent.
            candidate_evidence_map: dict | None = (
                system_output.metadata.get("candidate_evidence")
                if system_output.metadata
                else None
            )
            # OLD: resolved = normalize_track1(
            # NEW: also unpack escalated_ids returned by normalize_track1
            resolved, escalated_ids = normalize_track1(
                system_output,
                source_texts=artifact.source_texts,
                candidate_text_map=candidate_text_map,
                judge=self._judge,
                candidate_evidence_map=candidate_evidence_map,
            )

            # Step c: compute metrics
            result = compute_track1_unit_result(
                artifact=artifact,
                resolved_decisions=resolved,
                judge_verdicts=[],  # no supplementary judge verdicts at unit level
                system_id=adapter.system_id,
            )
            unit_results.append(result)
            logger.info(
                "orchestrator.track1_unit_done",
                system_id=adapter.system_id,
                unit_id=artifact.unit_id,
                uar=result.uar,
                hrr=result.hrr,
                grr=result.grr,
                golds_rejected=result.golds_rejected,
                golds_accepted=result.golds_accepted,
                # NEW: add escalation summary per unit
                escalated_to_judge=len(escalated_ids),
                escalated_ids=escalated_ids,
            )

        agg = aggregate_track1(unit_results)
        # NEW: collect all IDs that were escalated across every unit for the end-of-run summary
        all_escalated = [
            d["candidate_id"]
            for r in unit_results
            for d in r.model_dump()["final_decisions"]
            if d["signal_source"] == "llm_judge"
        ]
        logger.info(
            "orchestrator.track1_done",
            system_id=adapter.system_id,
            units=agg["unit_count"],
            mean_uar=agg["mean_uar"],
            mean_hrr=agg["mean_hrr"],
            mean_grr=agg["mean_grr"],
            total_escalated_to_judge=len(all_escalated),
            escalated_ids=all_escalated,
        )
        return SystemEvalSummary(
            system_id=adapter.system_id,
            track_id="track1",
            unit_count=agg["unit_count"],
            mean_uar=agg["mean_uar"],
            mean_hrr=agg["mean_hrr"],
            mean_grr=agg["mean_grr"],
            unit_results=[r.model_dump() for r in unit_results],
        )

    # ── Track 2 ──────────────────────────────────────────────────────────────

    def _run_track2(
        self,
        adapter: SystemAdapter,
        unit_id: str | None,
        variant_id: str | None,
    ) -> SystemEvalSummary:
        artifacts = load_track2_artifacts(
            self._benchmark_dir,
            unit_id=unit_id,
            variant_id=variant_id,
        )
        if not artifacts:
            logger.warning(
                "orchestrator.track2_no_artifacts",
                system_id=adapter.system_id,
                benchmark_dir=str(self._benchmark_dir),
            )
            return SystemEvalSummary(
                system_id=adapter.system_id,
                track_id="track2",
                unit_count=0,
            )

        unit_results: list[Track2UnitResult] = []

        for artifact in artifacts:
            logger.info(
                "orchestrator.track2_artifact",
                system_id=adapter.system_id,
                unit_id=artifact.unit_id,
                seeds=len(artifact.seed_registry),
            )

            # Step a: system extracts requirements from poisoned source texts
            system_output = adapter.extract_requirements(artifact)

            # Step b: build plain-text extraction block for judge
            extraction_text = build_extraction_text(system_output)

            # Step c: judge each seed
            verdicts = self._judge.judge_all_seeds(
                seeds=artifact.seed_registry,
                extraction_text=extraction_text,
            )

            # Step d: compute DLR
            result = compute_track2_unit_result(
                artifact=artifact,
                judge_verdicts=verdicts,
                system_id=adapter.system_id,
            )
            unit_results.append(result)
            logger.info(
                "orchestrator.track2_unit_done",
                system_id=adapter.system_id,
                unit_id=artifact.unit_id,
                dlr=result.defect_leakage_rate,
                leaked=result.seeds_leaked,
                detected=result.seeds_detected,
            )

        agg = aggregate_track2(unit_results)
        return SystemEvalSummary(
            system_id=adapter.system_id,
            track_id="track2",
            unit_count=agg["unit_count"],
            mean_dlr=agg["mean_dlr"],
            unit_results=[r.model_dump() for r in unit_results],
        )
