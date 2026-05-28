"""PURE pipeline.

End-to-end flow for each PURE document:

  1. Load document (PDF / DOCX / HTML / TXT).
  2. Profile structure.
  3. Detect sections heuristically.
  4. Build a global context string from context-type sections.
  5a. Section-aware extraction  — only on requirement-dense sections.
  5b. Raw-text extraction       — full document via sliding-window chunks.
  6. Merge all candidates into a deduplicated gold-requirement set.
  7. Build a scenario brief (using global context for grounding).
  8. Generate 3-artifact source bundle.
  9. Validate coverage + leakage.
  10. Repair if needed (bounded by max_repair_rounds).
  11. Save benchmark unit (JSON + TXT).
"""

from __future__ import annotations

import structlog

from reqlens_benchmark_builder.azure_client import AzureOpenAIClient
from reqlens_benchmark_builder.config import get_settings
from reqlens_benchmark_builder.generation.scenario_brief_builder import build_scenario_brief
from reqlens_benchmark_builder.generation.source_bundle_generator import generate_source_bundle
from reqlens_benchmark_builder.loaders.pure_loader import load_pure_documents
from reqlens_benchmark_builder.pure.document_profiler import profile_document
from reqlens_benchmark_builder.pure.hierarchical_chunker import chunk_document_raw, chunk_sections
from reqlens_benchmark_builder.pure.requirement_extractor import extract_requirements_from_chunks
from reqlens_benchmark_builder.pure.requirement_merger import merge_requirements
from reqlens_benchmark_builder.pure.section_detector import Section, detect_sections
from reqlens_benchmark_builder.schemas.benchmark_models import (
    BenchmarkUnit,
    CoverageEntry,
    ValidationSummary,
)
from reqlens_benchmark_builder.validation.coverage_validator import validate_coverage
from reqlens_benchmark_builder.validation.repair import repair_source_bundle
from reqlens_benchmark_builder.validation.unsupported_validator import validate_unsupported
from reqlens_benchmark_builder.writers.io import write_benchmark_unit

logger = structlog.get_logger(__name__)

# Section types whose text is useful for building the global context brief
_CONTEXT_SECTION_TYPES = {"context", "usecase", "other"}


def _build_global_context(sections: list[Section], max_chars: int) -> str:
    """Extract global context text from non-requirement sections.

    Prioritises context/usecase sections, then falls back to the document head.
    """
    context_parts: list[str] = []
    for s in sections:
        if s.section_type in _CONTEXT_SECTION_TYPES:
            context_parts.append(f"[{s.title}]\n{s.text}")
        if sum(len(p) for p in context_parts) >= max_chars:
            break

    if not context_parts:
        # No context sections found — use the first chunk of the full document
        for s in sections[:3]:
            context_parts.append(s.text)

    raw = "\n\n".join(context_parts)
    return raw[:max_chars]


def _build_validation_summary(
    coverage_report: dict,
    unsupported_report: dict,
    repair_rounds: int,
    settings,
) -> ValidationSummary:
    rate   = coverage_report.get("coverage_rate", 0.0)
    missed = coverage_report.get("missing_req_ids", [])
    leaks  = unsupported_report.get("count", 0)
    passed = (
        rate >= settings.min_coverage_rate
        and leaks <= settings.max_unsupported_count
    )
    entries = [
        CoverageEntry(**e)
        for e in coverage_report.get("coverage", [])
        if isinstance(e, dict)
    ]
    return ValidationSummary(
        coverage_rate=rate,
        missing_req_ids=missed,
        unsupported_count=leaks,
        repair_rounds_used=repair_rounds,
        coverage_entries=entries,
        unsupported_implied=unsupported_report.get(
            "unsupported_implied_requirements", []
        ),
        passed=passed,
    )


def run_pure_pipeline() -> list[BenchmarkUnit]:
    """Run the full PURE benchmark-building pipeline.

    Returns a list of completed ``BenchmarkUnit`` objects (one per document).
    """
    settings = get_settings()
    llm      = AzureOpenAIClient()
    units: list[BenchmarkUnit] = []

    try:
        docs = load_pure_documents()
    except FileNotFoundError as exc:
        logger.error("pure_pipeline.load_failed", error=str(exc))
        return []

    for doc in docs:
        unit_id = f"PURE_{doc.doc_id}"
        logger.info(
            "pure_pipeline.start",
            unit_id=unit_id,
            file_type=doc.file_type,
            chars=doc.char_count,
        )

        # Step 2 – profile
        profile = profile_document(doc.text)
        logger.info(
            "pure_pipeline.profile",
            unit_id=unit_id,
            structure=profile.structure_strength,
            headings=profile.heading_density,
            shall=profile.shall_count,
        )

        # Step 3 – section detection
        sections = detect_sections(doc.text)

        # Step 4 – global context
        global_context = _build_global_context(sections, settings.pure_profile_summary_chars)

        # Step 5a – section-aware extraction (requirement-dense sections only)
        dense_sections = [s for s in sections if s.is_requirement_dense]
        logger.info(
            "pure_pipeline.section_extraction",
            unit_id=unit_id,
            dense_sections=len(dense_sections),
            total_sections=len(sections),
        )
        section_chunks = chunk_sections(dense_sections) if dense_sections else []
        section_candidates = (
            extract_requirements_from_chunks(
                llm, doc.doc_id, section_chunks, strategy="section"
            )
            if section_chunks
            else []
        )

        # Step 5b – raw-text extraction (full document, filtered by signal count)
        raw_chunks = chunk_document_raw(doc.text)
        raw_candidates = extract_requirements_from_chunks(
            llm, doc.doc_id, raw_chunks, strategy="raw_chunk"
        )

        all_candidates = section_candidates + raw_candidates
        logger.info(
            "pure_pipeline.extraction_done",
            unit_id=unit_id,
            section_candidates=len(section_candidates),
            raw_candidates=len(raw_candidates),
            total=len(all_candidates),
        )

        # Step 6 – merge
        gold_reqs = merge_requirements(llm, doc.doc_id, all_candidates)
        if not gold_reqs:
            logger.warning("pure_pipeline.no_gold_requirements", unit_id=unit_id)
            continue

        gold_dicts = [r.model_dump() for r in gold_reqs]
        logger.info(
            "pure_pipeline.gold_set",
            unit_id=unit_id,
            gold_count=len(gold_dicts),
        )

        # Step 7 – scenario brief (grounded by global context)
        brief = build_scenario_brief(
            llm, unit_id, gold_dicts, global_context=global_context
        )

        # Step 8 – source bundle
        source_texts = generate_source_bundle(llm, unit_id, brief, gold_dicts)
        source_dicts = [s.model_dump() for s in source_texts]

        # Step 9 – validate
        coverage_report    = validate_coverage(llm, unit_id, source_dicts, gold_dicts)
        unsupported_report = validate_unsupported(llm, unit_id, source_dicts, gold_dicts)

        # Step 10 – repair loop
        repair_rounds = 0
        while repair_rounds < settings.max_repair_rounds:
            rate  = coverage_report.get("coverage_rate", 0.0)
            leaks = unsupported_report.get("count", 0)
            if (
                rate >= settings.min_coverage_rate
                and leaks <= settings.max_unsupported_count
            ):
                break

            logger.info(
                "pure_pipeline.repair",
                unit_id=unit_id,
                round=repair_rounds + 1,
                coverage_rate=rate,
                unsupported_count=leaks,
            )
            source_texts = repair_source_bundle(
                llm=llm,
                unit_id=unit_id,
                brief=brief,
                source_texts=source_dicts,
                gold_requirements=gold_dicts,
                coverage_report=coverage_report,
                unsupported_report=unsupported_report,
            )
            source_dicts       = [s.model_dump() for s in source_texts]
            coverage_report    = validate_coverage(llm, unit_id, source_dicts, gold_dicts)
            unsupported_report = validate_unsupported(llm, unit_id, source_dicts, gold_dicts)
            repair_rounds += 1

        # Step 11 – assemble and save
        validation = _build_validation_summary(
            coverage_report, unsupported_report, repair_rounds, settings
        )
        unit = BenchmarkUnit(
            id=unit_id,
            origin="PURE",
            source_texts=source_texts,
            gold_requirements=gold_reqs,
            validation=validation,
            brief=brief,
            metadata={
                "file_path":       str(doc.path),
                "file_type":       doc.file_type,
                "char_count":      doc.char_count,
                "profile":         profile.model_dump(),
                "section_count":   len(sections),
                "dense_sections":  len(dense_sections),
                "gold_count":      len(gold_reqs),
                "section_candidates": len(section_candidates),
                "raw_candidates":     len(raw_candidates),
                "llm_usage":          llm.usage_summary(),
            },
        )

        out_dir = settings.output_path / "pure" / unit_id
        write_benchmark_unit(out_dir, unit)
        units.append(unit)

        logger.info(
            "pure_pipeline.unit_done",
            unit_id=unit_id,
            gold_reqs=len(gold_reqs),
            coverage_rate=validation.coverage_rate,
            passed=validation.passed,
        )

    logger.info(
        "pure_pipeline.complete",
        total_units=len(units),
        passed=sum(1 for u in units if u.validation.passed),
        llm_usage=llm.usage_summary(),
    )
    return units
