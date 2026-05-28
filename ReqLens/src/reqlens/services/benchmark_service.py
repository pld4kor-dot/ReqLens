"""Benchmark service – run evaluation benchmarks."""

from __future__ import annotations

import structlog

from reqlens.domain.ids import generate_id
from reqlens.domain.models import BenchmarkRun
from reqlens.storage.repositories import BenchmarkRunRepository

logger = structlog.get_logger(__name__)


class BenchmarkService:
    def __init__(self, repo: BenchmarkRunRepository) -> None:
        self.repo = repo

    def record_run(
        self,
        benchmark_type: str,
        dataset: str,
        metrics: dict,
        config: dict | None = None,
    ) -> BenchmarkRun:
        run = BenchmarkRun(
            benchmark_type=benchmark_type,
            dataset=dataset,
            metrics=metrics,
            config=config or {},
        )
        return self.repo.create(run)

    def get_run(self, run_id: str) -> BenchmarkRun | None:
        return self.repo.get(run_id)
