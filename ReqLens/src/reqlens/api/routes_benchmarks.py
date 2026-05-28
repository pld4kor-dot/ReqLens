"""API routes – benchmarks."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel
from sqlalchemy.orm import Session

from reqlens.storage.db import get_db_session
from reqlens.storage.repositories import BenchmarkRunRepository
from reqlens.services.benchmark_service import BenchmarkService

router = APIRouter()


class BenchmarkRunRequest(BaseModel):
    benchmark_type: str
    dataset: str = ""
    config: dict = {}


class BenchmarkRunResponse(BaseModel):
    id: str
    benchmark_type: str
    dataset: str
    metrics: dict
    config: dict
    created_at: str


@router.post("/promise/run", response_model=BenchmarkRunResponse)
async def run_promise_benchmark(
    body: BenchmarkRunRequest,
    session: Session = Depends(get_db_session),
) -> BenchmarkRunResponse:
    service = BenchmarkService(BenchmarkRunRepository(session))
    run = service.record_run(
        benchmark_type="promise_classification",
        dataset=body.dataset or "PROMISE",
        metrics={"status": "pending"},
        config=body.config,
    )
    return BenchmarkRunResponse(
        id=run.id, benchmark_type=run.benchmark_type,
        dataset=run.dataset, metrics=run.metrics, config=run.config,
        created_at=run.created_at.isoformat(),
    )


@router.post("/reqfromsrs/run", response_model=BenchmarkRunResponse)
async def run_reqfromsrs_benchmark(
    body: BenchmarkRunRequest,
    session: Session = Depends(get_db_session),
) -> BenchmarkRunResponse:
    service = BenchmarkService(BenchmarkRunRepository(session))
    run = service.record_run(
        benchmark_type="reqfromsrs_classification",
        dataset=body.dataset or "ReqFromSRS",
        metrics={"status": "pending"},
        config=body.config,
    )
    return BenchmarkRunResponse(
        id=run.id, benchmark_type=run.benchmark_type,
        dataset=run.dataset, metrics=run.metrics, config=run.config,
        created_at=run.created_at.isoformat(),
    )


@router.get("/{benchmark_run_id}", response_model=BenchmarkRunResponse)
async def get_benchmark_run(
    benchmark_run_id: str,
    session: Session = Depends(get_db_session),
) -> BenchmarkRunResponse:
    service = BenchmarkService(BenchmarkRunRepository(session))
    run = service.get_run(benchmark_run_id)
    if run is None:
        raise HTTPException(status_code=404, detail="Benchmark run not found")
    return BenchmarkRunResponse(
        id=run.id, benchmark_type=run.benchmark_type,
        dataset=run.dataset, metrics=run.metrics, config=run.config,
        created_at=run.created_at.isoformat(),
    )
