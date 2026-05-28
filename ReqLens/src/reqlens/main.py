"""FastAPI application entry point."""

from __future__ import annotations

from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI
import structlog

from reqlens.config.logging import setup_logging
from reqlens.config.settings import get_settings


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Startup / shutdown lifecycle hook."""
    settings = get_settings()
    setup_logging(settings.log_level)
    log = structlog.get_logger("reqlens.startup")

    from reqlens.storage.db import create_all_tables
    try:
        create_all_tables()
        log.info("database_tables_ready")
    except Exception as exc:
        log.error("database_table_creation_failed", error=str(exc))
        raise

    yield


def create_app() -> FastAPI:
    """Application factory."""
    app = FastAPI(
        title="ReqLens",
        version="2.0.0a1",
        description="Evidence-grounded, graph-aware, multi-agent RE tool",
        lifespan=lifespan,
    )

    # -- Register routers --------------------------------------------
    from reqlens.api.routes_agents import router as agents_router          # NEW
    from reqlens.api.routes_benchmarks import router as benchmarks_router
    from reqlens.api.routes_documents import router as documents_router
    from reqlens.api.routes_export import router as export_router
    from reqlens.api.routes_graph import router as graph_router
    from reqlens.api.routes_projects import router as projects_router
    from reqlens.api.routes_requirements import router as requirements_router
    from reqlens.api.routes_review import router as review_router
    from reqlens.api.routes_impact import router as impact_router
    from reqlens.api.routes_versions import router as versions_router

    app.include_router(agents_router, tags=["agents"])                       # NEW
    app.include_router(projects_router, prefix="/projects", tags=["projects"])
    app.include_router(documents_router, tags=["documents"])
    app.include_router(requirements_router, tags=["requirements"])
    app.include_router(graph_router, tags=["graph"])
    app.include_router(review_router, tags=["review"])
    app.include_router(export_router, tags=["export"])
    app.include_router(benchmarks_router, prefix="/benchmarks", tags=["benchmarks"])
    app.include_router(impact_router, tags=["impact"])
    app.include_router(versions_router, tags=["versions"])

    @app.get("/health")
    async def health() -> dict[str, str]:
        return {"status": "ok", "version": "2.0.0a1"}

    return app


app = create_app()


def cli() -> None:
    """CLI entry-point (``reqlens`` command)."""
    import uvicorn

    settings = get_settings()
    uvicorn.run(
        "reqlens.main:app",
        host="0.0.0.0",
        port=8001,
        reload=settings.is_dev,
    )


if __name__ == "__main__":
    cli()
