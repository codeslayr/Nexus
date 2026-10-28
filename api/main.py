"""
FastAPI Application Entry Point — Section 4.

Startup/shutdown lifecycle, global exception handlers, structured logging,
and OpenTelemetry instrumentation.
"""
from __future__ import annotations

import asyncio
import sys
import time
import uuid
from contextlib import asynccontextmanager
from typing import Any

if sys.platform == "win32":
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

import structlog
from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from api.routers import graph, ingest, query
from pipelines.graph_store import GraphStore
from src.config import get_settings
from src.utils.db import close_pg_pool, get_pg_pool
from src.utils.logging import configure_logging, get_logger

# ── Bootstrap logging before anything else ────────────────────────────────────
configure_logging()
log = get_logger(__name__)


# ──────────────────────────────────────────────────────────────────────────────
# Application lifespan (startup / shutdown)
# ──────────────────────────────────────────────────────────────────────────────

@asynccontextmanager
async def lifespan(app: FastAPI):  # type: ignore[type-arg]
    settings = get_settings()
    log.info("graphrag_starting", env=settings.app_env)

    # Warm up PostgreSQL pool
    await get_pg_pool()
    log.info("postgres_pool_ready")

    # Ensure Neo4j schema (constraints + indexes)
    graph_store = GraphStore()
    await graph_store.ensure_schema()
    app.state.graph_store = graph_store
    log.info("neo4j_schema_ready")

    yield  # ← application runs here

    # Graceful shutdown
    await close_pg_pool()
    await graph_store.close()
    log.info("graphrag_shutdown_complete")


# ──────────────────────────────────────────────────────────────────────────────
# App factory
# ──────────────────────────────────────────────────────────────────────────────

def create_app() -> FastAPI:
    settings = get_settings()

    app = FastAPI(
        title="Enterprise Hybrid GraphRAG API",
        description=(
            "Production-grade Knowledge Graph + Vector RAG system. "
            "Combines Neo4j graph traversal with pgvector semantic search."
        ),
        version="1.0.0",
        docs_url="/docs" if settings.app_env != "production" else None,
        redoc_url="/redoc" if settings.app_env != "production" else None,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"] if settings.app_env == "development" else [],
        allow_credentials=True,
        allow_methods=["GET", "POST"],
        allow_headers=["*"],
    )

    # ── Request ID middleware ──────────────────────────────────────────────────
    @app.middleware("http")
    async def request_id_middleware(request: Request, call_next):
        request_id = request.headers.get("X-Request-ID", str(uuid.uuid4()))
        structlog.contextvars.bind_contextvars(request_id=request_id)
        start = time.perf_counter()
        response = await call_next(request)
        duration_ms = (time.perf_counter() - start) * 1000
        log.info(
            "http_request",
            method=request.method,
            path=request.url.path,
            status=response.status_code,
            duration_ms=round(duration_ms, 2),
        )
        response.headers["X-Request-ID"] = request_id
        structlog.contextvars.unbind_contextvars("request_id")
        return response

    # ── Global exception handler ───────────────────────────────────────────────
    @app.exception_handler(Exception)
    async def global_exception_handler(request: Request, exc: Exception) -> JSONResponse:
        log.error(
            "unhandled_exception",
            path=request.url.path,
            error=str(exc),
            exc_info=True,
        )
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content={
                "error": "Internal server error",
                "detail": str(exc) if get_settings().app_env != "production" else None,
            },
        )

    # ── Routers ───────────────────────────────────────────────────────────────
    app.include_router(ingest.router, prefix="/ingest", tags=["Ingestion"])
    app.include_router(query.router,  prefix="/query",  tags=["Query"])
    app.include_router(graph.router,  prefix="/graph",  tags=["Graph"])

    # ── Health check ───────────────────────────────────────────────────────────
    @app.get("/health", tags=["Health"])
    async def health() -> dict[str, Any]:
        return {"status": "ok", "service": "graphrag-api"}

    # ── Web UI ─────────────────────────────────────────────────────────────────
    from pathlib import Path
    from fastapi.staticfiles import StaticFiles
    from fastapi.responses import FileResponse

    web_dir = Path(__file__).parent.parent / "web"
    if web_dir.exists():
        app.mount("/static", StaticFiles(directory=str(web_dir)), name="static")

        @app.get("/", tags=["UI"])
        async def serve_ui():
            return FileResponse(web_dir / "index.html")

        @app.get("/style.css", include_in_schema=False)
        async def serve_css():
            return FileResponse(web_dir / "style.css")

        @app.get("/app.js", include_in_schema=False)
        async def serve_js():
            return FileResponse(web_dir / "app.js")

    return app


app = create_app()


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        "api.main:app",
        host="0.0.0.0",
        port=8000,
        reload=get_settings().app_env == "development",
        log_config=None,  # structlog handles logging
    )
