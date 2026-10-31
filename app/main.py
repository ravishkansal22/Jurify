"""Jurify FastAPI entrypoint (Me 1).

Boots with CORS, request-ID middleware, GET /health, optional Me 2 API router,
and optional Friend 1 static UI at /ui.
"""

from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from typing import AsyncIterator

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.core.config import get_settings
from app.core.middleware import RequestLoggingMiddleware

settings = get_settings()


@asynccontextmanager
async def lifespan(_app: FastAPI) -> AsyncIterator[None]:
    # Touch settings early so misconfiguration fails at startup, not first request.
    get_settings()
    yield
    get_settings.cache_clear()
    try:
        from app.llm.gemma_service import get_gemma_service

        get_gemma_service.cache_clear()
    except Exception:  # noqa: BLE001 — shutdown must not raise
        pass


app = FastAPI(
    title="Jurify — Legal Document Intelligence",
    description=(
        "Explain legal documents with Gemma 4 (plain English/Hindi, key clauses, "
        "risk flags). Demo analyses are stored in-memory only (no Postgres)."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
app.add_middleware(RequestLoggingMiddleware)

# Me 2 owns API routes. Prefer app.api.v1.router; fall back to the current
# documents router so the app still boots / existing tests stay green.
try:
    from app.api.v1.router import router as v1_router

    app.include_router(v1_router)
except Exception:
    try:
        from app.api.routes.documents import router as documents_router

        app.include_router(documents_router)
    except Exception:
        pass


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "model": settings.llm.gemma_model_name}


_frontend_dir = Path("frontend")
if _frontend_dir.is_dir():
    app.mount("/ui", StaticFiles(directory=str(_frontend_dir), html=True), name="ui")
