"""Aggregated v1 API router (imported by Me 1's app.main)."""

from __future__ import annotations

from fastapi import APIRouter

from app.api.v1.routes.documents import router as documents_router

router = APIRouter()
router.include_router(documents_router)
