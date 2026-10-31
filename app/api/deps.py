"""FastAPI dependencies for the Legal Document Intelligence demo."""

from __future__ import annotations

from functools import lru_cache

from app.core.config import Settings, get_settings
from app.services.legal_document_intel_service import LegalDocumentIntelService


@lru_cache
def get_document_intel_service() -> LegalDocumentIntelService:
    settings: Settings = get_settings()
    return LegalDocumentIntelService(settings)
