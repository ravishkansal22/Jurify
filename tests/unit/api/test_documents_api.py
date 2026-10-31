"""API contract tests — LLM is always mocked."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.deps import get_document_intel_service
from app.core.config import LLMSettings, Settings, get_settings
from app.llm.gemma_client import GemmaAPIError
from app.main import app
from app.schemas.legal_document_intel import (
    NOT_LEGAL_ADVICE_DISCLAIMER,
    DocumentAnalyzeResponse,
    DocumentAskResponse,
    KeyClause,
    RiskFlag,
)
from app.services.legal_document_intel_service import LegalDocumentIntelService
from app.store.analysis_store import InMemoryAnalysisStore


@pytest.fixture(autouse=True)
def _clear_caches() -> None:
    get_settings.cache_clear()
    get_document_intel_service.cache_clear()
    yield
    get_settings.cache_clear()
    get_document_intel_service.cache_clear()
    app.dependency_overrides.clear()


def _mock_service() -> MagicMock:
    service = MagicMock(spec=LegalDocumentIntelService)
    service.model_id = "gemma-4-26b-a4b-it"
    return service


def test_health_returns_ok_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMMA_MODEL_NAME", "gemma-4-26b-a4b-it")
    get_settings.cache_clear()
    # Rebuild settings on health via get_settings already imported in main —
    # override by patching the module-level settings used at request time.
    from app import main as main_mod

    main_mod.settings = Settings(
        _env_file=None,
        llm=LLMSettings(
            _env_file=None,
            gemma_model_name="gemma-4-26b-a4b-it",
            google_api_key="<your-key>",  # type: ignore[arg-type]
        ),
    )
    client = TestClient(app)
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok", "model": "gemma-4-26b-a4b-it"}


def test_analyze_requires_text_or_file() -> None:
    service = _mock_service()
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post("/api/v1/documents/analyze", data={"language": "en"})
    assert resp.status_code == 422
    assert "text or file" in resp.json()["detail"]
    service.analyze.assert_not_called()


def test_analyze_text_success() -> None:
    service = _mock_service()
    service.analyze = AsyncMock(
        return_value=DocumentAnalyzeResponse(
            analysis_id="aid-1",
            document_type="rental_agreement",
            plain_summary="A lease.",
            key_clauses=[
                KeyClause(title="Rent", quote="Pay 10k", explanation="Monthly rent")
            ],
            risk_flags=[
                RiskFlag(
                    clause_title="Deposit",
                    quote="Deposit is non-refundable",
                    risk_level="high",
                    reason="Unusual non-refundable deposit",
                )
            ],
            questions_to_ask=["Is the deposit refundable?"],
            language="en",
            model="gemma-4-26b-a4b-it",
        )
    )
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "Pay 10k. Deposit is non-refundable", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["analysis_id"] == "aid-1"
    assert body["risk_flags"][0]["quote"] == "Deposit is non-refundable"
    assert body["disclaimer"] == NOT_LEGAL_ADVICE_DISCLAIMER
    service.analyze.assert_awaited_once()


def test_analyze_file_too_large() -> None:
    service = _mock_service()
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    huge = b"x" * (5 * 1024 * 1024 + 1)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"language": "en"},
        files={"file": ("scan.png", huge, "image/png")},
    )
    assert resp.status_code == 413
    assert "5 MB" in resp.json()["detail"]


def test_analyze_gemma_failure_returns_502() -> None:
    service = _mock_service()
    service.analyze = AsyncMock(side_effect=GemmaAPIError("upstream down"))
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "some lease text", "language": "en"},
    )
    assert resp.status_code == 502
    assert resp.json() == {"detail": "upstream down"}


def test_ask_unknown_analysis_404() -> None:
    store = InMemoryAnalysisStore()
    settings = Settings(
        _env_file=None,
        llm=LLMSettings(
            _env_file=None,
            gemma_model_name="gemma-4-26b-a4b-it",
            google_api_key="<your-key>",  # type: ignore[arg-type]
        ),
    )
    real = LegalDocumentIntelService(settings, client=MagicMock(), store=store)
    app.dependency_overrides[get_document_intel_service] = lambda: real
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": "missing", "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 404


def test_ask_success() -> None:
    service = _mock_service()
    service.ask = AsyncMock(
        return_value=DocumentAskResponse(
            answer="Rent is 10k.",
            evidence_quotes=["Pay 10k"],
            confidence="high",
        )
    )
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": "aid-1", "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Rent is 10k."
    assert body["disclaimer"] == NOT_LEGAL_ADVICE_DISCLAIMER


def test_ask_gemma_failure_502() -> None:
    service = _mock_service()
    service.ask = AsyncMock(side_effect=GemmaAPIError("boom"))
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": "aid-1", "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"] == "boom"


def test_analyze_rejects_oversized_text() -> None:
    service = _mock_service()
    app.dependency_overrides[get_document_intel_service] = lambda: service
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "a" * 20_001, "language": "en"},
    )
    assert resp.status_code == 422
