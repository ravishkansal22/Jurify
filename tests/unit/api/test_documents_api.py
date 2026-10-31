"""API contract tests — GemmaService is always faked via dependency_overrides."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi.testclient import TestClient

from app.api.v1.store import get_analysis_store
from app.core.config import LLMSettings, Settings, get_settings
from app.llm.gemma_service import GemmaError, GemmaService, get_gemma_service
from app.main import app
from app.schemas.document_analysis import (
    NOT_LEGAL_ADVICE_DISCLAIMER,
    AnalyzeLLMResult,
    AskLLMResult,
    KeyClause,
    RiskFlag,
)


@pytest.fixture(autouse=True)
def _reset() -> None:
    get_settings.cache_clear()
    get_gemma_service.cache_clear()
    get_analysis_store().clear()
    app.dependency_overrides.clear()
    yield
    get_settings.cache_clear()
    get_gemma_service.cache_clear()
    get_analysis_store().clear()
    app.dependency_overrides.clear()


def _fake_gemma(**kwargs: object) -> MagicMock:
    gemma = MagicMock(spec=GemmaService)
    for key, value in kwargs.items():
        setattr(gemma, key, value)
    return gemma


def test_health_returns_ok_and_model(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("GEMMA_MODEL_NAME", "gemma-4-26b-a4b-it")
    get_settings.cache_clear()
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
    gemma = _fake_gemma(generate_structured=AsyncMock())
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post("/api/v1/documents/analyze", data={"language": "en"})
    assert resp.status_code == 422
    assert "text or file" in resp.json()["detail"]
    gemma.generate_structured.assert_not_called()


def test_analyze_text_success() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            return_value=AnalyzeLLMResult(
                document_type="rental_agreement",
                plain_summary="A simple lease.",
                key_clauses=[
                    KeyClause(
                        title="Rent",
                        quote="Pay 10k monthly",
                        explanation="Monthly rent",
                    )
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
            )
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    text = "Pay 10k monthly. Deposit is non-refundable."
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": text, "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_type"] == "rental_agreement"
    assert body["risk_flags"][0]["quote"] == "Deposit is non-refundable"
    assert body["disclaimer"] == NOT_LEGAL_ADVICE_DISCLAIMER
    assert body["language"] == "en"
    assert body["analysis_id"]
    gemma.generate_structured.assert_awaited_once()
    call_kwargs = gemma.generate_structured.await_args.kwargs
    # Default ANALYZE_THINKING is OFF after live latency measurement.
    assert call_kwargs["thinking"] is False
    assert call_kwargs["image"] is None


def test_analyze_with_image_success() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            return_value=AnalyzeLLMResult(
                document_type="notice",
                plain_summary="A notice letter.",
                key_clauses=[],
                risk_flags=[],
                questions_to_ask=["When is the deadline?", "Who sent this?", "What happens next?"],
            )
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"language": "en"},
        files={"file": ("scan.png", b"\x89PNG\r\n\x1a\nfakedata", "image/png")},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["document_type"] == "notice"
    assert body["analysis_id"]
    call_kwargs = gemma.generate_structured.await_args.kwargs
    assert call_kwargs["thinking"] is False
    assert call_kwargs["image"] is not None
    assert call_kwargs["image"][1] == "image/png"


def test_analyze_file_too_large() -> None:
    gemma = _fake_gemma(generate_structured=AsyncMock())
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    huge = b"x" * (5 * 1024 * 1024 + 1)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"language": "en"},
        files={"file": ("scan.png", huge, "image/png")},
    )
    assert resp.status_code == 413
    assert "5 MB" in resp.json()["detail"]
    gemma.generate_structured.assert_not_called()


def test_analyze_bad_mime_422() -> None:
    gemma = _fake_gemma(generate_structured=AsyncMock())
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"language": "en"},
        files={"file": ("doc.pdf", b"%PDF-1.4", "application/pdf")},
    )
    assert resp.status_code == 422
    assert "png" in resp.json()["detail"].lower()


def test_analyze_gemma_failure_returns_502() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(side_effect=GemmaError("upstream down"))
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "some lease text", "language": "en"},
    )
    assert resp.status_code == 502
    assert resp.json() == {"detail": "upstream down"}


def test_ask_unknown_analysis_404() -> None:
    gemma = _fake_gemma(generate_structured=AsyncMock())
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": "missing", "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 404


def test_ask_success() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            side_effect=[
                AnalyzeLLMResult(
                    document_type="rental_agreement",
                    plain_summary="Lease.",
                    key_clauses=[
                        KeyClause(title="Rent", quote="Pay 10k", explanation="Rent")
                    ],
                    risk_flags=[],
                    questions_to_ask=["Q1", "Q2", "Q3"],
                ),
                AskLLMResult(
                    answer="Rent is 10k.",
                    evidence_quotes=["Pay 10k"],
                    confidence="high",
                ),
            ]
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    analyze = client.post(
        "/api/v1/documents/analyze",
        data={"text": "Pay 10k. Deposit refundable.", "language": "en"},
    )
    aid = analyze.json()["analysis_id"]
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": aid, "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["answer"] == "Rent is 10k."
    assert body["evidence_quotes"] == ["Pay 10k"]
    assert body["disclaimer"] == NOT_LEGAL_ADVICE_DISCLAIMER
    ask_kwargs = gemma.generate_structured.await_args_list[1].kwargs
    assert ask_kwargs["thinking"] is False


def test_analyze_id_stored_equals_returned_and_ask_accepts_it() -> None:
    """Regression: returned analysis_id must be stored (never 'pending') and work for /ask."""
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            side_effect=[
                AnalyzeLLMResult(
                    document_type="rental_agreement",
                    plain_summary="Lease.",
                    key_clauses=[],
                    risk_flags=[],
                    questions_to_ask=["Q1", "Q2", "Q3"],
                ),
                AskLLMResult(
                    answer="Deposit is refundable.",
                    evidence_quotes=["Deposit refundable"],
                    confidence="medium",
                ),
            ]
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    analyze = client.post(
        "/api/v1/documents/analyze",
        data={"text": "Pay 10k. Deposit refundable.", "language": "en"},
    )
    assert analyze.status_code == 200
    aid = analyze.json()["analysis_id"]
    assert aid
    assert aid != "pending"
    stored = get_analysis_store().get(aid)
    assert stored is not None
    assert stored.analysis.analysis_id == aid
    ask = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": aid, "question": "Is the deposit refundable?", "language": "en"},
    )
    assert ask.status_code == 200


def test_analyze_thinking_env_override(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("ANALYZE_THINKING", "1")
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            return_value=AnalyzeLLMResult(
                document_type="rental_agreement",
                plain_summary="Lease.",
                key_clauses=[],
                risk_flags=[],
                questions_to_ask=["Q1", "Q2", "Q3"],
            )
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "Pay 10k monthly.", "language": "en"},
    )
    assert resp.status_code == 200
    assert gemma.generate_structured.await_args.kwargs["thinking"] is True


def test_openapi_lists_exactly_contract_endpoints() -> None:
    client = TestClient(app)
    schema = client.get("/openapi.json").json()
    paths = schema["paths"]
    assert set(paths.keys()) == {
        "/health",
        "/api/v1/documents/analyze",
        "/api/v1/documents/ask",
    }
    assert "get" in paths["/health"]
    assert "post" in paths["/api/v1/documents/analyze"]
    assert "post" in paths["/api/v1/documents/ask"]


def test_ask_gemma_failure_502() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            side_effect=[
                AnalyzeLLMResult(
                    document_type="rental_agreement",
                    plain_summary="Lease.",
                    key_clauses=[],
                    risk_flags=[],
                    questions_to_ask=["Q1", "Q2", "Q3"],
                ),
                GemmaError("boom"),
            ]
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    aid = client.post(
        "/api/v1/documents/analyze",
        data={"text": "some text here", "language": "en"},
    ).json()["analysis_id"]
    resp = client.post(
        "/api/v1/documents/ask",
        json={"analysis_id": aid, "question": "What is rent?", "language": "en"},
    )
    assert resp.status_code == 502
    assert resp.json()["detail"] == "boom"


def test_analyze_rejects_oversized_text() -> None:
    gemma = _fake_gemma(generate_structured=AsyncMock())
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "a" * 20_001, "language": "en"},
    )
    assert resp.status_code == 422


def test_hallucinated_quote_is_dropped() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            return_value=AnalyzeLLMResult(
                document_type="rental_agreement",
                plain_summary="Lease.",
                key_clauses=[],
                risk_flags=[
                    RiskFlag(
                        clause_title="Fake",
                        quote="Tenant shall forfeit kidney",
                        risk_level="high",
                        reason="Invented",
                    ),
                    RiskFlag(
                        clause_title="Real",
                        quote="rent is due on the 1st",
                        risk_level="medium",
                        reason="Timing",
                    ),
                ],
                questions_to_ask=["Q1", "Q2", "Q3"],
            )
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={
            "text": "The rent is due on the 1st of each month.",
            "language": "en",
        },
    )
    assert resp.status_code == 200
    flags = resp.json()["risk_flags"]
    assert len(flags) == 1
    assert flags[0]["quote"] == "rent is due on the 1st"


def test_hindi_language_passthrough() -> None:
    gemma = _fake_gemma(
        generate_structured=AsyncMock(
            return_value=AnalyzeLLMResult(
                document_type="rental_agreement",
                plain_summary="यह एक किराये का समझौता है।",
                key_clauses=[],
                risk_flags=[],
                questions_to_ask=["प्रश्न 1", "प्रश्न 2", "प्रश्न 3"],
            )
        )
    )
    app.dependency_overrides[get_gemma_service] = lambda: gemma
    client = TestClient(app)
    resp = client.post(
        "/api/v1/documents/analyze",
        data={"text": "किराया 10000 रुपये मासिक है।", "language": "hi"},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["language"] == "hi"
    assert "किराये" in body["plain_summary"]
    prompt = gemma.generate_structured.await_args.args[0]
    assert "Devanagari" in prompt or "Hindi" in prompt
