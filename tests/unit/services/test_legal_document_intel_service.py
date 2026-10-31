from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock

import pytest

from app.core.config import LLMSettings, Settings
from app.llm.gemma_client import GemmaAPIError
from app.services.legal_document_intel_service import (
    AnalysisNotFoundError,
    LegalDocumentIntelService,
)
from app.store.analysis_store import InMemoryAnalysisStore


def _settings() -> Settings:
    return Settings(
        _env_file=None,
        llm=LLMSettings(
            _env_file=None,
            llm_provider="gemini",
            gemma_model_name="gemma-4-26b-a4b-it",
            google_api_key="<your-key>",  # type: ignore[arg-type]
        ),
    )


def _fake_analyze_json() -> str:
    return json.dumps(
        {
            "document_type": "rental_agreement",
            "plain_summary": "A one-year rental agreement.",
            "key_clauses": [
                {
                    "title": "Rent",
                    "quote": "Tenant shall pay rent of INR 15000 on the 1st of each month.",
                    "explanation": "Monthly rent due on the 1st.",
                }
            ],
            "risk_flags": [
                {
                    "clause_title": "Lock-in",
                    "quote": "Tenant may not vacate for 24 months.",
                    "risk_level": "high",
                    "reason": "Long lock-in with no exit option stated.",
                }
            ],
            "questions_to_ask": ["Is early termination allowed?"],
        }
    )


@pytest.mark.asyncio
async def test_analyze_text_stores_record_and_returns_contract() -> None:
    store = InMemoryAnalysisStore()
    client = MagicMock()
    client.generate_text = AsyncMock(return_value=_fake_analyze_json())
    client.generate_with_image = AsyncMock()

    service = LegalDocumentIntelService(_settings(), client=client, store=store)
    result = await service.analyze(
        text="Tenant shall pay rent of INR 15000 on the 1st of each month. "
        "Tenant may not vacate for 24 months.",
        language="en",
    )

    assert result.document_type == "rental_agreement"
    assert result.risk_flags[0].quote == "Tenant may not vacate for 24 months."
    assert "not legal advice" in result.disclaimer.lower()
    assert store.get(result.analysis_id) is not None
    client.generate_text.assert_awaited_once()
    client.generate_with_image.assert_not_awaited()


@pytest.mark.asyncio
async def test_analyze_with_image_calls_multimodal() -> None:
    store = InMemoryAnalysisStore()
    client = MagicMock()
    client.generate_with_image = AsyncMock(return_value=_fake_analyze_json())

    service = LegalDocumentIntelService(_settings(), client=client, store=store)
    result = await service.analyze(
        text=None,
        language="hi",
        image_bytes=b"\x89PNG",
        image_mime="image/png",
    )

    assert result.language == "hi"
    client.generate_with_image.assert_awaited_once()


@pytest.mark.asyncio
async def test_ask_grounds_in_stored_document() -> None:
    store = InMemoryAnalysisStore()
    client = MagicMock()
    client.generate_text = AsyncMock(
        side_effect=[
            _fake_analyze_json(),
            json.dumps(
                {
                    "answer": "Rent is INR 15000 due on the 1st.",
                    "evidence_quotes": [
                        "Tenant shall pay rent of INR 15000 on the 1st of each month."
                    ],
                    "confidence": "high",
                }
            ),
        ]
    )

    service = LegalDocumentIntelService(_settings(), client=client, store=store)
    analyzed = await service.analyze(
        text="Tenant shall pay rent of INR 15000 on the 1st of each month.",
        language="en",
    )
    answer = await service.ask(
        analysis_id=analyzed.analysis_id,
        question="How much is rent?",
        language="en",
    )
    assert answer.confidence == "high"
    assert answer.evidence_quotes
    assert "not legal advice" in answer.disclaimer.lower()


@pytest.mark.asyncio
async def test_ask_unknown_analysis_id() -> None:
    service = LegalDocumentIntelService(
        _settings(), client=MagicMock(), store=InMemoryAnalysisStore()
    )
    with pytest.raises(AnalysisNotFoundError):
        await service.ask(analysis_id="missing", question="Hi?", language="en")


@pytest.mark.asyncio
async def test_analyze_propagates_gemma_failure() -> None:
    client = MagicMock()
    client.generate_text = AsyncMock(side_effect=GemmaAPIError("down"))
    service = LegalDocumentIntelService(
        _settings(), client=client, store=InMemoryAnalysisStore()
    )
    with pytest.raises(GemmaAPIError):
        await service.analyze(text="hello lease", language="en")
