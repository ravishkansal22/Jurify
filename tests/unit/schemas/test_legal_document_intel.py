from __future__ import annotations

from app.schemas.legal_document_intel import (
    NOT_LEGAL_ADVICE_DISCLAIMER,
    DocumentAnalyzeResponse,
    DocumentAskRequest,
    DocumentAskResponse,
    HealthResponse,
    KeyClause,
    RiskFlag,
)


def test_analyze_response_includes_disclaimer_by_default() -> None:
    resp = DocumentAnalyzeResponse(
        analysis_id="a1",
        document_type="rental_agreement",
        plain_summary="A simple lease.",
        key_clauses=[
            KeyClause(title="Rent", quote="Rent is due on the 1st.", explanation="Pay monthly.")
        ],
        risk_flags=[
            RiskFlag(
                clause_title="Lock-in",
                quote="No exit for 24 months.",
                risk_level="high",
                reason="Long lock-in with no exit.",
            )
        ],
        questions_to_ask=["Can I terminate early?"],
        language="en",
        model="gemma-4-26b-a4b-it",
    )
    assert resp.disclaimer == NOT_LEGAL_ADVICE_DISCLAIMER
    assert "not legal advice" in resp.disclaimer.lower()


def test_ask_request_and_response_shapes() -> None:
    req = DocumentAskRequest(analysis_id="a1", question="What is the rent?", language="hi")
    assert req.language == "hi"
    resp = DocumentAskResponse(
        answer="किराया 1 तारीख को देय है।",
        evidence_quotes=["Rent is due on the 1st."],
        confidence="high",
    )
    assert resp.disclaimer == NOT_LEGAL_ADVICE_DISCLAIMER


def test_health_response() -> None:
    assert HealthResponse(model="gemma-4-26b-a4b-it").model_dump() == {
        "status": "ok",
        "model": "gemma-4-26b-a4b-it",
    }
