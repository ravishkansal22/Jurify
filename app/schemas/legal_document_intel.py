"""API contracts for Legal Document Intelligence (Gemma 4 hackday demo).

Frozen response shapes for POST /api/v1/documents/analyze and /ask.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from app.schemas._common import NyayaBaseModel

ResponseLanguage = Literal["en", "hi"]
RiskLevel = Literal["low", "medium", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]

# Every user-facing response must include this disclaimer.
NOT_LEGAL_ADVICE_DISCLAIMER = (
    "This is not legal advice. Jurify explains documents for informational "
    "purposes only and does not create an attorney-client relationship. "
    "Consult a qualified lawyer for advice about your situation."
)


class KeyClause(NyayaBaseModel):
    title: str
    quote: str
    explanation: str


class RiskFlag(NyayaBaseModel):
    clause_title: str
    quote: str
    risk_level: RiskLevel
    reason: str


class DocumentAnalyzeResponse(NyayaBaseModel):
    analysis_id: str
    document_type: str
    plain_summary: str
    key_clauses: list[KeyClause]
    risk_flags: list[RiskFlag]
    questions_to_ask: list[str]
    language: ResponseLanguage
    model: str
    disclaimer: str = NOT_LEGAL_ADVICE_DISCLAIMER


class DocumentAskRequest(NyayaBaseModel):
    analysis_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=2000)
    language: ResponseLanguage = "en"


class DocumentAskResponse(NyayaBaseModel):
    answer: str
    evidence_quotes: list[str]
    confidence: ConfidenceLevel
    disclaimer: str = NOT_LEGAL_ADVICE_DISCLAIMER


class HealthResponse(NyayaBaseModel):
    status: Literal["ok"] = "ok"
    model: str
