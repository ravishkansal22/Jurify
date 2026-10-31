"""API + LLM schemas for Legal Document Intelligence (Me 2 contract)."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

ResponseLanguage = Literal["en", "hi"]
RiskLevel = Literal["low", "medium", "high"]
ConfidenceLevel = Literal["low", "medium", "high"]

NOT_LEGAL_ADVICE_DISCLAIMER = (
    "This is not legal advice. Jurify explains documents for informational "
    "purposes only and does not create an attorney-client relationship. "
    "Consult a qualified lawyer for advice about your situation."
)


class KeyClause(BaseModel):
    title: str
    quote: str
    explanation: str


class RiskFlag(BaseModel):
    clause_title: str
    quote: str
    risk_level: RiskLevel
    reason: str


class AnalyzeLLMResult(BaseModel):
    """Structured payload Gemma fills for /analyze (server adds id/model/disclaimer)."""

    document_type: str
    plain_summary: str
    key_clauses: list[KeyClause] = Field(default_factory=list)
    risk_flags: list[RiskFlag] = Field(default_factory=list)
    questions_to_ask: list[str] = Field(default_factory=list)


class AskLLMResult(BaseModel):
    """Structured payload Gemma fills for /ask (server adds disclaimer)."""

    answer: str
    evidence_quotes: list[str] = Field(default_factory=list)
    confidence: ConfidenceLevel


class AnalyzeResponse(BaseModel):
    analysis_id: str
    document_type: str
    plain_summary: str
    key_clauses: list[KeyClause]
    risk_flags: list[RiskFlag]
    questions_to_ask: list[str]
    language: ResponseLanguage
    model: str
    disclaimer: str = NOT_LEGAL_ADVICE_DISCLAIMER


class AskRequest(BaseModel):
    analysis_id: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1, max_length=2000)
    language: ResponseLanguage = "en"


class AskResponse(BaseModel):
    answer: str
    evidence_quotes: list[str]
    confidence: ConfidenceLevel
    disclaimer: str = NOT_LEGAL_ADVICE_DISCLAIMER
