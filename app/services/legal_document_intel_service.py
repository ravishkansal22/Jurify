"""Legal Document Intelligence service (Gemma 4 demo).

Uses Gemma4Client + an in-memory analysis store. No Postgres on this path.
Never invents laws, sections, or citations — prompts enforce grounding.
"""

from __future__ import annotations

import json
import re
import uuid
from typing import Any

from app.core.config import Settings
from app.llm.gemma_client import Gemma4Client, GemmaAPIError
from app.schemas.legal_document_intel import (
    NOT_LEGAL_ADVICE_DISCLAIMER,
    DocumentAnalyzeResponse,
    DocumentAskResponse,
    KeyClause,
    ResponseLanguage,
    RiskFlag,
)
from app.store.analysis_store import AnalysisRecord, InMemoryAnalysisStore, analysis_store

SYSTEM_ANALYZE = """You are Jurify, a Legal Document Intelligence assistant powered by Gemma 4.
Explain legal documents in plain language. Extract key clauses and flag risky clauses.
RULES (must follow):
- Never invent laws, statute sections, case citations, or legal authorities.
- Every risk_flag.quote and key_clauses.quote MUST be an exact verbatim passage from the document (or from visible text in an uploaded image).
- If something is unclear or not in the document, say so; do not guess.
- This is NOT legal advice. Do not tell the user what they "must" do legally.
- Respond with a single JSON object only (no markdown fences) matching:
{
  "document_type": string,
  "plain_summary": string,
  "key_clauses": [{"title": string, "quote": string, "explanation": string}],
  "risk_flags": [{"clause_title": string, "quote": string, "risk_level": "low"|"medium"|"high", "reason": string}],
  "questions_to_ask": [string]
}
Write plain_summary, explanations, reasons, and questions_to_ask in the requested language.
"""

SYSTEM_ASK = """You are Jurify, answering questions about a previously analyzed legal document.
RULES:
- Ground every answer ONLY in the document text provided. Quote evidence verbatim.
- Never invent laws, sections, or citations.
- If the document does not contain enough information, say so and set confidence to "low".
- This is NOT legal advice.
- Respond with a single JSON object only (no markdown fences):
{
  "answer": string,
  "evidence_quotes": [string],
  "confidence": "low"|"medium"|"high"
}
Write "answer" in the requested language.
"""


class AnalysisNotFoundError(KeyError):
    """Unknown analysis_id."""


class LegalDocumentIntelService:
    def __init__(
        self,
        settings: Settings,
        client: Gemma4Client | None = None,
        store: InMemoryAnalysisStore | None = None,
    ) -> None:
        self._settings = settings
        self._client = client
        # Do not use `store or ...` — an empty store is falsy via __len__.
        self._store = analysis_store if store is None else store

    def _get_client(self) -> Gemma4Client:
        if self._client is None:
            self._client = Gemma4Client(self._settings)
        return self._client

    @property
    def model_id(self) -> str:
        return self._settings.llm.gemma_model_name

    async def analyze(
        self,
        *,
        text: str | None,
        language: ResponseLanguage,
        image_bytes: bytes | None = None,
        image_mime: str | None = None,
    ) -> DocumentAnalyzeResponse:
        lang_name = "Hindi" if language == "hi" else "English"
        user_prompt = (
            f"Language for explanations: {lang_name} ({language}).\n"
            "Analyze the following legal document content.\n"
        )
        if text:
            user_prompt += f"\n--- DOCUMENT TEXT ---\n{text}\n--- END ---\n"
        if image_bytes:
            user_prompt += (
                "\nAn image of the document is attached. Read all visible text "
                "and use it as the document source (together with any text above).\n"
            )

        client = self._get_client()
        try:
            if image_bytes and image_mime:
                raw = await client.generate_with_image(
                    user_prompt,
                    image_bytes=image_bytes,
                    mime_type=image_mime,
                    system_instruction=SYSTEM_ANALYZE,
                    thinking_level="high",
                    response_mime_type="application/json",
                )
            else:
                raw = await client.generate_text(
                    user_prompt,
                    system_instruction=SYSTEM_ANALYZE,
                    thinking_level="high",
                    response_mime_type="application/json",
                )
        except GemmaAPIError:
            raise

        data = _parse_json_object(raw)
        analysis_id = str(uuid.uuid4())
        document_text = text or data.get("_ocr_text") or ""
        # Prefer model-visible reconstruction if only image was provided.
        if not document_text and image_bytes:
            document_text = data.get("plain_summary", "")

        key_clauses = [
            KeyClause(
                title=str(c.get("title", "")),
                quote=str(c.get("quote", "")),
                explanation=str(c.get("explanation", "")),
            )
            for c in data.get("key_clauses", [])
            if isinstance(c, dict)
        ]
        risk_flags = [
            RiskFlag(
                clause_title=str(r.get("clause_title", "")),
                quote=str(r.get("quote", "")),
                risk_level=_clamp_risk(r.get("risk_level")),
                reason=str(r.get("reason", "")),
            )
            for r in data.get("risk_flags", [])
            if isinstance(r, dict)
        ]
        questions = [str(q) for q in data.get("questions_to_ask", []) if q]

        response = DocumentAnalyzeResponse(
            analysis_id=analysis_id,
            document_type=str(data.get("document_type", "unknown")),
            plain_summary=str(data.get("plain_summary", "")),
            key_clauses=key_clauses,
            risk_flags=risk_flags,
            questions_to_ask=questions,
            language=language,
            model=self.model_id,
            disclaimer=NOT_LEGAL_ADVICE_DISCLAIMER,
        )

        # DEMO: persist in process-local dict only (no Postgres).
        self._store.put(
            AnalysisRecord(
                analysis_id=analysis_id,
                document_text=text or response.plain_summary,
                language=language,
                model=self.model_id,
                analyze_payload=response.model_dump(),
            )
        )
        return response

    async def ask(
        self,
        *,
        analysis_id: str,
        question: str,
        language: ResponseLanguage,
    ) -> DocumentAskResponse:
        record = self._store.get(analysis_id)
        if record is None:
            raise AnalysisNotFoundError(analysis_id)

        lang_name = "Hindi" if language == "hi" else "English"
        prompt = (
            f"Language for the answer: {lang_name} ({language}).\n"
            f"Question: {question}\n\n"
            f"--- DOCUMENT ---\n{record.document_text}\n--- END ---\n\n"
            f"Prior analysis JSON (for context, still ground in DOCUMENT):\n"
            f"{json.dumps(record.analyze_payload, ensure_ascii=False)}\n"
        )
        client = self._get_client()
        raw = await client.generate_text(
            prompt,
            system_instruction=SYSTEM_ASK,
            thinking_level="high",
            response_mime_type="application/json",
        )
        data = _parse_json_object(raw)
        return DocumentAskResponse(
            answer=str(data.get("answer", "")),
            evidence_quotes=[str(q) for q in data.get("evidence_quotes", []) if q],
            confidence=_clamp_confidence(data.get("confidence")),
            disclaimer=NOT_LEGAL_ADVICE_DISCLAIMER,
        )


def _parse_json_object(raw: str) -> dict[str, Any]:
    text = raw.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    try:
        value = json.loads(text)
    except json.JSONDecodeError as exc:
        match = re.search(r"\{.*\}", text, flags=re.DOTALL)
        if not match:
            raise GemmaAPIError(f"Gemma 4 returned non-JSON: {exc}") from exc
        try:
            value = json.loads(match.group(0))
        except json.JSONDecodeError as exc2:
            raise GemmaAPIError(f"Gemma 4 returned non-JSON: {exc2}") from exc2
    if not isinstance(value, dict):
        raise GemmaAPIError("Gemma 4 JSON was not an object")
    return value


def _clamp_risk(value: Any) -> str:
    v = str(value or "medium").lower()
    return v if v in ("low", "medium", "high") else "medium"


def _clamp_confidence(value: Any) -> str:
    v = str(value or "medium").lower()
    return v if v in ("low", "medium", "high") else "medium"
