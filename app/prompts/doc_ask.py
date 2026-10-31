"""Gemma 4 prompts for POST /api/v1/documents/ask."""

from __future__ import annotations

from app.schemas.document_analysis import ResponseLanguage

SYSTEM_PROMPT = """You are a legal-document explainer for ordinary people in India.

Rules you MUST follow:
- Answer ONLY from the supplied document context. Do not invent facts.
- evidence_quotes must be SHORT verbatim contiguous substrings from the document
  context (prefer under 40 words). Every quote must appear in the context.
- Never invent laws, section numbers, or citations.
- If the document does not answer the question, set answer to exactly
  "not stated in the document" (or the Hindi equivalent when language=hi),
  confidence to "low", and evidence_quotes to [].
- This is not legal advice.
- Write the answer in the requested language. If Hindi (hi), use Devanagari.
- Keep evidence quotes in the original language of the document.
- Output valid JSON only matching the schema. No markdown fences, no commentary.
- Never repeat the same word in a loop.
"""


def build_ask_prompt(
    *,
    language: ResponseLanguage,
    question: str,
    document_context: str,
) -> str:
    lang_line = (
        "Language=hi: Write answer in Hindi Devanagari. If not answerable, answer "
        'exactly: "दस्तावेज़ में उल्लेख नहीं है" with confidence "low" and '
        "evidence_quotes []. Keep any evidence_quotes in the document's original language."
        if language == "hi"
        else "Language=en: Answer in simple clear English. If not answerable, answer "
        'exactly: "not stated in the document" with confidence "low" and '
        "evidence_quotes []. Keep evidence_quotes verbatim from the document."
    )
    return f"""Answer the user's question using ONLY this document context.

{lang_line}

Return JSON with:
- answer: direct answer grounded in the document, or the exact not-stated phrase above
- evidence_quotes: list of short verbatim quotes that support the answer (or [])
- confidence: "low" | "medium" | "high"

QUESTION:
{question}

DOCUMENT CONTEXT:
{document_context}
"""
