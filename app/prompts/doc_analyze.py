"""Gemma 4 prompts for POST /api/v1/documents/analyze."""

from __future__ import annotations

from app.schemas.document_analysis import ResponseLanguage

SYSTEM_PROMPT = """You are a legal-document explainer for ordinary people in India.

Rules you MUST follow:
- Use ONLY the supplied document (text and/or image). Do not use outside knowledge.
- Quote exact contiguous substrings from the document for every quote field.
- Prefer SHORT quotes (under 40 words) that still identify the clause.
- Never invent laws, section numbers, case names, or citations.
- If you are unsure, say so clearly.
- This is not legal advice. Do not tell the user what they should do legally.
- Write explanations in the requested language. If Hindi (hi), use Devanagari script.
- Keep clause quotes and risk-flag quotes in the ORIGINAL language of the document
  (do not translate quotes into Hindi).
- Output valid JSON only matching the schema. No markdown fences, no commentary.
- Never repeat the same word or phrase in a loop. Keep plain_summary under 120 words.
- Always populate key_clauses (at least 2 when the document has them) and
  questions_to_ask (3 to 5). Include risk_flags for one-sided or harsh terms when present.
"""


def build_analyze_prompt(*, language: ResponseLanguage, has_image: bool, text: str | None) -> str:
    lang_line = (
        "Language=hi: Write plain_summary, each explanation, each reason, and "
        "questions_to_ask in Hindi Devanagari ONLY. Do NOT translate quotes — "
        "copy quotes verbatim from the document (usually English). "
        "plain_summary must be 2–4 short sentences with NO repeated filler words. "
        "Never leave key_clauses or questions_to_ask empty when the document "
        "mentions rent, deposit, notice, entry, or rent increase — include them."
        if language == "hi"
        else "Language=en: Write plain_summary, explanations, reasons, and "
        "questions_to_ask in simple clear English. Copy quotes exactly from the document."
    )
    source_bits: list[str] = []
    if text:
        source_bits.append(f"DOCUMENT TEXT:\n{text}")
    if has_image:
        source_bits.append(
            "An image of the document is attached. Read the visible text from the image."
        )
    source = "\n\n".join(source_bits) if source_bits else "(no text; image only)"

    return f"""Analyze this legal document for a non-lawyer in India.

{lang_line}

Return JSON with ALL of these fields filled:
- document_type: short snake_case label (e.g. rental_agreement, notice, employment_contract)
- plain_summary: under 120 words, everyday language, no word loops
- key_clauses: array of important clauses that appear in the document
  (rent, deposit, notice period, termination, penalties, lock-in, entry rights,
  rent increase, payment terms — only what exists). Each item:
  {{"title": "...", "quote": "<verbatim short substring>", "explanation": "..."}}
- risk_flags: unfair / harsh / easy-to-miss clauses. Each item:
  {{"clause_title": "...", "quote": "<verbatim short substring>",
    "risk_level": "low"|"medium"|"high", "reason": "..."}}
  If none, use [].
- questions_to_ask: 3 to 5 practical clarification questions (strings)

Do not invent clauses. Every quote MUST appear character-for-character (aside from
whitespace) inside the document text above.

{source}
"""
