"""Demo-only in-memory analysis store (Me 2).

Analyses live in a process-local dict guarded by an asyncio-friendly threading
lock. Data is lost on restart. Oldest entries are evicted when the cap is hit.
This is intentionally NOT Postgres — hackday demo path only.
"""

from __future__ import annotations

import threading
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

from app.schemas.document_analysis import AnalyzeResponse

DEFAULT_MAX_ENTRIES = 200


@dataclass
class StoredAnalysis:
    source_text: str
    analysis: AnalyzeResponse


class AnalysisStore:
    """Thread-safe LRU-ish dict: analysis_id -> StoredAnalysis."""

    def __init__(self, max_entries: int = DEFAULT_MAX_ENTRIES) -> None:
        self._max_entries = max_entries
        self._items: OrderedDict[str, StoredAnalysis] = OrderedDict()
        self._lock = threading.Lock()

    def create(
        self,
        *,
        source_text: str,
        analysis: AnalyzeResponse,
        analysis_id: str | None = None,
    ) -> AnalyzeResponse:
        aid = analysis_id or str(uuid.uuid4())
        stored_analysis = analysis.model_copy(update={"analysis_id": aid})
        with self._lock:
            while len(self._items) >= self._max_entries:
                self._items.popitem(last=False)
            self._items[aid] = StoredAnalysis(
                source_text=source_text, analysis=stored_analysis
            )
            self._items.move_to_end(aid)
        return stored_analysis

    def get(self, analysis_id: str) -> StoredAnalysis | None:
        with self._lock:
            item = self._items.get(analysis_id)
            if item is not None:
                self._items.move_to_end(analysis_id)
            return item

    def clear(self) -> None:
        with self._lock:
            self._items.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._items)


_store: AnalysisStore | None = None
_store_lock = threading.Lock()


def get_analysis_store() -> AnalysisStore:
    global _store
    with _store_lock:
        if _store is None:
            _store = AnalysisStore()
        return _store


def build_qa_context_from_analysis(analysis: AnalyzeResponse) -> str:
    """For image-only analyzes: use summary + key clauses as later Q&A context."""
    parts: list[str] = [f"Summary: {analysis.plain_summary}"]
    for clause in analysis.key_clauses:
        parts.append(f"- {clause.title}: {clause.quote} ({clause.explanation})")
    return "\n".join(parts)


def normalize_for_quote_match(value: str) -> str:
    return " ".join(value.casefold().split())


def quote_appears_in_source(quote: str, source_text: str) -> bool:
    if not quote or not source_text:
        return False
    return normalize_for_quote_match(quote) in normalize_for_quote_match(source_text)


def filter_verified_risk_flags(
    flags: list[Any],
    source_text: str | None,
) -> list[Any]:
    """Drop risk flags whose quotes cannot be found in source text.

    When source_text is None/empty (image-only with no OCR text), keep flags
    but mark the reason so the client can see they were not text-verified.
    """
    from app.schemas.document_analysis import RiskFlag

    kept: list[RiskFlag] = []
    for flag in flags:
        if not isinstance(flag, RiskFlag):
            flag = RiskFlag.model_validate(flag)
        if source_text and source_text.strip():
            if quote_appears_in_source(flag.quote, source_text):
                kept.append(flag)
            # else: drop fabricated quote
        else:
            reason = flag.reason
            if not reason.startswith("[unverified quote]"):
                reason = f"[unverified quote] {reason}"
            kept.append(flag.model_copy(update={"reason": reason}))
    return kept


def filter_verified_quotes(quotes: list[str], source_text: str) -> list[str]:
    if not source_text.strip():
        return quotes
    return [q for q in quotes if quote_appears_in_source(q, source_text)]
