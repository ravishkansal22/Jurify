"""In-memory analysis store for the Gemma 4 Legal Document Intelligence demo.

DEMO ONLY: analyses live in a process-local dict. No Postgres is needed for
the demo path. Data is lost on process restart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock
from typing import Any


@dataclass
class AnalysisRecord:
    analysis_id: str
    document_text: str
    language: str
    model: str
    analyze_payload: dict[str, Any] = field(default_factory=dict)


class InMemoryAnalysisStore:
    """Thread-safe dict-backed store. Not durable — demo only."""

    def __init__(self) -> None:
        self._records: dict[str, AnalysisRecord] = {}
        self._lock = Lock()

    def put(self, record: AnalysisRecord) -> None:
        with self._lock:
            self._records[record.analysis_id] = record

    def get(self, analysis_id: str) -> AnalysisRecord | None:
        with self._lock:
            return self._records.get(analysis_id)

    def clear(self) -> None:
        with self._lock:
            self._records.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._records)


# Process-wide singleton for the demo API.
analysis_store = InMemoryAnalysisStore()
