from __future__ import annotations

from app.store.analysis_store import AnalysisRecord, InMemoryAnalysisStore


def test_in_memory_store_put_get_clear() -> None:
    # DEMO ONLY: process-local dict — no Postgres.
    store = InMemoryAnalysisStore()
    store.put(
        AnalysisRecord(
            analysis_id="id-1",
            document_text="Rent is 10,000.",
            language="en",
            model="gemma-4-26b-a4b-it",
            analyze_payload={"document_type": "rental_agreement"},
        )
    )
    got = store.get("id-1")
    assert got is not None
    assert got.document_text == "Rent is 10,000."
    assert store.get("missing") is None
    assert len(store) == 1
    store.clear()
    assert len(store) == 0
