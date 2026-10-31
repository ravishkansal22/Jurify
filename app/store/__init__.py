"""Demo-only storage helpers.

The Legal Document Intelligence hackday path uses an in-memory dict store.
No Postgres is required for analyze/ask demos.
"""

from app.store.analysis_store import AnalysisRecord, InMemoryAnalysisStore

__all__ = ["AnalysisRecord", "InMemoryAnalysisStore"]
