"""Me-2-owned demo settings that do not require edits to app.core.config.

ANALYZE_THINKING defaults to False after live measurement: thinking ON on the
sample rental agreement exceeded 25s (hung >11 minutes before abort). /ask
always uses thinking OFF in the route.
"""

from __future__ import annotations

import os


def analyze_thinking_enabled() -> bool:
    """Return whether /analyze should pass thinking=True to GemmaService.

    Env: ANALYZE_THINKING=1/true/yes/on enables; 0/false/no/off disables.
    Default: False (OFF) — chosen by live latency measurement (>25s threshold).
    """
    raw = os.getenv("ANALYZE_THINKING")
    if raw is None or not raw.strip():
        return False
    return raw.strip().casefold() in {"1", "true", "yes", "on"}
