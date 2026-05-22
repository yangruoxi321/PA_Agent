"""Normalize common Stage 2 AI JSON variants before schema validation."""
from __future__ import annotations

import copy
import logging
from typing import Any

from pa_agent.ai.trace_normalize import normalize_stage2_traces

logger = logging.getLogger(__name__)


def normalize_stage2(obj: dict[str, Any]) -> dict[str, Any]:
    """Return a copy of *obj* with decision_trace quirks corrected."""
    out = copy.deepcopy(obj)
    normalize_stage2_traces(out)
    decision = out.get("decision")
    if isinstance(decision, dict) and decision.get("order_type") == "不下单":
        # A no-order decision has no executable trade; tolerate model-provided
        # win-rate estimates in legacy payloads by clearing them before schema
        # validation while keeping price-field mistakes strict.
        decision["estimated_win_rate"] = None

    bar_analysis = out.get("bar_analysis")
    if isinstance(bar_analysis, dict):
        signal_bar = bar_analysis.get("signal_bar")
        if isinstance(signal_bar, dict) and not signal_bar.get("bar"):
            signal_bar["bar"] = None
            signal_bar.setdefault("quality", "invalid")
            signal_bar.setdefault("pattern", "none")

        entry_bar = bar_analysis.get("entry_bar")
        if isinstance(entry_bar, dict):
            strength = str(entry_bar.get("strength", "") or "").strip().lower()
            has_bar = bool(entry_bar.get("bar"))
            if strength == "not_triggered" or not has_bar:
                # Pending limit/breakout orders do not have an actual entry bar
                # yet. Normalize common model variants before schema checks.
                entry_bar["strength"] = "not_triggered"
                entry_bar.setdefault("bar", None)
                entry_bar.setdefault("freshness", "pending")
                if entry_bar.get("follow_through") in (None, "", "pending"):
                    entry_bar["follow_through"] = "pending"
    return out
