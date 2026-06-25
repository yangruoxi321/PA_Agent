"""Routing + guard wiring for the failed-breakout-below / liquidity-grab setup."""
from __future__ import annotations

from dataclasses import dataclass

from pa_agent.ai.router import route_strategy_files
from pa_agent.ai.liquidity_grab import (
    PATTERN_FAILED_BREAKOUT_ABOVE,
    PATTERN_FAILED_BREAKOUT_BELOW,
    PATTERN_LIQUIDITY_GRAB,
    PATTERN_LIQUIDITY_GRAB_ABOVE,
    PATTERN_LIQUIDITY_GRAB_PENDING,
    guard_failed_breakout,
    guard_failed_breakout_below,  # back-compat alias
)


@dataclass(frozen=True)
class Bar:
    seq: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


@dataclass(frozen=True)
class Frame:
    symbol: str
    bars: tuple


def _baseline(start_seq: int, n: int, price: float = 100.8, vol: float = 800.0) -> list[Bar]:
    return [
        Bar(seq=start_seq + i, open=price, high=price + 0.4, low=price - 0.3,
            close=price + 0.1, volume=vol)
        for i in range(n)
    ]


def _candidate_frame(symbol: str = "AAPL") -> Frame:
    follow = Bar(seq=1, open=100.2, high=101.5, low=100.1, close=101.2, volume=800)
    stab = Bar(seq=2, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    return Frame(symbol=symbol, bars=tuple([follow, stab] + _baseline(3, 8)))


# ── Router ────────────────────────────────────────────────────────────────────


def test_router_loads_grab_playbook():
    stage1 = {
        "cycle_position": "trading_range",
        "direction": "neutral",
        "detected_patterns": [PATTERN_FAILED_BREAKOUT_BELOW],
    }
    files = route_strategy_files(stage1)
    assert "文件23-假突破与流动性扫单.txt" in files
    assert "文件18-突破失败与突破测试.txt" in files
    assert "文件22-信号失败后的磁力位.txt" in files
    # boundary/range setup → range strategy available
    assert "震荡区间交易策略.txt" in files
    # no duplicate file names
    assert len(files) == len(set(files))


def test_router_loads_grab_playbook_above():
    stage1 = {
        "cycle_position": "trading_range",
        "direction": "neutral",
        "detected_patterns": [PATTERN_FAILED_BREAKOUT_ABOVE],
    }
    files = route_strategy_files(stage1)
    assert "文件23-假突破与流动性扫单.txt" in files
    assert "文件18-突破失败与突破测试.txt" in files
    assert len(files) == len(set(files))


def _above_candidate_frame(symbol: str = "AAPL") -> Frame:
    follow = Bar(seq=1, open=99.8, high=99.9, low=98.5, close=98.8, volume=800)
    stab = Bar(seq=2, open=99.7, high=100.6, low=99.5, close=99.75, volume=2000)
    return Frame(symbol=symbol, bars=tuple([follow, stab] + _baseline(3, 8, price=99.2)))


def test_guard_confirms_above_short_candidate():
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [],
        "bar_analysis": {"always_in": "neutral"},
        "resistance_levels": ["100.0"],
    }
    v = guard_failed_breakout(stage1, _above_candidate_frame("AAPL"))
    assert v is not None and v.status == "candidate"
    assert v.side == "above" and v.direction == "short"
    assert PATTERN_FAILED_BREAKOUT_ABOVE in stage1["detected_patterns"]
    assert PATTERN_LIQUIDITY_GRAB_ABOVE in stage1["detected_patterns"]
    assert stage1["liquidity_grab"]["side"] == "above"
    assert stage1["liquidity_grab"]["direction"] == "short"


# ── Guard: confirm / add tags / attach block ──────────────────────────────────


def test_guard_confirms_and_tags_candidate():
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [],
        "bar_analysis": {"always_in": "neutral"},
        "support_levels": ["100.0"],
    }
    v = guard_failed_breakout_below(stage1, _candidate_frame("AAPL"))
    assert v is not None and v.status == "candidate"
    assert PATTERN_FAILED_BREAKOUT_BELOW in stage1["detected_patterns"]
    assert PATTERN_LIQUIDITY_GRAB in stage1["detected_patterns"]
    assert stage1["liquidity_grab"]["status"] == "candidate"
    assert stage1["liquidity_grab"]["pierced_level"] == 100.0


def test_guard_strips_tag_when_geometry_fails():
    # LLM hallucinated the tag, but there's no support pierced → strip it.
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [PATTERN_FAILED_BREAKOUT_BELOW, "wedge"],
        "bar_analysis": {"always_in": "neutral"},
        "support_levels": ["50.0"],  # far below price, nothing pierced
    }
    frame = Frame(symbol="AAPL", bars=tuple(_baseline(1, 8)))
    guard_failed_breakout_below(stage1, frame)
    assert PATTERN_FAILED_BREAKOUT_BELOW not in stage1["detected_patterns"]
    assert "wedge" in stage1["detected_patterns"]  # unrelated tag preserved


def test_guard_tick_volume_symbol_caps_confidence():
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [],
        "bar_analysis": {"always_in": "neutral"},
        "support_levels": ["100.0"],
    }
    v = guard_failed_breakout_below(stage1, _candidate_frame("XAUUSDm"))
    assert v.status == "candidate"
    assert v.volume_quality == "tick_or_missing"
    assert stage1["liquidity_grab"]["confidence_cap"] <= 15


def test_guard_fundamental_crack_downgrades():
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [PATTERN_FAILED_BREAKOUT_BELOW],
        "bar_analysis": {"always_in": "neutral"},
        "support_levels": ["100.0"],
        "context_assessment": {"stance": "diverges", "confidence_adjustment": -20},
    }
    v = guard_failed_breakout_below(stage1, _candidate_frame("AAPL"))
    assert v.status == "downgraded"
    assert PATTERN_FAILED_BREAKOUT_BELOW not in stage1["detected_patterns"]


def test_guard_pending_marks_no_buy():
    fresh_stab = Bar(seq=1, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    prior = Bar(seq=2, open=100.7, high=101.0, low=100.5, close=100.8, volume=850)
    frame = Frame(symbol="AAPL", bars=tuple([fresh_stab, prior] + _baseline(3, 6)))
    stage1 = {
        "cycle_position": "trading_range",
        "detected_patterns": [PATTERN_FAILED_BREAKOUT_BELOW],
        "bar_analysis": {"always_in": "neutral"},
        "support_levels": ["100.0"],
    }
    v = guard_failed_breakout_below(stage1, frame)
    assert v.status == "pending"
    assert PATTERN_FAILED_BREAKOUT_BELOW not in stage1["detected_patterns"]
    assert PATTERN_LIQUIDITY_GRAB_PENDING in stage1["detected_patterns"]
