"""Unit tests for the UPPER-boundary failed-breakout (冲高诱多扫单 / short) mirror.

Mirror of test_liquidity_grab.py. Bars are newest-first; the stab pierces a
RESISTANCE, fails, long UPPER wick + small body + close LOWER part, down-follow.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pa_agent.ai.liquidity_grab import GrabConfig, detect_failed_breakout_above


@dataclass(frozen=True)
class Bar:
    seq: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    closed: bool = True


def _baseline(start_seq: int, n: int, *, price: float = 99.2, vol: float = 800.0) -> list[Bar]:
    """Quiet bars sitting below resistance, to warm the volume average."""
    return [
        Bar(seq=start_seq + i, open=price, high=price + 0.3, low=price - 0.4,
            close=price - 0.1, volume=vol)
        for i in range(n)
    ]


RESISTANCE = [100.0]  # the upper boundary for all cases


def _clean_candidate_bars() -> list[Bar]:
    """Textbook冲高诱多扫单: stab pokes above 100 a touch, fails back below, long
    upper wick + small body + close lower third, follow bar DOWN + absorbs."""
    follow = Bar(seq=1, open=99.8, high=99.9, low=98.5, close=98.8, volume=800)
    stab = Bar(seq=2, open=99.7, high=100.6, low=99.5, close=99.75, volume=2000)
    return [follow, stab] + _baseline(3, 8)


def test_clean_candidate_real_volume():
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), RESISTANCE, regime="trading_range", always_in="neutral",
        volume_reliable=True,
    )
    assert v.status == "candidate"
    assert v.side == "above"
    assert v.direction == "short"
    assert v.pierced_level == 100.0
    assert v.candidate_seq == 2
    assert v.confidence_cap == 30
    assert not v.failed_criteria


def test_pierce_too_deep_is_downgraded():
    # stab rockets far above resistance → real break, not a grab (①).
    follow = Bar(seq=1, open=102.0, high=102.2, low=101.0, close=101.2, volume=800)
    stab = Bar(seq=2, open=99.9, high=103.0, low=99.8, close=101.8, volume=2000)
    bars = [follow, stab] + _baseline(3, 6)
    v = detect_failed_breakout_above(bars, RESISTANCE, regime="trading_range")
    assert v.status == "downgraded"
    assert any(c.startswith("①") for c in v.failed_criteria)


def test_no_down_follow_is_downgraded():
    # Valid stab geometry but the next bar closes UP → no confirmation (⑥).
    follow = Bar(seq=1, open=99.7, high=100.5, low=99.6, close=100.4, volume=900)
    stab = Bar(seq=2, open=99.7, high=100.6, low=99.5, close=99.75, volume=2000)
    bars = [follow, stab] + _baseline(3, 6)
    v = detect_failed_breakout_above(bars, RESISTANCE, regime="trading_range", volume_reliable=True)
    assert v.status == "downgraded"
    assert any(c.startswith("⑥") for c in v.failed_criteria)


def test_fresh_stab_without_follow_is_pending():
    fresh_stab = Bar(seq=1, open=99.7, high=100.6, low=99.5, close=99.75, volume=2000)
    prior = Bar(seq=2, open=99.0, high=99.3, low=98.9, close=99.1, volume=850)
    bars = [fresh_stab, prior] + _baseline(3, 6)
    v = detect_failed_breakout_above(bars, RESISTANCE, regime="trading_range", volume_reliable=True)
    assert v.status == "pending"
    assert v.confidence_cap == 0


def test_forming_bar_cannot_be_candidate():
    forming = Bar(seq=0, open=99.7, high=100.7, low=99.6, close=100.5, volume=2500, closed=False)
    prior = Bar(seq=1, open=99.0, high=99.3, low=98.9, close=99.1, volume=850)
    bars = [forming, prior] + _baseline(2, 6)
    v = detect_failed_breakout_above(bars, RESISTANCE, regime="trading_range", volume_reliable=True)
    assert v.status != "candidate"


def test_always_in_long_is_downgraded():
    # Opposing-direction Always-In for a short setup is AIL.
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), RESISTANCE, regime="trading_range", always_in="long",
        volume_reliable=True,
    )
    assert v.status == "downgraded"
    assert any(c.startswith("⑤") for c in v.failed_criteria)


def test_fundamental_bull_crack_vetoes():
    # For a SHORT setup, a bullish catalyst (positive adjustment) is the crack.
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), RESISTANCE, regime="trading_range", always_in="neutral",
        fundamental_crack=True, volume_reliable=True,
    )
    assert v.status == "downgraded"
    assert any(c.startswith("基本面") for c in v.failed_criteria)


def test_tick_volume_caps_confidence():
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), RESISTANCE, regime="trading_range", always_in="neutral",
        volume_reliable=False,
    )
    assert v.status == "candidate"
    assert v.volume_quality == "tick_or_missing"
    assert v.confidence_cap <= GrabConfig().tick_volume_cap


def test_no_resistance_is_not_applicable():
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), [], regime="trading_range", volume_reliable=True,
    )
    assert v.status == "not_applicable"


def test_mid_range_stab_is_downgraded():
    # Resistance 100 is a mid-range swing high: window also holds ~106 top and ~95
    # bottom, so 100 is in the MIDDLE third → ⑦ rejects (§6.3/§14).
    follow = Bar(seq=1, open=99.8, high=99.9, low=99.0, close=99.1, volume=800)
    stab = Bar(seq=2, open=99.7, high=100.6, low=99.5, close=99.75, volume=2000)
    upper = [Bar(seq=3 + i, open=105.0, high=106.0, low=104.6, close=105.4, volume=800)
             for i in range(3)]
    lower = [Bar(seq=6 + i, open=95.5, high=96.0, low=95.0, close=95.4, volume=800)
             for i in range(3)]
    bars = [follow, stab] + upper + lower
    v = detect_failed_breakout_above(bars, RESISTANCE, regime="trading_range", volume_reliable=True)
    assert v.status == "downgraded"
    assert any(c.startswith("⑦") for c in v.failed_criteria)


def test_upper_boundary_stab_passes_position_gate():
    v = detect_failed_breakout_above(
        _clean_candidate_bars(), RESISTANCE, regime="trading_range", volume_reliable=True,
    )
    assert v.status == "candidate"
    assert any(c.startswith("⑦") for c in v.passed_criteria)
