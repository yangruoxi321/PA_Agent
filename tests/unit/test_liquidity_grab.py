"""Unit tests for the failed-breakout-below / liquidity-grab detector.

Bars are newest-first (``bars[0]`` = latest closed). Each case documents which
criterion is exercised so verdicts can be eyeballed against the spec.
"""
from __future__ import annotations

from dataclasses import dataclass

import pytest

from pa_agent.ai.liquidity_grab import (
    GrabConfig,
    detect_failed_breakout_below,
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


def _baseline(start_seq: int, n: int, *, price: float = 100.8, vol: float = 800.0) -> list[Bar]:
    """Quiet bars sitting above support, used to warm the volume average."""
    return [
        Bar(seq=start_seq + i, open=price, high=price + 0.4, low=price - 0.3,
            close=price + 0.1, volume=vol)
        for i in range(n)
    ]


# Support at 100 for all geometry cases.
SUPPORTS = [100.0]


def _clean_candidate_bars() -> list[Bar]:
    """A textbook failed-breakout-below: stab pierces 100 a touch, recovers,
    long lower wick, small body, close upper third, follow bar up + absorbs."""
    follow = Bar(seq=1, open=100.2, high=101.5, low=100.1, close=101.2, volume=800)
    stab = Bar(seq=2, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    return [follow, stab] + _baseline(3, 8)


def test_clean_candidate_real_volume():
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), SUPPORTS, regime="trading_range", always_in="neutral",
        volume_reliable=True,
    )
    assert v.status == "candidate"
    assert v.pierced_level == 100.0
    assert v.candidate_seq == 2
    assert v.volume_quality == "real"
    assert v.confidence_cap == 30
    assert not v.failed_criteria


def test_pierce_too_deep_is_downgraded():
    # stab gouges far below support → real break, not a grab (criterion ①).
    follow = Bar(seq=1, open=98.0, high=99.0, low=97.8, close=98.8, volume=800)
    stab = Bar(seq=2, open=100.1, high=100.2, low=97.0, close=98.2, volume=2000)
    bars = [follow, stab] + _baseline(3, 6)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range")
    assert v.status == "downgraded"
    assert any(c.startswith("①") for c in v.failed_criteria)


def test_no_follow_through_is_downgraded():
    # Valid stab geometry but the next bar closes down → no confirmation (⑥).
    follow = Bar(seq=1, open=100.3, high=100.4, low=99.6, close=99.7, volume=900)
    stab = Bar(seq=2, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    bars = [follow, stab] + _baseline(3, 6)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    assert v.status == "downgraded"
    assert any(c.startswith("⑥") for c in v.failed_criteria)


def test_fresh_stab_without_follow_is_pending():
    # Newest closed bar is itself a fresh stab that recovered same-bar; there is
    # no up-follow yet → pending, never candidate, zero confidence (iron law).
    fresh_stab = Bar(seq=1, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    prior = Bar(seq=2, open=100.7, high=101.0, low=100.5, close=100.8, volume=850)
    bars = [fresh_stab, prior] + _baseline(3, 6)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    assert v.status == "pending"
    assert v.confidence_cap == 0


def test_forming_bar_cannot_be_a_candidate():
    # An in-progress down-stab (closed=False) must be filtered out entirely; the
    # detector never confirms a buy on a forming bar.
    forming = Bar(seq=0, open=100.3, high=100.4, low=99.3, close=99.5, volume=2500, closed=False)
    prior = Bar(seq=1, open=100.7, high=101.0, low=100.5, close=100.8, volume=850)
    bars = [forming, prior] + _baseline(2, 6)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    assert v.status != "candidate"


def test_always_in_short_is_downgraded():
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), SUPPORTS, regime="trading_range", always_in="short",
        volume_reliable=True,
    )
    assert v.status == "downgraded"
    assert any(c.startswith("⑤") for c in v.failed_criteria)


def test_fundamental_crack_vetoes_candidate():
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), SUPPORTS, regime="trading_range", always_in="neutral",
        fundamental_crack=True, volume_reliable=True,
    )
    assert v.status == "downgraded"
    assert any(c.startswith("基本面") for c in v.failed_criteria)


def test_tick_volume_caps_confidence():
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), SUPPORTS, regime="trading_range", always_in="neutral",
        volume_reliable=False,
    )
    assert v.status == "candidate"
    assert v.volume_quality == "tick_or_missing"
    assert v.confidence_cap <= GrabConfig().tick_volume_cap


def test_missing_volume_is_unavailable_and_capped():
    follow = Bar(seq=1, open=100.2, high=101.5, low=100.1, close=101.2, volume=0)
    stab = Bar(seq=2, open=100.3, high=100.5, low=99.4, close=100.25, volume=0)
    bars = [follow, stab] + _baseline(3, 6, vol=0.0)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range")
    assert v.volume_quality == "unavailable"
    assert v.confidence_cap <= GrabConfig().tick_volume_cap


def test_no_support_is_not_applicable():
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), [], regime="trading_range", volume_reliable=True,
    )
    assert v.status == "not_applicable"


def test_support_text_band_is_parsed():
    # structure_levels may store "99.8-100.2"; the mid (100.0) should be used.
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), ["99.8-100.2"], regime="trading_range", volume_reliable=True,
    )
    assert v.status == "candidate"
    assert v.pierced_level == pytest.approx(100.0)


def test_unrelated_pullback_is_not_applicable():
    # Price never dips below support → not this setup at all.
    bars = _baseline(1, 8)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    assert v.status == "not_applicable"


def test_mid_range_stab_is_downgraded():
    # The pierced support (100) is a mid-range swing low: the window also holds a
    # range bottom near 95 and a top near 106, so 100 sits in the MIDDLE third.
    # §6.3/§14: must not be a candidate even with perfect candle geometry (⑦).
    follow = Bar(seq=1, open=100.2, high=101.0, low=100.1, close=100.9, volume=800)
    stab = Bar(seq=2, open=100.3, high=100.5, low=99.4, close=100.25, volume=2000)
    upper = [Bar(seq=3 + i, open=105.0, high=106.0, low=104.6, close=105.4, volume=800)
             for i in range(3)]
    lower = [Bar(seq=6 + i, open=95.5, high=96.0, low=95.0, close=95.6, volume=800)
             for i in range(3)]
    bars = [follow, stab] + upper + lower
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    assert v.status == "downgraded"
    assert any(c.startswith("⑦") for c in v.failed_criteria)


def test_lower_boundary_stab_passes_position_gate():
    # Same support 100, but now it IS the range bottom (window low ~99.4), so the
    # position gate ⑦ passes and the clean candidate stands.
    v = detect_failed_breakout_below(
        _clean_candidate_bars(), SUPPORTS, regime="trading_range", volume_reliable=True,
    )
    assert v.status == "candidate"
    assert any(c.startswith("⑦") for c in v.passed_criteria)


def test_bad_shape_is_downgraded():
    # Pierces and recovers, but it's a big-body bear bar (no long lower wick) → ③.
    follow = Bar(seq=1, open=100.1, high=100.6, low=100.0, close=100.5, volume=800)
    stab = Bar(seq=2, open=100.4, high=100.45, low=99.5, close=99.95, volume=2000)
    bars = [follow, stab] + _baseline(3, 6)
    v = detect_failed_breakout_below(bars, SUPPORTS, regime="trading_range", volume_reliable=True)
    # close (99.95) is below support 100 → ② also fails; assert ③ shape flagged.
    assert v.status == "downgraded"
    assert any(c.startswith("③") for c in v.failed_criteria)
