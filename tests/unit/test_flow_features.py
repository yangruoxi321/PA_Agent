"""单元测试：量价资金面 (需求 4.1-4.4, 4.6)。"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from pa_agent.context.flow_features import (
    DIVERGENCE_BOTH_UP,
    DIVERGENCE_NONE,
    DIVERGENCE_PRICE_DOWN_VOL_UP,
    DIVERGENCE_PRICE_UP_VOL_DOWN,
    compute_flow_features,
    format_flow_for_prompt,
)

pytestmark = pytest.mark.unit


@dataclass
class _Bar:
    close: float
    volume: float


@dataclass
class _Frame:
    bars: tuple


def _frame(pairs: list[tuple[float, float]]) -> _Frame:
    """pairs 为 newest-first 的 (close, volume) 列表。"""
    return _Frame(bars=tuple(_Bar(close=c, volume=v) for c, v in pairs))


def test_rel_volume_basic() -> None:
    # 最新量 200，近 4 根均量 (200+100+100+100)/4 = 125
    frame = _frame([(10.0, 200.0), (10.0, 100.0), (10.0, 100.0), (10.0, 100.0)])
    feat = compute_flow_features(frame, avg_window=4)
    assert feat["available"] is True
    assert feat["latest_volume"] == 200.0
    assert feat["avg_volume_n"] == 125.0
    assert feat["rel_volume"] == pytest.approx(1.6, rel=1e-3)


def test_window_clamped_to_bar_count() -> None:
    frame = _frame([(10.0, 100.0), (10.0, 100.0)])
    feat = compute_flow_features(frame, avg_window=20)
    assert feat["avg_window"] == 2


def test_divergence_price_up_vol_down() -> None:
    # 旧→新：价从 9 升到 11（价涨），量从 300 降到 100（量缩）
    frame = _frame([(11.0, 100.0), (10.0, 200.0), (9.0, 300.0)])
    feat = compute_flow_features(frame)
    assert feat["vol_price_divergence"] == DIVERGENCE_PRICE_UP_VOL_DOWN


def test_divergence_price_down_vol_up() -> None:
    frame = _frame([(9.0, 300.0), (10.0, 200.0), (11.0, 100.0)])
    feat = compute_flow_features(frame)
    assert feat["vol_price_divergence"] == DIVERGENCE_PRICE_DOWN_VOL_UP


def test_divergence_both_up() -> None:
    frame = _frame([(11.0, 300.0), (10.0, 200.0), (9.0, 100.0)])
    feat = compute_flow_features(frame)
    assert feat["vol_price_divergence"] == DIVERGENCE_BOTH_UP


def test_insufficient_data_safe_default() -> None:
    feat = compute_flow_features(_frame([]))
    assert feat["available"] is False
    assert feat["vol_price_divergence"] == DIVERGENCE_NONE
    feat1 = compute_flow_features(_frame([(10.0, 100.0)]))
    assert feat1["available"] is False


def test_none_frame_does_not_raise() -> None:
    feat = compute_flow_features(None)
    assert feat["available"] is False


def test_zero_avg_volume_rel_none() -> None:
    frame = _frame([(10.0, 0.0), (10.0, 0.0)])
    feat = compute_flow_features(frame)
    assert feat["rel_volume"] is None


def test_format_empty_when_unavailable() -> None:
    assert format_flow_for_prompt(compute_flow_features(_frame([]))) == ""
    assert format_flow_for_prompt({}) == ""


def test_format_contains_volume_and_divergence() -> None:
    frame = _frame([(11.0, 300.0), (10.0, 200.0), (9.0, 100.0)])
    text = format_flow_for_prompt(compute_flow_features(frame, avg_window=3))
    assert "量价资金面" in text
    assert "相对均量" in text
    assert "量价齐升" in text
