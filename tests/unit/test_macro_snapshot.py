"""单元测试：宏观快照 (需求 5)。所有网络调用用 mock。"""

from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from pa_agent.context import macro_snapshot as ms
from pa_agent.context.market_classifier import Market

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache():
    ms.clear_macro_cache()
    yield
    ms.clear_macro_cache()


class _FakeHist:
    """模拟 DataFrame：支持 len() 和 ["Close"]。"""

    def __init__(self, closes: list[float]):
        self._closes = closes

    def __len__(self) -> int:
        return len(self._closes)

    def __getitem__(self, key: str):
        assert key == "Close"
        return self._closes


def _install_fake_yf(monkeypatch, closes_by_code: dict):
    def _ticker(code):
        closes = closes_by_code.get(code, [])
        hist = _FakeHist(closes)
        return SimpleNamespace(history=lambda period, interval: hist)

    fake = SimpleNamespace(Ticker=mock.Mock(side_effect=_ticker))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


def test_macro_us_renders_changes(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = {
        "^GSPC": [5000.0, 5050.0],
        "^IXIC": [16000.0, 15840.0],
        "^VIX": [15.0, 16.0],
        "^TNX": [4.0, 4.1],
        "DX-Y.NYB": [104.0, 104.5],
    }
    _install_fake_yf(monkeypatch, closes)
    snap = ms.fetch_macro_snapshot(Market.US, use_cache=False)
    assert snap["available"] is True
    text = ms.format_macro_for_prompt(snap)
    assert "宏观环境快照" in text
    assert "标普500" in text
    assert "+1.00%" in text  # 5000→5050
    assert "-1.00%" in text  # 16000→15840


def test_macro_hk_basket(monkeypatch: pytest.MonkeyPatch) -> None:
    closes = {
        "^HSI": [18000.0, 18180.0],
        "^GSPC": [5000.0, 5000.0],
        "^TNX": [4.0, 4.0],
        "DX-Y.NYB": [104.0, 104.0],
    }
    _install_fake_yf(monkeypatch, closes)
    snap = ms.fetch_macro_snapshot(Market.HK, use_cache=False)
    text = ms.format_macro_for_prompt(snap)
    assert "恒生指数" in text


def test_single_bar_no_change_pct(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yf(monkeypatch, {"DX-Y.NYB": [104.0], "^TNX": [4.0]})
    snap = ms.fetch_macro_snapshot(Market.OTHER, use_cache=False)
    assert snap["available"] is True
    # change_pct 为 None → 渲染为 —
    text = ms.format_macro_for_prompt(snap)
    assert "—" in text


def test_yfinance_missing_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    import builtins

    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("no yfinance")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "yfinance", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    snap = ms.fetch_macro_snapshot(Market.US, use_cache=False)
    assert snap["available"] is False
    assert ms.format_macro_for_prompt(snap) == ""


def test_fetch_exception_per_index_skipped(monkeypatch: pytest.MonkeyPatch) -> None:
    def _ticker(code):
        if code == "^GSPC":
            raise RuntimeError("boom")
        return SimpleNamespace(history=lambda period, interval: _FakeHist([1.0, 1.1]))

    fake = SimpleNamespace(Ticker=mock.Mock(side_effect=_ticker))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    snap = ms.fetch_macro_snapshot(Market.US, use_cache=False)
    # 至少有部分指数成功，不抛异常
    assert snap["available"] is True
    names = [it["code"] for it in snap["items"]]
    assert "^GSPC" not in names


def test_format_empty_when_unavailable() -> None:
    assert ms.format_macro_for_prompt({"available": False, "items": []}) == ""
    assert ms.format_macro_for_prompt({}) == ""
