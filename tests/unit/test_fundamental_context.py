"""单元测试：统一入口 fundamental_context (需求 6, 9.3, 9.4)。"""

from __future__ import annotations

from dataclasses import dataclass
from unittest import mock

import pytest

from pa_agent.context import fundamental_context as fc
from pa_agent.context.market_classifier import Market

pytestmark = pytest.mark.unit


class _Settings:
    """模拟 PromptSettings 的最小开关集。"""

    enable_fundamental_context = True
    fundamental_include_news = False
    fundamental_include_macro = True
    fundamental_include_sentiment = True
    fundamental_include_flow = True
    fundamental_news_max_items = 3
    fundamental_flow_avg_window = 20
    fundamental_cache_ttl_minutes = 360


@dataclass
class _Bar:
    close: float
    volume: float


@dataclass
class _Frame:
    bars: tuple


def _frame() -> _Frame:
    return _Frame(bars=tuple(_Bar(10.0 + i * 0.1, 100.0 + i) for i in range(5)))


def test_master_switch_off_returns_empty() -> None:
    s = _Settings()
    s.enable_fundamental_context = False
    assert fc.build_for_symbol("AAPL", settings=s) == ""


def test_us_routes_to_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ctx = {
        "available": True,
        "fundamentals": {"trailingPE": 25.0, "currency": "USD"},
        "sentiment": {"recommendationKey": "buy"},
        "flow": {"shortPercentOfFloat": 0.05},
    }
    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: fake_ctx)
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    out = fc.build_for_symbol("AAPL", settings=_Settings())
    assert "基本面与分析师观点" in out
    assert "PE(TTM) 25.00" in out
    assert "做空占流通" in out


def test_hk_routes_to_yfinance(monkeypatch: pytest.MonkeyPatch) -> None:
    called = {}

    def _fetch(symbol, market, **k):
        called["market"] = market
        return {
            "available": True,
            "fundamentals": {"longName": "Tencent"},
            "sentiment": {},
            "flow": {},
        }

    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", _fetch)
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    fc.build_for_symbol("0700.HK", settings=_Settings())
    assert called["market"] is Market.HK


def test_other_market_skips_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = mock.Mock()
    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", spy)
    monkeypatch.setattr(
        fc.macro_snapshot,
        "fetch_macro_snapshot",
        lambda *a, **k: {
            "available": True,
            "items": [{"name": "美元指数", "value": 104.0, "change_pct": 0.1}],
        },
    )
    out = fc.build_for_symbol("XAUUSD", settings=_Settings())
    spy.assert_not_called()  # OTHER 不调个股基本面
    assert "宏观环境快照" in out


def test_a_share_skips_equity(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = mock.Mock()
    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", spy)
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    fc.build_for_symbol("600519", settings=_Settings())
    spy.assert_not_called()


def test_flow_included_with_frame(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: {"available": False}
    )
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    out = fc.build_for_symbol("AAPL", settings=_Settings(), frame=_frame())
    assert "量价资金面" in out


def test_subexception_does_not_propagate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fc.yfinance_fundamentals,
        "fetch_yf_fundamentals",
        mock.Mock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    out = fc.build_for_symbol("AAPL", settings=_Settings())
    assert isinstance(out, str)  # 不抛，返回 str


def test_macro_off_excludes_macro(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Settings()
    s.fundamental_include_macro = False
    spy = mock.Mock()
    monkeypatch.setattr(fc.macro_snapshot, "fetch_macro_snapshot", spy)
    monkeypatch.setattr(
        fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: {"available": False}
    )
    fc.build_for_symbol("AAPL", settings=s)
    spy.assert_not_called()


def test_sentiment_off_excludes_section(monkeypatch: pytest.MonkeyPatch) -> None:
    s = _Settings()
    s.fundamental_include_sentiment = False
    fake_ctx = {
        "available": True,
        "fundamentals": {"trailingPE": 25.0, "currency": "USD"},
        "sentiment": {"recommendationKey": "buy"},
        "flow": {},
    }
    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: fake_ctx)
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    out = fc.build_for_symbol("AAPL", settings=s)
    assert "分析师评级" not in out


def test_sections_for_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    fake_ctx = {
        "available": True,
        "fundamentals": {"longName": "Apple", "trailingPE": 25.0, "currency": "USD"},
        "sentiment": {},
        "flow": {},
    }
    monkeypatch.setattr(fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: fake_ctx)
    monkeypatch.setattr(
        fc.macro_snapshot, "fetch_macro_snapshot", lambda *a, **k: {"available": False, "items": []}
    )
    sections = fc.build_sections_for_symbol("AAPL", settings=_Settings())
    assert any(t == "估值" for t, _ in sections)


def test_guidance_present_when_content(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        fc.yfinance_fundamentals, "fetch_yf_fundamentals", lambda *a, **k: {"available": False}
    )
    monkeypatch.setattr(
        fc.macro_snapshot,
        "fetch_macro_snapshot",
        lambda *a, **k: {
            "available": True,
            "items": [{"name": "美元指数", "value": 104.0, "change_pct": 0.1}],
        },
    )
    out = fc.build_for_symbol("XAUUSD", settings=_Settings())
    assert "以价格行为为主" in out
