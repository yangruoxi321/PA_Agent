"""单元测试：yfinance 基本面/情绪/资金面 (需求 3, 4.5, 9.1, 9.2)。

所有网络调用用 mock，禁止真实联网。yfinance 包可能未安装，
因此用注入 sys.modules 的伪模块来模拟。
"""

from __future__ import annotations

import builtins
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

from pa_agent.context import yfinance_fundamentals as yff
from pa_agent.context.market_classifier import Market

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _clear_cache():
    yff.clear_yf_fundamentals_cache()
    yield
    yff.clear_yf_fundamentals_cache()


def _install_fake_yfinance(monkeypatch: pytest.MonkeyPatch, info: dict, news=None):
    """往 sys.modules 注入伪 yfinance，Ticker 返回带 info/news 的对象。"""
    ticker_obj = SimpleNamespace(info=info, news=news or [])
    fake = SimpleNamespace(Ticker=mock.Mock(return_value=ticker_obj))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    return fake


# ── to_yf_symbol ──────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol,expected",
    [
        ("700", "0700.HK"),
        ("HKEX:700", "0700.HK"),
        ("07709", "07709.HK"),
        ("0700.HK", "0700.HK"),
        ("9988", "9988.HK"),
    ],
)
def test_to_yf_symbol_hk(symbol: str, expected: str) -> None:
    assert yff.to_yf_symbol(symbol, Market.HK) == expected


@pytest.mark.parametrize(
    "symbol,expected",
    [("aapl", "AAPL"), ("AAPL", "AAPL"), ("brk.b", "BRK.B"), ("NASDAQ:tsla", "TSLA")],
)
def test_to_yf_symbol_us(symbol: str, expected: str) -> None:
    assert yff.to_yf_symbol(symbol, Market.US) == expected


# ── fetch + render ────────────────────────────────────────────────────────────


def test_fetch_and_render_with_partial_fields(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {
        "longName": "Apple Inc.",
        "sector": "Technology",
        "trailingPE": 30.5,
        "marketCap": 3_000_000_000_000,
        "currency": "USD",
        "shortPercentOfFloat": 0.012,
        "recommendationKey": "buy",
        "numberOfAnalystOpinions": 40,
    }
    _install_fake_yfinance(monkeypatch, info)
    ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False)
    assert ctx["available"] is True
    text = yff.format_yf_fundamentals_for_prompt(ctx)
    assert "PE(TTM) 30.50" in text
    assert "3.00T" in text
    assert "做空占流通 1.20%" in text
    assert "评级 buy" in text


def test_missing_fields_skipped_not_error(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"trailingPE": 15.0}
    _install_fake_yfinance(monkeypatch, info)
    ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False)
    text = yff.format_yf_fundamentals_for_prompt(ctx)
    assert "PE(TTM) 15.00" in text
    assert "市值" not in text
    assert "ROE" not in text


def test_no_valid_fields_returns_empty_string(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, {})
    ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False)
    assert ctx["available"] is False
    assert yff.format_yf_fundamentals_for_prompt(ctx) == ""


def test_yfinance_not_installed_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    real_import = builtins.__import__

    def _fake_import(name, *args, **kwargs):
        if name == "yfinance":
            raise ImportError("no yfinance")
        return real_import(name, *args, **kwargs)

    monkeypatch.delitem(sys.modules, "yfinance", raising=False)
    monkeypatch.setattr(builtins, "__import__", _fake_import)
    ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False)
    assert ctx["available"] is False
    assert ctx["yf_symbol"] == "AAPL"
    assert yff.format_yf_fundamentals_for_prompt(ctx) == ""


def test_fetch_exception_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    _install_fake_yfinance(monkeypatch, {"trailingPE": 1.0})
    with mock.patch.object(yff, "_safe_info", side_effect=RuntimeError("boom")):
        ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False)
    assert ctx["available"] is False


def test_cache_hit_avoids_second_fetch(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"trailingPE": 10.0, "currency": "USD"}
    fake = _install_fake_yfinance(monkeypatch, info)
    yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=True)
    yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=True)
    assert fake.Ticker.call_count == 1


def test_sections_for_gui(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"longName": "Tencent", "trailingPE": 20.0, "currency": "HKD"}
    _install_fake_yfinance(monkeypatch, info)
    ctx = yff.fetch_yf_fundamentals("0700.HK", Market.HK, use_cache=False)
    sections = yff.format_yf_fundamentals_sections(ctx)
    assert any(title == "估值" for title, _ in sections)


def test_news_fetched_and_rendered(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"trailingPE": 10.0, "currency": "USD"}
    news = [
        {"title": "Apple launches new product", "providerPublishTime": 1700000000},
        {"title": "Analysts upgrade AAPL", "providerPublishTime": 1700000100},
    ]
    ticker_obj = SimpleNamespace(info=info, news=news)
    fake = SimpleNamespace(Ticker=mock.Mock(return_value=ticker_obj))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    ctx = yff.fetch_yf_fundamentals(
        "AAPL", Market.US, use_cache=False, include_news=True, news_max_items=2
    )
    assert len(ctx["news"]) == 2
    sections = yff.format_yf_fundamentals_sections(ctx)
    assert any(title == "近期新闻" for title, _ in sections)


def test_news_not_fetched_when_disabled(monkeypatch: pytest.MonkeyPatch) -> None:
    info = {"trailingPE": 10.0, "currency": "USD"}
    news = [{"title": "X", "providerPublishTime": 1}]
    ticker_obj = SimpleNamespace(info=info, news=news)
    fake = SimpleNamespace(Ticker=mock.Mock(return_value=ticker_obj))
    monkeypatch.setitem(sys.modules, "yfinance", fake)
    ctx = yff.fetch_yf_fundamentals("AAPL", Market.US, use_cache=False, include_news=False)
    assert ctx["news"] == []
