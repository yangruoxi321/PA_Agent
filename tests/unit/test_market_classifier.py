"""单元测试：市场分类器 (需求 2)。"""

from __future__ import annotations

import pytest

from pa_agent.context.market_classifier import (
    Market,
    classify_market,
    market_from_exchange,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "exchange,expected",
    [
        ("NASDAQ", Market.US),
        ("nyse", Market.US),
        ("HKEX", Market.HK),
        ("SEHK", Market.HK),
        ("SSE", Market.A_SHARE),
        ("SZSE", Market.A_SHARE),
        ("TSE", Market.JP),
        ("tyo", Market.JP),
        ("KRX", Market.KR),
        ("KOSDAQ", Market.KR),
        ("OANDA", None),  # 外汇所不覆盖
        ("BINANCE", None),  # 加密所不覆盖
        ("", None),
        (None, None),
    ],
)
def test_market_from_exchange(exchange, expected) -> None:
    assert market_from_exchange(exchange) is expected


def test_exchange_priority_overrides_symbol() -> None:
    # 选了 NASDAQ：即使代码像港股数字，也以交易所为准判美股。
    assert classify_market("700", exchange="NASDAQ") is Market.US
    # 选了 HKEX：字母代码也按港股。
    assert classify_market("WDC", exchange="HKEX") is Market.HK
    # 指数 CFD 误判修复：选了 US 交易所 → 美股，而非误判港股。
    assert classify_market("US500m", exchange="NASDAQ") is Market.US


def test_forex_metal_crypto_not_overridden_by_exchange() -> None:
    # 黄金/外汇/加密即便被打上股票交易所，也不当股票(避免 yfinance 404 空等)。
    assert classify_market("XAUUSD", exchange="NASDAQ") is Market.OTHER
    assert classify_market("XAUUSDm", exchange="NASDAQ") is Market.OTHER
    assert classify_market("EURUSD", exchange="NYSE") is Market.OTHER
    assert classify_market("BTCUSDT", exchange="NASDAQ") is Market.OTHER


def test_exchange_auto_falls_back_to_symbol() -> None:
    # 空/自动交易所 → 回退按代码判定。
    assert classify_market("WDC", exchange="") is Market.US
    assert classify_market("600519", exchange=None) is Market.A_SHARE
    # 外汇所不覆盖 → 仍按代码(WDC 字母 → 美股)。
    assert classify_market("WDC", exchange="OANDA") is Market.US


@pytest.mark.parametrize(
    "symbol",
    ["600519", "000001", "688981", "sh600519", "sz000001", "600519.SH", "000001.SZ"],
)
def test_a_share(symbol: str) -> None:
    assert classify_market(symbol) is Market.A_SHARE


@pytest.mark.parametrize(
    "symbol",
    ["HKEX:700", "700", "07709", "0700.HK", "00700", "9988.HK"],
)
def test_hk(symbol: str) -> None:
    assert classify_market(symbol) is Market.HK


@pytest.mark.parametrize(
    "symbol,exchange",
    [
        ("7203", "TSE"),       # 选交易所
        ("TSE:7203", None),    # 交易所前缀
        ("7203.T", None),      # yfinance 后缀
        ("6758.T", None),
    ],
)
def test_jp(symbol: str, exchange) -> None:
    assert classify_market(symbol, exchange=exchange) is Market.JP


@pytest.mark.parametrize(
    "symbol,exchange",
    [
        ("005930", "KRX"),     # 选交易所
        ("KRX:005930", None),  # 交易所前缀
        ("005930.KS", None),   # KOSPI 后缀
        ("247540.KQ", None),   # KOSDAQ 后缀
    ],
)
def test_kr(symbol: str, exchange) -> None:
    assert classify_market(symbol, exchange=exchange) is Market.KR


def test_jp_kr_numeric_needs_exchange() -> None:
    # 数字代码冲突：无交易所时日股 4 位→港股、韩股 6 位→A 股（必须靠选交易所区分）。
    assert classify_market("7203") is Market.HK
    assert classify_market("005930") is Market.A_SHARE


@pytest.mark.parametrize(
    "symbol",
    ["AAPL", "brk.b", "TSLA", "NVDA", "NASDAQ:AAPL", "BRK.B"],
)
def test_us(symbol: str) -> None:
    assert classify_market(symbol) is Market.US


@pytest.mark.parametrize(
    "symbol",
    ["XAUUSD", "EURUSD", "BTCUSDT", "ETH-USD", "XAUUSDm"],
)
def test_other(symbol: str) -> None:
    assert classify_market(symbol) is Market.OTHER


def test_empty_and_none() -> None:
    assert classify_market("") is Market.OTHER
    assert classify_market("   ") is Market.OTHER
    assert classify_market(None) is Market.OTHER  # type: ignore[arg-type]


def test_data_source_hint_a_share() -> None:
    # 数据源强信号：A 股专用源
    assert classify_market("000001", data_source="akshare") is Market.A_SHARE
    assert classify_market("600519", data_source="eastmoney") is Market.A_SHARE


def test_data_source_hint_mt5_is_other() -> None:
    # mt5 数据源下，非 A 股/港股的字母代码倾向 OTHER
    assert classify_market("XAUUSD", data_source="mt5") is Market.OTHER


def test_market_enum_values() -> None:
    assert Market.A_SHARE.value == "a_share"
    assert Market.HK.value == "hk"
    assert Market.US.value == "us"
    assert Market.JP.value == "jp"
    assert Market.KR.value == "kr"
    assert Market.OTHER.value == "other"
