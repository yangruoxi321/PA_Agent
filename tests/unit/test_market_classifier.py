"""单元测试：市场分类器 (需求 2)。"""

from __future__ import annotations

import pytest

from pa_agent.context.market_classifier import Market, classify_market

pytestmark = pytest.mark.unit


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
    assert Market.OTHER.value == "other"
