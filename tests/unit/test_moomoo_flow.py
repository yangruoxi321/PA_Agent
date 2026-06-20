"""单元测试：moomoo 主力资金流 provider（代码映射/格式化/降级）。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pa_agent.context import moomoo_flow as mf
from pa_agent.context.market_classifier import Market

pytestmark = pytest.mark.unit


# ── 代码映射 ──────────────────────────────────────────────────────────────────


@pytest.mark.parametrize(
    "symbol,market,expected",
    [
        ("WDC", Market.US, "US.WDC"),
        ("NASDAQ:AAPL", Market.US, "US.AAPL"),
        ("700", Market.HK, "HK.00700"),
        ("0700.HK", Market.HK, "HK.00700"),
        ("9988", Market.HK, "HK.09988"),
        ("600519", Market.A_SHARE, "SH.600519"),
        ("000001", Market.A_SHARE, "SZ.000001"),
        ("300750", Market.A_SHARE, "SZ.300750"),
        ("XAUUSD", Market.OTHER, None),
        ("", Market.US, None),
    ],
)
def test_to_moomoo_code(symbol, market, expected) -> None:
    assert mf.to_moomoo_code(symbol, market) == expected


# ── 格式化 ────────────────────────────────────────────────────────────────────


def _sample() -> dict:
    return {
        "code": "US.WDC",
        "available": True,
        "dist": {
            "in_super": 61_910_750.0,
            "out_super": 50_918_879.0,
            "in_big": 257_751_692.0,
            "out_big": 249_680_096.0,
            "in_mid": 300_616_317.0,
            "out_mid": 318_107_751.0,
            "in_small": 469_109_967.0,
            "out_small": 497_629_849.0,
        },
        "net_in_flow": -26_950_000.0,
    }


def test_format_sections_nets_and_signs() -> None:
    secs = mf.format_moomoo_flow_sections(_sample())
    assert len(secs) == 1
    title, body = secs[0]
    assert title == "主力资金流(特大/大/中/小单)"
    # 主力(特大+大) = +1099万 + 807万 ≈ +1906万，净正 → 绿
    assert "主力净流入 +1906万" in body
    assert "🟢" in body
    # 散户(中+小) 净负
    assert "散户净流入 -4601万" in body
    assert "当日累计净流入 -2695万" in body


def test_format_empty_on_unavailable() -> None:
    assert mf.format_moomoo_flow_sections(None) == []
    assert mf.format_moomoo_flow_sections({"available": False}) == []
    assert mf.format_moomoo_flow_for_prompt(None) == ""


def test_format_prompt_has_heading() -> None:
    md = mf.format_moomoo_flow_for_prompt(_sample())
    assert md.startswith("## 主力资金流")
    assert "主力净流入" in md


# ── 降级 ──────────────────────────────────────────────────────────────────────


def test_other_market_returns_none() -> None:
    # 非股票市场不抓，直接 None（不触网、不依赖 OpenD）。
    assert mf.fetch_moomoo_flow("XAUUSD", Market.OTHER) is None


def test_missing_sdk_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    # 模拟未安装 moomoo-api：import 失败 → None，不抛。
    monkeypatch.setitem(sys.modules, "moomoo", None)
    mf.clear_moomoo_flow_cache()
    assert mf.fetch_moomoo_flow("WDC", Market.US, use_cache=False) is None


# ── 带假 SDK 的成功路径 ───────────────────────────────────────────────────────


class _FakeRow:
    def __init__(self, d):
        self._d = d

    def to_dict(self):
        return dict(self._d)


class _Iloc:
    def __init__(self, rows):
        self._rows = rows

    def __getitem__(self, i):
        return _FakeRow(self._rows[i])


class _FakeDF:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    @property
    def iloc(self):
        return _Iloc(self._rows)


class _FakeCtx:
    def __init__(self, *a, **k):
        pass

    def get_capital_distribution(self, code):
        return 0, _FakeDF([
            {
                "capital_in_super": 100.0, "capital_out_super": 40.0,
                "capital_in_big": 50.0, "capital_out_big": 20.0,
                "capital_in_mid": 10.0, "capital_out_mid": 30.0,
                "capital_in_small": 5.0, "capital_out_small": 25.0,
            }
        ])

    def get_capital_flow(self, code, **k):
        return 0, _FakeDF([{"in_flow": 50.0}])

    def close(self):
        pass


def test_fetch_with_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(RET_OK=0, OpenQuoteContext=_FakeCtx)
    monkeypatch.setitem(sys.modules, "moomoo", fake)
    mf.clear_moomoo_flow_cache()
    data = mf.fetch_moomoo_flow("WDC", Market.US, use_cache=False)
    assert data is not None and data["available"] is True
    assert data["dist"]["in_super"] == 100.0
    assert data["net_in_flow"] == 50.0
    # 主力净 = (100-40)+(50-20)=90 正；散户净 = (10-30)+(5-25)=-40
    secs = mf.format_moomoo_flow_sections(data)
    assert "主力净流入" in secs[0][1]
