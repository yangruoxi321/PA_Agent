"""单元测试：moomoo 深度基本面 provider（解析/格式化/降级/假SDK）。"""

from __future__ import annotations

import sys
from types import SimpleNamespace

import pytest

from pa_agent.context import moomoo_fundamentals as mfd
from pa_agent.context.market_classifier import Market

pytestmark = pytest.mark.unit


# ── 财报解析：去重 + 顺序 ─────────────────────────────────────────────────────


def test_parse_financials_dedup_and_order() -> None:
    data = {
        "structure_list": [
            {"field_id": 1, "display_name": "总收入"},
            {"field_id": 2, "display_name": "营业总收入"},
            {"field_id": 3, "display_name": "毛利"},
            {"field_id": 4, "display_name": "营业利润"},
            {"field_id": 5, "display_name": "净利润"},
            {"field_id": 6, "display_name": "归属于母公司股东净利润"},
        ],
        "report_list": [
            {
                "period_text": "2026/Q3",
                "date_time_str": "2026-04-02",
                "currency_code": "USD",
                "item_list": [
                    {"field_id": 1, "data": 3.34e9, "yoy": 45.5},
                    {"field_id": 2, "data": 3.34e9, "yoy": 45.5},
                    {"field_id": 3, "data": 1.68e9, "yoy": 83.8},
                    {"field_id": 4, "data": 1.24e9, "yoy": 120.9},
                    {"field_id": 5, "data": 3.21e9, "yoy": 511.6},
                    {"field_id": 6, "data": 3.21e9, "yoy": 511.6},
                ],
            }
        ],
    }
    fin = mfd._parse_financials(data)
    names = [it["name"] for it in fin["items"]]
    # 总收入只一条（不是 总收入+营业总收入）；净利润只一条（归属母公司优先，显示为“净利润”）
    assert names == ["总收入", "毛利", "营业利润", "净利润"]
    assert fin["currency"] == "USD"


# ── 格式化 ────────────────────────────────────────────────────────────────────


def _sample() -> dict:
    return {
        "code": "US.WDC",
        "available": True,
        "profile": {"公司名称": "西部数据", "所属市场": "纳斯达克", "CEO": "Irving Tan", "员工数量": "40000"},
        "financials": {
            "period": "2026/Q3",
            "currency": "USD",
            "items": [
                {"name": "总收入", "value": 3.34e9, "yoy": 45.5},
                {"name": "净利润", "value": 3.21e9, "yoy": 511.6},
            ],
        },
        "valuation": {
            "PE": {"current": 44.44, "average": 25.31, "percentile": 100.0, "forward": 46.71},
            "PS": {"current": 21.84, "average": 7.18, "percentile": 100.0, "forward": 15.64},
        },
        "analyst": {"highest": 685.0, "average": 596.15, "lowest": 450.0, "rating": 4,
                    "total": 13, "buy": 100.0, "hold": 0.0, "sell": 0.0},
        "revenue": [{"name": "HDD", "ratio": 100.0}],
    }


def test_format_sections_rich() -> None:
    secs = mfd.format_moomoo_fundamentals_sections(_sample())
    titles = [t for t, _ in secs]
    assert "公司简介" in titles
    assert any(t.startswith("财报") for t in titles)
    assert "估值历史分位" in titles
    assert "分析师一致预期" in titles
    body = dict(secs)
    assert "3.34B" in body["财报（2026/Q3）"]
    assert "YoY +45.5%" in body["财报（2026/Q3）"]
    assert "分位100%·极高" in body["估值历史分位"]
    assert "一致评级 买入" in body["分析师一致预期"]
    assert "买入 100%" in body["分析师一致预期"]


def test_format_snapshot_metrics_short_inst() -> None:
    data = dict(_sample())
    data["snapshot"] = {
        "total_market_val": 257_212_146_616.0,
        "pe_ttm_ratio": 44.44,
        "pb_ratio": 26.57,
        "dividend_ratio_ttm": 0.06,  # 已是百分数
        "dividend_ttm": 0.45,
        "highest52weeks_price": 799.87,
        "lowest52weeks_price": 58.5,
        "last_price": 746.23,
        "turnover_rate": 4.96,
        "amplitude": 8.53,
    }
    data["metrics"] = {
        "period": "2026/Q3",
        "净资产收益率（ROE）": 85.87,
        "总资产净利率（ROA）": 40.61,
        "毛利率": 45.43,
        "归母净利率": 54.16,
        "流动比率": 1.49,
        "速动比率": 1.11,
        "自由现金流与收入比率": 24.67,
    }
    data["short"] = {"short_percent": 9.25, "shares_short": 31_870_000.0, "days_to_cover": 5.02}
    data["institution"] = {"holder_pct": 97.86, "institution_quantity": 1666}
    body = dict(mfd.format_moomoo_fundamentals_sections(data))
    assert "市值 257.21B" in body["估值现状"]
    # 股息率不再误×100：0.06 → 0.06%（不是 6.00%）
    assert "股息率(TTM) 0.06%" in body["估值现状"]
    assert "ROE 85.87%" in body["盈利与财务（2026/Q3）"]
    assert "流动比率 1.49" in body["盈利与财务（2026/Q3）"]
    assert "52周 58.50 ~ 799.87" in body["区间与风险"]
    assert "做空 9.25%" in body["做空与机构"]
    assert "31.87M" in body["做空与机构"]
    assert "机构持股 97.86%" in body["做空与机构"]


def test_parse_metrics_filters_nan_and_targets() -> None:
    data = {
        "structure_list": [
            {"field_id": 1, "display_name": "净资产收益率（ROE）"},
            {"field_id": 2, "display_name": "无关指标"},
        ],
        "report_list": [
            {"period_text": "2026/Q3", "item_list": [
                {"field_id": 1, "data": 85.87},
                {"field_id": 2, "data": 1.0},
            ]}
        ],
    }
    m = mfd._parse_metrics(data)
    assert m["净资产收益率（ROE）"] == 85.87
    assert "无关指标" not in m


def test_format_empty_on_unavailable() -> None:
    assert mfd.format_moomoo_fundamentals_sections(None) == []
    assert mfd.format_moomoo_fundamentals_sections({"available": False}) == []
    assert mfd.format_moomoo_fundamentals_for_prompt(None) == ""


def test_prompt_markdown_heading() -> None:
    md = mfd.format_moomoo_fundamentals_for_prompt(_sample())
    assert md.startswith("## 基本面（moomoo")
    assert "### 估值历史分位" in md


# ── 降级 ──────────────────────────────────────────────────────────────────────


def test_other_market_none() -> None:
    assert mfd.fetch_moomoo_fundamentals("XAUUSD", Market.OTHER) is None


def test_missing_sdk_degrades(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setitem(sys.modules, "moomoo", None)
    mfd.clear_moomoo_fundamentals_cache()
    assert mfd.fetch_moomoo_fundamentals("WDC", Market.US, use_cache=False) is None


# ── 假 SDK 成功路径 ───────────────────────────────────────────────────────────


class _Row(dict):
    def get(self, k, default=None):  # noqa: A003
        return dict.get(self, k, default)


class _ProfileDF:
    def __init__(self, rows):
        self._rows = rows

    def __len__(self):
        return len(self._rows)

    def iterrows(self):
        for i, r in enumerate(self._rows):
            yield i, _Row(r)


class _FakeCtx:
    def __init__(self, *a, **k):
        pass

    def get_company_profile(self, code):
        return 0, _ProfileDF([
            {"name": "公司名称", "value": "西部数据"},
            {"name": "CEO", "value": "Irving Tan"},
        ])

    def get_financials_statements(self, code, **k):
        return 0, {
            "structure_list": [{"field_id": 1, "display_name": "总收入"}],
            "report_list": [{"period_text": "2026/Q3", "currency_code": "USD",
                             "item_list": [{"field_id": 1, "data": 3.34e9, "yoy": 45.5}]}],
        }

    def get_valuation_detail(self, code, **k):
        return 0, {"trend": {"current_value": 44.4, "average_value": 25.3,
                             "valuation_percentile": 100.0, "forward_value": 46.7}}

    def get_research_analyst_consensus(self, code):
        return 0, {"highest": 685.0, "average": 596.15, "lowest": 450.0, "rating": 4,
                   "total": 13, "buy": 100.0, "hold": 0.0, "sell": 0.0}

    def get_financials_revenue_breakdown(self, code):
        return 0, {"breakdown_list": [{"item_list": [{"name": "HDD", "ratio": 100.0}]}]}

    def close(self):
        pass


def test_fetch_with_fake_sdk(monkeypatch: pytest.MonkeyPatch) -> None:
    fake = SimpleNamespace(RET_OK=0, OpenQuoteContext=_FakeCtx)
    monkeypatch.setitem(sys.modules, "moomoo", fake)
    mfd.clear_moomoo_fundamentals_cache()
    d = mfd.fetch_moomoo_fundamentals("WDC", Market.US, use_cache=False)
    assert d is not None and d["available"] is True
    assert d["profile"]["公司名称"] == "西部数据"
    assert d["financials"]["items"][0]["name"] == "总收入"
    assert d["valuation"]["PE"]["percentile"] == 100.0
    assert d["analyst"]["total"] == 13
    secs = mfd.format_moomoo_fundamentals_sections(d)
    assert any(t == "公司简介" for t, _ in secs)
