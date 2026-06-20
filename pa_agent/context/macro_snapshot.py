"""宏观环境快照：大盘指数 / 利率 / 美元指数 / VIX (yfinance)。

按市场选取指数篮子，用近 2 根日线计算涨跌%。缺包或任何失败静默降级。
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any

from pa_agent.context.market_classifier import Market
from pa_agent.context.yfinance_fundamentals import _run_with_timeout

logger = logging.getLogger(__name__)

_TTL_S = 60 * 60  # 宏观缓存 1 小时
# 单个指数抓取超时(秒)
_INDEX_TIMEOUT_S = 6

# (yf 代码, 中文名) 篮子
_BASKETS: dict[Market, list[tuple[str, str]]] = {
    Market.HK: [
        ("^HSI", "恒生指数"),
        ("^GSPC", "标普500"),
        ("^TNX", "美债10Y"),
        ("DX-Y.NYB", "美元指数"),
    ],
    Market.US: [
        ("^GSPC", "标普500"),
        ("^IXIC", "纳斯达克"),
        ("^VIX", "VIX恐慌"),
        ("^TNX", "美债10Y"),
        ("DX-Y.NYB", "美元指数"),
    ],
    Market.A_SHARE: [
        ("000001.SS", "上证指数"),
        ("399001.SZ", "深证成指"),
    ],
    Market.OTHER: [
        ("DX-Y.NYB", "美元指数"),
        ("^TNX", "美债10Y"),
    ],
}

# 缓存：{market_value: (monotonic_ts, snap)}
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}


def clear_macro_cache() -> None:
    """清空宏观缓存。"""
    _CACHE.clear()


def fetch_macro_snapshot(market: Market, *, use_cache: bool = True) -> dict[str, Any]:
    """抓取宏观快照。

    返回 ``{"available": bool, "items": [{"name","value","change_pct"}, ...]}``。
    缺包/失败时 ``available=False``，绝不抛异常。
    """
    key = market.value
    if use_cache:
        cached = _CACHE.get(key)
        if cached and (time.monotonic() - cached[0]) < _TTL_S:
            return dict(cached[1])

    snap: dict[str, Any] = {"market": key, "available": False, "items": []}

    basket = _BASKETS.get(market) or _BASKETS[Market.OTHER]

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — macro snapshot skipped")
        return snap

    items: list[dict[str, Any]] = []
    for code, name in basket:
        row = _fetch_one(yf, code, name)
        if row is not None:
            items.append(row)

    snap["items"] = items
    snap["available"] = bool(items)

    if use_cache and items:
        _CACHE[key] = (time.monotonic(), dict(snap))
    return snap


def _fetch_one(yf: Any, code: str, name: str) -> dict[str, Any] | None:
    """取单个指数近 2 根日线算涨跌%；失败返回 None。"""
    try:
        ticker = yf.Ticker(code)
        hist = _run_with_timeout(
            lambda: ticker.history(period="5d", interval="1d"),
            timeout=_INDEX_TIMEOUT_S,
            default=None,
        )
        if hist is None or len(hist) < 1:
            return None
        # 过滤 NaN 收盘：yfinance 对指数常返回最新一行为 NaN(当日未收盘/数据缺口)，
        # 直接取会得到 nan。只保留有效收盘值。
        closes: list[float] = []
        for c in hist["Close"]:
            try:
                cf = float(c)
            except (TypeError, ValueError):
                continue
            if not math.isnan(cf):
                closes.append(cf)
        if not closes:
            return None
        latest = closes[-1]
        change_pct: float | None = None
        if len(closes) >= 2 and closes[-2] != 0:
            change_pct = round((latest - closes[-2]) / closes[-2] * 100, 2)
        return {
            "code": code,
            "name": name,
            "value": round(latest, 2),
            "change_pct": change_pct,
        }
    except Exception:
        logger.debug("macro fetch failed for %s", code, exc_info=True)
        return None


def format_macro_for_prompt(snap: dict[str, Any]) -> str:
    """渲染宏观快照为 markdown；无数据返回 ""。"""
    if not snap or not snap.get("available"):
        return ""
    items = snap.get("items") or []
    if not items:
        return ""
    lines = ["## 宏观环境快照(程序抓取，供参考)"]
    for it in items:
        name = it.get("name", "—")
        value = it.get("value", "—")
        chg = it.get("change_pct")
        chg_s = f"{chg:+.2f}%" if chg is not None else "—"
        lines.append(f"- {name} {value} ({chg_s})")
    return "\n".join(lines)
