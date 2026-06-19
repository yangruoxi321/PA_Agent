"""由品种代码(+数据源)判定市场：A股 / 港股 / 美股 / 其他。

复用 ``pa_agent.data.market_defaults`` 中已有的 A 股、港股、加密判定逻辑，
避免重复造轮子。判定优先级见 ``classify_market`` 文档。
"""

from __future__ import annotations

import re
from enum import Enum

from pa_agent.data.market_defaults import (
    _is_ashare_tv_code,
    _is_hk_tv_code,
    _looks_like_ashare_code,
    is_likely_crypto_symbol,
    normalize_ashare_tv_code,
    normalize_hk_tv_code,
)


class Market(str, Enum):
    """标的所属市场。"""

    A_SHARE = "a_share"
    HK = "hk"
    US = "us"
    OTHER = "other"


# 黄金/外汇等明确归 OTHER 的提示词
_FOREX_METAL_HINTS = ("XAU", "XAG", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

# 美股代码：纯字母，或字母+点+字母(如 BRK.B)，长度合理
_US_TICKER_RE = re.compile(r"^[A-Za-z]{1,6}(\.[A-Za-z]{1,3})?$")


def _strip_exchange_prefix(symbol: str) -> str:
    """去掉 ``HKEX:700`` / ``NASDAQ:AAPL`` 这类交易所前缀。"""
    s = (symbol or "").strip()
    if ":" in s:
        return s.split(":", 1)[1].strip()
    return s


def _looks_like_hk(symbol: str) -> bool:
    """港股：HKEX: 前缀、.HK 后缀、或纯数字 1–5 位。"""
    s = (symbol or "").strip()
    if not s:
        return False
    if s.upper().startswith("HKEX:") or s.upper().startswith("HK:"):
        return True
    body = _strip_exchange_prefix(s)
    if body.upper().endswith(".HK"):
        return True
    code = normalize_hk_tv_code(body)
    return _is_hk_tv_code(code)


def _looks_like_us(symbol: str) -> bool:
    """美股：纯字母或字母数字带点(AAPL/BRK.B)，且非外汇/金属/加密。"""
    body = _strip_exchange_prefix(symbol).strip()
    if not body:
        return False
    if body.upper().endswith(".HK"):
        return False
    upper = body.upper().replace(".", "")
    if is_likely_crypto_symbol(body):
        return False
    # 形如 XAUUSD / EURUSD 这类 6 位外汇对，含两段货币提示 → 非美股
    if len(upper) == 6 and sum(h in upper for h in _FOREX_METAL_HINTS) >= 2:
        return False
    return bool(_US_TICKER_RE.match(body))


def classify_market(symbol: str, data_source: str | None = None) -> Market:
    """判定 *symbol* 所属市场。

    优先级(从上到下)：

    1. A 股：6 位纯数字，或带 ``sh``/``sz`` 前缀、``.SH``/``.SZ`` 后缀。
    2. 港股：``HKEX:xxxx``、``.HK`` 后缀、或纯数字 1–5 位。
    3. 美股：纯字母或字母数字带点(``AAPL``、``BRK.B``)，且非外汇/金属/加密。
    4. 其他：黄金/外汇/加密、无法识别、空串 → :attr:`Market.OTHER`。

    *data_source* 作为辅助信号：``akshare``/``eastmoney`` 倾向 A 股；
    ``mt5`` 倾向 OTHER。
    """
    s = (symbol or "").strip()
    ds = (data_source or "").strip().lower()

    if not s:
        return Market.OTHER

    body = _strip_exchange_prefix(s)

    # 数据源强信号：A 股专用源
    if ds in ("akshare", "eastmoney"):
        code = normalize_ashare_tv_code(body)
        if _is_ashare_tv_code(code) or _looks_like_ashare_code(body):
            return Market.A_SHARE

    # 1. A 股(6 位数字 / sh,sz)。注意要在港股之前。
    if _looks_like_ashare_code(body):
        return Market.A_SHARE
    if body.upper().endswith((".SH", ".SZ")):
        return Market.A_SHARE

    # 2. 港股
    if _looks_like_hk(s):
        return Market.HK

    # mt5 数据源：非股票，倾向 OTHER
    if ds == "mt5":
        return Market.OTHER

    # 3. 美股
    if _looks_like_us(s):
        return Market.US

    # 4. 其他(含加密/外汇/无法识别)
    return Market.OTHER
