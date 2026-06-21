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
    JP = "jp"
    KR = "kr"
    OTHER = "other"


# 黄金/外汇等明确归 OTHER 的提示词
_FOREX_METAL_HINTS = ("XAU", "XAG", "USD", "EUR", "GBP", "JPY", "AUD", "CAD", "CHF", "NZD")

# 交易所 → 市场映射（用户在「交易所」框明确选择时优先生效）。
# 未列出的（OANDA/BINANCE 等外汇/加密所）返回 None，回退到按代码判定。
_EXCHANGE_MARKET: dict[str, Market] = {
    # 港股
    "HKEX": Market.HK, "HK": Market.HK, "SEHK": Market.HK,
    # 美股
    "NASDAQ": Market.US, "NYSE": Market.US, "AMEX": Market.US,
    "NYSEARCA": Market.US, "ARCA": Market.US, "BATS": Market.US, "US": Market.US,
    # A 股
    "SSE": Market.A_SHARE, "SHSE": Market.A_SHARE, "SH": Market.A_SHARE,
    "SHANGHAI": Market.A_SHARE, "SZSE": Market.A_SHARE, "XSHE": Market.A_SHARE,
    "SZ": Market.A_SHARE, "SHENZHEN": Market.A_SHARE,
    # 日股（东京证券交易所）
    "TSE": Market.JP, "TYO": Market.JP, "JPX": Market.JP,
    "XTKS": Market.JP, "TOKYO": Market.JP, "JP": Market.JP,
    # 韩股（韩国交易所；KOSPI/KOSDAQ 统一 KRX）
    "KRX": Market.KR, "KSC": Market.KR, "KOSPI": Market.KR,
    "KOSDAQ": Market.KR, "XKRX": Market.KR, "KR": Market.KR,
}


def market_from_exchange(exchange: str | None) -> Market | None:
    """已知交易所 → 市场；未知/空 → None（交回代码判定）。"""
    if not exchange:
        return None
    return _EXCHANGE_MARKET.get(exchange.strip().upper())


def _is_non_equity_symbol(symbol: str) -> bool:
    """明确的外汇/金属/加密(如 XAUUSD/EURUSD/BTCUSDT/XAUUSDm)。

    这类符号在某些数据源里会被打上股票交易所(如 NASDAQ)，但它们**不是股票**，
    去 yfinance 查必然 404。用于让交易所优先判定对它们网开一面 → 归 OTHER。
    """
    body = _strip_exchange_prefix(symbol).strip()
    if not body:
        return False
    if is_likely_crypto_symbol(body):
        return True
    upper = body.upper().replace(".", "")
    # 容忍 MT5 后缀(如 XAUUSDm 的尾 M)：7 位且以 M 结尾时取前 6 位
    base = upper[:-1] if len(upper) == 7 and upper.endswith("M") else upper
    if base in ("GOLD", "XAU", "XAG", "SILVER"):
        return True
    # 6 位外汇/金属对：含 ≥2 段货币/金属提示(XAUUSD、EURUSD…)
    if len(base) == 6 and sum(h in base for h in _FOREX_METAL_HINTS) >= 2:
        return True
    return False

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


def _market_from_suffix(body: str) -> Market | None:
    """按 yfinance 风格后缀识别日韩：``.T``/``.JP`` → 日股，``.KS``/``.KQ`` → 韩股。

    必须在 4 位(日股)/6 位(韩股)数字识别之前调用，否则会被港股/A 股误判。
    """
    u = (body or "").strip().upper()
    if u.endswith(".T") or u.endswith(".JP"):
        return Market.JP
    if u.endswith((".KS", ".KQ")):
        return Market.KR
    return None


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


def classify_market(
    symbol: str,
    data_source: str | None = None,
    exchange: str | None = None,
) -> Market:
    """判定 *symbol* 所属市场。

    优先级(从上到下)：

    0. **交易所优先**：用户在「交易所」框明确选了已知交易所(NASDAQ/NYSE/HKEX/
       SSE/SZSE…)时，以交易所为准，覆盖代码判定。外汇/加密所或「自动」不覆盖。
    1. A 股：6 位纯数字，或带 ``sh``/``sz`` 前缀、``.SH``/``.SZ`` 后缀。
    2. 港股：``HKEX:xxxx``、``.HK`` 后缀、或纯数字 1–5 位。
    3. 美股：纯字母或字母数字带点(``AAPL``、``BRK.B``)，且非外汇/金属/加密。
    4. 其他：黄金/外汇/加密、无法识别、空串 → :attr:`Market.OTHER`。

    *data_source* 作为辅助信号：``akshare``/``eastmoney`` 倾向 A 股；
    ``mt5`` 倾向 OTHER。
    """
    # 0. 交易所优先：显式 exchange 参数 > symbol 自带前缀(如 TSE:7203/KRX:005930)。
    #    外汇/金属/加密(黄金被某些源标成 NASDAQ 等)例外，它们不是股票，仍归 OTHER。
    ex = exchange
    if not ex:
        raw = symbol or ""
        if ":" in raw:
            ex = raw.split(":", 1)[0]
    forced = market_from_exchange(ex)
    if forced is not None and not _is_non_equity_symbol(symbol):
        return forced

    s = (symbol or "").strip()
    ds = (data_source or "").strip().lower()

    if not s:
        return Market.OTHER

    body = _strip_exchange_prefix(s)

    # 日韩后缀（在数字识别前：日股 4 位、韩股 6 位与港股/A 股数字冲突）
    suffixed = _market_from_suffix(body)
    if suffixed is not None:
        return suffixed

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
