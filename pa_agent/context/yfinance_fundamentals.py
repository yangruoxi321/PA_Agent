"""港股/美股基本面、分析师情绪、机构/做空资金面 (yfinance)。

依赖 ``yfinance`` (可选依赖组 equity)。缺包或任何抓取失败都静默降级，
绝不向主流程抛异常。带模块级缓存避免重复打网。
"""

from __future__ import annotations

import logging
import time
from typing import Any

from pa_agent.context.market_classifier import Market

logger = logging.getLogger(__name__)

# 默认基本面缓存 TTL(秒)，可被 settings.fundamental_cache_ttl_minutes 覆盖
_DEFAULT_TTL_S = 360 * 60
# 单次抓取超时(秒)
_FETCH_TIMEOUT_S = 8

# 缓存：{yf_symbol: (monotonic_ts, ctx)}
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# 从 .info 提取的字段分组
_FUNDAMENTAL_KEYS = (
    "longName",
    "sector",
    "industry",
    "trailingPE",
    "forwardPE",
    "marketCap",
    "trailingEps",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "dividendYield",
    "profitMargins",
    "currency",
)
_SENTIMENT_KEYS = (
    "recommendationKey",
    "numberOfAnalystOpinions",
    "targetMeanPrice",
    "currentPrice",
)
_FLOW_KEYS = (
    "heldPercentInstitutions",
    "heldPercentInsiders",
    "shortPercentOfFloat",
    "shortRatio",
    "sharesShort",
    "averageVolume",
    "averageVolume10days",
)


def to_yf_symbol(symbol: str, market: Market) -> str:
    """把内部品种代码映射为 yfinance 代码。

    - 港股：抽数字 → 长度 ≤4 则 ``zfill(4)`` → 加 ``.HK``
      (``700``/``HKEX:700`` → ``0700.HK``；``07709`` → ``07709.HK``)。
    - 美股：去交易所前缀后原样大写(``aapl`` → ``AAPL``；``brk.b`` → ``BRK.B``)。
    - 其他：原样返回。
    """
    s = (symbol or "").strip()
    if ":" in s:
        s = s.split(":", 1)[1].strip()

    if market is Market.HK:
        import re

        # 已带 .HK 的，规整数字部分
        body = s[:-3] if s.upper().endswith(".HK") else s
        digits = re.sub(r"\D", "", body)
        if not digits:
            return s.upper()
        if len(digits) <= 4:
            digits = digits.zfill(4)
        return f"{digits}.HK"

    if market is Market.US:
        return s.upper()

    return s


def clear_yf_fundamentals_cache(symbol: str | None = None) -> None:
    """清缓存：``symbol=None`` 清全部，否则按 yf 代码清单条。"""
    if symbol is None:
        _CACHE.clear()
        return
    # symbol 可能是原始代码或 yf 代码；尝试两种 key
    _CACHE.pop(symbol, None)
    _CACHE.pop(symbol.upper(), None)


def _empty_ctx(symbol: str, yf_symbol: str, market: Market) -> dict[str, Any]:
    return {
        "symbol": symbol,
        "yf_symbol": yf_symbol,
        "market": market.value,
        "available": False,
        "fundamentals": {},
        "sentiment": {},
        "flow": {},
        "news": [],
    }


def fetch_yf_fundamentals(
    symbol: str,
    market: Market,
    *,
    use_cache: bool = True,
    ttl_seconds: int = _DEFAULT_TTL_S,
    include_news: bool = False,
    news_max_items: int = 3,
) -> dict[str, Any]:
    """抓取 yfinance 基本面/情绪/机构做空字段。

    全程 ``try/except``：缺包、超时、无数据、缺字段都降级，绝不抛。
    返回 dict 始终含 ``symbol``/``yf_symbol``/``market``/``available``。
    """
    yf_symbol = to_yf_symbol(symbol, market)

    if use_cache:
        cached = _CACHE.get(yf_symbol)
        if cached and (time.monotonic() - cached[0]) < ttl_seconds:
            return dict(cached[1])

    ctx = _empty_ctx(symbol, yf_symbol, market)

    try:
        import yfinance as yf
    except ImportError:
        logger.warning("yfinance not installed — fundamental context skipped")
        return ctx

    try:
        ticker = yf.Ticker(yf_symbol)
        info = _safe_info(ticker)
        if not info:
            return ctx

        fundamentals = {k: info.get(k) for k in _FUNDAMENTAL_KEYS}
        sentiment = {k: info.get(k) for k in _SENTIMENT_KEYS}
        flow = {k: info.get(k) for k in _FLOW_KEYS}

        ctx["fundamentals"] = fundamentals
        ctx["sentiment"] = sentiment
        ctx["flow"] = flow
        ctx["available"] = any(v is not None for v in info.values()) if info else False

        if include_news and news_max_items > 0:
            ctx["news"] = _safe_news(ticker, news_max_items)
    except Exception:
        logger.warning("yfinance fetch failed for %s", yf_symbol, exc_info=True)
        return _empty_ctx(symbol, yf_symbol, market)

    if use_cache:
        _CACHE[yf_symbol] = (time.monotonic(), dict(ctx))
    return ctx


def _safe_info(ticker: Any) -> dict[str, Any]:
    """安全读取 ``ticker.info``，失败返回 {}。"""
    try:
        info = ticker.info
        return dict(info) if info else {}
    except Exception:
        return {}


def _safe_news(ticker: Any, max_items: int) -> list[dict[str, Any]]:
    """安全读取 ``ticker.news`` 前 N 条标题+时间，失败返回 []。"""
    try:
        raw = ticker.news or []
    except Exception:
        return []
    out: list[dict[str, Any]] = []
    for item in raw[:max_items]:
        if not isinstance(item, dict):
            continue
        title = item.get("title") or (item.get("content") or {}).get("title")
        published = item.get("providerPublishTime") or item.get("pubDate")
        if title:
            out.append({"title": title, "publishedAt": published})
    return out


def _fmt_market_cap(val: Any, currency: str | None) -> str | None:
    """市值格式化：美元用 B/T，其他用亿/万亿。"""
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    cur = (currency or "").upper()
    if cur in ("USD", ""):
        if abs(n) >= 1e12:
            return f"{n / 1e12:.2f}T {cur}".strip()
        if abs(n) >= 1e9:
            return f"{n / 1e9:.2f}B {cur}".strip()
        if abs(n) >= 1e6:
            return f"{n / 1e6:.2f}M {cur}".strip()
        return f"{n:.0f} {cur}".strip()
    # 港币等：亿/万亿
    if abs(n) >= 1e12:
        return f"{n / 1e12:.2f}万亿"
    if abs(n) >= 1e8:
        return f"{n / 1e8:.2f}亿"
    return f"{n:.0f}"


def _fmt_pct(val: Any) -> str | None:
    try:
        return f"{float(val) * 100:.2f}%"
    except (TypeError, ValueError):
        return None


def _fmt_num(val: Any, digits: int = 2) -> str | None:
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return None


def format_yf_fundamentals_sections(ctx: dict[str, Any]) -> list[tuple[str, str]]:
    """拆为 (标题, 正文) 列表供 GUI；无有效内容返回 []。"""
    if not ctx or not ctx.get("available"):
        return []

    sections: list[tuple[str, str]] = []
    f = ctx.get("fundamentals") or {}
    s = ctx.get("sentiment") or {}
    flow = ctx.get("flow") or {}

    # 估值与基本面
    fund_lines: list[str] = []
    name = f.get("longName")
    sector = f.get("sector")
    industry = f.get("industry")
    if name or sector or industry:
        parts = [p for p in (name, sector, industry) if p]
        fund_lines.append("- " + " · ".join(str(p) for p in parts))
    pe_t = _fmt_num(f.get("trailingPE"))
    pe_f = _fmt_num(f.get("forwardPE"))
    if pe_t or pe_f:
        fund_lines.append(f"- PE(TTM) {pe_t or '—'} · 预期PE {pe_f or '—'}")
    mc = _fmt_market_cap(f.get("marketCap"), f.get("currency"))
    if mc:
        fund_lines.append(f"- 市值 {mc}")
    eps = _fmt_num(f.get("trailingEps"))
    if eps:
        fund_lines.append(f"- EPS(TTM) {eps}")
    rg = _fmt_pct(f.get("revenueGrowth"))
    eg = _fmt_pct(f.get("earningsGrowth"))
    if rg or eg:
        fund_lines.append(f"- 营收增速 {rg or '—'} · 盈利增速 {eg or '—'}")
    roe = _fmt_pct(f.get("returnOnEquity"))
    pm = _fmt_pct(f.get("profitMargins"))
    if roe or pm:
        fund_lines.append(f"- ROE {roe or '—'} · 净利率 {pm or '—'}")
    dy = _fmt_pct(f.get("dividendYield"))
    if dy:
        fund_lines.append(f"- 股息率 {dy}")
    if fund_lines:
        sections.append(("估值与基本面", "\n".join(fund_lines)))

    # 分析师评级
    sent_lines: list[str] = []
    rec = s.get("recommendationKey")
    num = s.get("numberOfAnalystOpinions")
    if rec or num:
        sent_lines.append(f"- 评级 {rec or '—'} · 分析师数 {num or '—'}")
    tgt = _fmt_num(s.get("targetMeanPrice"))
    cur_p = _fmt_num(s.get("currentPrice"))
    if tgt or cur_p:
        sent_lines.append(f"- 平均目标价 {tgt or '—'} · 现价 {cur_p or '—'}")
    if sent_lines:
        sections.append(("分析师评级", "\n".join(sent_lines)))

    # 资金面(机构/做空)
    flow_lines: list[str] = []
    inst = _fmt_pct(flow.get("heldPercentInstitutions"))
    insider = _fmt_pct(flow.get("heldPercentInsiders"))
    if inst or insider:
        flow_lines.append(f"- 机构持股 {inst or '—'} · 内部人持股 {insider or '—'}")
    short_pct = _fmt_pct(flow.get("shortPercentOfFloat"))
    short_ratio = _fmt_num(flow.get("shortRatio"))
    if short_pct or short_ratio:
        flow_lines.append(f"- 做空占流通 {short_pct or '—'} · 做空比率 {short_ratio or '—'}")
    if flow_lines:
        sections.append(("资金面(机构/做空)", "\n".join(flow_lines)))

    # 近期新闻(Phase 2)
    news = ctx.get("news") or []
    news_lines: list[str] = []
    for item in news:
        if not isinstance(item, dict):
            continue
        title = item.get("title")
        if title:
            news_lines.append(f"- {title}")
    if news_lines:
        sections.append(("近期新闻", "\n".join(news_lines)))

    return sections


def format_yf_fundamentals_for_prompt(ctx: dict[str, Any]) -> str:
    """渲染为紧凑 markdown；无任何有效字段返回 ""。"""
    sections = format_yf_fundamentals_sections(ctx)
    if not sections:
        return ""
    lines = ["## 基本面与分析师观点(程序抓取，供参考)", ""]
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip()
