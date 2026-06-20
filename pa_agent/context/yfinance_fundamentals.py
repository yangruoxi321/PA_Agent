"""港股/美股基本面、分析师情绪、机构/做空资金面 (yfinance)。

依赖 ``yfinance`` (可选依赖组 equity)。缺包或任何抓取失败都静默降级，
绝不向主流程抛异常。带模块级缓存避免重复打网。
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable

from pa_agent.context.market_classifier import Market

logger = logging.getLogger(__name__)

# 默认基本面缓存 TTL(秒)，可被 settings.fundamental_cache_ttl_minutes 覆盖
_DEFAULT_TTL_S = 360 * 60
# 单次抓取超时(秒)。yfinance ``.info`` 冷启动/慢网常需 10-25s，过短会误杀；
# 因抓取走后台预取(不阻塞分析)，给足时间更稳妥。
_FETCH_TIMEOUT_S = 30

# 缓存：{yf_symbol: (monotonic_ts, ctx)}
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# 共享线程池：给 yfinance 阻塞调用套真实超时(.info/.history 本身不支持 timeout)。
# 超时后底层线程任其后台跑完，调用方不再等待，避免拖慢分析主流程。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=4, thread_name_prefix="yf-fetch"
)


def _run_with_timeout(fn: Callable[[], Any], *, timeout: float, default: Any) -> Any:
    """在后台线程执行 ``fn``，超时或异常返回 ``default``，绝不抛。"""
    fut = _EXECUTOR.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("yfinance call timed out after %ss", timeout)
        return default
    except Exception:  # noqa: BLE001
        logger.debug("yfinance call failed", exc_info=True)
        return default

# 从 .info 提取的字段分组
_FUNDAMENTAL_KEYS = (
    "longName",
    "sector",
    "industry",
    "currency",
    # 估值
    "trailingPE",
    "forwardPE",
    "priceToBook",
    "priceToSalesTrailing12Months",
    "trailingPegRatio",
    "enterpriseToEbitda",
    "marketCap",
    "enterpriseValue",
    # 盈利与成长
    "trailingEps",
    "forwardEps",
    "revenueGrowth",
    "earningsGrowth",
    "returnOnEquity",
    "grossMargins",
    "operatingMargins",
    "profitMargins",
    # 财务健康
    "totalCash",
    "totalDebt",
    "debtToEquity",
    "currentRatio",
    "freeCashflow",
    # 区间与风险
    "fiftyTwoWeekHigh",
    "fiftyTwoWeekLow",
    "beta",
    "sharesOutstanding",
    "floatShares",
    # 股息
    "dividendYield",
    "dividendRate",
)
_SENTIMENT_KEYS = (
    "recommendationKey",
    "numberOfAnalystOpinions",
    "targetMeanPrice",
    "targetHighPrice",
    "targetLowPrice",
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
        want_news = bool(include_news and news_max_items > 0)
        t0 = time.monotonic()
        result = _run_with_timeout(
            lambda: _build_ctx_blocking(
                yf, symbol, yf_symbol, market, want_news, news_max_items,
                cache_key=yf_symbol, use_cache=use_cache,
            ),
            timeout=_FETCH_TIMEOUT_S,
            default=None,
        )
        elapsed = time.monotonic() - t0
        if result is None:
            # 本次等超时：后台线程仍在抓，成功后会自写缓存，下次(分析完成/再看)命中。
            logger.warning(
                "yfinance equity TIMEOUT for %s (>%ss) - bg fetch continues, "
                "caches on success for next hit",
                yf_symbol,
                _FETCH_TIMEOUT_S,
            )
            return ctx
        if not result.get("available"):
            logger.warning(
                "yfinance equity EMPTY for %s (%.1fs) - Yahoo rate-limit or no fundamentals",
                yf_symbol,
                elapsed,
            )
            return result
        logger.info("yfinance equity OK for %s (%.1fs)", yf_symbol, elapsed)
        return result
    except Exception:
        logger.warning("yfinance fetch failed for %s", yf_symbol, exc_info=True)
        return _empty_ctx(symbol, yf_symbol, market)


def _build_ctx_blocking(
    yf: Any,
    symbol: str,
    yf_symbol: str,
    market: Market,
    want_news: bool,
    news_max_items: int,
    *,
    cache_key: str,
    use_cache: bool,
) -> dict[str, Any]:
    """实际的 yfinance 阻塞抓取，由 :func:`_run_with_timeout` 在子线程执行。

    构建完整 ctx，并在抓到有效数据时**自行写入缓存**——这样即使外层调用已
    超时返回空，后台线程跑完仍会把结果留在缓存里，下次触发直接命中。
    """
    ctx = _empty_ctx(symbol, yf_symbol, market)
    ticker = yf.Ticker(yf_symbol)
    info = _safe_info(ticker)
    if not info:
        return ctx
    ctx["fundamentals"] = {k: info.get(k) for k in _FUNDAMENTAL_KEYS}
    ctx["sentiment"] = {k: info.get(k) for k in _SENTIMENT_KEYS}
    ctx["flow"] = {k: info.get(k) for k in _FLOW_KEYS}
    ctx["available"] = any(v is not None for v in info.values())
    if want_news:
        ctx["news"] = _safe_news(ticker, news_max_items)
    if use_cache and ctx["available"]:
        _CACHE[cache_key] = (time.monotonic(), dict(ctx))
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


def _fmt_count(val: Any) -> str | None:
    """股数等计数：亿 / 万，不带货币。"""
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    if abs(n) >= 1e8:
        return f"{n / 1e8:.2f}亿"
    if abs(n) >= 1e4:
        return f"{n / 1e4:.2f}万"
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
    cur = f.get("currency")

    def _add(lines: list[str], label: str, *parts: str | None) -> None:
        """有任一非空值时追加一行 ``- label v1 · v2…``。"""
        vals = [p for p in parts if p]
        if vals:
            lines.append(f"- {label} " + " · ".join(vals))

    # 估值
    val_lines: list[str] = []
    name = f.get("longName")
    sector = f.get("sector")
    industry = f.get("industry")
    if name or sector or industry:
        parts = [str(p) for p in (name, sector, industry) if p]
        val_lines.append("- " + " · ".join(parts))
    pe_t = _fmt_num(f.get("trailingPE"))
    pe_f = _fmt_num(f.get("forwardPE"))
    peg = _fmt_num(f.get("trailingPegRatio"))
    if pe_t or pe_f or peg:
        val_lines.append(
            f"- PE(TTM) {pe_t or '—'} · 预期PE {pe_f or '—'} · PEG {peg or '—'}"
        )
    pb = _fmt_num(f.get("priceToBook"))
    ps = _fmt_num(f.get("priceToSalesTrailing12Months"))
    ev_eb = _fmt_num(f.get("enterpriseToEbitda"))
    if pb or ps or ev_eb:
        val_lines.append(
            f"- 市净率PB {pb or '—'} · 市销率PS {ps or '—'} · EV/EBITDA {ev_eb or '—'}"
        )
    mc = _fmt_market_cap(f.get("marketCap"), cur)
    ev = _fmt_market_cap(f.get("enterpriseValue"), cur)
    _add(val_lines, "市值", mc, f"企业价值EV {ev}" if ev else None)
    if val_lines:
        sections.append(("估值", "\n".join(val_lines)))

    # 盈利与成长
    grow_lines: list[str] = []
    eps_t = _fmt_num(f.get("trailingEps"))
    eps_f = _fmt_num(f.get("forwardEps"))
    if eps_t or eps_f:
        grow_lines.append(f"- EPS(TTM) {eps_t or '—'} · 预期EPS {eps_f or '—'}")
    rg = _fmt_pct(f.get("revenueGrowth"))
    eg = _fmt_pct(f.get("earningsGrowth"))
    if rg or eg:
        grow_lines.append(f"- 营收增速 {rg or '—'} · 盈利增速 {eg or '—'}")
    roe = _fmt_pct(f.get("returnOnEquity"))
    gm = _fmt_pct(f.get("grossMargins"))
    om = _fmt_pct(f.get("operatingMargins"))
    pm = _fmt_pct(f.get("profitMargins"))
    if roe:
        grow_lines.append(f"- ROE {roe}")
    if gm or om or pm:
        grow_lines.append(
            f"- 毛利率 {gm or '—'} · 经营利润率 {om or '—'} · 净利率 {pm or '—'}"
        )
    if grow_lines:
        sections.append(("盈利与成长", "\n".join(grow_lines)))

    # 财务健康
    fin_lines: list[str] = []
    cash = _fmt_market_cap(f.get("totalCash"), cur)
    debt = _fmt_market_cap(f.get("totalDebt"), cur)
    de = _fmt_num(f.get("debtToEquity"))
    if cash or debt or de:
        fin_lines.append(
            f"- 现金 {cash or '—'} · 总负债 {debt or '—'} · 负债权益比 {de or '—'}"
        )
    cr = _fmt_num(f.get("currentRatio"))
    fcf = _fmt_market_cap(f.get("freeCashflow"), cur)
    if cr or fcf:
        fin_lines.append(f"- 流动比率 {cr or '—'} · 自由现金流 {fcf or '—'}")
    if fin_lines:
        sections.append(("财务健康", "\n".join(fin_lines)))

    # 区间与风险
    range_lines: list[str] = []
    hi = f.get("fiftyTwoWeekHigh")
    lo = f.get("fiftyTwoWeekLow")
    cur_p_raw = s.get("currentPrice")
    hi_s, lo_s = _fmt_num(hi), _fmt_num(lo)
    if hi_s or lo_s:
        line = f"- 52周 {lo_s or '—'} ~ {hi_s or '—'}"
        try:
            if cur_p_raw is not None and hi:
                line += f"(距高 {(float(cur_p_raw) / float(hi) - 1) * 100:+.1f}%)"
        except (TypeError, ValueError, ZeroDivisionError):
            pass
        range_lines.append(line)
    beta = _fmt_num(f.get("beta"))
    if beta:
        range_lines.append(f"- Beta {beta}")
    so = _fmt_count(f.get("sharesOutstanding"))
    fl = _fmt_count(f.get("floatShares"))
    if so or fl:
        range_lines.append(f"- 总股本 {so or '—'} · 流通股 {fl or '—'}")
    if range_lines:
        sections.append(("区间与风险", "\n".join(range_lines)))

    # 股息
    dy = _fmt_pct(f.get("dividendYield"))
    dr = _fmt_num(f.get("dividendRate"))
    if dy or dr:
        sections.append(("股息", f"- 股息率 {dy or '—'} · 每股股息 {dr or '—'}"))

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
    tgt_hi = _fmt_num(s.get("targetHighPrice"))
    tgt_lo = _fmt_num(s.get("targetLowPrice"))
    if tgt_hi or tgt_lo:
        sent_lines.append(f"- 目标价区间 {tgt_lo or '—'} ~ {tgt_hi or '—'}")
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
