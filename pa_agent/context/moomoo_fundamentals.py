"""港股/美股/A股深度基本面 via moomoo OpenAPI（OpenD）。

覆盖：公司简介 / 财报核心+增速 / 估值历史分位(PE·PS) / 分析师一致预期 / 营收分部。
比 yfinance 厚得多且自带中文。可选依赖 ``moomoo-api`` + 本地 OpenD。
缺包 / OpenD 未连接 / 无权限 / 任何异常都返回 ``None``，由上层回退 yfinance。
单次复用一条 OpenD 连接拉全部接口；带模块级缓存与超时，绝不阻塞分析。
"""

from __future__ import annotations

import concurrent.futures
import logging
import time
from typing import Any, Callable

from pa_agent.context.market_classifier import Market
from pa_agent.context.moomoo_flow import to_moomoo_code

logger = logging.getLogger(__name__)

_DEFAULT_TTL_S = 360 * 60
# 一条连接顺序拉 5 个接口，给足超时（仍远低于一次分析）。
_FETCH_TIMEOUT_S = 20

_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="moomoo-fund"
)

_FUND_MARKETS = (Market.US, Market.HK, Market.A_SHARE)

# 分析师评级（moomoo rating 1–5）
_RATING_LABEL = {1: "卖出", 2: "减持", 3: "持有", 4: "买入", 5: "强烈买入"}


def _run_with_timeout(fn: Callable[[], Any], *, timeout: float, default: Any) -> Any:
    fut = _EXECUTOR.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("moomoo fundamentals call timed out after %ss", timeout)
        return default
    except Exception:  # noqa: BLE001
        logger.debug("moomoo fundamentals call failed", exc_info=True)
        return default


def clear_moomoo_fundamentals_cache(code: str | None = None) -> None:
    if code is None:
        _CACHE.clear()
    else:
        _CACHE.pop(code, None)


def fetch_moomoo_fundamentals(
    symbol: str,
    market: Market,
    *,
    host: str = "127.0.0.1",
    port: int = 11111,
    use_cache: bool = True,
    ttl_seconds: int = _DEFAULT_TTL_S,
) -> dict[str, Any] | None:
    """拉取 moomoo 深度基本面；不适用/缺包/未连接/异常→``None``（上层回退 yfinance）。"""
    if market not in _FUND_MARKETS:
        return None
    code = to_moomoo_code(symbol, market)
    if not code:
        return None

    if use_cache:
        cached = _CACHE.get(code)
        if cached and (time.monotonic() - cached[0]) < ttl_seconds:
            return dict(cached[1])

    result = _run_with_timeout(
        lambda: _fetch_blocking(code, host, port, use_cache=use_cache),
        timeout=_FETCH_TIMEOUT_S,
        default=None,
    )
    if result is None:
        logger.info("moomoo fundamentals unavailable for %s", code)
    else:
        logger.info("moomoo fundamentals OK for %s", code)
    return result


def _fetch_blocking(
    code: str, host: str, port: int, *, use_cache: bool
) -> dict[str, Any] | None:
    try:
        import moomoo as ft
    except ImportError:
        logger.debug("moomoo-api not installed — moomoo fundamentals skipped")
        return None

    ctx = None
    try:
        ctx = ft.OpenQuoteContext(host=host, port=port)
        out: dict[str, Any] = {
            "code": code,
            "available": False,
            "profile": {},
            "financials": {},
            "metrics": {},
            "snapshot": {},
            "valuation": {},
            "analyst": {},
            "short": {},
            "institution": {},
            "revenue": [],
        }

        # 公司简介（DataFrame: name/value/field_type）
        try:
            ret, df = ctx.get_company_profile(code)
            if ret == ft.RET_OK and df is not None and len(df) > 0:
                out["profile"] = _parse_profile(df)
        except Exception:  # noqa: BLE001
            logger.debug("profile fetch failed for %s", code, exc_info=True)

        # 行情快照（市值/52周/换手/PB/股息率，一接口全有）
        try:
            ret, df = ctx.get_market_snapshot([code])
            if ret == ft.RET_OK and df is not None and len(df) > 0:
                out["snapshot"] = df.iloc[0].to_dict()
        except Exception:  # noqa: BLE001
            logger.debug("snapshot fetch failed for %s", code, exc_info=True)

        # 财报（statement_type=1 利润表, financial_type=10 季度+年度）
        try:
            ret, data = ctx.get_financials_statements(
                code, statement_type=1, financial_type=10, num=4
            )
            if ret == ft.RET_OK and isinstance(data, dict):
                out["financials"] = _parse_financials(data)
        except Exception:  # noqa: BLE001
            logger.debug("financials fetch failed for %s", code, exc_info=True)

        # 主要指标（statement_type=4：ROE/利润率/流动比率/自由现金流比率）
        try:
            ret, data = ctx.get_financials_statements(
                code, statement_type=4, financial_type=10, num=1
            )
            if ret == ft.RET_OK and isinstance(data, dict):
                out["metrics"] = _parse_metrics(data)
        except Exception:  # noqa: BLE001
            logger.debug("metrics fetch failed for %s", code, exc_info=True)

        # 做空（最新一期）。返回 (ret, us_df, hk_df)，按市场取对应非空 df。
        try:
            res = ctx.get_short_interest(code)
            if isinstance(res, tuple) and len(res) == 3 and res[0] == ft.RET_OK:
                for df in (res[1], res[2]):
                    if df is not None and len(df) > 0:
                        out["short"] = df.iloc[0].to_dict()
                        break
        except Exception:  # noqa: BLE001
            logger.debug("short interest fetch failed for %s", code, exc_info=True)

        # 机构持股（最新一期）
        try:
            ret, df = ctx.get_shareholders_institutional(code)
            if ret == ft.RET_OK and df is not None and len(df) > 0:
                out["institution"] = df.iloc[0].to_dict()
        except Exception:  # noqa: BLE001
            logger.debug("institutional fetch failed for %s", code, exc_info=True)

        # 估值历史分位（PE=1, PS=3；interval 3=近1年）
        for vt, label in ((1, "PE"), (3, "PS")):
            try:
                ret, data = ctx.get_valuation_detail(
                    code, valuation_type=vt, interval_type=3
                )
                if ret == ft.RET_OK and isinstance(data, dict):
                    t = data.get("trend") or {}
                    if t.get("current_value") is not None:
                        out["valuation"][label] = {
                            "current": t.get("current_value"),
                            "average": t.get("average_value"),
                            "percentile": t.get("valuation_percentile"),
                            "forward": t.get("forward_value"),
                        }
            except Exception:  # noqa: BLE001
                logger.debug("valuation %s fetch failed for %s", label, code, exc_info=True)

        # 分析师一致预期（dict）
        try:
            ret, data = ctx.get_research_analyst_consensus(code)
            if ret == ft.RET_OK and isinstance(data, dict) and any(data.values()):
                out["analyst"] = dict(data)
        except Exception:  # noqa: BLE001
            logger.debug("analyst fetch failed for %s", code, exc_info=True)

        # 营收分部
        try:
            ret, data = ctx.get_financials_revenue_breakdown(code)
            if ret == ft.RET_OK and isinstance(data, dict):
                out["revenue"] = _parse_revenue(data)
        except Exception:  # noqa: BLE001
            logger.debug("revenue breakdown fetch failed for %s", code, exc_info=True)

        out["available"] = bool(
            out["profile"] or out["financials"] or out["valuation"]
            or out["analyst"] or out["snapshot"] or out["metrics"]
        )
        if not out["available"]:
            return None
        if use_cache:
            _CACHE[code] = (time.monotonic(), dict(out))
        return out
    except Exception:  # noqa: BLE001 — OpenD 未启动/网络/SDK 任何错误都降级
        logger.debug("moomoo fundamentals fetch error for %s", code, exc_info=True)
        return None
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass


# ── 解析 ──────────────────────────────────────────────────────────────────────


def _parse_profile(df: Any) -> dict[str, Any]:
    out: dict[str, Any] = {}
    try:
        for _, row in df.iterrows():
            name = row.get("name") if hasattr(row, "get") else None
            val = row.get("value") if hasattr(row, "get") else None
            if name:
                out[str(name)] = val
    except Exception:  # noqa: BLE001
        return {}
    return out


#: 财报展示科目 → 候选源字段名（按优先级，取最新一期的第一个命中）
_FIN_PLAN = (
    ("总收入", ("总收入", "营业总收入")),
    ("毛利", ("毛利",)),
    ("营业利润", ("营业利润",)),
    ("净利润", ("归属于母公司股东净利润", "净利润")),
)


def _parse_financials(data: dict[str, Any]) -> dict[str, Any]:
    structure = data.get("structure_list") or []
    reports = data.get("report_list") or []
    if not reports:
        return {}
    id_to_name = {e.get("field_id"): (e.get("display_name") or "") for e in structure}
    rpt = reports[0]  # 最新一期
    by_name = {
        id_to_name.get(it.get("field_id"), ""): it for it in (rpt.get("item_list") or [])
    }
    lines: list[dict[str, Any]] = []
    for disp, candidates in _FIN_PLAN:
        for cand in candidates:
            item = by_name.get(cand)
            if item is not None and item.get("data") is not None:
                lines.append(
                    {"name": disp, "value": item.get("data"), "yoy": item.get("yoy")}
                )
                break
    return {
        "period": rpt.get("period_text"),
        "date": rpt.get("date_time_str"),
        "currency": rpt.get("currency_code") or "",
        "items": lines,
    }


#: 主要指标想要的科目（按显示名包含匹配，取最新一期）
_METRIC_TARGETS = (
    "净资产收益率（ROE）",
    "总资产净利率（ROA）",
    "归母净利率",
    "毛利率",
    "流动比率",
    "速动比率",
    "自由现金流与收入比率",
)


def _parse_metrics(data: dict[str, Any]) -> dict[str, Any]:
    structure = data.get("structure_list") or []
    reports = data.get("report_list") or []
    if not reports:
        return {}
    id_to_name = {e.get("field_id"): (e.get("display_name") or "") for e in structure}
    rpt = reports[0]
    by_name = {
        id_to_name.get(it.get("field_id"), ""): it for it in (rpt.get("item_list") or [])
    }
    out: dict[str, Any] = {"period": rpt.get("period_text")}
    for tgt in _METRIC_TARGETS:
        it = by_name.get(tgt)
        if it is not None and it.get("data") is not None:
            out[tgt] = it.get("data")
    return out


def _parse_revenue(data: dict[str, Any]) -> list[dict[str, Any]]:
    bl = data.get("breakdown_list") or []
    if not bl:
        return []
    items = (bl[0] or {}).get("item_list") or []
    out = []
    for it in items[:6]:
        name = it.get("name")
        ratio = it.get("ratio")
        if name and ratio is not None:
            out.append({"name": name, "ratio": ratio})
    return out


# ── 格式化 ────────────────────────────────────────────────────────────────────


def _fmt_amount(val: Any, currency: str = "") -> str | None:
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    cur = (currency or "").upper()
    a = abs(n)
    sign = "-" if n < 0 else ""
    if cur in ("USD", ""):
        if a >= 1e9:
            return f"{sign}{a / 1e9:.2f}B"
        if a >= 1e6:
            return f"{sign}{a / 1e6:.2f}M"
        return f"{sign}{a:.0f}"
    if a >= 1e12:
        return f"{sign}{a / 1e12:.2f}万亿"
    if a >= 1e8:
        return f"{sign}{a / 1e8:.2f}亿"
    return f"{sign}{a:.0f}"


def _fmt_pct(val: Any) -> str | None:
    try:
        return f"{float(val):+.1f}%"
    except (TypeError, ValueError):
        return None


def _fmt_num(val: Any, digits: int = 2) -> str | None:
    try:
        return f"{float(val):.{digits}f}"
    except (TypeError, ValueError):
        return None


def _fmt_pctval(val: Any) -> str | None:
    """值本身已是百分数（如 ROE 85.87 → 85.87%），NaN/非数返回 None。"""
    try:
        f = float(val)
    except (TypeError, ValueError):
        return None
    if f != f:  # NaN
        return None
    return f"{f:.2f}%"


def _fmt_shares(val: Any) -> str | None:
    """股数：亿 / 万 / 百万，NaN/非数返回 None。"""
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    if n != n or n <= 0:
        return None
    if n >= 1e8:
        return f"{n / 1e8:.2f}亿"
    if n >= 1e6:
        return f"{n / 1e6:.2f}M"
    if n >= 1e4:
        return f"{n / 1e4:.2f}万"
    return f"{n:.0f}"


def _mcap(val: Any, currency: str = "") -> str | None:
    """市值：美元 B/T，其它 亿/万亿。NaN/非数返回 None。"""
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    if n != n or n <= 0:
        return None
    cur = (currency or "").upper()
    if cur in ("USD", ""):
        if n >= 1e12:
            return f"{n / 1e12:.2f}T"
        return f"{n / 1e9:.2f}B"
    if n >= 1e12:
        return f"{n / 1e12:.2f}万亿"
    return f"{n / 1e8:.2f}亿"


def format_moomoo_fundamentals_sections(data: dict[str, Any] | None) -> list[tuple[str, str]]:
    """拆为 (标题, 正文) 列表供 GUI；无有效内容返回 []。"""
    if not data or not data.get("available"):
        return []
    sections: list[tuple[str, str]] = []
    p = data.get("profile") or {}
    fin = data.get("financials") or {}
    val = data.get("valuation") or {}
    ana = data.get("analyst") or {}
    rev = data.get("revenue") or []
    snap = data.get("snapshot") or {}
    met = data.get("metrics") or {}
    short = data.get("short") or {}
    inst = data.get("institution") or {}

    # 公司简介
    prof_lines: list[str] = []
    head = [str(p[k]) for k in ("公司名称", "所属市场") if p.get(k)]
    if head:
        prof_lines.append("- " + " · ".join(head))
    sub = []
    if p.get("CEO"):
        sub.append(f"CEO {p['CEO']}")
    if p.get("员工数量"):
        sub.append(f"员工 {p['员工数量']}")
    if p.get("上市日期"):
        sub.append(f"上市 {p['上市日期']}")
    if sub:
        prof_lines.append("- " + " · ".join(sub))
    desc = p.get("公司简介")
    if desc:
        d = str(desc).strip().replace("\n", " ")
        prof_lines.append(f"- {d[:120]}{'…' if len(d) > 120 else ''}")
    if prof_lines:
        sections.append(("公司简介", "\n".join(prof_lines)))

    # 财报核心 + 增速
    if fin.get("items"):
        cur = fin.get("currency", "")
        fl: list[str] = []
        period = fin.get("period") or ""
        for it in fin["items"]:
            amt = _fmt_amount(it.get("value"), cur)
            if amt is None:
                continue
            yoy = _fmt_pct(it.get("yoy"))
            nm = it.get("name", "")
            fl.append(f"- {nm} {amt}" + (f"（YoY {yoy}）" if yoy else ""))
        if fl:
            sections.append((f"财报（{period}）", "\n".join(fl)))

    # 估值历史分位
    if val:
        vl: list[str] = []
        for label in ("PE", "PS"):
            v = val.get(label)
            if not v:
                continue
            cur = _fmt_num(v.get("current"))
            avg = _fmt_num(v.get("average"))
            pct = v.get("percentile")
            fwd = _fmt_num(v.get("forward"))
            extra = []
            if avg:
                extra.append(f"近1年均{avg}")
            if pct is not None:
                tag = "极高" if pct >= 90 else ("偏高" if pct >= 70 else ("极低" if pct <= 10 else ""))
                extra.append(f"分位{pct:.0f}%{('·' + tag) if tag else ''}")
            if fwd:
                extra.append(f"预期{fwd}")
            vl.append(f"- {label} {cur or '—'}" + (f"（{' · '.join(extra)}）" if extra else ""))
        if vl:
            sections.append(("估值历史分位", "\n".join(vl)))

    # 估值现状（市值/PE/PB/股息率，来自快照）
    if snap:
        sl: list[str] = []
        mc = _mcap(snap.get("total_market_val"))
        pe_ttm = _fmt_num(snap.get("pe_ttm_ratio"))
        pb = _fmt_num(snap.get("pb_ratio"))
        if mc or pe_ttm or pb:
            sl.append(
                f"- 市值 {mc or '—'} · PE(TTM) {pe_ttm or '—'} · PB {pb or '—'}"
            )
        dy = snap.get("dividend_ratio_ttm")  # 已是百分数（0.06 = 0.06%）
        dv = snap.get("dividend_ttm")
        if dy not in (None, 0) or dv not in (None, 0):
            dy_s = _fmt_pctval(dy) or "—"
            sl.append(f"- 股息率(TTM) {dy_s} · 每股股息 {_fmt_num(dv) or '—'}")
        if sl:
            sections.append(("估值现状", "\n".join(sl)))

    # 盈利与财务健康（主要指标）
    if met:
        ml: list[str] = []
        roe = _fmt_pctval(met.get("净资产收益率（ROE）"))
        roa = _fmt_pctval(met.get("总资产净利率（ROA）"))
        if roe or roa:
            ml.append(f"- ROE {roe or '—'} · ROA {roa or '—'}")
        gm = _fmt_pctval(met.get("毛利率"))
        nm = _fmt_pctval(met.get("归母净利率"))
        if gm or nm:
            ml.append(f"- 毛利率 {gm or '—'} · 净利率 {nm or '—'}")
        cr = _fmt_num(met.get("流动比率"))
        qr = _fmt_num(met.get("速动比率"))
        fcf = _fmt_pctval(met.get("自由现金流与收入比率"))
        if cr or qr or fcf:
            ml.append(
                f"- 流动比率 {cr or '—'} · 速动比率 {qr or '—'} · 自由现金流/收入 {fcf or '—'}"
            )
        if ml:
            sections.append((f"盈利与财务（{met.get('period') or ''}）", "\n".join(ml)))

    # 区间与风险（52周/振幅/换手，来自快照）
    if snap:
        rl: list[str] = []
        hi = _fmt_num(snap.get("highest52weeks_price"))
        lo = _fmt_num(snap.get("lowest52weeks_price"))
        last = snap.get("last_price")
        if hi or lo:
            line = f"- 52周 {lo or '—'} ~ {hi or '—'}"
            try:
                if last and snap.get("highest52weeks_price"):
                    line += f"（距高 {(float(last) / float(snap['highest52weeks_price']) - 1) * 100:+.1f}%）"
            except (TypeError, ValueError, ZeroDivisionError):
                pass
            rl.append(line)
        tr = _fmt_num(snap.get("turnover_rate"))
        amp = _fmt_num(snap.get("amplitude"))
        if tr or amp:
            rl.append(f"- 换手率 {tr or '—'}% · 振幅 {amp or '—'}%")
        if rl:
            sections.append(("区间与风险", "\n".join(rl)))

    # 做空与机构
    fi: list[str] = []
    sp = short.get("short_percent") if short else None
    ss = short.get("shares_short") if short else None
    dtc = short.get("days_to_cover") if short else None
    if sp is not None or ss is not None:
        fi.append(
            f"- 做空 {_fmt_num(sp) + '%' if sp is not None else '—'}"
            f" · 做空股数 {_fmt_shares(ss) or '—'} · 回补 {_fmt_num(dtc) or '—'}天"
        )
    if inst:
        hp = inst.get("holder_pct")
        iq = inst.get("institution_quantity")
        if hp is not None or iq is not None:
            hp_s = _fmt_num(hp) + "%" if hp is not None else "—"
            fi.append(f"- 机构持股 {hp_s} · 机构数 {int(iq) if iq else '—'} 家")
    if fi:
        sections.append(("做空与机构", "\n".join(fi)))

    # 分析师一致预期
    if ana:
        al: list[str] = []
        rating = ana.get("rating")
        rlabel = _RATING_LABEL.get(rating, "")
        total = ana.get("total")
        if rlabel or total:
            al.append(f"- 一致评级 {rlabel or '—'} · 分析师 {total or '—'} 家")
        hi = _fmt_num(ana.get("highest"))
        avg = _fmt_num(ana.get("average"))
        lo = _fmt_num(ana.get("lowest"))
        if hi or avg or lo:
            al.append(f"- 目标价 低 {lo or '—'} · 均 {avg or '—'} · 高 {hi or '—'}")
        buy, hold, sell = ana.get("buy"), ana.get("hold"), ana.get("sell")
        if buy is not None or hold is not None or sell is not None:
            al.append(
                f"- 评级分布 买入 {buy:.0f}% · 持有 {hold:.0f}% · 卖出 {sell:.0f}%"
                if all(x is not None for x in (buy, hold, sell))
                else "- 评级分布（部分缺失）"
            )
        if al:
            sections.append(("分析师一致预期", "\n".join(al)))

    # 营收分部
    if rev and len(rev) >= 1:
        rl = [f"- {it['name']} {it['ratio']:.1f}%" for it in rev]
        if rl:
            sections.append(("营收分部", "\n".join(rl)))

    return sections


def format_moomoo_fundamentals_for_prompt(data: dict[str, Any] | None) -> str:
    """渲染为紧凑 markdown；无有效内容返回 ""。"""
    sections = format_moomoo_fundamentals_sections(data)
    if not sections:
        return ""
    lines = ["## 基本面（moomoo，程序抓取，供参考）", ""]
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip()
