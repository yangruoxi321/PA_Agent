"""港股/美股/A股主力资金流（特大/大/中/小单）via moomoo OpenAPI（OpenD）。

可选依赖 ``moomoo-api`` + 本地 OpenD 网关（默认 ``127.0.0.1:11111``）。
缺包 / OpenD 未启动 / 无行情权限 / 任何异常都静默降级返回 ``None``，
绝不向主流程抛异常。带模块级缓存与超时，避免阻塞分析。
"""

from __future__ import annotations

import concurrent.futures
import logging
import re
import time
from typing import Any, Callable

from pa_agent.context.market_classifier import Market

logger = logging.getLogger(__name__)

_DEFAULT_TTL_S = 360 * 60
# OpenD 本机查询通常 <1s；给足超时覆盖连接握手/冷启动，仍远低于个股 yfinance。
_FETCH_TIMEOUT_S = 12

# 缓存：{moomoo_code: (monotonic_ts, data)}
_CACHE: dict[str, tuple[float, dict[str, Any]]] = {}

# 给 OpenD 阻塞调用套真实超时（SDK 调用本身不支持 timeout）。
_EXECUTOR = concurrent.futures.ThreadPoolExecutor(
    max_workers=2, thread_name_prefix="moomoo-flow"
)

# 有主力资金流的市场（外汇/加密/指数 CFD 无）。
_FLOW_MARKETS = (Market.US, Market.HK, Market.A_SHARE)


def _run_with_timeout(fn: Callable[[], Any], *, timeout: float, default: Any) -> Any:
    fut = _EXECUTOR.submit(fn)
    try:
        return fut.result(timeout=timeout)
    except concurrent.futures.TimeoutError:
        logger.warning("moomoo OpenD call timed out after %ss", timeout)
        return default
    except Exception:  # noqa: BLE001
        logger.debug("moomoo OpenD call failed", exc_info=True)
        return default


def to_moomoo_code(symbol: str, market: Market) -> str | None:
    """内部品种代码 → moomoo 代码（``US.WDC`` / ``HK.00700`` / ``SH.600519``）。

    - 美股：去交易所前缀后大写。
    - 港股：抽数字补足 5 位（moomoo 用 5 位，如 ``HK.00700``）。
    - A 股：6 位数字，按代码推断沪/深 → ``SH.`` / ``SZ.``。
    - 其它：``None``。
    """
    s = (symbol or "").strip()
    if ":" in s:
        s = s.split(":", 1)[1].strip()
    if not s:
        return None

    if market is Market.US:
        return f"US.{s.upper()}"

    if market is Market.HK:
        body = s[:-3] if s.upper().endswith(".HK") else s
        digits = re.sub(r"\D", "", body)
        if not digits:
            return None
        return f"HK.{digits.zfill(5)}"

    if market is Market.A_SHARE:
        digits = re.sub(r"\D", "", s)
        if len(digits) != 6:
            return None
        from pa_agent.data.market_defaults import infer_ashare_tv_exchange

        prefix = "SH" if infer_ashare_tv_exchange(digits) == "SSE" else "SZ"
        return f"{prefix}.{digits}"

    return None


def clear_moomoo_flow_cache(code: str | None = None) -> None:
    if code is None:
        _CACHE.clear()
    else:
        _CACHE.pop(code, None)


def fetch_moomoo_flow(
    symbol: str,
    market: Market,
    *,
    host: str = "127.0.0.1",
    port: int = 11111,
    use_cache: bool = True,
    ttl_seconds: int = _DEFAULT_TTL_S,
) -> dict[str, Any] | None:
    """抓取主力资金分布（特大/大/中/小单净流入）+ 当日累计净流入。

    成功返回 dict（含 ``available=True``）；不适用市场/缺包/未连接/异常→``None``。
    """
    if market not in _FLOW_MARKETS:
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
        logger.info("moomoo flow unavailable for %s (OpenD/权限/超时)", code)
    else:
        logger.info("moomoo flow OK for %s", code)
    return result


def _fetch_blocking(
    code: str, host: str, port: int, *, use_cache: bool
) -> dict[str, Any] | None:
    """实际 OpenD 阻塞抓取，由 :func:`_run_with_timeout` 在子线程执行。"""
    try:
        import moomoo as ft
    except ImportError:
        logger.debug("moomoo-api not installed — moomoo flow skipped")
        return None

    ctx = None
    try:
        ctx = ft.OpenQuoteContext(host=host, port=port)
        ret, dist = ctx.get_capital_distribution(code)
        if ret != ft.RET_OK or dist is None or len(dist) == 0:
            return None
        row = dist.iloc[0].to_dict()

        def _num(key: str) -> float | None:
            v = row.get(key)
            try:
                return float(v)
            except (TypeError, ValueError):
                return None

        dist_d = {
            "in_super": _num("capital_in_super"),
            "out_super": _num("capital_out_super"),
            "in_big": _num("capital_in_big"),
            "out_big": _num("capital_out_big"),
            "in_mid": _num("capital_in_mid"),
            "out_mid": _num("capital_out_mid"),
            "in_small": _num("capital_in_small"),
            "out_small": _num("capital_out_small"),
        }
        if all(v is None for v in dist_d.values()):
            return None

        result: dict[str, Any] = {
            "code": code,
            "available": True,
            "dist": dist_d,
            "update_time": row.get("update_time"),
            "net_in_flow": None,
        }

        # 当日累计净流入（资金流时间序列末点的 in_flow）。失败不影响主结果。
        try:
            ret2, flow = ctx.get_capital_flow(code)
            if ret2 == ft.RET_OK and flow is not None and len(flow) > 0:
                last = flow.iloc[-1].to_dict()
                try:
                    result["net_in_flow"] = float(last.get("in_flow"))
                except (TypeError, ValueError):
                    pass
        except Exception:  # noqa: BLE001
            logger.debug("get_capital_flow failed for %s", code, exc_info=True)

        if use_cache:
            _CACHE[code] = (time.monotonic(), dict(result))
        return result
    except Exception:  # noqa: BLE001 — OpenD 未启动/网络/SDK 任何错误都降级
        logger.debug("moomoo OpenD fetch error for %s", code, exc_info=True)
        return None
    finally:
        if ctx is not None:
            try:
                ctx.close()
            except Exception:  # noqa: BLE001
                pass


def _fmt_amount(val: Any) -> str | None:
    """金额格式化：亿 / 万（带正负号）。"""
    try:
        n = float(val)
    except (TypeError, ValueError):
        return None
    sign = "+" if n >= 0 else "-"
    a = abs(n)
    if a >= 1e8:
        return f"{sign}{a / 1e8:.2f}亿"
    if a >= 1e4:
        return f"{sign}{a / 1e4:.0f}万"
    return f"{sign}{a:.0f}"


def _nets(data: dict[str, Any]) -> dict[str, float | None]:
    d = data.get("dist") or {}

    def net(a: str, b: str) -> float | None:
        x, y = d.get(a), d.get(b)
        if x is None or y is None:
            return None
        return x - y

    return {
        "super": net("in_super", "out_super"),
        "big": net("in_big", "out_big"),
        "mid": net("in_mid", "out_mid"),
        "small": net("in_small", "out_small"),
    }


def format_moomoo_flow_sections(data: dict[str, Any] | None) -> list[tuple[str, str]]:
    """拆为 (标题, 正文) 列表供 GUI；无有效内容返回 []。"""
    if not data or not data.get("available"):
        return []
    n = _nets(data)
    lines: list[str] = []

    main = sum(v for v in (n["super"], n["big"]) if v is not None) if (
        n["super"] is not None or n["big"] is not None
    ) else None
    retail = sum(v for v in (n["mid"], n["small"]) if v is not None) if (
        n["mid"] is not None or n["small"] is not None
    ) else None

    if main is not None:
        sup = _fmt_amount(n["super"]) or "—"
        big = _fmt_amount(n["big"]) or "—"
        flag = "🟢" if main >= 0 else "🔴"
        lines.append(f"- 主力净流入 {_fmt_amount(main)} {flag}（特大 {sup} · 大单 {big}）")
    if retail is not None:
        mid = _fmt_amount(n["mid"]) or "—"
        sml = _fmt_amount(n["small"]) or "—"
        lines.append(f"- 散户净流入 {_fmt_amount(retail)}（中单 {mid} · 小单 {sml}）")

    net_flow = data.get("net_in_flow")
    nf = _fmt_amount(net_flow)
    if nf:
        lines.append(f"- 当日累计净流入 {nf}")

    if not lines:
        return []
    return [("主力资金流(特大/大/中/小单)", "\n".join(lines))]


def format_moomoo_flow_for_prompt(data: dict[str, Any] | None) -> str:
    """渲染为紧凑 markdown；无有效内容返回 ""。"""
    sections = format_moomoo_flow_sections(data)
    if not sections:
        return ""
    title, body = sections[0]
    return f"## {title}（程序抓取，供参考）\n{body}"
