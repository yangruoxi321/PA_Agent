"""统一入口：按市场路由基本面/资金面，拼接宏观，整体降级。

供 prompt 组装器(`build_for_symbol`)与 GUI(`build_sections_for_symbol`)调用。
任何失败都返回安全结果(空串 / 空列表)，永不向调用方抛异常。
"""

from __future__ import annotations

import logging
from typing import Any

from pa_agent.context import macro_snapshot, yfinance_fundamentals
from pa_agent.context.flow_features import compute_flow_features, format_flow_for_prompt
from pa_agent.context.market_classifier import Market, classify_market

logger = logging.getLogger(__name__)

_GUIDANCE = (
    "> 以下为程序抓取的基本面/宏观/情绪信息，作为价格行为分析的辅助。"
    "判断冲突时以价格行为为主；基本面/宏观仅用于“确认或背离”的加权"
    "(如技术看多但估值极高+宏观逆风→适度下调信心)。"
)

# 个股基本面/资金面可用的市场
_EQUITY_MARKETS = (Market.HK, Market.US)


def _get(settings: Any, name: str, default: Any) -> Any:
    if settings is None:
        return default
    return getattr(settings, name, default)


def _ttl_seconds(settings: Any) -> int:
    minutes = _get(settings, "fundamental_cache_ttl_minutes", 360)
    try:
        return int(minutes) * 60
    except (TypeError, ValueError):
        return 360 * 60


def build_sections_for_symbol(
    symbol: str,
    *,
    data_source: str | None = None,
    settings: Any = None,
    use_cache: bool = True,
    frame: Any = None,
) -> list[tuple[str, str]]:
    """返回 (标题, 正文) 列表供 GUI 展示。失败返回 []。"""
    if not _get(settings, "enable_fundamental_context", True):
        return []
    sections: list[tuple[str, str]] = []
    try:
        market = classify_market(symbol, data_source)

        if market in _EQUITY_MARKETS:
            ctx = yfinance_fundamentals.fetch_yf_fundamentals(
                symbol,
                market,
                use_cache=use_cache,
                ttl_seconds=_ttl_seconds(settings),
                include_news=bool(_get(settings, "fundamental_include_news", False)),
                news_max_items=int(_get(settings, "fundamental_news_max_items", 3)),
            )
            sections.extend(_equity_sections(ctx, settings))

        # 量价资金面(基于 frame，所有市场可用)
        if frame is not None and _get(settings, "fundamental_include_flow", True):
            feat = compute_flow_features(
                frame, avg_window=int(_get(settings, "fundamental_flow_avg_window", 20))
            )
            body = format_flow_for_prompt(feat)
            if body:
                sections.append(("量价资金面", body.split("\n", 1)[-1]))

        # 宏观
        if _get(settings, "fundamental_include_macro", True):
            snap = macro_snapshot.fetch_macro_snapshot(market, use_cache=use_cache)
            if snap.get("available"):
                lines = [
                    (
                        f"{it.get('name')} {it.get('value')} " f"({it.get('change_pct'):+.2f}%)"
                        if it.get("change_pct") is not None
                        else f"{it.get('name')} {it.get('value')}"
                    )
                    for it in snap.get("items", [])
                ]
                if lines:
                    sections.append(("宏观环境", "\n".join(f"- {x}" for x in lines)))
    except Exception:
        logger.warning("build_sections_for_symbol failed for %s", symbol, exc_info=True)
    return sections


def _equity_sections(ctx: dict[str, Any], settings: Any) -> list[tuple[str, str]]:
    """从 yfinance ctx 取分栏，按开关裁剪情绪/资金面/新闻。"""
    raw = yfinance_fundamentals.format_yf_fundamentals_sections(ctx)
    include_sentiment = bool(_get(settings, "fundamental_include_sentiment", True))
    include_flow = bool(_get(settings, "fundamental_include_flow", True))
    include_news = bool(_get(settings, "fundamental_include_news", False))
    out: list[tuple[str, str]] = []
    for title, body in raw:
        if title == "分析师评级" and not include_sentiment:
            continue
        if title == "资金面(机构/做空)" and not include_flow:
            continue
        if title == "近期新闻" and not include_news:
            continue
        out.append((title, body))
    return out


def build_for_symbol(
    symbol: str,
    *,
    data_source: str | None = None,
    settings: Any = None,
    use_cache: bool = True,
    frame: Any = None,
) -> str:
    """返回注入用 markdown；任何失败返回 ''；按 settings 开关裁剪；永不抛异常。"""
    try:
        if not _get(settings, "enable_fundamental_context", True):
            return ""

        market = classify_market(symbol, data_source)
        blocks: list[str] = []

        # 个股基本面/情绪/机构做空(仅港股/美股)
        if market in _EQUITY_MARKETS:
            try:
                ctx = yfinance_fundamentals.fetch_yf_fundamentals(
                    symbol,
                    market,
                    use_cache=use_cache,
                    ttl_seconds=_ttl_seconds(settings),
                    include_news=bool(_get(settings, "fundamental_include_news", False)),
                    news_max_items=int(_get(settings, "fundamental_news_max_items", 3)),
                )
                block = _render_equity_block(ctx, settings)
                if block:
                    blocks.append(block)
            except Exception:
                logger.warning("equity block failed for %s", symbol, exc_info=True)

        # 量价资金面(基于 frame，所有市场)
        if frame is not None and _get(settings, "fundamental_include_flow", True):
            try:
                feat = compute_flow_features(
                    frame, avg_window=int(_get(settings, "fundamental_flow_avg_window", 20))
                )
                block = format_flow_for_prompt(feat)
                if block:
                    blocks.append(block)
            except Exception:
                logger.warning("flow block failed for %s", symbol, exc_info=True)

        # 宏观
        if _get(settings, "fundamental_include_macro", True):
            try:
                snap = macro_snapshot.fetch_macro_snapshot(market, use_cache=use_cache)
                block = macro_snapshot.format_macro_for_prompt(snap)
                if block:
                    blocks.append(block)
            except Exception:
                logger.warning("macro block failed for %s", symbol, exc_info=True)

        if not blocks:
            return ""
        return _GUIDANCE + "\n\n" + "\n\n".join(blocks)
    except Exception:
        logger.warning("build_for_symbol failed for %s", symbol, exc_info=True)
        return ""


def _render_equity_block(ctx: dict[str, Any], settings: Any) -> str:
    """渲染港股/美股块，按开关裁剪情绪/资金面小节。"""
    sections = _equity_sections(ctx, settings)
    if not sections:
        return ""
    lines = ["## 基本面与分析师观点(程序抓取，供参考)", ""]
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip()
