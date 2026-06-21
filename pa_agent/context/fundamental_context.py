"""统一入口：按市场路由基本面/资金面，拼接宏观，整体降级。

供 prompt 组装器(`build_for_symbol`)与 GUI(`build_sections_for_symbol`)调用。
任何失败都返回安全结果(空串 / 空列表)，永不向调用方抛异常。
"""

from __future__ import annotations

import logging
import threading
from typing import Any

from pa_agent.context import (
    macro_snapshot,
    moomoo_flow,
    moomoo_fundamentals,
    yfinance_fundamentals,
)
from pa_agent.context.flow_features import compute_flow_features, format_flow_for_prompt
from pa_agent.context.market_classifier import Market, classify_market

logger = logging.getLogger(__name__)

# 并行抓取 join 上限(秒)：个股/宏观各自内部已有超时，这里只兜底防卡死。
_GATHER_JOIN_TIMEOUT_S = 35.0

# 基本面块开头标记——校验层据此判断本次 prompt 是否注入了基本面。
FUNDAMENTAL_BLOCK_MARKER = "以下为程序抓取的基本面"

# 基本面分析方法引导语：外置为可编辑 txt（prompt_engineering/基本面分析方法.txt），
# 改文件即生效，无需改代码。文件缺失/损坏/缺标记时回退内置兜底，绝不让主流程崩。
_GUIDANCE_FILENAME = "基本面分析方法.txt"
# 仅防崩占位（不含任何分析方法，方法全在 txt）。保留 marker 供校验层识别注入。
_GUIDANCE_FALLBACK = (
    "> " + FUNDAMENTAL_BLOCK_MARKER + "/资金面/宏观/情绪信息（分析方法见 "
    "基本面分析方法.txt；该文件当前缺失，暂以价格行为为主、基本面仅作交叉验证）。"
)


def _load_guidance() -> str:
    """读取可编辑的基本面分析方法 txt；缺失/损坏/缺标记时回退内置兜底。"""
    try:
        from pa_agent.config.paths import PROMPT_DIR

        text = (PROMPT_DIR / _GUIDANCE_FILENAME).read_text(encoding="utf-8").strip()
        if text and FUNDAMENTAL_BLOCK_MARKER in text:
            return text
        logger.warning(
            "%s 缺少必需标记『%s』，回退内置引导语",
            _GUIDANCE_FILENAME,
            FUNDAMENTAL_BLOCK_MARKER,
        )
    except Exception:  # noqa: BLE001
        logger.warning("%s 读取失败，回退内置引导语", _GUIDANCE_FILENAME, exc_info=True)
    return _GUIDANCE_FALLBACK

# 个股基本面/资金面可用的市场（日韩走 yfinance；日股若开通 moomoo 日本行情权限亦走 moomoo）
_EQUITY_MARKETS = (Market.HK, Market.US, Market.JP, Market.KR)


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


def _gather_parallel(
    symbol: str,
    market: Market,
    *,
    settings: Any,
    use_cache: bool,
    frame: Any,
) -> dict[str, Any]:
    """并行抓取个股(慢)与宏观，互不阻塞；量价特征(瞬时)主线程算。

    返回 ``{"equity_ctx", "macro_snap", "flow_feat", "moomoo_flow"}``，缺项为 None。
    各子任务内部已有超时，join 仅作兜底，绝不抛异常。
    """
    res: dict[str, Any] = {
        "equity_ctx": None,
        "moomoo_fund": None,
        "macro_snap": None,
        "flow_feat": None,
        "moomoo_flow": None,
    }

    def _eq() -> None:
        if market not in _EQUITY_MARKETS:
            return
        # moomoo 深度基本面（公司/财报/估值/盈利财务/区间/做空机构/分析师/营收）。
        # 在线则全用 moomoo（更厚/更快/统一），不再打 yfinance；未连接才回退。
        if _get(settings, "enable_moomoo_fundamentals", False):
            try:
                res["moomoo_fund"] = moomoo_fundamentals.fetch_moomoo_fundamentals(
                    symbol,
                    market,
                    host=str(_get(settings, "moomoo_opend_host", "127.0.0.1")),
                    port=int(_get(settings, "moomoo_opend_port", 11111)),
                    use_cache=use_cache,
                    ttl_seconds=_ttl_seconds(settings),
                )
            except Exception:  # noqa: BLE001
                logger.warning("moomoo fundamentals failed for %s", symbol, exc_info=True)
        want_news = bool(_get(settings, "fundamental_include_news", False))
        # moomoo 命中且不要新闻 → 跳过 yfinance（更快）。要新闻则仍抓（新闻仅 yfinance 有）。
        if res["moomoo_fund"] is not None and not want_news:
            return
        try:
            res["equity_ctx"] = yfinance_fundamentals.fetch_yf_fundamentals(
                symbol,
                market,
                use_cache=use_cache,
                ttl_seconds=_ttl_seconds(settings),
                include_news=want_news,
                news_max_items=int(_get(settings, "fundamental_news_max_items", 3)),
            )
        except Exception:  # noqa: BLE001
            logger.warning("equity fetch failed for %s", symbol, exc_info=True)

    def _macro() -> None:
        if not _get(settings, "fundamental_include_macro", True):
            return
        try:
            res["macro_snap"] = macro_snapshot.fetch_macro_snapshot(
                market, use_cache=use_cache
            )
        except Exception:  # noqa: BLE001
            logger.warning("macro fetch failed for %s", symbol, exc_info=True)

    def _mm() -> None:
        if not _get(settings, "enable_moomoo_flow", False):
            return
        try:
            res["moomoo_flow"] = moomoo_flow.fetch_moomoo_flow(
                symbol,
                market,
                host=str(_get(settings, "moomoo_opend_host", "127.0.0.1")),
                port=int(_get(settings, "moomoo_opend_port", 11111)),
                use_cache=use_cache,
                ttl_seconds=_ttl_seconds(settings),
            )
        except Exception:  # noqa: BLE001
            logger.warning("moomoo flow fetch failed for %s", symbol, exc_info=True)

    threads = [
        threading.Thread(target=_eq, name="ctx-equity", daemon=True),
        threading.Thread(target=_macro, name="ctx-macro", daemon=True),
        threading.Thread(target=_mm, name="ctx-moomoo", daemon=True),
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join(timeout=_GATHER_JOIN_TIMEOUT_S)

    # 量价资金面：纯计算、瞬时，主线程做即可。
    if frame is not None and _get(settings, "fundamental_include_flow", True):
        try:
            res["flow_feat"] = compute_flow_features(
                frame, avg_window=int(_get(settings, "fundamental_flow_avg_window", 20))
            )
        except Exception:  # noqa: BLE001
            logger.warning("flow compute failed for %s", symbol, exc_info=True)
    return res


def build_sections_for_symbol(
    symbol: str,
    *,
    data_source: str | None = None,
    exchange: str | None = None,
    settings: Any = None,
    use_cache: bool = True,
    frame: Any = None,
) -> list[tuple[str, str]]:
    """返回 (标题, 正文) 列表供 GUI 展示。失败返回 []。"""
    if not _get(settings, "enable_fundamental_context", True):
        return []
    sections: list[tuple[str, str]] = []
    try:
        market = classify_market(symbol, data_source, exchange)
        data = _gather_parallel(
            symbol, market, settings=settings, use_cache=use_cache, frame=frame
        )

        # moomoo 在线 → 全用 moomoo（更厚/更快/统一）；离线才回退 yfinance。
        if data["moomoo_fund"] is not None:
            sections.extend(
                moomoo_fundamentals.format_moomoo_fundamentals_sections(data["moomoo_fund"])
            )
            # 新闻仅 yfinance 有 → moomoo 在场时单独补「近期新闻」节。
            if data["equity_ctx"] is not None:
                for title, body in _equity_sections(data["equity_ctx"], settings):
                    if title == "近期新闻":
                        sections.append((title, body))
        elif data["equity_ctx"] is not None:
            sections.extend(_equity_sections(data["equity_ctx"], settings))

        # 量价资金面(基于 frame，所有市场可用)
        feat = data["flow_feat"]
        if feat is not None:
            body = format_flow_for_prompt(feat)
            if body:
                sections.append(("量价资金面", body.split("\n", 1)[-1]))

        # 主力资金流(moomoo：特大/大/中/小单，仅港股/美股/A股且开启时)
        mm = data["moomoo_flow"]
        if mm:
            sections.extend(moomoo_flow.format_moomoo_flow_sections(mm))

        # 宏观
        snap = data["macro_snap"]
        if snap and snap.get("available"):
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
    exchange: str | None = None,
    settings: Any = None,
    use_cache: bool = True,
    frame: Any = None,
) -> str:
    """返回注入用 markdown；任何失败返回 ''；按 settings 开关裁剪；永不抛异常。"""
    try:
        if not _get(settings, "enable_fundamental_context", True):
            return ""

        market = classify_market(symbol, data_source, exchange)
        data = _gather_parallel(
            symbol, market, settings=settings, use_cache=use_cache, frame=frame
        )
        blocks: list[str] = []

        # 个股基本面(仅港股/美股)。moomoo 在线全用 moomoo；离线回退 yfinance。
        if data["moomoo_fund"] is not None:
            block = moomoo_fundamentals.format_moomoo_fundamentals_for_prompt(
                data["moomoo_fund"]
            )
            if block:
                blocks.append(block)
            # 新闻仅 yfinance 有 → moomoo 在场时单独补「近期新闻」块。
            if data["equity_ctx"] is not None:
                news = _render_equity_block(
                    data["equity_ctx"], settings, only_titles=("近期新闻",)
                )
                if news:
                    blocks.append(news)
        elif data["equity_ctx"] is not None:
            block = _render_equity_block(data["equity_ctx"], settings)
            if block:
                blocks.append(block)

        # 量价资金面(基于 frame，所有市场)
        feat = data["flow_feat"]
        if feat is not None:
            block = format_flow_for_prompt(feat)
            if block:
                blocks.append(block)

        # 主力资金流(moomoo：特大/大/中/小单，仅港股/美股/A股且开启时)
        mm = data["moomoo_flow"]
        if mm:
            block = moomoo_flow.format_moomoo_flow_for_prompt(mm)
            if block:
                blocks.append(block)

        # 宏观
        snap = data["macro_snap"]
        if snap:
            block = macro_snapshot.format_macro_for_prompt(snap)
            if block:
                blocks.append(block)

        if not blocks:
            return ""
        return _load_guidance() + "\n\n" + "\n\n".join(blocks)
    except Exception:
        logger.warning("build_for_symbol failed for %s", symbol, exc_info=True)
        return ""


def _render_equity_block(
    ctx: dict[str, Any], settings: Any, *, only_titles: tuple[str, ...] = ()
) -> str:
    """渲染港股/美股块，按开关裁剪情绪/资金面小节；only_titles 时只保留指定小节。"""
    sections = _equity_sections(ctx, settings)
    if only_titles:
        sections = [(t, b) for t, b in sections if t in only_titles]
    if not sections:
        return ""
    # 仅渲染新闻等补充节时换标题，避免与 moomoo「基本面」重复。
    heading = "## 近期新闻(程序抓取，供参考)" if only_titles else "## 基本面与分析师观点(程序抓取，供参考)"
    lines = [heading, ""]
    for title, body in sections:
        lines.append(f"### {title}")
        lines.append(body)
        lines.append("")
    return "\n".join(lines).rstrip()
