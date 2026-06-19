"""量价资金面特征：仅基于已有 K 线计算，不发起任何网络请求。

提供成交量、相对均量、量价背离等"资金面"线索，作为价格行为分析的辅助。
对所有市场(含外汇/加密)均可用，因为只依赖 K 线序列。
"""

from __future__ import annotations

import math
from typing import Any

# 背离判定类型
DIVERGENCE_PRICE_UP_VOL_DOWN = "价涨量缩"
DIVERGENCE_PRICE_DOWN_VOL_UP = "价跌量增"
DIVERGENCE_BOTH_UP = "量价齐升"
DIVERGENCE_BOTH_DOWN = "量价齐跌"
DIVERGENCE_NONE = "无明显"

# 计算量价背离时使用的最近 K 线根数
_DIVERGENCE_LOOKBACK = 3


def _safe_default(avg_window: int) -> dict[str, Any]:
    return {
        "available": False,
        "latest_volume": None,
        "avg_volume_n": None,
        "avg_window": avg_window,
        "rel_volume": None,
        "vol_price_divergence": DIVERGENCE_NONE,
    }


def compute_flow_features(frame: Any, *, avg_window: int = 20) -> dict[str, Any]:
    """基于 *frame* 计算量价资金面特征。

    ``frame.bars`` 为 newest-first(bars[0] 最新已收盘棒)。返回 dict：
    - ``latest_volume``：最新棒成交量
    - ``avg_volume_n``：近 ``avg_window`` 根均量
    - ``rel_volume``：相对均量比(latest / avg)
    - ``vol_price_divergence``：量价背离类型

    数据不足或异常时返回安全默认(``available=False``)，绝不抛异常。
    """
    try:
        bars = getattr(frame, "bars", None) or ()
        n = len(bars)
        if n < 2:
            return _safe_default(avg_window)

        latest = bars[0]
        latest_vol = float(getattr(latest, "volume", 0.0) or 0.0)

        window = min(avg_window, n)
        vols = [float(getattr(b, "volume", 0.0) or 0.0) for b in bars[:window]]
        avg_vol = sum(vols) / len(vols) if vols else 0.0
        rel_vol = (latest_vol / avg_vol) if avg_vol > 0 else None

        divergence = _classify_divergence(bars)

        return {
            "available": True,
            "latest_volume": latest_vol,
            "avg_volume_n": round(avg_vol, 2) if not math.isnan(avg_vol) else None,
            "avg_window": window,
            "rel_volume": round(rel_vol, 3) if rel_vol is not None else None,
            "vol_price_divergence": divergence,
        }
    except Exception:
        return _safe_default(avg_window)


def _classify_divergence(bars: Any) -> str:
    """用最近数根 K 线的收盘价与成交量走向判定量价背离。"""
    lookback = min(_DIVERGENCE_LOOKBACK, len(bars))
    if lookback < 2:
        return DIVERGENCE_NONE

    # bars newest-first；取最近 lookback 根，按时间正序(旧→新)比较趋势
    recent = list(bars[:lookback])[::-1]
    closes = [float(getattr(b, "close", 0.0) or 0.0) for b in recent]
    vols = [float(getattr(b, "volume", 0.0) or 0.0) for b in recent]

    price_up = closes[-1] > closes[0]
    price_down = closes[-1] < closes[0]
    vol_up = vols[-1] > vols[0]
    vol_down = vols[-1] < vols[0]

    if price_up and vol_down:
        return DIVERGENCE_PRICE_UP_VOL_DOWN
    if price_down and vol_up:
        return DIVERGENCE_PRICE_DOWN_VOL_UP
    if price_up and vol_up:
        return DIVERGENCE_BOTH_UP
    if price_down and vol_down:
        return DIVERGENCE_BOTH_DOWN
    return DIVERGENCE_NONE


def format_flow_for_prompt(feat: dict[str, Any]) -> str:
    """渲染量价资金面为紧凑 markdown；无有效数据返回 ""。"""
    if not feat or not feat.get("available"):
        return ""

    lines = ["## 量价资金面(程序基于K线计算，供参考)"]

    latest = feat.get("latest_volume")
    avg = feat.get("avg_volume_n")
    rel = feat.get("rel_volume")
    window = feat.get("avg_window")

    if latest is not None and avg is not None:
        rel_s = f"{rel:.2f}x" if rel is not None else "—"
        lines.append(f"- 最新成交量 {latest:.0f} · 近{window}根均量 {avg:.0f} · 相对均量 {rel_s}")

    divergence = feat.get("vol_price_divergence")
    if divergence and divergence != DIVERGENCE_NONE:
        lines.append(f"- 量价关系：{divergence}")

    if len(lines) == 1:
        return ""
    return "\n".join(lines)
