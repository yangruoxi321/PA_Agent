"""Deterministic detector for *failed breakout* / liquidity-grab candidates.

A **boundary** setup that only fires at the edge of a recognised structure
(前低/缺口边/区间下沿/关键均线 below; 前高/缺口边/区间上沿/关键均线 above). Two
mirror-image directions share one tested core:

* ``below`` — 下破假突破/下沿扫单: a bar pierces a **support** a touch, fails
  (recovers above), long **lower** wick + small body + close in the **upper**
  part, up-follow → boundary limit **LONG**. A bullish reversal of trapped bears.
* ``above`` — 冲高假突破/上沿诱多扫单: a bar pierces a **resistance** a touch,
  fails (recovers below), long **upper** wick + small body + close in the
  **lower** part, down-follow → boundary limit **SHORT**. A bearish reversal of
  trapped bulls (the bull-trap / buy-climax failure).

It is a pure, side-effect-free function so it can be unit-tested against bars.

Iron law (enforced structurally, not by convention)
---------------------------------------------------
The judgement is made **only after the stab bar has closed and a follow bar has
also closed**. With a newest-first ``bars`` list of *closed* bars:

* ``bars[0]`` = the follow-through confirmation bar (newest closed)
* ``bars[1]`` = the candidate stab bar (the one that pierced the level)

Because a confirmed candidate *requires* ``bars[0]`` to be a closed follow bar
**after** the stab, it is structurally impossible to flag a forming /
in-progress stab as a candidate. If the freshest closed bar is itself the stab
(no follow-through closed yet) the verdict is ``pending`` — never ``candidate``.
The caller must never turn ``pending`` into an order.

The function never *originates* a trade beyond tagging the pattern; it can only
**downgrade** (mirrors the fundamentals rule: context can veto, never invent a
signal).
"""
from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Sequence

logger = logging.getLogger(__name__)

SIDE_BELOW = "below"
SIDE_ABOVE = "above"

# ── detected_patterns keys ────────────────────────────────────────────────────
# Lower-boundary (下沿做多)
PATTERN_FAILED_BREAKOUT_BELOW = "failed_breakout_below"
PATTERN_LIQUIDITY_GRAB = "liquidity_grab_candidate"
PATTERN_LIQUIDITY_GRAB_PENDING = "liquidity_grab_pending"
# Upper-boundary (上沿做空)
PATTERN_FAILED_BREAKOUT_ABOVE = "failed_breakout_above"
PATTERN_LIQUIDITY_GRAB_ABOVE = "liquidity_grab_above_candidate"
PATTERN_LIQUIDITY_GRAB_ABOVE_PENDING = "liquidity_grab_above_pending"

_GRAB_TAGS_BELOW = frozenset(
    {PATTERN_FAILED_BREAKOUT_BELOW, PATTERN_LIQUIDITY_GRAB, "liquidity_grab", "扫单候选"}
)
_GRAB_TAGS_ABOVE = frozenset(
    {PATTERN_FAILED_BREAKOUT_ABOVE, PATTERN_LIQUIDITY_GRAB_ABOVE, "诱多扫单候选"}
)
_GRAB_TAGS_ALL = _GRAB_TAGS_BELOW | _GRAB_TAGS_ABOVE | {
    PATTERN_LIQUIDITY_GRAB_PENDING,
    PATTERN_LIQUIDITY_GRAB_ABOVE_PENDING,
}

# Regimes in which a failed boundary breakout is tradable. Spike is excluded (too
# violent); the opposing Always-In is vetoed separately (per side).
_DEFAULT_ALLOWED_REGIMES: frozenset[str] = frozenset(
    {
        "trading_range",
        "trending_tr",
        "micro_channel",
        "tight_channel",
        "normal_channel",
        "broad_channel",
    }
)


@dataclass(frozen=True)
class GrabConfig:
    """Thresholds for the failed-breakout detector (all ratios are fractions)."""

    #: Pierce depth must be <= this * candidate bar range — "只刺穿一点点".
    max_pierce_range_frac: float = 0.6
    #: Reversal wick (lower for below / upper for above) must be >= this * range.
    min_lower_wick_frac: float = 0.45
    #: Body must be <= this * candidate bar range — 实体小.
    max_body_frac: float = 0.5
    #: Close must sit at/beyond this position toward the reversal side (0..1).
    #: below: close_pos >= this; above: close_pos <= 1-this.
    min_close_pos: float = 0.5
    #: Close position considered "ideal" (upper/lower third) — full confidence.
    ideal_close_pos: float = 0.66
    #: Close back across the level must happen within this many bars of the stab.
    recover_within: int = 2
    #: Stab volume >= avg(window) * this to count as a stop-run volume spike.
    vol_spike_mult: float = 1.3
    #: Bars used to compute the average volume baseline.
    vol_avg_window: int = 20
    #: §6.3/§14 boundary gate. below: pierced support pos <= this; above: pierced
    #: resistance pos >= 1-this. 0.40 ≈ outer third + tolerance (middle third
    #: spans 0.333..0.667). position = (level - window_low)/(window_high-window_low).
    lower_boundary_max_pos: float = 0.40
    #: Max confidence (0..30) this setup alone justifies (existing ±30 scale).
    base_confidence: int = 30
    #: Confidence ceiling when volume is tick-based / missing (criterion ④ unreliable).
    tick_volume_cap: int = 15
    #: Penalty when the volume pattern is real but not spike-then-absorb.
    weak_volume_penalty: int = 8
    #: Tolerance (as a fraction of candidate bar range) for "closed back across level".
    recover_tolerance_frac: float = 0.1
    allowed_regimes: frozenset[str] = _DEFAULT_ALLOWED_REGIMES


@dataclass(frozen=True)
class GrabVerdict:
    """Result of :func:`detect_failed_breakout`.

    ``status``:
        * ``candidate``       – all hard criteria met, follow-through confirmed.
        * ``pending``         – the freshest closed bar is a valid stab but no
                                 follow bar has closed yet (iron law: not an order).
        * ``downgraded``      – a stab exists but at least one hard criterion fails.
        * ``not_applicable``  – no stab / no level pierced / not enough bars.
    """

    status: str
    side: str = SIDE_BELOW
    direction: str = "long"  # "long" for below, "short" for above
    pierced_level: float | None = None
    candidate_seq: int | None = None
    failed_criteria: tuple[str, ...] = field(default_factory=tuple)
    passed_criteria: tuple[str, ...] = field(default_factory=tuple)
    confidence_cap: int = 0
    volume_quality: str = "real"  # "real" | "tick_or_missing" | "unavailable"
    notes: tuple[str, ...] = field(default_factory=tuple)

    @property
    def is_candidate(self) -> bool:
        return self.status == "candidate"


# ── Bar geometry helpers ──────────────────────────────────────────────────────


@dataclass(frozen=True)
class _Metrics:
    seq: int
    open: float
    high: float
    low: float
    close: float
    volume: float
    rng: float
    body: float
    lower_wick: float
    upper_wick: float
    close_pos: float  # 0 at low, 1 at high


def _metrics(bar: Any) -> _Metrics:
    o = float(bar.open)
    h = float(bar.high)
    low = float(bar.low)
    c = float(bar.close)
    rng = max(h - low, 0.0)
    body = abs(c - o)
    lower_wick = min(o, c) - low
    upper_wick = h - max(o, c)
    close_pos = (c - low) / rng if rng > 0 else 0.0
    return _Metrics(
        seq=int(getattr(bar, "seq", 0)),
        open=o,
        high=h,
        low=low,
        close=c,
        volume=float(getattr(bar, "volume", 0.0) or 0.0),
        rng=rng,
        body=body,
        lower_wick=max(lower_wick, 0.0),
        upper_wick=max(upper_wick, 0.0),
        close_pos=close_pos,
    )


def _nearest_pierced_level(
    stab: _Metrics, levels: Sequence[float], *, side: str, tol: float
) -> float | None:
    """Return the level the bar pierced by the smallest margin.

    below: support with ``low < level`` (wick went under), nearest above the low.
    above: resistance with ``high > level`` (wick went over), nearest below the high.
    """
    if side == SIDE_BELOW:
        pierced = [lvl for lvl in levels if lvl is not None and stab.low < (lvl + tol) and lvl > stab.low]
        if not pierced:
            return None
        return min(pierced, key=lambda lvl: lvl - stab.low)
    pierced = [lvl for lvl in levels if lvl is not None and stab.high > (lvl - tol) and lvl < stab.high]
    if not pierced:
        return None
    return min(pierced, key=lambda lvl: stab.high - lvl)


def _coerce_levels(levels: Sequence[Any]) -> list[float]:
    out: list[float] = []
    for raw in levels or []:
        try:
            # structure_levels stores text like "3421.5" or "3420-3425"; take the
            # numeric mean of any numbers present.
            nums = [float(m) for m in re.findall(r"\d+(?:\.\d+)?", str(raw))]
            if nums:
                out.append(sum(nums) / len(nums))
        except (TypeError, ValueError):
            continue
    return out


def _avg_volume(metrics: Sequence[_Metrics], *, window: int, exclude_idx: int) -> float:
    vols = [
        m.volume
        for i, m in enumerate(metrics)
        if i != exclude_idx and m.volume > 0
    ][:window]
    if not vols:
        return 0.0
    return sum(vols) / len(vols)


def _classify_volume(
    metrics: Sequence[_Metrics],
    *,
    stab_idx: int,
    follow_idx: int,
    cfg: GrabConfig,
    volume_reliable: bool | None,
) -> tuple[str, bool, list[str]]:
    """Return (volume_quality, spike_then_absorb, notes). Direction-agnostic."""
    notes: list[str] = []
    stab = metrics[stab_idx]
    follow = metrics[follow_idx]
    all_zero = all(m.volume <= 0 for m in metrics)
    if all_zero:
        notes.append("成交量全为 0 / 缺失，条件④（放量→缩量吸收）无法判定")
        return "unavailable", False, notes
    if volume_reliable is False:
        notes.append("成交量为 tick volume（MT5/外汇/黄金），条件④仅供参考、置信度封顶")
        avg = _avg_volume(metrics, window=cfg.vol_avg_window, exclude_idx=stab_idx)
        spike = avg > 0 and stab.volume >= avg * cfg.vol_spike_mult
        absorb = follow.volume < stab.volume
        if spike and absorb:
            notes.append("（tick 量形态：刺穿放量后缩量，方向吻合但不作硬条件）")
        return "tick_or_missing", (spike and absorb), notes
    avg = _avg_volume(metrics, window=cfg.vol_avg_window, exclude_idx=stab_idx)
    spike = avg > 0 and stab.volume >= avg * cfg.vol_spike_mult
    absorb = follow.volume < stab.volume
    if not spike:
        notes.append("刺穿未明显放量（条件④偏弱），置信度下调")
    elif not absorb:
        notes.append("放量后跟随棒未缩量（吸收不明确），置信度下调")
    return "real", (spike and absorb), notes


# ── Core detector ──────────────────────────────────────────────────────────────


def _evaluate_stab(
    metrics: Sequence[_Metrics],
    *,
    side: str,
    stab_idx: int,
    follow_idx: int | None,
    levels: Sequence[float],
    regime: str,
    always_in: str,
    fundamental_crack: bool,
    cfg: GrabConfig,
    volume_reliable: bool | None,
) -> GrabVerdict | None:
    """Evaluate the bar at ``stab_idx`` as a failed-breakout stab on ``side``.

    Returns ``None`` when this bar did not even pierce a level (caller falls
    through). ``follow_idx`` is the (newer) follow bar, or ``None`` (→ pending).
    """
    is_below = side == SIDE_BELOW
    direction = "long" if is_below else "short"
    stab = metrics[stab_idx]
    if stab.rng <= 0:
        return None

    tol = stab.rng * cfg.recover_tolerance_frac
    level = _nearest_pierced_level(stab, levels, side=side, tol=tol)
    if level is None:
        return None

    failed: list[str] = []
    passed: list[str] = []
    notes: list[str] = []

    # ① pierced an existing level by only a little
    pierce_depth = (level - stab.low) if is_below else (stab.high - level)
    if pierce_depth <= cfg.max_pierce_range_frac * stab.rng:
        passed.append("①刺穿已识别" + ("支撑" if is_below else "阻力") + "且仅一点点")
    else:
        failed.append("①刺穿过深（更像真破，非扫单）")

    # ⑦ §6.3/§14: the defended level must be an OUTER-boundary level, not mid-range.
    window_low = min(m.low for m in metrics)
    window_high = max(m.high for m in metrics)
    range_height = window_high - window_low
    if range_height > 0:
        level_pos = (level - window_low) / range_height
        if is_below:
            pos_ok = level_pos <= cfg.lower_boundary_max_pos
            ok_text, bad_text = "⑦支撑位于结构下沿（下1/3）", "⑦不在结构下沿（中间1/3不交易，§6.3/§14）"
        else:
            pos_ok = level_pos >= (1.0 - cfg.lower_boundary_max_pos)
            ok_text, bad_text = "⑦阻力位于结构上沿（上1/3）", "⑦不在结构上沿（中间1/3不交易，§6.3/§14）"
        passed.append(ok_text) if pos_ok else failed.append(bad_text)
    else:
        failed.append("⑦无法确定区间位置（区间高度为0）")

    # ③ shape: long reversal wick + small body + close toward reversal side
    wick = stab.lower_wick if is_below else stab.upper_wick
    close_pos_ok = (
        stab.close_pos >= cfg.min_close_pos
        if is_below
        else stab.close_pos <= (1.0 - cfg.min_close_pos)
    )
    shape_ok = (
        wick >= cfg.min_lower_wick_frac * stab.rng
        and stab.body <= cfg.max_body_frac * stab.rng
        and close_pos_ok
    )
    if shape_ok:
        passed.append("③长" + ("下影+小实体+收上半部" if is_below else "上影+小实体+收下半部"))
    else:
        failed.append("③K线形态不符（影线/实体/收盘位置）")

    # ② recovered back across the level within recover_within bars (incl. same bar)
    def _recovered(m: _Metrics) -> bool:
        return m.close >= level - tol if is_below else m.close <= level + tol

    recover_ok = _recovered(stab)
    if not recover_ok and follow_idx is not None:
        for j in range(max(follow_idx, 0), stab_idx):
            if (stab_idx - j) <= cfg.recover_within and _recovered(metrics[j]):
                recover_ok = True
                break
    if recover_ok:
        passed.append("②收回" + ("支撑上方（下破失败）" if is_below else "阻力下方（上破失败）"))
    else:
        failed.append("②未能收回" + ("支撑上方" if is_below else "阻力下方"))

    # ⑤ regime: range / opposing structure, not the same-direction Always-In
    regime_norm = (regime or "").strip().lower()
    ai_norm = (always_in or "neutral").strip().lower()
    forbidden_ai = "short" if is_below else "long"
    if ai_norm == forbidden_ai:
        failed.append(
            "⑤大背景 Always-In-"
            + ("Short（下破更可能为真）" if is_below else "Long（上破更可能为真）")
        )
    elif regime_norm in cfg.allowed_regimes:
        passed.append("⑤区间/" + ("上升" if is_below else "下降") + "结构背景")
    else:
        failed.append(f"⑤背景不支持（regime={regime_norm or 'unknown'}）")

    # ④ volume (soft: caps confidence, never blocks alone)
    volume_quality, vol_pattern_ok, vol_notes = _classify_volume(
        metrics,
        stab_idx=stab_idx,
        follow_idx=follow_idx if follow_idx is not None else stab_idx,
        cfg=cfg,
        volume_reliable=volume_reliable,
    )
    notes.extend(vol_notes)
    if volume_quality == "real" and vol_pattern_ok:
        passed.append("④刺穿放量后缩量被吸收")

    # ⑥ follow-through: an opposing-direction bar closed after the stab
    follow_ok = False
    if follow_idx is not None:
        fol = metrics[follow_idx]
        if is_below:
            follow_ok = fol.close > fol.open and fol.close > stab.close
        else:
            follow_ok = fol.close < fol.open and fol.close < stab.close
    if follow_idx is None:
        notes.append("尚无收盘跟随棒——按铁律只能 pending，不得据此下单")
    elif follow_ok:
        passed.append("⑥次根向" + ("上" if is_below else "下") + "跟随确认")
    else:
        failed.append("⑥次根无向" + ("上" if is_below else "下") + "跟随")

    # ── Confidence cap ────────────────────────────────────────────────────────
    cap = cfg.base_confidence
    if volume_quality != "real":
        cap = min(cap, cfg.tick_volume_cap)
    elif not vol_pattern_ok:
        cap = max(0, cap - cfg.weak_volume_penalty)
    ideal_ok = (
        stab.close_pos >= cfg.ideal_close_pos
        if is_below
        else stab.close_pos <= (1.0 - cfg.ideal_close_pos)
    )
    if not ideal_ok:
        cap = max(0, cap - 5)

    # ── Fundamental veto (downgrade only) ─────────────────────────────────────
    if fundamental_crack:
        failed.append(
            "基本面裂缝（"
            + ("看空催化/指引转坏" if is_below else "看多催化/指引转好")
            + "）→ 否决，"
            + ("下破" if is_below else "上破")
            + "更可能为真"
        )
        notes.append("context_assessment.diverges + 反向催化 → 扫单候选被否决")

    # ── Status resolution (iron law) ──────────────────────────────────────────
    hard_failed = [
        c
        for c in failed
        if c[0] in {"①", "②", "③", "⑤", "⑥", "⑦"} or c.startswith("基本面")
    ]

    if follow_idx is None:
        non_follow_hard = [c for c in hard_failed if not c.startswith("⑥")]
        status = "pending" if not non_follow_hard else "downgraded"
        cap = 0
    elif hard_failed:
        status = "downgraded"
    else:
        status = "candidate"

    return GrabVerdict(
        status=status,
        side=side,
        direction=direction,
        pierced_level=level,
        candidate_seq=stab.seq,
        failed_criteria=tuple(failed),
        passed_criteria=tuple(passed),
        confidence_cap=cap if status == "candidate" else (0 if status == "pending" else cap),
        volume_quality=volume_quality,
        notes=tuple(notes),
    )


def detect_failed_breakout(
    bars: Sequence[Any],
    levels: Sequence[Any],
    *,
    side: str,
    regime: str,
    always_in: str = "neutral",
    fundamental_crack: bool = False,
    volume_reliable: bool | None = None,
    config: GrabConfig | None = None,
) -> GrabVerdict:
    """Classify a failed-breakout / liquidity-grab candidate on ``side``.

    Args:
        bars: Newest-first sequence of **closed** ``KlineBar``-like objects.
            ``bars[0]`` is the newest closed bar. Forming bars must not be passed.
        levels: Code-side S/R from ``structure_levels`` (numbers or text like
            ``"3420-3425"``). below → support_levels; above → resistance_levels.
        side: ``"below"`` (long, pierce support) or ``"above"`` (short, pierce
            resistance).
        regime: Stage-1 ``cycle_position``.
        always_in: Stage-1 ``bar_analysis.always_in``.
        fundamental_crack: True when ``context_assessment`` diverges with a
            genuine opposing catalyst → the candidate is vetoed (downgraded).
        volume_reliable: ``True`` real volume, ``False`` tick volume
            (MT5/forex/gold), ``None`` to infer.
        config: Threshold overrides.
    """
    cfg = config or GrabConfig()
    direction = "long" if side == SIDE_BELOW else "short"
    closed = [b for b in bars if getattr(b, "closed", True)]
    if len(closed) < 2:
        return GrabVerdict(status="not_applicable", side=side, direction=direction,
                           notes=("收盘K线不足，无法判定",))

    metrics = [_metrics(b) for b in closed]
    coerced = _coerce_levels(levels)
    if not coerced:
        return GrabVerdict(
            status="not_applicable", side=side, direction=direction,
            volume_quality="real" if volume_reliable is not False else "tick_or_missing",
            notes=(("无已识别" + ("支撑" if side == SIDE_BELOW else "阻力")
                    + "结构，拒绝用随机影线硬凑（条件①缺失）"),),
        )

    # Primary: stab = bars[1], follow = bars[0] (iron law: follow already closed).
    primary = _evaluate_stab(
        metrics, side=side, stab_idx=1, follow_idx=0, levels=coerced, regime=regime,
        always_in=always_in, fundamental_crack=fundamental_crack, cfg=cfg,
        volume_reliable=volume_reliable,
    )
    if primary is not None and primary.status in {"candidate", "downgraded"}:
        return primary

    # Secondary: freshest closed bar is itself a fresh stab with no follow yet → pending.
    fresh = _evaluate_stab(
        metrics, side=side, stab_idx=0, follow_idx=None, levels=coerced, regime=regime,
        always_in=always_in, fundamental_crack=fundamental_crack, cfg=cfg,
        volume_reliable=volume_reliable,
    )
    if fresh is not None:
        return fresh
    if primary is not None:
        return primary

    return GrabVerdict(
        status="not_applicable", side=side, direction=direction,
        volume_quality="real" if volume_reliable is not False else "tick_or_missing",
        notes=(("最近K线未刺穿任何已识别"
                + ("支撑，非下破假突破" if side == SIDE_BELOW else "阻力，非冲高假突破")),),
    )


def detect_failed_breakout_below(
    bars: Sequence[Any], support_levels: Sequence[Any], **kwargs: Any
) -> GrabVerdict:
    """Lower-boundary failed breakout (long). See :func:`detect_failed_breakout`."""
    return detect_failed_breakout(bars, support_levels, side=SIDE_BELOW, **kwargs)


def detect_failed_breakout_above(
    bars: Sequence[Any], resistance_levels: Sequence[Any], **kwargs: Any
) -> GrabVerdict:
    """Upper-boundary failed breakout (short). See :func:`detect_failed_breakout`."""
    return detect_failed_breakout(bars, resistance_levels, side=SIDE_ABOVE, **kwargs)


# ── Pipeline guard (downgrade-only, mirrors the fundamentals rule) ─────────────


def _fundamental_crack(stage1: dict[str, Any], side: str) -> bool:
    """True when context_assessment shows a material divergence opposing the setup.

    below (long setup): a bearish catalyst (confidence_adjustment <= -10) makes a
    lower breakout more likely real → veto. above (short setup): a bullish
    catalyst (>= +10) makes an upper breakout more likely real → veto. Pure noise
    (small adjustment) does not interfere.
    """
    ca = stage1.get("context_assessment")
    if not isinstance(ca, dict):
        return False
    if str(ca.get("stance", "")).strip().lower() != "diverges":
        return False
    adj = ca.get("confidence_adjustment")
    try:
        if adj is None:
            return False
        adj = float(adj)
    except (TypeError, ValueError):
        return False
    return adj <= -10 if side == SIDE_BELOW else adj >= 10


def _volume_reliable_for_symbol(symbol: str) -> bool | None:
    """tick volume (forex/metal/crypto via MT5) → False; equities → None (infer)."""
    try:
        from pa_agent.context.market_classifier import _is_non_equity_symbol

        if _is_non_equity_symbol(symbol or ""):
            return False
    except Exception:  # noqa: BLE001
        pass
    return None


_STATUS_RANK = {"candidate": 3, "pending": 2, "downgraded": 1, "not_applicable": 0}

_SIDE_TAGS = {
    SIDE_BELOW: {
        "candidate": (PATTERN_FAILED_BREAKOUT_BELOW, PATTERN_LIQUIDITY_GRAB),
        "pending": PATTERN_LIQUIDITY_GRAB_PENDING,
    },
    SIDE_ABOVE: {
        "candidate": (PATTERN_FAILED_BREAKOUT_ABOVE, PATTERN_LIQUIDITY_GRAB_ABOVE),
        "pending": PATTERN_LIQUIDITY_GRAB_ABOVE_PENDING,
    },
}


def guard_failed_breakout(stage1: dict[str, Any], kline_frame: Any) -> GrabVerdict | None:
    """Reconcile the LLM's grab tags (both sides) with the deterministic detector.

    Runs the below (support) and above (resistance) detectors, picks the most
    actionable, and edits ``stage1`` in place: confirms + tags a clean candidate
    (attaching a ``liquidity_grab`` block with side/direction), marks a fresh stab
    pending, or strips an unsupported LLM tag (downgrade only — never adds an
    order). Returns the chosen verdict (or ``None`` when there are no bars).
    """
    if kline_frame is None:
        return None
    bars = getattr(kline_frame, "bars", None)
    if not bars:
        return None

    patterns = [str(p).strip().lower() for p in (stage1.get("detected_patterns") or [])]
    llm_tagged = any(p in _GRAB_TAGS_ALL for p in patterns)

    regime = str(stage1.get("cycle_position", "") or "")
    bar_analysis = stage1.get("bar_analysis") if isinstance(stage1.get("bar_analysis"), dict) else {}
    always_in = str((bar_analysis or {}).get("always_in", "neutral") or "neutral")
    symbol = str(getattr(kline_frame, "symbol", "") or "")
    vol_reliable = _volume_reliable_for_symbol(symbol)

    verdicts = {
        SIDE_BELOW: detect_failed_breakout(
            bars, stage1.get("support_levels") or [], side=SIDE_BELOW, regime=regime,
            always_in=always_in, fundamental_crack=_fundamental_crack(stage1, SIDE_BELOW),
            volume_reliable=vol_reliable,
        ),
        SIDE_ABOVE: detect_failed_breakout(
            bars, stage1.get("resistance_levels") or [], side=SIDE_ABOVE, regime=regime,
            always_in=always_in, fundamental_crack=_fundamental_crack(stage1, SIDE_ABOVE),
            volume_reliable=vol_reliable,
        ),
    }

    # Pick the most actionable side (candidate > pending > downgraded); tie → higher cap.
    chosen = max(
        verdicts.values(),
        key=lambda v: (_STATUS_RANK.get(v.status, 0), v.confidence_cap),
    )

    if not llm_tagged and chosen.status != "candidate":
        return chosen

    out_patterns: list[str] = [p for p in patterns if p not in _GRAB_TAGS_ALL]

    if chosen.status == "candidate":
        for tag in _SIDE_TAGS[chosen.side]["candidate"]:
            if tag not in out_patterns:
                out_patterns.append(tag)
        stage1["liquidity_grab"] = {
            "status": "candidate",
            "side": chosen.side,
            "direction": chosen.direction,
            "pierced_level": chosen.pierced_level,
            "candidate_seq": chosen.candidate_seq,
            "confidence_cap": chosen.confidence_cap,
            "volume_quality": chosen.volume_quality,
            "passed": list(chosen.passed_criteria),
            "notes": list(chosen.notes),
        }
        logger.info(
            "failed_breakout_%s confirmed @%s level=%s cap=%s vol=%s",
            chosen.side, chosen.candidate_seq, chosen.pierced_level,
            chosen.confidence_cap, chosen.volume_quality,
        )
    elif chosen.status == "pending":
        pending_tag = _SIDE_TAGS[chosen.side]["pending"]
        if pending_tag not in out_patterns:
            out_patterns.append(pending_tag)
        stage1["liquidity_grab"] = {
            "status": "pending",
            "side": chosen.side,
            "direction": chosen.direction,
            "pierced_level": chosen.pierced_level,
            "notes": ["尚无收盘跟随棒，按铁律不得据此下单", *chosen.notes],
        }
        logger.info("failed_breakout_%s pending (no follow-through) — tag stripped", chosen.side)
    else:  # downgraded / not_applicable → strip LLM tag, record reason
        stage1["liquidity_grab"] = {
            "status": chosen.status,
            "side": chosen.side,
            "direction": chosen.direction,
            "failed": list(chosen.failed_criteria),
            "notes": list(chosen.notes),
        }
        if llm_tagged:
            logger.info(
                "failed_breakout downgraded (%s): %s",
                chosen.side, "; ".join(chosen.failed_criteria) or chosen.status,
            )

    stage1["detected_patterns"] = out_patterns
    return chosen


# Backwards-compatible alias (now bidirectional).
guard_failed_breakout_below = guard_failed_breakout
