"""单元测试：prompt 组装器多维上下文注入 (需求 1, 7, 10.4)。"""

from __future__ import annotations

from pathlib import Path

import pytest

from pa_agent.ai.prompt_assembler import PromptAssembler
from pa_agent.data.base import IndicatorBundle, KlineBar, KlineFrame
from pa_agent.records.schema import AnalysisRecord, RecordMeta

pytestmark = pytest.mark.unit

_MARKER = "【FUND-CONTEXT-MARKER】基本面注入测试"


def _make_frame(n: int = 8) -> KlineFrame:
    bars = tuple(
        KlineBar(
            seq=i + 1,
            ts_open=float(1_700_000_000 - i * 3600),
            open=2600.0 + i,
            high=2610.0 + i,
            low=2590.0 + i,
            close=2605.0 + i,
            volume=1000.0 + i,
            closed=(i != 0),
        )
        for i in range(n)
    )
    indicators = IndicatorBundle(
        ema20=tuple(2600.0 + i for i in range(n)),
        atr14=tuple(5.0 for _ in range(n)),
    )
    return KlineFrame(
        symbol="AAPL",
        timeframe="1h",
        bars=bars,
        indicators=indicators,
        snapshot_ts_local_ms=1_700_000_000_000,
    )


def _write_prompt_files(tmp_path: Path) -> None:
    for fname in [
        "提示词大纲_人设与思维方式.txt",
        "市场诊断框架.txt",
        "二元决策.txt",
        "二元决策_闸门.txt",
        "文件16-K线信号识别.txt",
        "逐棒分析检查单.txt",
        "文件17-止损和止盈与仓位管理.txt",
    ]:
        (tmp_path / fname).write_text(f"[CONTENT OF {fname}]", encoding="utf-8")


class _FakeProvider:
    def __init__(self, text: str):
        self._text = text
        self.last_exchange: str | None = "unset"

    def build_for_symbol(self, symbol, *, exchange=None, settings=None, frame=None):
        self.last_exchange = exchange
        return self._text


class _BoomProvider:
    def build_for_symbol(self, symbol, *, exchange=None, settings=None, frame=None):
        raise RuntimeError("boom")


class _Settings:
    enable_fundamental_context = True


def _make_record(frame: KlineFrame) -> AnalysisRecord:
    meta = RecordMeta(
        timestamp_local_iso="2026-01-01T00:00:00",
        timestamp_local_ms=1_700_000_000_000,
        symbol=frame.symbol,
        timeframe=frame.timeframe,
        bar_count=len(frame.bars),
        ai_provider={},
    )
    return AnalysisRecord(
        meta=meta,
        kline_data=[],
        htf_text="",
        stage1_messages=[{"role": "user", "content": "prev user"}],
        stage1_response={"content": '{"cycle_position":"trading_range"}'},
        stage1_diagnosis={"cycle_position": "trading_range", "direction": "neutral"},
        stage2_messages=[],
        stage2_response=None,
        stage2_decision=None,
        strategy_files_used=[],
        experience_loaded=[],
        exception=None,
        usage_total={},
    )


# ── 注入存在性 ────────────────────────────────────────────────────────────────


def test_marker_in_full_stage1(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    asm = PromptAssembler(
        prompt_dir=tmp_path,
        prompt_settings=_Settings(),
        fundamental_provider=_FakeProvider(_MARKER),
    )
    messages = asm.build_stage1(_make_frame())
    assert _MARKER in messages[1]["content"]


def test_current_exchange_forwarded_to_provider(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    provider = _FakeProvider(_MARKER)
    asm = PromptAssembler(
        prompt_dir=tmp_path,
        prompt_settings=_Settings(),
        fundamental_provider=provider,
    )
    asm.current_exchange = "NASDAQ"
    asm.build_stage1(_make_frame())
    assert provider.last_exchange == "NASDAQ"


def test_blank_current_exchange_forwarded_as_none(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    provider = _FakeProvider(_MARKER)
    asm = PromptAssembler(
        prompt_dir=tmp_path,
        prompt_settings=_Settings(),
        fundamental_provider=provider,
    )
    # 默认 current_exchange = "" → 传 None（不覆盖，回退按代码判定）
    asm.build_stage1(_make_frame())
    assert provider.last_exchange is None


def test_marker_in_incremental(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    asm = PromptAssembler(
        prompt_dir=tmp_path,
        prompt_settings=_Settings(),
        fundamental_provider=_FakeProvider(_MARKER),
    )
    frame = _make_frame()
    rec = _make_record(frame)
    text = asm._build_incremental_stage1_user_prompt(frame, rec, 1)
    assert _MARKER in text


def test_marker_in_incremental_continuation(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    asm = PromptAssembler(
        prompt_dir=tmp_path,
        prompt_settings=_Settings(),
        fundamental_provider=_FakeProvider(_MARKER),
    )
    frame = _make_frame()
    rec = _make_record(frame)
    text = asm._build_incremental_stage1_continuation_user_prompt(frame, rec, 1)
    assert _MARKER in text


# ── 降级与回归 ────────────────────────────────────────────────────────────────


def test_provider_none_no_block(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    asm = PromptAssembler(prompt_dir=tmp_path, prompt_settings=_Settings())
    messages = asm.build_stage1(_make_frame())
    assert _MARKER not in messages[1]["content"]


def test_switch_off_no_block(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    s = _Settings()
    s.enable_fundamental_context = False
    asm = PromptAssembler(
        prompt_dir=tmp_path, prompt_settings=s, fundamental_provider=_FakeProvider(_MARKER)
    )
    messages = asm.build_stage1(_make_frame())
    assert _MARKER not in messages[1]["content"]


def test_provider_exception_does_not_raise(tmp_path: Path) -> None:
    _write_prompt_files(tmp_path)
    asm = PromptAssembler(
        prompt_dir=tmp_path, prompt_settings=_Settings(), fundamental_provider=_BoomProvider()
    )
    messages = asm.build_stage1(_make_frame())  # 不应抛
    assert _MARKER not in messages[1]["content"]


def test_regression_provider_none_byte_identical(tmp_path: Path) -> None:
    """provider=None 时输出与"完全不配置 provider"逐字节一致。"""
    _write_prompt_files(tmp_path)
    frame = _make_frame()

    baseline = PromptAssembler(prompt_dir=tmp_path, prompt_settings=_Settings())
    with_none = PromptAssembler(
        prompt_dir=tmp_path, prompt_settings=_Settings(), fundamental_provider=None
    )
    assert baseline.build_stage1(frame)[1]["content"] == with_none.build_stage1(frame)[1]["content"]
