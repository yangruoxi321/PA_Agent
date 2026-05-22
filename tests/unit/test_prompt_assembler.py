"""Unit tests for PromptAssembler (task 7.3)."""
from __future__ import annotations

import json
import math
import pytest
from pathlib import Path
from unittest.mock import patch, MagicMock

from pa_agent.ai.prompt_assembler import PromptAssembler
from pa_agent.data.base import KlineBar, KlineFrame, IndicatorBundle


def _make_frame(n: int = 5) -> KlineFrame:
    bars = tuple(
        KlineBar(
            seq=i + 1,
            ts_open=float(1_700_000_000 - i * 3600),
            open=2600.0 + i,
            high=2610.0 + i,
            low=2590.0 + i,
            close=2605.0 + i,
            volume=1000.0,
            closed=(i != 0),
        )
        for i in range(n)
    )
    indicators = IndicatorBundle(
        ema20=tuple(2600.0 + i for i in range(n)),
        atr14=tuple(5.0 for _ in range(n)),
    )
    return KlineFrame(
        symbol="XAUUSD",
        timeframe="1h",
        bars=bars,
        indicators=indicators,
        snapshot_ts_local_ms=1_700_000_000_000,
    )


@pytest.fixture()
def assembler(tmp_path: Path) -> PromptAssembler:
    """PromptAssembler with fake prompt files."""
    for fname in [
        "提示词大纲_人设与思维方式.txt",
        "市场诊断框架.txt",
        "二元决策.txt",
        "文件16-K线信号识别.txt",
        "逐棒分析检查单.txt",
        "文件17-止损和止盈与仓位管理.txt",
        "文件18-突破失败与突破测试.txt",
        "文件19-H1H2-L1L2计数.txt",
        "文件20-AlwaysIn与20GB.txt",
        "文件21-铁丝网与无交易环境.txt",
        "文件22-信号失败后的磁力位.txt",
        "上涨通道分析识别.txt",
        "上涨通道交易策略.txt",
        "下跌通道分析识别.txt",
        "下跌通道交易策略.txt",
        "极速上涨分析识别.txt",
        "极速上涨交易策略.txt",
        "极速下跌分析识别.txt",
        "极速下跌交易策略.txt",
        "震荡区间分析识别.txt",
        "震荡区间交易策略.txt",
        "文件13-窄通道与宽通道策略.txt",
        "文件14-楔形形态分析交易.txt",
        "文件15-二次入场机会.txt",
    ]:
        (tmp_path / fname).write_text(f"[CONTENT OF {fname}]", encoding="utf-8")
    return PromptAssembler(prompt_dir=tmp_path)


def test_stage1_system_prompt_order(assembler: PromptAssembler):
    """Stage 1 system: shared persona + binary tree; user: framework + signals."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    system = messages[0]["content"]
    user = messages[1]["content"]
    pos_persona = system.find("提示词大纲_人设与思维方式")
    pos_binary_sys = system.find("二元决策")
    assert pos_persona >= 0
    assert 0 <= pos_persona < pos_binary_sys, "Binary tree should follow persona in system"
    assert "市场诊断框架" not in system

    pos_diag = user.find("市场诊断框架")
    pos_signal = user.find("文件16-K线信号识别")
    pos_bar_by_bar = user.find("逐棒分析检查单")
    assert "[CONTENT OF 二元决策.txt]" not in user, (
        "Full binary tree file is only in system (shared with stage 2)"
    )
    assert "[CONTENT OF 二元决策.txt]" in system
    assert 0 <= pos_diag < pos_signal, "Stage 1 user task files are out of order"
    assert 0 <= pos_signal < pos_bar_by_bar, "Bar-by-bar checklist should follow signal file"
    assert "文件18-突破失败与突破测试" in user
    assert "文件19-H1H2-L1L2计数" in user
    assert "文件20-AlwaysIn与20GB" in user
    assert "文件21-铁丝网与无交易环境" in user
    assert "文件22-信号失败后的磁力位" in user


def test_stage1_user_prompt_contains_required_fields(assembler: PromptAssembler):
    """Stage 1 user prompt must contain symbol, timeframe, bar count."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    user = messages[1]["content"]
    assert "XAUUSD" in user
    assert "1h" in user
    assert "序号" in user
    assert "K线几何特征" in user
    assert "doji" in user
    assert "更高时间框架" not in user


def test_stage2_user_prompt_includes_gate_trace(assembler: PromptAssembler):
    frame = _make_frame()
    stage1_json = {
        "cycle_position": "normal_channel",
        "direction": "bullish",
        "gate_result": "proceed",
        "gate_trace": [{"node_id": "0.1", "question": "q", "answer": "是", "reason": "r"}],
    }
    messages = assembler.build_stage2(frame, stage1_json, [], [])
    user = messages[1]["content"]
    assert "gate_result=proceed" in user
    assert "gate_trace" in user or "0.1" in user


def test_stage2_system_prompt_order(assembler: PromptAssembler):
    """Stage 2 system reuses stage-1 system (persona + binary); user: strategy → risk."""
    frame = _make_frame()
    stage1_json = {"cycle_position": "normal_channel", "direction": "bullish"}
    strategy_files = ["上涨通道分析识别.txt", "上涨通道交易策略.txt", "文件13-窄通道与宽通道策略.txt"]
    messages = assembler.build_stage2(frame, stage1_json, strategy_files, [])
    system = messages[0]["content"]
    user = messages[1]["content"]

    pos_persona = system.find("提示词大纲_人设与思维方式")
    pos_binary_sys = system.find("二元决策")
    assert pos_persona >= 0
    assert 0 <= pos_persona < pos_binary_sys

    assert "[CONTENT OF 二元决策.txt]" not in user, (
        "Full binary tree file is not duplicated in stage 2 user turn"
    )
    assert "[CONTENT OF 二元决策.txt]" in system
    pos_strategy = user.find("上涨通道分析识别")
    pos_bar_by_bar = user.find("逐棒分析检查单")
    pos_signal = user.find("文件16-K线信号识别")
    pos_risk = user.find("文件17-止损和止盈与仓位管理")
    assert 0 <= pos_strategy < pos_risk, "Stage 2 user task files are out of order"
    assert 0 <= pos_bar_by_bar < pos_signal < pos_risk


def test_stage2_user_prompt_contains_stage1_json(assembler: PromptAssembler):
    """Stage 2 user prompt must embed the Stage 1 JSON."""
    frame = _make_frame()
    stage1_json = {"cycle_position": "spike", "direction": "bearish"}
    messages = assembler.build_stage2(frame, stage1_json, [], [])
    user = messages[1]["content"]
    assert "spike" in user
    assert "bearish" in user


def test_stage2_user_prompt_always_includes_full_strategy_pack(
    assembler: PromptAssembler,
):
    """Accuracy-first mode loads all strategy references in Stage 2."""
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [])
    user = messages[1]["content"]
    assert "上涨通道分析识别" in user
    assert "下跌通道分析识别" in user
    assert "极速上涨分析识别" in user
    assert "极速下跌分析识别" in user
    assert "震荡区间分析识别" in user
    assert "文件18-突破失败与突破测试" in user
    assert "文件22-信号失败后的磁力位" in user


def test_stage1_output_reminder_present(assembler: PromptAssembler):
    """Stage 1 user turn must contain the output format reminder."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    user = messages[1]["content"]
    assert "cycle_position" in user
    assert "diagnosis_confidence" in user
    assert "bar_by_bar_summary" in user
    assert "逐K摘要硬规则" in user
    assert "gate_trace" in user
    assert "gate_result" in user


def test_stage2_output_contract_present(assembler: PromptAssembler):
    """Stage 2 user turn must contain the output contract with null rule."""
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [])
    user = messages[1]["content"]
    assert "不下单" in user
    assert "order_direction" in user
    assert "bar_analysis" in user
    assert "entry_basis_bar" in user
    assert "突破单 entry_price 硬规则" in user
    assert "§9 逐K信号链与新鲜度硬规则" in user
    assert "K线几何特征" in user
    assert "EMA缺口数" in user
    assert "decision_trace" in user
    assert "terminal" in user


def test_stage2_experience_entries_included(assembler: PromptAssembler):
    """Stage 2 user turn must include experience entries when provided."""
    frame = _make_frame()
    entries = [{"cycle_position": "spike", "outcome": "success"}]
    messages = assembler.build_stage2(frame, {}, [], entries)
    user = messages[1]["content"]
    assert "经验库" in user
    assert "案例 1" in user


def test_stage2_system_prompt_only_matches_build_stage2(assembler: PromptAssembler):
    """stage2_system_prompt_only must return the same system content as build_stage2."""
    frame = _make_frame()
    strategy_files = ["上涨通道分析识别.txt"]
    entries = [{"note": "test"}]
    messages = assembler.build_stage2(frame, {}, strategy_files, entries)
    system_from_build = messages[0]["content"]
    system_only = assembler.stage2_system_prompt_only(strategy_files, entries)
    assert system_from_build == system_only


def test_kline_table_contains_nan_as_na(assembler: PromptAssembler):
    """K-line table renders NaN indicator values as 'N/A'."""
    bars = (
        KlineBar(seq=1, ts_open=1_700_000_000.0, open=2600.0, high=2610.0,
                 low=2590.0, close=2605.0, volume=1000.0, closed=False),
    )
    indicators = IndicatorBundle(
        ema20=(float("nan"),),
        atr14=(float("nan"),),
    )
    frame = KlineFrame(
        symbol="XAUUSD", timeframe="1h", bars=bars,
        indicators=indicators, snapshot_ts_local_ms=1_700_000_000_000,
    )
    messages = assembler.build_stage1(frame)
    user = messages[1]["content"]
    assert "N/A" in user


def test_stage1_message_roles(assembler: PromptAssembler):
    """build_stage1 must return exactly [system, user] messages."""
    frame = _make_frame()
    messages = assembler.build_stage1(frame)
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_stage2_message_roles(assembler: PromptAssembler):
    """build_stage2 must return exactly [system, user] messages."""
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [])
    assert len(messages) == 2
    assert messages[0]["role"] == "system"
    assert messages[1]["role"] == "user"


def test_stage2_continuation_omits_stage1_user_prompt(assembler: PromptAssembler):
    """Stage 2 must not duplicate the huge Stage 1 user turn; K-line table lives in Stage 2 user."""
    frame = _make_frame()
    stage1_messages = assembler.build_stage1(frame)
    stage1_json = {"cycle_position": "spike", "direction": "bearish", "gate_result": "proceed"}

    messages = assembler.build_stage2_continuation(
        frame=frame,
        stage1_messages=stage1_messages,
        stage1_reply_content='{"cycle_position":"spike","direction":"bearish"}',
        stage1_json=stage1_json,
        strategy_files=["上涨通道分析识别.txt"],
        experience_entries=[],
    )

    assert [m["role"] for m in messages] == ["system", "assistant", "user"]
    assert messages[0]["content"] == stage1_messages[0]["content"]
    assert "cycle_position" in messages[1]["content"]
    assert "K线数据" in messages[2]["content"]
    assert "沿用上一轮阶段一用户消息中的同一份 K线数据" not in messages[2]["content"]
    assert "上涨通道分析识别" in messages[2]["content"]
    assert "【最后一步·必做】" in messages[2]["content"]


def test_stage2_prompt_includes_balanced_stance_guidance(assembler: PromptAssembler):
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [], decision_stance="balanced")
    user = messages[1]["content"]
    assert "交易倾向" in user
    assert "均衡" in user
    assert "次优但可执行" in user


def test_stage2_prompt_conservative_omits_balanced_only_hints(assembler: PromptAssembler):
    frame = _make_frame()
    messages = assembler.build_stage2(frame, {}, [], [], decision_stance="conservative")
    user = messages[1]["content"]
    assert "当前系统默认" in user
    assert "次优但可执行" not in user


def test_incremental_stage1_prompt_includes_previous_record_and_new_bars(
    assembler: PromptAssembler,
):
    """Incremental Stage 1 prompt carries previous analysis and new bars."""
    from pa_agent.records.schema import AnalysisRecord, RecordMeta

    frame = _make_frame(5)
    previous = AnalysisRecord(
        meta=RecordMeta(
            timestamp_local_iso="2026-01-01T00:00:00.000",
            timestamp_local_ms=1,
            symbol="XAUUSD",
            timeframe="1h",
            bar_count=5,
            ai_provider={},
        ),
        kline_data=[],
        htf_text="",
        stage1_messages=[],
        stage1_response=None,
        stage1_diagnosis={"cycle_position": "normal_channel"},
        stage2_messages=[],
        stage2_response=None,
        stage2_decision={"decision": {"order_type": "不下单"}},
        strategy_files_used=["上涨通道分析识别.txt"],
        experience_loaded=[],
        exception=None,
        usage_total={},
    )

    messages = assembler.build_incremental_stage1(frame, previous, 2)

    assert [m["role"] for m in messages] == ["system", "user"]
    user = messages[1]["content"]
    assert "阶段一增量任务" in user
    assert "新增已收盘K线:2" in user
    assert "上一轮已完成分析" in user
    assert "normal_channel" in user
    assert "当前完整 K线数据" in user
    assert "当前完整 K线几何特征" in user
