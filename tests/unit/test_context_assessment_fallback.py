"""单元测试：context_assessment 兜底（有基本面却没填 → 重试/补壳）。"""

from __future__ import annotations

import json

import pytest

from pa_agent.ai.json_validator import JsonValidator, Ok, ValidationError
from pa_agent.config.settings import ValidationSettings

pytestmark = pytest.mark.unit


def _validator() -> JsonValidator:
    # 关闭其它 coherence 检查，只留 schema + context_assessment 兜底（它在开关外）。
    return JsonValidator(
        ValidationSettings(
            stage1_coherence_checks=False,
            trace_semantic_checks=False,
            strict_bar_by_bar_features=False,
            retry_max_semantic=1,
        )
    )


def _stage1(**over) -> str:
    base = {
        "cycle_position": "trading_range",
        "direction": "neutral",
        "diagnosis_confidence": 55,
        "market_phase": "stable",
        "detected_patterns": [],
        "key_signals": ["x"],
        "htf_context": "h",
        "entry_setup": "none",
        "strategy_files_needed": [],
        "bar_by_bar_summary": [
            {
                "bar": "K1",
                "role": "structure",
                "bar_type": "doji",
                "context_effect": "neutral",
                "follow_through": "pending",
                "trapped_side": "none",
                "reason": "r",
            }
        ],
        "gate_trace": [
            {
                "node_id": "1.1",
                "question": "q",
                "answer": "是",
                "reason": "r",
                "bar_range": "K8-K1",
            }
        ],
        "gate_result": "wait",
    }
    base.update(over)
    return json.dumps(base, ensure_ascii=False)


_VALID_CA = {"stance": "diverges", "confidence_adjustment": -8, "note": "估值极高背离"}


def test_no_fundamental_does_not_require_ca() -> None:
    # 无基本面注入 → 不强制，缺 context_assessment 也通过（回归安全）。
    r = _validator().validate("stage1", _stage1(), had_fundamental=False)
    assert isinstance(r, Ok)
    assert r.obj.get("context_assessment") in (None, {}) or "context_assessment" not in r.obj


def test_had_fundamental_missing_ca_first_attempt_retries() -> None:
    # 有基本面但没填 + 首轮 → 触发重试（返回 ValidationError，含该字段）。
    r = _validator().validate("stage1", _stage1(), had_fundamental=True, _attempt=0)
    assert isinstance(r, ValidationError)
    assert any("context_assessment" in f for f in (r.invalid_fields or []))


def test_had_fundamental_missing_ca_last_attempt_fills_shell() -> None:
    # 达到语义重试上限 → 不再重试，补程序兜底壳放行（不卡流程）。
    r = _validator().validate("stage1", _stage1(), had_fundamental=True, _attempt=1)
    assert isinstance(r, Ok)
    ca = r.obj.get("context_assessment")
    assert isinstance(ca, dict) and ca.get("stance") == "na"
    assert ca.get("_auto_filled") is True


def test_had_fundamental_valid_ca_passes() -> None:
    # AI 正常填了有效 context_assessment → 直接通过，保留原值（无壳标记）。
    r = _validator().validate(
        "stage1", _stage1(context_assessment=_VALID_CA), had_fundamental=True, _attempt=0
    )
    assert isinstance(r, Ok)
    ca = r.obj.get("context_assessment")
    assert ca.get("stance") == "diverges"
    assert "_auto_filled" not in ca
