"""Tests for order-opportunity detection."""
from __future__ import annotations

from pa_agent.gui.order_opportunity import (
    format_order_alert_message,
    has_order_opportunity,
)


def test_has_order_opportunity_for_active_orders() -> None:
    for order_type in ("限价单", "突破单", "市价单"):
        assert has_order_opportunity({"order_type": order_type})


def test_no_order_opportunity_when_wait() -> None:
    assert not has_order_opportunity({"order_type": "不下单"})
    assert not has_order_opportunity({})


def test_format_order_alert_message_includes_prices() -> None:
    text = format_order_alert_message(
        {
            "order_direction": "做多",
            "order_type": "突破单",
            "entry_price": 2650.5,
            "stop_loss_price": 2640,
            "take_profit_price": 2670,
            "reasoning": "测试理由",
        }
    )
    assert "做多" in text
    assert "突破单" in text
    assert "2650.5" in text
    assert "决策" in text
