"""Unit tests for Anthropic Messages API adapter."""
from __future__ import annotations

from unittest.mock import MagicMock, patch

from pa_agent.ai.anthropic_compat import (
    _parse_anthropic_message,
    _prepare_anthropic_messages,
    anthropic_messages_endpoint,
)
from pa_agent.ai.deepseek_client import DeepSeekClient
from pa_agent.config.settings import AIProviderSettings


def test_anthropic_messages_endpoint_appends_messages_path() -> None:
    assert anthropic_messages_endpoint("https://nekocode.ai/v1") == (
        "https://nekocode.ai/v1/messages"
    )
    assert anthropic_messages_endpoint("https://example.com/v1/messages") == (
        "https://example.com/v1/messages"
    )


def test_prepare_anthropic_messages_hoists_system() -> None:
    messages = [
        {"role": "system", "content": "sys-a"},
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]
    api_messages, system = _prepare_anthropic_messages(messages)
    assert system == "sys-a"
    assert api_messages == [
        {"role": "user", "content": "hi"},
        {"role": "assistant", "content": "yo"},
    ]


def test_parse_anthropic_message_extracts_text_and_thinking() -> None:
    content, reasoning = _parse_anthropic_message(
        {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "answer"},
            ]
        }
    )
    assert content == "answer"
    assert reasoning == "hmm"


def test_chat_routes_anthropic_format_to_adapter() -> None:
    settings = AIProviderSettings()
    settings.base_url = "https://nekocode.ai/v1"
    settings.api_format = "anthropic"
    settings.model = "claude-opus-4-8"
    settings.api_key = "sk-test"
    client = DeepSeekClient(settings)

    mock_reply = MagicMock()
    with patch(
        "pa_agent.ai.anthropic_compat.anthropic_messages_chat",
        return_value=mock_reply,
    ) as anthropic_chat:
        reply = client.chat([{"role": "user", "content": "hi"}])

    anthropic_chat.assert_called_once()
    assert reply is mock_reply


def test_chat_keeps_openai_format_on_default_setting() -> None:
    settings = AIProviderSettings()
    settings.base_url = "https://nekocode.ai/v1"
    settings.api_format = "openai"
    client = DeepSeekClient(settings)

    mock_resp = MagicMock()
    mock_resp.choices = [MagicMock()]
    mock_resp.choices[0].message.content = "ok"
    mock_resp.choices[0].message.reasoning_content = None
    mock_resp.usage = MagicMock(prompt_tokens=1, completion_tokens=1, total_tokens=2)
    mock_resp.usage.prompt_tokens_details = None
    mock_resp.id = "x"
    mock_resp.model = "m"

    mock_openai = MagicMock()
    mock_openai.return_value.chat.completions.create.return_value = mock_resp

    with patch("pa_agent.ai.deepseek_client._OpenAI", mock_openai):
        with patch(
            "pa_agent.ai.anthropic_compat.anthropic_messages_chat"
        ) as anthropic_chat:
            client.chat([{"role": "user", "content": "hi"}])

    anthropic_chat.assert_not_called()
    mock_openai.return_value.chat.completions.create.assert_called_once()


def test_stream_chat_routes_anthropic_format_to_adapter() -> None:
    settings = AIProviderSettings()
    settings.base_url = "https://nekocode.ai/v1"
    settings.api_format = "anthropic"
    settings.model = "claude-opus-4-8"
    settings.api_key = "sk-test"
    client = DeepSeekClient(settings)

    mock_reply = MagicMock()
    with patch(
        "pa_agent.ai.anthropic_compat.anthropic_messages_stream_chat",
        return_value=mock_reply,
    ) as anthropic_stream:
        reply = client.stream_chat([{"role": "user", "content": "hi"}])

    anthropic_stream.assert_called_once()
    assert reply is mock_reply