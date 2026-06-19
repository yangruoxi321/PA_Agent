"""Anthropic Messages API adapter for providers selected as api_format=anthropic."""
from __future__ import annotations

import json
import logging
import time
from typing import Any, Callable, TYPE_CHECKING

from pa_agent.ai.deepseek_client import (
    AIReply,
    AIUsage,
    CancelledError,
    ProviderEndpointError,
    _looks_like_html_response,
)
from pa_agent.config.settings import AIProviderSettings

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

logger = logging.getLogger(__name__)

_ANTHROPIC_VERSION = "2023-06-01"


def anthropic_messages_endpoint(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if url.lower().endswith("/messages"):
        return url
    return f"{url}/messages"


def _prepare_anthropic_messages(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        role = msg.get("role")
        content = msg.get("content", "")
        if role == "system":
            if isinstance(content, str) and content.strip():
                system_parts.append(content)
            continue
        if role in ("user", "assistant"):
            api_messages.append({"role": role, "content": content})
    system_param = "\n\n".join(system_parts) if system_parts else None
    return api_messages, system_param


def _anthropic_headers(api_key: str) -> dict[str, str]:
    return {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "anthropic-version": _ANTHROPIC_VERSION,
        "Accept": "application/json, text/event-stream",
    }


def _raise_if_html_payload(text: str | None, *, base_url: str) -> None:
    if _looks_like_html_response(text):
        raise ProviderEndpointError(
            "Provider returned an HTML web page instead of an Anthropic API "
            f"response. If this gateway uses OpenAI /chat/completions, set "
            f"「API 格式」to OpenAI. Base URL: {base_url}."
        )


def _parse_anthropic_message(data: dict[str, Any]) -> tuple[str, str]:
    content_parts: list[str] = []
    thinking_parts: list[str] = []
    for block in data.get("content") or []:
        if not isinstance(block, dict):
            continue
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str) and text:
                content_parts.append(text)
        elif block_type == "thinking":
            thought = block.get("thinking")
            if isinstance(thought, str) and thought:
                thinking_parts.append(thought)
    return "".join(content_parts), "".join(thinking_parts)


def _usage_from_anthropic(raw_usage: dict[str, Any] | None) -> AIUsage:
    usage = raw_usage or {}
    prompt_tokens = int(usage.get("input_tokens") or 0)
    completion_tokens = int(usage.get("output_tokens") or 0)
    cached_prompt_tokens = int(usage.get("cache_read_input_tokens") or 0)
    total_tokens = prompt_tokens + completion_tokens
    return AIUsage(
        prompt_tokens=prompt_tokens,
        cached_prompt_tokens=cached_prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=total_tokens,
    )


def _build_request_body(
    settings: AIProviderSettings,
    *,
    api_messages: list[dict[str, Any]],
    system_param: str | None,
    max_tokens: int,
    extra_body: dict[str, Any],
    stream: bool,
) -> dict[str, Any]:
    body: dict[str, Any] = {
        "model": settings.model,
        "max_tokens": max_tokens,
        "messages": api_messages,
    }
    if system_param:
        body["system"] = system_param
    thinking = extra_body.get("thinking")
    if isinstance(thinking, dict) and thinking:
        body["thinking"] = thinking
    if stream:
        body["stream"] = True
    return body


def _decode_response_text(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str):
        return text
    content = getattr(response, "content", b"")
    if isinstance(content, bytes):
        return content.decode("utf-8", errors="replace")
    return str(content)


def anthropic_messages_chat(
    settings: AIProviderSettings,
    messages: list[dict[str, Any]],
    *,
    extra_body: dict[str, Any],
    max_tokens: int,
    cancel_token: "CancelToken | None" = None,
    timeout_s: float = 600.0,
    logger_: logging.Logger | None = None,
) -> AIReply:
    if cancel_token is not None and cancel_token.is_set():
        raise CancelledError("Request cancelled before API call")

    import httpx

    log = logger_ or logger
    api_messages, system_param = _prepare_anthropic_messages(messages)
    url = anthropic_messages_endpoint(settings.base_url)
    body = _build_request_body(
        settings,
        api_messages=api_messages,
        system_param=system_param,
        max_tokens=max_tokens,
        extra_body=extra_body,
        stream=False,
    )

    log.debug(
        "anthropic_messages_chat: model=%s max_tokens=%s system=%s msgs=%d url=%s",
        settings.model,
        max_tokens,
        bool(system_param),
        len(api_messages),
        url,
    )

    t0 = time.monotonic()
    try:
        response = httpx.post(
            url,
            headers=_anthropic_headers(settings.api_key),
            json=body,
            timeout=timeout_s,
        )
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        log.error("anthropic_messages_chat error after %.0f ms: %s", latency_ms, exc)
        raise

    latency_ms = (time.monotonic() - t0) * 1000
    raw_text = _decode_response_text(response)
    _raise_if_html_payload(raw_text, base_url=settings.base_url)

    if response.status_code >= 400:
        raise RuntimeError(
            f"Anthropic API HTTP {response.status_code}: {raw_text[:500]}"
        )

    data = response.json()
    content, reasoning_content = _parse_anthropic_message(data)
    usage = _usage_from_anthropic(data.get("usage"))
    request_id = str(data.get("id") or "")
    model_name = str(data.get("model") or settings.model or "")

    raw: dict[str, Any] = {
        "id": request_id,
        "model": model_name,
        "content": content,
        "reasoning_content": reasoning_content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "cached_prompt_tokens": usage.cached_prompt_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "cache_hit_rate_pct": round(usage.cache_hit_rate * 100, 1),
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "latency_ms": latency_ms,
    }

    return AIReply(
        content=content,
        reasoning_content=reasoning_content,
        raw=raw,
        usage=usage,
        request_id=request_id,
        latency_ms=latency_ms,
    )


def _handle_anthropic_stream_event(
    event: dict[str, Any],
    *,
    on_reasoning_token: Callable[[str], None] | None,
    on_content_token: Callable[[str], None] | None,
    content: str,
    reasoning_content: str,
    request_id: str,
    model_name: str,
    usage: AIUsage,
) -> tuple[str, str, str, str, AIUsage]:
    event_type = event.get("type")
    if event_type == "message_start":
        message = event.get("message") or {}
        request_id = str(message.get("id") or request_id)
        model_name = str(message.get("model") or model_name)
        usage = _usage_from_anthropic(message.get("usage")) or usage
        return content, reasoning_content, request_id, model_name, usage

    if event_type == "content_block_delta":
        delta = event.get("delta") or {}
        delta_type = delta.get("type")
        if delta_type == "thinking_delta":
            chunk = delta.get("thinking") or ""
            if chunk:
                reasoning_content += chunk
                if on_reasoning_token is not None:
                    on_reasoning_token(chunk)
        elif delta_type == "text_delta":
            chunk = delta.get("text") or ""
            if chunk:
                content += chunk
                if on_content_token is not None:
                    on_content_token(chunk)
        return content, reasoning_content, request_id, model_name, usage

    if event_type == "message_delta":
        usage = _usage_from_anthropic(event.get("usage")) or usage
        return content, reasoning_content, request_id, model_name, usage

    return content, reasoning_content, request_id, model_name, usage


def anthropic_messages_stream_chat(
    settings: AIProviderSettings,
    messages: list[dict[str, Any]],
    *,
    extra_body: dict[str, Any],
    max_tokens: int,
    on_reasoning_token: Callable[[str], None] | None = None,
    on_content_token: Callable[[str], None] | None = None,
    cancel_token: "CancelToken | None" = None,
    timeout_s: float = 600.0,
    logger_: logging.Logger | None = None,
) -> AIReply:
    if cancel_token is not None and cancel_token.is_set():
        raise CancelledError("Request cancelled before API call")

    import httpx

    log = logger_ or logger
    api_messages, system_param = _prepare_anthropic_messages(messages)
    url = anthropic_messages_endpoint(settings.base_url)
    body = _build_request_body(
        settings,
        api_messages=api_messages,
        system_param=system_param,
        max_tokens=max_tokens,
        extra_body=extra_body,
        stream=True,
    )

    log.info(
        "anthropic_messages_stream_chat: model=%s max_tokens=%s system=%s msgs=%d",
        settings.model,
        max_tokens,
        bool(system_param),
        len(api_messages),
    )

    content = ""
    reasoning_content = ""
    request_id = ""
    model_name = ""
    usage = AIUsage()
    t0 = time.monotonic()

    try:
        with httpx.stream(
            "POST",
            url,
            headers=_anthropic_headers(settings.api_key),
            json=body,
            timeout=timeout_s,
        ) as response:
            if response.status_code >= 400:
                raw_text = _decode_response_text(response)
                _raise_if_html_payload(raw_text, base_url=settings.base_url)
                raise RuntimeError(
                    f"Anthropic API HTTP {response.status_code}: {raw_text[:500]}"
                )

            for line in response.iter_lines():
                if cancel_token is not None and cancel_token.is_set():
                    raise CancelledError("Request cancelled during streaming")

                if not line or not line.startswith("data:"):
                    continue
                payload = line[5:].strip()
                if not payload or payload == "[DONE]":
                    continue
                try:
                    event = json.loads(payload)
                except json.JSONDecodeError:
                    continue
                (
                    content,
                    reasoning_content,
                    request_id,
                    model_name,
                    usage,
                ) = _handle_anthropic_stream_event(
                    event,
                    on_reasoning_token=on_reasoning_token,
                    on_content_token=on_content_token,
                    content=content,
                    reasoning_content=reasoning_content,
                    request_id=request_id,
                    model_name=model_name,
                    usage=usage,
                )
    except CancelledError:
        raise
    except Exception as exc:
        latency_ms = (time.monotonic() - t0) * 1000
        log.error(
            "anthropic_messages_stream_chat error after %.0f ms: %s",
            latency_ms,
            exc,
        )
        raise

    latency_ms = (time.monotonic() - t0) * 1000
    _raise_if_html_payload(content, base_url=settings.base_url)

    raw: dict[str, Any] = {
        "id": request_id,
        "model": model_name or settings.model,
        "content": content,
        "reasoning_content": reasoning_content,
        "usage": {
            "prompt_tokens": usage.prompt_tokens,
            "cached_prompt_tokens": usage.cached_prompt_tokens,
            "cache_miss_tokens": usage.cache_miss_tokens,
            "cache_hit_rate_pct": round(usage.cache_hit_rate * 100, 1),
            "completion_tokens": usage.completion_tokens,
            "total_tokens": usage.total_tokens,
        },
        "latency_ms": latency_ms,
    }

    log.info(
        "anthropic_messages_stream_chat done: latency=%.0f ms "
        "reasoning_chars=%d content_chars=%d",
        latency_ms,
        len(reasoning_content),
        len(content),
    )

    return AIReply(
        content=content,
        reasoning_content=reasoning_content,
        raw=raw,
        usage=usage,
        request_id=request_id,
        latency_ms=latency_ms,
    )