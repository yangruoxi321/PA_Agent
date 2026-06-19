"""DeepSeek AI client (OpenAI-compatible API)."""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any, Callable, TYPE_CHECKING

if TYPE_CHECKING:
    from pa_agent.util.threading import CancelToken

from pa_agent.config.settings import AIProviderSettings
from pa_agent.util.mask_secret import mask_secret
from pa_agent.ai.mimo_compat import (
    ReasoningCache,
    is_mimo_provider,
    mimo_max_output_tokens,
    patch_messages_for_mimo,
    resolve_mimo_thinking_extra_body,
    response_message_dict,
    store_reasoning_from_response,
)

try:
    from openai import OpenAI as _OpenAI  # type: ignore[import]
except ImportError as _exc:
    _OpenAI = None  # type: ignore[assignment,misc]
    _OPENAI_IMPORT_ERROR = _exc
else:
    _OPENAI_IMPORT_ERROR = None

logger = logging.getLogger(__name__)

_MIMO_REASONING_CACHE = ReasoningCache()


@dataclass
class AIUsage:
    """Token usage from a single API call."""
    prompt_tokens: int = 0
    cached_prompt_tokens: int = 0
    completion_tokens: int = 0
    total_tokens: int = 0

    @property
    def cache_hit_rate(self) -> float:
        """Fraction of prompt tokens served from KV cache (0.0–1.0).

        DeepSeek 硬盘缓存命中率。值越高，费用越低。
        0.0 = 无缓存命中；1.0 = 全部命中缓存。
        """
        if self.prompt_tokens <= 0:
            return 0.0
        return self.cached_prompt_tokens / self.prompt_tokens

    @property
    def cache_miss_tokens(self) -> int:
        """Prompt tokens that were NOT served from cache (billed at full rate)."""
        return max(0, self.prompt_tokens - self.cached_prompt_tokens)


@dataclass
class AIReply:
    """Structured response from a single AI API call."""
    content: str
    reasoning_content: str
    raw: dict[str, Any]          # full raw response dict for debug tab
    usage: AIUsage
    request_id: str
    latency_ms: float


class CancelledError(Exception):
    """Raised when a cancel_token is set before or during an API call."""


class EmptyAIResponseError(Exception):
    """Raised when the provider returns an empty, token-less response."""


class ProviderEndpointError(Exception):
    """Raised when base_url points to a web UI/non-API endpoint."""


def _is_empty_ai_response(
    *,
    content: str,
    reasoning_content: str,
    request_id: str,
    model_name: str,
    usage: AIUsage,
) -> bool:
    return (
        not (content or "").strip()
        and not (reasoning_content or "").strip()
        and not (request_id or "").strip()
        and not (model_name or "").strip()
        and usage.prompt_tokens == 0
        and usage.completion_tokens == 0
        and usage.total_tokens == 0
    )


def _decode_provider_text(value: Any) -> str | None:
    """Return text for providers that yield raw string/bytes instead of SDK objects."""
    if isinstance(value, str):
        return value
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    return None


def _looks_like_html_response(text: str | None) -> bool:
    stripped = (text or "").lstrip().lower()
    return stripped.startswith("<!doctype html") or stripped.startswith("<html")


def _raise_if_html_response(text: str | None, *, base_url: str) -> None:
    if not _looks_like_html_response(text):
        return
    raise ProviderEndpointError(
        "Provider returned an HTML web page instead of an OpenAI-compatible API "
        f"response. Base URL may point to the management UI, not the API endpoint: "
        f"{base_url}. Try the API base URL, usually ending with /v1."
    )


def _is_deepseek_native(base_url: str) -> bool:
    return "deepseek.com" in (base_url or "").lower()


def _api_base_url(base_url: str) -> str:
    """Return the API base URL expected by the OpenAI SDK."""
    url = (base_url or "").strip().rstrip("/")
    suffix = "/chat/completions"
    if url.lower().endswith(suffix):
        return url[: -len(suffix)].rstrip("/")
    return url


def _openai_client(settings: AIProviderSettings) -> Any:
    """Build OpenAI-compatible client with headers accepted by stricter gateways."""
    return _OpenAI(
        base_url=_api_base_url(settings.base_url),
        api_key=settings.api_key,
        default_headers={
            "Accept": "application/json, text/event-stream",
            "User-Agent": (
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "PA_Agent/1.0 Safari/537.36"
            ),
        },
    )


def _is_deepseek_model(model: str) -> bool:
    """True for DeepSeek model ids; excludes QClaw ``openclaw`` and WorkBuddy ``openclaw_wb`` Agent aliases."""
    m = (model or "").lower()
    if m in ("openclaw", "openclaw_wb"):
        return False
    if m.startswith("openclaw/") or m.startswith("openclaw_wb/"):
        return False
    return "deepseek" in m


def _is_qclaw_openclaw_agent(settings: AIProviderSettings) -> bool:
    """True when requests go through QClaw's public-gateway OpenClaw Agent."""
    from pa_agent.ai.qclaw_connector import detect_qclaw, is_openclaw_model

    return bool(is_openclaw_model(settings.model) and detect_qclaw())


def _openclaw_agent_request_extra(settings: AIProviderSettings) -> dict[str, Any]:
    """Ask QClaw/WorkBuddy Agent to answer in-chat only (no exec/write tool loop)."""
    if _is_qclaw_openclaw_agent(settings) or _is_workbuddy_agent(settings):
        return {"tool_choice": "none"}
    return {}


def _is_workbuddy_agent(settings: AIProviderSettings) -> bool:
    """True when requests go through WorkBuddy's model route."""
    from pa_agent.ai.workbuddy_connector import is_workbuddy_route

    return is_workbuddy_route(settings)


def _effective_api_model(settings: AIProviderSettings) -> str:
    """Model id sent to the upstream API (resolve WorkBuddy aliases)."""
    if _is_workbuddy_agent(settings):
        from pa_agent.ai.workbuddy_connector import resolve_workbuddy_api_model

        return resolve_workbuddy_api_model(settings.model)
    return settings.model


def _workbuddy_agent_request_extra(settings: AIProviderSettings) -> dict[str, Any]:
    """Add WorkBuddy-specific request parameters.

    Returns empty dict if not using WorkBuddy agent route.
    WorkBuddy uses the same tool_choice: none strategy as QClaw.
    """
    return _openclaw_agent_request_extra(settings)


def _is_kkai_openai_proxy(base_url: str) -> bool:
    """KKAI (api.kkone.vip) OpenAI-compatible gateway."""
    url = (base_url or "").lower()
    return "kkone.vip" in url


def _is_packyapi(base_url: str) -> bool:
    return "packyapi.com" in (base_url or "").lower()


def _is_minimax(base_url: str) -> bool:
    """MiniMax (api.minimax.io) OpenAI-compatible gateway."""
    url = (base_url or "").lower()
    return "minimax.io" in url or "minimax.com" in url


def _is_openrouter(base_url: str) -> bool:
    """OpenRouter (openrouter.ai) OpenAI-compatible gateway."""
    return "openrouter.ai" in (base_url or "").lower()


# OpenRouter exposes a unified reasoning effort across upstream providers;
# it accepts low/medium/high (no "max"), so map "max" → "high".
_OPENROUTER_EFFORT: dict[str, str] = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "high",
    "xhigh": "high",
}


def _openrouter_effort(reasoning_effort: str | None) -> str:
    key = (reasoning_effort or "medium").strip().lower()
    return _OPENROUTER_EFFORT.get(key, "medium")


# Unified completion cap (max_tokens) for all providers / API formats.
_DEFAULT_MAX_OUTPUT_TOKENS = 128_000


def _model_uses_claude_adaptive(model: str) -> bool:
    """Claude models that require thinking.type=adaptive (not budget_tokens)."""
    m = (model or "").lower()
    return any(
        token in m
        for token in (
            "claude-opus-4-7",
            "claude-opus-4-6",
            "claude-sonnet-4-6",
        )
    )


_EFFORT_TO_ADAPTIVE_OUTPUT: dict[str, str] = {
    "none": "low",
    "minimal": "low",
    "low": "low",
    "medium": "medium",
    "high": "high",
    "max": "max",
    "xhigh": "max",
}


def _adaptive_output_effort(reasoning_effort: str | None) -> str:
    key = (reasoning_effort or "medium").strip().lower()
    return _EFFORT_TO_ADAPTIVE_OUTPUT.get(key, "medium")


def _effort_budget_tokens(effort: str | None, *, max_output: int) -> int:
    """Thinking budget; must stay below max_output (Anthropic/Packy rule)."""
    del effort  # reserved for future per-effort tuning
    return max(1024, max_output - 1)


def _thinking_enabled(extra_body: dict[str, Any], effort: str | None) -> bool:
    if extra_body:
        if extra_body.get("chat_template_kwargs", {}).get("enable_thinking"):
            return True
        return extra_body.get("thinking", {}).get("type") in ("enabled", "adaptive")
    return effort is not None and effort != "none"


def _packy_anthropic_messages_api(settings: AIProviderSettings) -> bool:
    """Packy claude-officially uses Anthropic Messages API (no role=system in messages)."""
    return _is_packyapi(settings.base_url) and "claude" in (settings.model or "").lower()


def _is_mimo(settings: AIProviderSettings) -> bool:
    return is_mimo_provider(settings.base_url, settings.model)


def _prepare_chat_messages(
    settings: AIProviderSettings,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Hoist system turns to top-level ``system`` for Anthropic-native Packy routes."""
    if not _packy_anthropic_messages_api(settings):
        return messages, None
    system_parts: list[str] = []
    api_messages: list[dict[str, Any]] = []
    for msg in messages:
        if msg.get("role") == "system":
            text = msg.get("content", "")
            if isinstance(text, str) and text.strip():
                system_parts.append(text)
            continue
        api_messages.append(msg)
    system_param = "\n\n".join(system_parts) if system_parts else None
    return api_messages, system_param


def _prepare_api_messages(
    settings: AIProviderSettings,
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], str | None]:
    """Normalize messages for the active provider before API submission."""
    api_messages, system_param = _prepare_chat_messages(settings, messages)
    if _is_mimo(settings):
        api_messages = patch_messages_for_mimo(
            api_messages,
            model=settings.model,
            reasoning_cache=_MIMO_REASONING_CACHE,
        )
    return api_messages, system_param


def _provider_max_output_tokens(settings: AIProviderSettings) -> int:
    """Per-gateway completion cap (max_tokens); avoids 400 from provider limits."""
    if _is_mimo(settings):
        return min(mimo_max_output_tokens(settings.model), _DEFAULT_MAX_OUTPUT_TOKENS)
    return _DEFAULT_MAX_OUTPUT_TOKENS


def _completion_max_tokens(
    settings: AIProviderSettings,
    *,
    extra_body: dict[str, Any],
    effort: str | None,
) -> int:
    """Total completion budget (thinking + content) for OpenAI-compatible APIs."""
    del effort, extra_body
    return _provider_max_output_tokens(settings)


def _resolve_thinking_params(
    settings: AIProviderSettings,
    *,
    thinking: bool | None,
    reasoning_effort: str | None,
) -> tuple[dict[str, Any], str | None]:
    """Return (extra_body, reasoning_effort) for chat.completions.create."""
    _thinking = thinking if thinking is not None else settings.thinking
    _effort = reasoning_effort if reasoning_effort is not None else settings.reasoning_effort
    model = settings.model or ""

    if _is_openrouter(settings.base_url):
        # OpenRouter normalizes reasoning across upstream providers via the
        # ``reasoning`` body param (not Anthropic ``thinking`` / DeepSeek
        # ``output_config``). Reasoning text surfaces as message.reasoning.
        # Checked before the DeepSeek branch because OpenRouter model ids look
        # like ``deepseek/deepseek-chat`` and would otherwise route to native.
        if _thinking:
            return {"reasoning": {"effort": _openrouter_effort(_effort)}}, None
        return {"reasoning": {"enabled": False}}, None

    if _is_deepseek_native(settings.base_url) or _is_deepseek_model(model):
        # DeepSeek v4+ requires thinking.type=adaptive + output_config.effort;
        # the old "enabled"/"disabled" values are no longer accepted.
        # Also covers DeepSeek models proxied through non-native gateways (e.g. QClaw).
        if _thinking:
            extra_body: dict[str, Any] = {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _adaptive_output_effort(_effort)},
            }
            return extra_body, _effort or "medium"
        else:
            extra_body = {
                "thinking": {"type": "disabled"},
            }
            return extra_body, None

    if _is_minimax(settings.base_url):
        # MiniMax (api.minimax.io):
        # - thinking.type only accepts "adaptive" (on) or "disabled" (off); no budget_tokens
        # - reasoning_split=True exposes thinking via reasoning_content / reasoning_details
        # - M2.x cannot disable thinking; "disabled" is accepted but ignored
        if _thinking:
            extra_body = {
                "thinking": {"type": "adaptive"},
                "reasoning_split": True,
            }
        else:
            extra_body = {
                "thinking": {"type": "disabled"},
                "reasoning_split": True,
            }
        # MiniMax does not use reasoning_effort
        return extra_body, None

    if _is_mimo(settings):
        # MiMo: DeepSeek-style reasoning via chat_template_kwargs.enable_thinking
        return resolve_mimo_thinking_extra_body(thinking=_thinking), (
            _effort or "medium" if _thinking else None
        )

    if not _thinking:
        return {}, None

    max_out = _completion_max_tokens(
        settings, extra_body={}, effort=_effort
    )

    if _is_packyapi(settings.base_url) and "claude" in model.lower():
        # Packy (e.g. claude-officially): budget_tokens only; reasoning_effort rejected.
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            None,
        )

    if _is_kkai_openai_proxy(settings.base_url):
        # KKAI claude-opus-4-5: reasoning_effort -> 503 paprika_mode on some routes.
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            None,
        )

    if _model_uses_claude_adaptive(model):
        # Yunwu / New-API style gateways: Opus 4.7+ needs adaptive thinking.
        return (
            {
                "thinking": {"type": "adaptive"},
                "output_config": {"effort": _adaptive_output_effort(_effort)},
            },
            _effort or "medium",
        )

    if "claude" in model.lower():
        budget = _effort_budget_tokens(_effort, max_output=max_out)
        return (
            {"thinking": {"type": "enabled", "budget_tokens": budget}},
            _effort or "medium",
        )

    # Other models on OpenAI-compatible proxies (o-series, deepseek-reasoner, etc.)
    return {}, _effort or "medium"


class DeepSeekClient:
    """Thin wrapper around the OpenAI-compatible DeepSeek API."""

    def __init__(self, settings: AIProviderSettings, logger_: logging.Logger | None = None) -> None:
        self._settings = settings
        self._log = logger_ or logger

    def update_provider(self, settings: AIProviderSettings) -> None:
        """Replace in-memory provider settings (e.g. after QClaw auto-fallback)."""
        self._settings = settings

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        context_window: int | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Send *messages* to the DeepSeek API and return a structured reply.

        Raises CancelledError if cancel_token is set before the call.
        Never sends temperature/top_p/presence_penalty/frequency_penalty.
        """
        # Check cancellation before making the network call
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        extra_body, _effort = _resolve_thinking_params(
            self._settings, thinking=thinking, reasoning_effort=reasoning_effort
        )
        extra_body = {**extra_body, **_openclaw_agent_request_extra(self._settings)}
        api_messages, system_param = _prepare_api_messages(self._settings, messages)
        if system_param:
            extra_body = {**extra_body, "system": system_param}
        _thinking_on = _thinking_enabled(extra_body, _effort)
        _max_tokens = _completion_max_tokens(
            self._settings, extra_body=extra_body, effort=_effort
        )

        if self._settings.api_format == "anthropic":
            from pa_agent.ai.anthropic_compat import anthropic_messages_chat

            return anthropic_messages_chat(
                self._settings,
                api_messages,
                extra_body=extra_body,
                max_tokens=_max_tokens,
                cancel_token=cancel_token,
                timeout_s=timeout_s,
                logger_=self._log,
            )

        masked_key = mask_secret(self._settings.api_key)
        self._log.debug(
            "DeepSeekClient.chat: model=%s thinking=%s effort=%s max_tokens=%s "
            "system_hoisted=%s key=...%s msgs=%d",
            self._settings.model,
            _thinking_on,
            _effort,
            _max_tokens,
            bool(system_param),
            masked_key[-4:] if len(masked_key) >= 4 else "****",
            len(api_messages),
        )

        if _OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        client = _openai_client(self._settings)

        t0 = time.monotonic()
        create_kwargs: dict[str, Any] = {
            "model": _effective_api_model(self._settings),
            "messages": api_messages,
            "timeout": timeout_s,
            "max_tokens": _max_tokens,
        }
        if extra_body:
            create_kwargs["extra_body"] = extra_body
        if _effort is not None:
            create_kwargs["reasoning_effort"] = _effort
        # When thinking mode is OFF, set temperature=0 for maximum instruction-following
        # fidelity and JSON format compliance.  Thinking mode is incompatible with
        # temperature (DeepSeek/Anthropic spec), so we only inject it when safe.
        if not _thinking_on:
            create_kwargs["temperature"] = 0
        try:
            response = client.chat.completions.create(
                **create_kwargs,
                # IMPORTANT: do NOT add temperature, top_p, presence_penalty,
                # frequency_penalty — they are incompatible with thinking mode.
            )
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("DeepSeekClient API error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        raw_text_response = _decode_provider_text(response)
        if raw_text_response is not None:
            _raise_if_html_response(raw_text_response, base_url=self._settings.base_url)
            usage = AIUsage()
            content = raw_text_response
            reasoning_content = ""
            request_id = ""
            model_name = ""
            if _is_empty_ai_response(
                content=content,
                reasoning_content=reasoning_content,
                request_id=request_id,
                model_name=model_name,
                usage=usage,
            ):
                raise EmptyAIResponseError(
                    "Provider returned an empty text response: no content, no reasoning, "
                    "no request id/model, and zero token usage."
                )
            raw = {
                "id": request_id,
                "model": model_name,
                "content": content,
                "reasoning_content": reasoning_content,
                "usage": {
                    "prompt_tokens": 0,
                    "cached_prompt_tokens": 0,
                    "cache_miss_tokens": 0,
                    "cache_hit_rate_pct": 0.0,
                    "completion_tokens": 0,
                    "total_tokens": 0,
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

        msg = response.choices[0].message
        content = msg.content or ""
        _raise_if_html_response(content, base_url=self._settings.base_url)
        reasoning_content = getattr(msg, "reasoning_content", None) or ""
        # MiniMax with reasoning_split=True may also use reasoning_details
        if not reasoning_content:
            details = getattr(msg, "reasoning_details", None)
            if details:
                parts = []
                for detail in details:
                    t = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
                    if t:
                        parts.append(t)
                reasoning_content = "".join(parts)
        # OpenRouter surfaces reasoning as message.reasoning (string)
        if not reasoning_content:
            reasoning_content = getattr(msg, "reasoning", None) or ""

        if _is_mimo(self._settings):
            store_reasoning_from_response(
                api_messages,
                response_message_dict(content, reasoning_content, msg),
                _MIMO_REASONING_CACHE,
            )

        # Build usage
        u = response.usage
        usage = AIUsage(
            prompt_tokens=getattr(u, "prompt_tokens", 0),
            cached_prompt_tokens=getattr(
                getattr(u, "prompt_tokens_details", None), "cached_tokens", 0
            ) if u else 0,
            completion_tokens=getattr(u, "completion_tokens", 0),
            total_tokens=getattr(u, "total_tokens", 0),
        )

        request_id = getattr(response, "id", "") or ""
        model_name = getattr(response, "model", "") or ""

        if _is_empty_ai_response(
            content=content,
            reasoning_content=reasoning_content,
            request_id=request_id,
            model_name=model_name,
            usage=usage,
        ):
            raise EmptyAIResponseError(
                "Provider returned an empty response: no content, no reasoning, "
                "no request id/model, and zero token usage."
            )

        # Build raw dict for debug tab — mask API key if it somehow appears
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

        self._log.debug(
            "DeepSeekClient.chat done: latency=%.0f ms tokens=%d/%d",
            latency_ms, usage.prompt_tokens, usage.completion_tokens,
        )

        # Log KV-cache hit rate so operators can monitor savings.
        # DeepSeek硬盘缓存：prompt_cache_hit_tokens 是命中缓存的 token 数。
        if usage.prompt_tokens > 0:
            hit_rate = usage.cached_prompt_tokens / usage.prompt_tokens * 100
            self._log.info(
                "KV-cache: hit=%d miss=%d total_prompt=%d hit_rate=%.1f%%",
                usage.cached_prompt_tokens,
                usage.prompt_tokens - usage.cached_prompt_tokens,
                usage.prompt_tokens,
                hit_rate,
            )

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )

    def stream_chat(
        self,
        messages: list[dict[str, Any]],
        *,
        on_reasoning_token: Callable[[str], None] | None = None,
        on_content_token: Callable[[str], None] | None = None,
        thinking: bool | None = None,
        reasoning_effort: str | None = None,
        cancel_token: "CancelToken | None" = None,
        timeout_s: float = 600.0,
    ) -> AIReply:
        """Stream *messages* to the DeepSeek API, calling callbacks per token.

        Follows the official DeepSeek streaming example exactly:
        - reasoning_content tokens arrive first (thinking phase)
        - content tokens arrive after (answer phase)
        - delta.reasoning_content is None (not empty string) when absent

        Parameters
        ----------
        on_reasoning_token:
            Called with each reasoning/thinking token chunk as it arrives.
        on_content_token:
            Called with each content token chunk as it arrives.

        Returns the same AIReply as chat() once the stream is complete.
        Raises CancelledError if cancel_token is set before or during the call.
        """
        if cancel_token is not None and cancel_token.is_set():
            raise CancelledError("Request cancelled before API call")

        extra_body, _effort = _resolve_thinking_params(
            self._settings, thinking=thinking, reasoning_effort=reasoning_effort
        )
        extra_body = {**extra_body, **_openclaw_agent_request_extra(self._settings)}
        api_messages, system_param = _prepare_api_messages(self._settings, messages)
        if system_param:
            extra_body = {**extra_body, "system": system_param}
        _thinking_on = _thinking_enabled(extra_body, _effort)
        _max_tokens = _completion_max_tokens(
            self._settings, extra_body=extra_body, effort=_effort
        )

        if self._settings.api_format == "anthropic":
            from pa_agent.ai.anthropic_compat import anthropic_messages_stream_chat

            return anthropic_messages_stream_chat(
                self._settings,
                api_messages,
                extra_body=extra_body,
                max_tokens=_max_tokens,
                on_reasoning_token=on_reasoning_token,
                on_content_token=on_content_token,
                cancel_token=cancel_token,
                timeout_s=timeout_s,
                logger_=self._log,
            )

        self._log.info(
            "DeepSeekClient.stream_chat: model=%s thinking=%s reasoning_effort=%s "
            "max_tokens=%s system_hoisted=%s msgs=%d",
            self._settings.model,
            _thinking_on,
            _effort,
            _max_tokens,
            bool(system_param),
            len(api_messages),
        )

        if _OpenAI is None:
            raise RuntimeError("openai package is not installed") from _OPENAI_IMPORT_ERROR

        client = _openai_client(self._settings)

        t0 = time.monotonic()
        reasoning_content = ""
        content = ""
        request_id = ""
        model_name = ""
        prompt_tokens = 0
        completion_tokens = 0
        total_tokens = 0
        cached_tokens = 0

        try:
            # Build kwargs with stream_options to get usage in the final chunk.
            # Some providers may not support it; if the create() call itself
            # rejects stream_options we retry without it.
            stream_kwargs: dict[str, Any] = {
                "model": _effective_api_model(self._settings),
                "messages": api_messages,
                "timeout": timeout_s,
                "max_tokens": _max_tokens,
                "stream": True,
                "stream_options": {"include_usage": True},
            }
            if extra_body:
                stream_kwargs["extra_body"] = extra_body
            if _effort is not None:
                stream_kwargs["reasoning_effort"] = _effort

            try:
                stream = client.chat.completions.create(**stream_kwargs)
            except Exception:
                # Retry without stream_options if provider rejects it
                self._log.debug("stream_options not supported; retrying without it")
                stream_kwargs.pop("stream_options", None)
                stream = client.chat.completions.create(**stream_kwargs)

            for chunk in stream:
                # Check cancellation on each chunk
                if cancel_token is not None and cancel_token.is_set():
                    raise CancelledError("Request cancelled during streaming")

                raw_chunk_text = _decode_provider_text(chunk)
                if raw_chunk_text is not None:
                    _raise_if_html_response(
                        raw_chunk_text, base_url=self._settings.base_url
                    )
                    content += raw_chunk_text
                    if on_content_token is not None:
                        on_content_token(raw_chunk_text)
                    continue

                # Extract usage from the final chunk (stream_options)
                if hasattr(chunk, "usage") and chunk.usage is not None:
                    u = chunk.usage
                    prompt_tokens = getattr(u, "prompt_tokens", 0) or prompt_tokens
                    completion_tokens = getattr(u, "completion_tokens", 0) or completion_tokens
                    total_tokens = getattr(u, "total_tokens", 0) or total_tokens
                    details = getattr(u, "prompt_tokens_details", None)
                    cached_tokens = getattr(details, "cached_tokens", 0) if details else cached_tokens

                if not getattr(chunk, "choices", None):
                    continue

                request_id = request_id or (getattr(chunk, "id", "") or "")
                model_name = model_name or (getattr(chunk, "model", "") or "")

                choice0 = chunk.choices[0]
                delta = getattr(choice0, "delta", None)
                if delta is None:
                    continue

                # Official pattern: reasoning_content is None when absent, not ""
                # reasoning_content arrives first (thinking phase), then content
                # MiniMax with reasoning_split=True uses delta.reasoning_details[].text
                # instead of delta.reasoning_content.
                r = getattr(delta, "reasoning_content", None)
                if not r:
                    # MiniMax streaming: reasoning_details is a list of dicts
                    details = getattr(delta, "reasoning_details", None)
                    if details:
                        for detail in details:
                            t = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
                            if t:
                                r = (r or "") + t
                if not r:
                    # OpenRouter streaming: reasoning arrives as delta.reasoning (string)
                    r = getattr(delta, "reasoning", None)
                if r:
                    reasoning_content += r
                    if on_reasoning_token is not None:
                        on_reasoning_token(r)
                elif delta.content:
                    content += delta.content
                    if on_content_token is not None:
                        on_content_token(delta.content)

        except CancelledError:
            raise
        except Exception as exc:
            latency_ms = (time.monotonic() - t0) * 1000
            self._log.error("DeepSeekClient stream error after %.0f ms: %s", latency_ms, exc)
            raise

        latency_ms = (time.monotonic() - t0) * 1000

        usage = AIUsage(
            prompt_tokens=prompt_tokens,
            cached_prompt_tokens=cached_tokens,
            completion_tokens=completion_tokens,
            total_tokens=total_tokens,
        )

        _raise_if_html_response(content, base_url=self._settings.base_url)

        if _is_empty_ai_response(
            content=content,
            reasoning_content=reasoning_content,
            request_id=request_id,
            model_name=model_name,
            usage=usage,
        ):
            self._log.warning(
                "Provider returned an empty stream; retrying once without stream_options"
            )
            reasoning_content = ""
            content = ""
            request_id = ""
            model_name = ""
            prompt_tokens = 0
            completion_tokens = 0
            total_tokens = 0
            cached_tokens = 0

            simple_stream_kwargs: dict[str, Any] = {
                "model": _effective_api_model(self._settings),
                "messages": api_messages,
                "timeout": timeout_s,
                "max_tokens": _max_tokens,
                "stream": True,
            }
            if extra_body:
                simple_stream_kwargs["extra_body"] = extra_body
            if _effort is not None:
                simple_stream_kwargs["reasoning_effort"] = _effort

            try:
                simple_stream = client.chat.completions.create(**simple_stream_kwargs)
                for chunk in simple_stream:
                    if cancel_token is not None and cancel_token.is_set():
                        raise CancelledError("Request cancelled during streaming")

                    raw_chunk_text = _decode_provider_text(chunk)
                    if raw_chunk_text is not None:
                        _raise_if_html_response(
                            raw_chunk_text, base_url=self._settings.base_url
                        )
                        content += raw_chunk_text
                        if on_content_token is not None:
                            on_content_token(raw_chunk_text)
                        continue

                    if hasattr(chunk, "usage") and chunk.usage is not None:
                        u = chunk.usage
                        prompt_tokens = getattr(u, "prompt_tokens", 0) or prompt_tokens
                        completion_tokens = getattr(u, "completion_tokens", 0) or completion_tokens
                        total_tokens = getattr(u, "total_tokens", 0) or total_tokens
                        details = getattr(u, "prompt_tokens_details", None)
                        cached_tokens = getattr(details, "cached_tokens", 0) if details else cached_tokens

                    if not getattr(chunk, "choices", None):
                        continue

                    request_id = request_id or (getattr(chunk, "id", "") or "")
                    model_name = model_name or (getattr(chunk, "model", "") or "")
                    choice0 = chunk.choices[0]
                    delta = getattr(choice0, "delta", None)
                    if delta is None:
                        continue

                    r = getattr(delta, "reasoning_content", None)
                    if not r:
                        details = getattr(delta, "reasoning_details", None)
                        if details:
                            for detail in details:
                                t = detail.get("text") if isinstance(detail, dict) else getattr(detail, "text", None)
                                if t:
                                    r = (r or "") + t
                    if not r:
                        r = getattr(delta, "reasoning", None)
                    if r:
                        reasoning_content += r
                        if on_reasoning_token is not None:
                            on_reasoning_token(r)
                    elif delta.content:
                        content += delta.content
                        if on_content_token is not None:
                            on_content_token(delta.content)
            except CancelledError:
                raise
            except Exception as exc:
                self._log.warning("Simplified stream retry failed: %s", exc)

            usage = AIUsage(
                prompt_tokens=prompt_tokens,
                cached_prompt_tokens=cached_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
            )

        _raise_if_html_response(content, base_url=self._settings.base_url)

        if _is_empty_ai_response(
            content=content,
            reasoning_content=reasoning_content,
            request_id=request_id,
            model_name=model_name,
            usage=usage,
        ):
            self._log.warning(
                "Provider returned empty streams; retrying once with non-stream chat"
            )
            fallback = self.chat(
                messages,
                thinking=thinking,
                reasoning_effort=reasoning_effort,
                cancel_token=cancel_token,
                timeout_s=timeout_s,
            )
            if fallback.reasoning_content and on_reasoning_token is not None:
                on_reasoning_token(fallback.reasoning_content)
            if fallback.content and on_content_token is not None:
                on_content_token(fallback.content)
            return fallback

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

        self._log.info(
            "DeepSeekClient.stream_chat done: latency=%.0f ms "
            "reasoning_chars=%d content_chars=%d deepseek_thinking=%s effort=%s",
            latency_ms,
            len(reasoning_content),
            len(content),
            _thinking_on,
            _effort,
        )

        # Log KV-cache hit rate for stream calls as well.
        if usage.prompt_tokens > 0:
            hit_rate = usage.cached_prompt_tokens / usage.prompt_tokens * 100
            self._log.info(
                "KV-cache: hit=%d miss=%d total_prompt=%d hit_rate=%.1f%%",
                usage.cached_prompt_tokens,
                usage.prompt_tokens - usage.cached_prompt_tokens,
                usage.prompt_tokens,
                hit_rate,
            )
        if not content.strip():
            self._log.warning(
                "API returned empty content (model=%s base_url=%s). "
                "Check 原始 tab Raw Response; for KKAI/Claude ensure model ID and token group match.",
                self._settings.model,
                self._settings.base_url,
            )
        if _thinking_on and len(reasoning_content) < 80:
            self._log.warning(
                "Thinking enabled but reasoning_content is very short (%d chars). "
                "For KKAI/Claude use reasoning_effort (not DeepSeek extra_body); "
                "check model ID, token group, and reasoning_effort=%s.",
                len(reasoning_content),
                _effort,
            )

        if _is_mimo(self._settings):
            store_reasoning_from_response(
                api_messages,
                {
                    "role": "assistant",
                    "content": content,
                    "reasoning_content": reasoning_content,
                },
                _MIMO_REASONING_CACHE,
            )

        return AIReply(
            content=content,
            reasoning_content=reasoning_content,
            raw=raw,
            usage=usage,
            request_id=request_id,
            latency_ms=latency_ms,
        )
