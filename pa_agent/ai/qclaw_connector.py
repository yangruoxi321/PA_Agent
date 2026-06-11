"""QClaw Gateway connector for PA Agent.

Detects the local QClaw Gateway, reads its endpoint and token from the
OpenClaw config file, and provides a function to populate AIProviderSettings
so PA Agent can use QClaw's models for two-stage analysis.

When possible, PA Agent starts a local reasoning relay (default port 19004)
inside the QClaw process tree. The relay forwards to QClaw's internal LLM
proxy and preserves ``reasoning_content`` in the stream.

Usage::

    from pa_agent.ai.qclaw_connector import detect_qclaw, qclaw_provider_settings

    if detect_qclaw():
        settings.provider = qclaw_provider_settings()
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from pa_agent.ai.qclaw_relay_manager import ensure_qclaw_relay

logger = logging.getLogger(__name__)

# Known QClaw config path (Windows)
_QCLAW_CONFIG_PATH = Path.home() / ".qclaw" / "openclaw.json"
# Fallback: Linux/macOS
_QCLAW_CONFIG_PATH_ALT = Path("~/.qclaw/openclaw.json").expanduser()

_RELAY_PROXY_PORT = 19004
_RELAY_PROXY_MODEL = "pool-deepseek-v4-flash"
_PUBLIC_GATEWAY_MODEL = "openclaw"


def is_openclaw_model(model: str | None) -> bool:
    """True when the user selected QClaw via model name ``openclaw``."""
    return (model or "").strip().lower() == _PUBLIC_GATEWAY_MODEL


def _find_qclaw_config() -> Path | None:
    """Return the first existing OpenClaw config file path, or None."""
    for candidate in (_QCLAW_CONFIG_PATH, _QCLAW_CONFIG_PATH_ALT):
        if candidate.exists():
            return candidate
    return None


def _read_qclaw_config(config_path: Path) -> dict | None:
    """Parse the OpenClaw JSON config file; returns None on error."""
    try:
        return json.loads(config_path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError) as exc:
        logger.debug("Failed to read QClaw config %s: %s", config_path, exc)
        return None


def detect_qclaw() -> bool:
    """Return True if QClaw Gateway is configured and chat completions are enabled."""
    config_path = _find_qclaw_config()
    if config_path is None:
        return False
    data = _read_qclaw_config(config_path)
    if data is None:
        return False
    gw = data.get("gateway", {})
    http = gw.get("http", {})
    eps = http.get("endpoints", {})
    chat = eps.get("chatCompletions", {})
    if not chat.get("enabled", False):
        logger.debug("QClaw chatCompletions endpoint not enabled")
        return False
    token = gw.get("auth", {}).get("token", "")
    if not token:
        logger.debug("QClaw gateway token is empty")
        return False
    return True


def _get_qclaw_gateway_info() -> tuple[str, int, str] | None:
    """Return (host, port, token) for the QClaw gateway, or None if unavailable."""
    config_path = _find_qclaw_config()
    if config_path is None:
        return None
    data = _read_qclaw_config(config_path)
    if data is None:
        return None
    gw = data.get("gateway", {})
    token = gw.get("auth", {}).get("token", "")
    if not token:
        return None
    port = int(gw.get("port", 51187))
    host = "127.0.0.1"
    bind = gw.get("bind", "127.0.0.1")
    if bind and bind not in ("0.0.0.0", "loopback"):
        host = bind
    return host, port, token


def _fetch_public_gateway_models(base_url: str, token: str, *, timeout: float = 2.0) -> list[str]:
    try:
        import httpx

        resp = httpx.get(
            f"{base_url.rstrip('/')}/models",
            headers={"Authorization": f"Bearer {token}"},
            timeout=timeout,
        )
        if resp.status_code != 200:
            return []
        data = resp.json()
        return [str(m.get("id", "")) for m in data.get("data", []) if m.get("id")]
    except Exception as exc:
        logger.debug("QClaw public gateway models probe failed for %s: %s", base_url, exc)
        return []


def _probe_public_gateway(base_url: str, token: str, *, timeout: float = 2.0) -> bool:
    return bool(_fetch_public_gateway_models(base_url, token, timeout=timeout))


def _pick_public_gateway_model(model_ids: list[str]) -> str:
    """Use flash when listed; otherwise openclaw (never auto-pick v4-pro pool)."""
    if _RELAY_PROXY_MODEL in model_ids:
        return _RELAY_PROXY_MODEL
    return _PUBLIC_GATEWAY_MODEL


def _resolve_qclaw_endpoint(
    host: str,
    port: int,
    token: str,
    *,
    prefer_relay: bool = True,
) -> tuple[str, str, str]:
    """Pick API base URL, model name, and human-readable mode label."""
    relay_base = f"http://127.0.0.1:{_RELAY_PROXY_PORT}"
    if prefer_relay:
        relay_ok, relay_msg = ensure_qclaw_relay(token, port=_RELAY_PROXY_PORT)
        if relay_ok:
            logger.info("QClaw relay ready at %s (%s)", relay_base, relay_msg)
            return relay_base, _RELAY_PROXY_MODEL, "中继代理（含 reasoning）"

    public_base = f"http://{host}:{port}/v1"
    model_ids = _fetch_public_gateway_models(public_base, token)
    if model_ids:
        public_model = _pick_public_gateway_model(model_ids)
        mode = "公开网关（无 reasoning 流）"
        if public_model != _PUBLIC_GATEWAY_MODEL:
            mode = f"公开网关 / {public_model}（无 reasoning 流）"
        else:
            logger.info(
                "QClaw public gateway has no pool-deepseek route; using model=%s "
                "(JSON compliance may be weaker than %s)",
                _PUBLIC_GATEWAY_MODEL,
                _RELAY_PROXY_MODEL,
            )
        logger.info("QClaw using public gateway at %s (model=%s)", public_base, public_model)
        return public_base, public_model, mode

    if prefer_relay:
        return relay_base, _RELAY_PROXY_MODEL, "中继代理（启动失败，仍将尝试连接）"
    return public_base, _PUBLIC_GATEWAY_MODEL, "公开网关"


def apply_qclaw_provider_to_settings(settings: Any) -> str | None:
    """Populate *settings.provider* from local QClaw (settings Save with model=openclaw).

    Returns None on success, or a user-facing error string.
    """
    if not detect_qclaw():
        return (
            "未检测到本地 QClaw Gateway。\n\n"
            "请确认：\n"
            "1. QClaw Gateway 正在运行\n"
            "2. config 中 chatCompletions 端点已启用\n"
            "3. gateway.auth.token 已配置"
        )

    resolved = qclaw_provider_settings()
    if resolved is None:
        return "QClaw 配置读取失败。"

    provider = settings.provider
    provider.model = resolved.model
    provider.base_url = resolved.base_url
    provider.api_key = resolved.api_key
    provider.thinking = resolved.thinking
    provider.reasoning_effort = resolved.reasoning_effort
    provider.context_window = resolved.context_window

    ok, health_msg = qclaw_health_check()
    if not ok:
        return f"QClaw 连通性检查失败：\n\n{health_msg}"
    return None


def qclaw_provider_settings(
    model: str | None = None,
    thinking: bool = True,
    reasoning_effort: str = "max",
    context_window: int = 2_000_000,
    *,
    prefer_relay: bool = True,
) -> "AIProviderSettings | None":
    """Return AIProviderSettings pre-filled for QClaw, or None if unavailable.

    Prefers the reasoning relay on port 19004 when it can be started inside the
    QClaw process tree. Falls back to the public gateway from config.
    """
    from pa_agent.config.settings import AIProviderSettings

    info = _get_qclaw_gateway_info()
    if info is None:
        return None
    host, port, token = info
    base_url, resolved_model, _mode = _resolve_qclaw_endpoint(
        host,
        port,
        token,
        prefer_relay=prefer_relay,
    )
    logger.info("QClaw detected at %s (model=%s)", base_url, resolved_model)
    return AIProviderSettings(
        model=model or resolved_model,
        base_url=base_url,
        api_key=token,
        thinking=thinking,
        reasoning_effort=reasoning_effort,
        context_window=context_window,
    )


def qclaw_health_check(*, prefer_relay: bool = True) -> tuple[bool, str]:
    """Perform a quick health check against the QClaw gateway.

    Returns a (ok, message) tuple.
    """
    info = _get_qclaw_gateway_info()
    if info is None:
        return False, "QClaw 配置文件未找到或 token 为空"

    host, port, token = info
    base_url, model, mode = _resolve_qclaw_endpoint(
        host,
        port,
        token,
        prefer_relay=prefer_relay,
    )

    try:
        import httpx

        if base_url.endswith(f":{_RELAY_PROXY_PORT}"):
            probe_url = f"{base_url.rstrip('/')}/health"
        else:
            probe_url = f"{base_url.rstrip('/')}/models"

        resp = httpx.get(
            probe_url,
            headers={"Authorization": f"Bearer {token}"},
            timeout=5.0,
        )
        if resp.status_code != 200:
            return False, f"QClaw 返回 HTTP {resp.status_code}: {resp.text[:200]}"

        if probe_url.endswith("/health"):
            detail = "reasoning 中继正常"
        else:
            models_data = resp.json()
            model_ids = [m.get("id", "?") for m in models_data.get("data", [])]
            detail = (
                f"可用模型: {', '.join(model_ids) if model_ids else '(列表为空)'}"
            )

        return (
            True,
            f"QClaw 连接正常（{mode}，推荐模型 {model}），{detail}",
        )
    except Exception as exc:
        return False, f"无法连接 QClaw ({base_url}): {exc}"
