import os
from typing import Any

import yaml

DEFAULT_TRADING_SYMBOLS = ["BTCUSDm", "XAUUSDm"]


_PLACEHOLDERS = {
    "YOUR_BOT_TOKEN_HERE",
    "YOUR_CHAT_ID_HERE",
}


def parse_symbol_list(raw: Any) -> list[str]:
    if isinstance(raw, (list, tuple)):
        return [str(item).strip() for item in raw if str(item).strip()]
    txt = str(raw or "").strip()
    if not txt:
        return []
    return [part.strip() for part in txt.split(",") if part.strip()]


def resolve_trading_symbols(
    cfg: dict[str, Any] | None,
    *,
    env_keys: tuple[str, ...] = (),
    fallback: list[str] | None = None,
) -> list[str]:
    fallback_symbols = list(fallback or DEFAULT_TRADING_SYMBOLS)
    for key in env_keys:
        raw = os.environ.get(key)
        if raw:
            symbols = parse_symbol_list(raw)
            if symbols:
                return symbols

    trading_cfg = (cfg or {}).get("trading", {}) if isinstance(cfg, dict) else {}
    symbols = parse_symbol_list(trading_cfg.get("symbols", []))
    return symbols or fallback_symbols


def load_project_config(project_root: str, live_mode: bool = False) -> dict[str, Any]:
    """
    Load config.yaml and enforce that live runs are not using placeholders.
    """
    cfg_path = os.path.join(project_root, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if live_mode:
        tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
        token = str(tel.get("token", "") or "").strip()
        chat_id = str(tel.get("chat_id", "") or "").strip()
        token_env = os.environ.get("TELEGRAM_TOKEN", "").strip()
        chat_env = os.environ.get("TELEGRAM_CHAT_ID", "").strip()

        token_is_env_ref = token.upper() == "ENV:TELEGRAM_TOKEN"
        chat_is_env_ref = chat_id.upper() == "ENV:TELEGRAM_CHAT_ID"

        if token in _PLACEHOLDERS or chat_id in _PLACEHOLDERS:
            raise RuntimeError(
                "Live mode blocked: config.yaml contains Telegram placeholders. "
                "Use real secrets via environment variables."
            )
        if token_is_env_ref and not token_env:
            raise RuntimeError(
                "Live mode blocked: TELEGRAM_TOKEN env var is not set while config.yaml uses ENV:TELEGRAM_TOKEN."
            )
        if chat_is_env_ref and not chat_env:
            raise RuntimeError(
                "Live mode blocked: TELEGRAM_CHAT_ID env var is not set while config.yaml uses ENV:TELEGRAM_CHAT_ID."
            )

    return cfg
