import os
from typing import Any

import yaml

DEFAULT_TRADING_SYMBOLS = ["BTCUSDm", "XAUUSDm"]


_ENV_REF_PREFIX = "ENV:"
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


def _normalize_scalar(raw: Any) -> str:
    return str(raw or "").strip()


def _is_env_ref(value: str) -> bool:
    return value.upper().startswith(_ENV_REF_PREFIX)


def _resolve_secret_value(
    raw_value: Any,
    *,
    field_name: str,
    required: bool,
    env_var_name: str | None = None,
) -> str:
    display_name = field_name.replace(".", " ")
    if env_var_name:
        env_value = _normalize_scalar(os.environ.get(env_var_name, ""))
        if env_value:
            return env_value

    value = _normalize_scalar(raw_value)
    if not value:
        if required:
            raise RuntimeError(
                f"Live mode blocked: {display_name} is not configured. "
                f"Set an environment variable or define {display_name} in config.yaml."
            )
        return ""

    if value in _PLACEHOLDERS:
        raise RuntimeError(
            f"Live mode blocked: {display_name} contains a placeholder value. "
            "Use a real secret or an ENV: reference."
        )

    if _is_env_ref(value):
        env_name = value.split(":", 1)[1].strip()
        if not env_name:
            raise RuntimeError(
                f"Live mode blocked: {display_name} uses an empty environment reference."
            )
        env_value = _normalize_scalar(os.environ.get(env_name, ""))
        if not env_value:
            raise RuntimeError(
                f"Live mode blocked: {display_name} uses {value} but {env_name} is not set."
            )
        return env_value

    return value


def load_project_config(project_root: str, live_mode: bool = False) -> dict[str, Any]:
    """
    Load config.yaml and enforce that live runs are not using placeholders.
    """
    cfg_path = os.environ.get("AGI_CONFIG") or os.path.join(project_root, "config.yaml")
    if not os.path.exists(cfg_path):
        raise FileNotFoundError(f"Missing config file: {cfg_path}")

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    if live_mode:
        tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
        mt5_cfg = cfg.get("mt5", {}) if isinstance(cfg, dict) else {}
        token = _resolve_secret_value(
            tel.get("token", ""),
            field_name="telegram.token",
            required=False,
            env_var_name="TELEGRAM_TOKEN",
        )
        chat_id = _resolve_secret_value(
            tel.get("chat_id", ""),
            field_name="telegram.chat_id",
            required=False,
            env_var_name="TELEGRAM_CHAT_ID",
        )
        if bool(token) != bool(chat_id):
            raise RuntimeError(
                "Live mode blocked: telegram.token and telegram.chat_id must both be set "
                "or both be empty. Use ENV: references or omit the telegram section."
            )

        effective_login = _resolve_secret_value(
            mt5_cfg.get("login", ""),
            field_name="mt5.login",
            required=True,
            env_var_name="MT5_LOGIN",
        )
        if effective_login == "0":
            raise RuntimeError(
                "Live mode blocked: MT5 login is not configured. "
                "Set MT5_LOGIN env var or mt5.login in config.yaml."
            )
        effective_password = _resolve_secret_value(
            mt5_cfg.get("password", ""),
            field_name="mt5.password",
            required=True,
            env_var_name="MT5_PASSWORD",
        )
        effective_server = _resolve_secret_value(
            mt5_cfg.get("server", ""),
            field_name="mt5.server",
            required=True,
            env_var_name="MT5_SERVER",
        )
        if not effective_password:
            raise RuntimeError(
                "Live mode blocked: MT5 password is not configured. "
                "Set MT5_PASSWORD env var or mt5.password in config.yaml."
            )
        if not effective_server:
            raise RuntimeError(
                "Live mode blocked: MT5 server is not configured. "
                "Set MT5_SERVER env var or mt5.server in config.yaml."
            )

    return cfg
