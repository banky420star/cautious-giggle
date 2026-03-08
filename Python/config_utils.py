import os
from typing import Any

import yaml


_PLACEHOLDERS = {
    "YOUR_BOT_TOKEN_HERE",
    "YOUR_CHAT_ID_HERE",
    "ENV:TELEGRAM_TOKEN",
    "ENV:TELEGRAM_CHAT_ID",
}


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

        if token in _PLACEHOLDERS or chat_id in _PLACEHOLDERS:
            raise RuntimeError(
                "Live mode blocked: config.yaml contains Telegram placeholders. "
                "Use real secrets via environment variables."
            )

    return cfg
