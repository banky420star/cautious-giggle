import shutil
import uuid
from pathlib import Path

import pytest

from Python.config_utils import load_project_config


def _mk_local_cfg_root() -> Path:
    root = Path(".tmp") / f"cfg_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_live_mode_rejects_placeholder_telegram_values(monkeypatch):
    for key in ("TELEGRAM_TOKEN", "TELEGRAM_CHAT_ID", "MT5_LOGIN", "MT5_PASSWORD", "MT5_SERVER", "AGI_CONFIG"):
        monkeypatch.delenv(key, raising=False)
    root = _mk_local_cfg_root()
    try:
        cfg = root / "config.yaml"
        cfg.write_text(
            """
telegram:
  token: "YOUR_BOT_TOKEN_HERE"
  chat_id: "YOUR_CHAT_ID_HERE"
""".strip(),
            encoding="utf-8",
        )

        with pytest.raises(RuntimeError):
            load_project_config(str(root), live_mode=True)
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_non_live_mode_allows_example_values(monkeypatch):
    monkeypatch.delenv("AGI_CONFIG", raising=False)
    root = _mk_local_cfg_root()
    try:
        cfg = root / "config.yaml"
        cfg.write_text(
            """
telegram:
  token: "YOUR_BOT_TOKEN_HERE"
  chat_id: "YOUR_CHAT_ID_HERE"
""".strip(),
            encoding="utf-8",
        )

        loaded = load_project_config(str(root), live_mode=False)
        assert loaded["telegram"]["token"] == "YOUR_BOT_TOKEN_HERE"
    finally:
        shutil.rmtree(root, ignore_errors=True)
