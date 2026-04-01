"""
Cautious Giggle — HFT Scalping Mode Launcher
Starts the Server_AGI in high-frequency scalping mode with M1 timeframe,
aggressive risk parameters, and fast decision loop (5-second intervals).

Usage:
    python start_hft.py [--live]

This runs alongside the standard M5 Server_AGI. Each has its own:
  - Config file (config_hft.yaml)
  - Lock file (server_agi_hft.lock)
  - Log files (server_hft.log, audit_events_hft.jsonl, trade_events_hft.jsonl)
  - Magic number (506 vs 505) to avoid trade conflicts
"""
import os
import sys
import subprocess

from Python.config_utils import load_project_config

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PYTHON = sys.executable
CONFIG_PATH = os.path.join(BASE_DIR, "config_hft.yaml")
ENV_PATH = os.path.join(BASE_DIR, ".env")


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as handle:
        for raw_line in handle:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            value = value.strip()
            if key and key not in os.environ:
                os.environ[key] = value


def _runtime_env() -> dict[str, str]:
    _load_env_file(ENV_PATH)
    env = os.environ.copy()
    if not env.get("AGI_TOKEN", "").strip():
        raise RuntimeError("AGI_TOKEN must be set in the environment or .env before starting HFT mode.")
    env.setdefault("AGI_CONFIG", CONFIG_PATH)
    env.setdefault("AGI_MODE_TAG", "hft")
    env.setdefault("AGI_LOOP_SEC", "5")
    env.setdefault("AGI_HEARTBEAT_SEC", "300")
    env.setdefault("AGI_SYMBOL_CARD_SEC", "60")
    env.setdefault("AGI_TRADE_LEARN_SEC", "300")
    prev_cfg = os.environ.get("AGI_CONFIG")
    try:
        os.environ.update(env)
        os.environ["AGI_CONFIG"] = CONFIG_PATH
        load_project_config(BASE_DIR, live_mode=True)
    finally:
        if prev_cfg is None:
            os.environ.pop("AGI_CONFIG", None)
        else:
            os.environ["AGI_CONFIG"] = prev_cfg
    return env

def main():
    args = [PYTHON, "-m", "Python.Server_AGI"]
    if "--live" in sys.argv:
        args.append("--live")

    env = _runtime_env()

    print(f"[HFT] Starting high-frequency scalping mode")
    print(f"[HFT] Config: {env['AGI_CONFIG']}")
    print(f"[HFT] Loop interval: {env['AGI_LOOP_SEC']} seconds")
    print(f"[HFT] Timeframe: M1")
    print(f"[HFT] Magic: 506")
    print(f"[HFT] Logs: logs/server_hft.log")
    if os.path.exists(ENV_PATH):
        print("[HFT] Loaded local .env overrides")
    print()

    proc = subprocess.run(args, cwd=BASE_DIR, env=env)
    sys.exit(proc.returncode)


if __name__ == "__main__":
    main()
