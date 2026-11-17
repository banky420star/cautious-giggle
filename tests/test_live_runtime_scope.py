import json
import re
from pathlib import Path

import pytest
import yaml


def _allowed_symbols() -> set[str]:
    """Configured trading symbols plus markets that graduated (passed their exam)."""
    allowed = {"BTCUSDm", "XAUUSDm"}
    cfg_path = Path("config.yaml")
    if cfg_path.exists():
        cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8")) or {}
        allowed |= set((cfg.get("trading") or {}).get("symbols") or [])
    grad = Path("logs") / "graduation.json"
    if grad.exists():
        st = (json.loads(grad.read_text(encoding="utf-8")).get("symbols") or {})
        allowed |= {s for s, v in st.items()
                    if v.get("status") == "graduated" or (v.get("status") in ("training", "retrain_queued") and v.get("was_live"))}
    return allowed


def test_recent_live_runtime_decisions_only_for_approved_markets():
    log_path = Path("logs") / "server.log"
    if not log_path.exists():
        pytest.skip("no live server log in this checkout")

    lines = log_path.read_text(encoding="utf-8", errors="replace").splitlines()
    recent = [line for line in lines if "DECISION " in line][-40:]
    if not recent:
        pytest.skip("no decisions logged yet")

    try:
        from tools import account_profiles as ap
        mode = ap.active()
    except Exception:
        mode = None
    if mode == "demo":
        pytest.skip("demo account: demo_learning trades every market on purpose (this guard is for live)")

    traded = {m.group(1) for line in recent if (m := re.search(r"DECISION (\S+)", line))}
    assert traded <= _allowed_symbols(), f"decisions for unapproved markets: {traded - _allowed_symbols()}"
