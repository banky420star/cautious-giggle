"""Demo exploration: let a model that failed the promotion gates trade small on a DEMO account
so the model memory can learn from real fills. Active wherever the demo rules apply: a demo account,
and a real-money account too when trading.demo_learning.same_on_live is set. Never
promotable to champion (the candidate dir carries an exploration.json marker)."""
from __future__ import annotations

import json
import os
import time

MARKER = "exploration.json"


def settings(cfg: dict) -> dict:
    d = ((cfg or {}).get("trading", {}) or {}).get("demo_exploration", {}) or {}
    return {"enabled": bool(d.get("enabled", False)), "size_scale": float(d.get("size_scale", 0.5))}


def account_is_demo(mt5) -> bool:
    """True only when the broker positively reports a demo account."""
    try:
        client = getattr(mt5, "client", None)
        if client is not None and hasattr(client, "account_details"):
            return bool((client.account_details() or {}).get("is_demo", False))
        info = mt5.account_info()
        return info is not None and int(getattr(info, "trade_mode", -1)) == 0  # ACCOUNT_TRADE_MODE_DEMO
    except Exception:
        return False


def same_on_live(cfg: dict | None) -> bool:
    """trading.demo_learning.same_on_live: a real-money account runs exactly like demo."""
    return bool((((cfg or {}).get("trading") or {}).get("demo_learning") or {}).get("same_on_live", False))


def demo_rules_apply(mt5, cfg: dict | None = None) -> bool:
    """Whether the demo settings (risk, stacking, all markets, learning, council, exploration, news)
    apply: always on a demo account, and on a live account when same_on_live is set."""
    return same_on_live(cfg) or account_is_demo(mt5)


def mark(candidate_dir: str, report: dict | None = None) -> None:
    with open(os.path.join(candidate_dir, MARKER), "w", encoding="utf-8") as f:
        json.dump({"ts": time.time(), "reason": "failed promotion gates; demo exploration only",
                   "gates": (report or {}).get("per_symbol_gates")}, f, default=str)


def is_exploration(model_dir: str | None) -> bool:
    return bool(model_dir) and os.path.exists(os.path.join(str(model_dir), MARKER))
