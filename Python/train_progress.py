"""Tiny live-progress channel from the trainers to the dashboard (logs/training_progress.json)."""
from __future__ import annotations

import json
import os
import time

PATH = os.path.join(os.environ.get("AGI_LOG_DIR") or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"), "training_progress.json")


class ProgressWriter:
    def __init__(self, kind: str, symbol: str, min_interval: float = 0.5):
        self.state = {"kind": kind, "symbol": symbol, "started": time.time(), "history": [], "recent": []}
        self.min_interval = float(min_interval)
        self._last = 0.0

    def update(self, force: bool = False, recent: float | None = None, **fields) -> None:
        self.state.update(fields)
        if recent is not None:
            self.state["recent"] = (self.state["recent"] + [float(recent)])[-240:]
        now = time.time()
        if not force and now - self._last < self.min_interval:
            return
        self._last = now
        self.state["ts"] = now
        # the shared file (latest writer wins) plus one per market, so parallel runs each keep a bar
        for path in (PATH, symbol_path(self.state.get("symbol"))):
            if not path:
                continue
            try:
                os.makedirs(os.path.dirname(path), exist_ok=True)
                tmp = f"{path}.{os.getpid()}.tmp"
                with open(tmp, "w", encoding="utf-8") as f:
                    json.dump(self.state, f)
                os.replace(tmp, path)
            except Exception:
                pass

    def epoch_done(self, **row) -> None:
        self.state["history"] = (self.state["history"] + [row])[-200:]
        self.update(force=True)

    def finish(self, **fields) -> None:
        self.update(force=True, done=True, **fields)


def symbol_path(symbol) -> str | None:
    safe = "".join(ch for ch in str(symbol or "") if ch.isalnum())
    return os.path.join(os.path.dirname(PATH), f"training_progress_{safe}.json") if safe else None


def read_symbol(symbol) -> dict:
    try:
        with open(symbol_path(symbol), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def read() -> dict:
    try:
        with open(PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
