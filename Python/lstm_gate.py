"""Is this market's LSTM proven? (tools/lstm_eval.py writes logs/lstm_eval.json daily.)

Unproven = the bot ignores it: no vote, no size cut, and the LLM traders are told so. A missing or unreadable
report counts as unproven: on 2026-11-24 every market's LSTM lost money after the spread on unseen data.
"""
from __future__ import annotations

import json
import os
import time

_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "lstm_eval.json")
_cache = [0.0, {}]


def report() -> dict:
    if time.time() - _cache[0] > 600:
        try:
            with open(_PATH, encoding="utf-8") as f:
                _cache[1] = (json.load(f) or {}).get("markets") or {}
        except (OSError, ValueError):
            _cache[1] = {}
        _cache[0] = time.time()
    return _cache[1]


def proven(symbol: str) -> bool:
    return bool((report().get(str(symbol)) or {}).get("proven"))
