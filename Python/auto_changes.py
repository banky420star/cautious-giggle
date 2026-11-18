"""Self-update log: every change the bot makes to itself, in plain English, as it happens.

The bot adapts on its own: voters are muted when their record in a market condition turns losing and come
back when it recovers; kinds of trade (setups) move between learning / proven / blocked; AI models are
queued for retraining and pass or fail their exams. This module notices each of those flips and writes one
line per change to logs/auto_changes.jsonl, which the war room shows. Risk limits are never changed
automatically (only by a person, in config.yaml).
"""
from __future__ import annotations

import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")
OUT = os.path.join(LOG, "auto_changes.jsonl")
STATE = os.path.join(LOG, "auto_changes_state.json")
NAMES = {"BTCUSDm": "Bitcoin", "XAUUSDm": "Gold", "EURUSDm": "EUR/USD", "USDJPYm": "USD/JPY", "XAGUSDm": "Silver",
         "ETHUSDm": "Ether", "USOILm": "Oil"}
SETUP_WORDS = {"full": "proven (normal size)", "probe": "learning (smallest size)", "retest": "re-test", "block": "BLOCKED"}

_state: dict | None = None
_last_slow = 0.0


def _load() -> dict:
    global _state
    if _state is None:
        try:
            with open(STATE, encoding="utf-8") as f:
                _state = json.load(f)
        except (OSError, ValueError):
            _state = {}
        _state.setdefault("muted", {})
        _state.setdefault("setups", {})
        _state.setdefault("grad", {})
    return _state


def _save():
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(_state, f)
    os.replace(tmp, STATE)


def log(kind: str, text: str, **extra) -> None:
    with open(OUT, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": time.time(), "kind": kind, "text": text, **extra}, default=str) + "\n")


def _describe(key: str) -> str:
    try:
        from Python.indicator_council import describe
        return describe(key)
    except Exception:
        return key


def observe_council(symbol: str, context: str | None, ignored: list, assess: dict) -> None:
    """Voters muted / back, per market and market condition (so normal condition changes aren't 'changes')."""
    try:
        st = _load()
        key = f"{symbol}|{context}"
        now, before = set(ignored or []), st["muted"].get(key)
        st["muted"][key] = sorted(now)
        changed = False
        if before is not None:
            mkt = NAMES.get(symbol, symbol)
            for v in sorted(now - set(before)):
                a = assess.get(v) or {}
                log("muted", f"Muted {_describe(v)} on {mkt} ({context}): its record there turned losing "
                             f"({a.get('edge_bps', '?')} bps per call after costs over {round(float(a.get('n_all', 0) or 0))} calls).",
                    symbol=symbol, voter=v)
                changed = True
            for v in sorted(set(before) - now):
                log("restored", f"{_describe(v)} counts again on {NAMES.get(symbol, symbol)} ({context}): its record recovered.",
                    symbol=symbol, voter=v)
                changed = True
        if changed or before is None:
            _save()
    except Exception:
        pass


def observe_slow(setup_rows: list[dict] | None = None) -> None:
    """Setup verdicts and training status; checked at most once a minute."""
    global _last_slow
    if time.time() - _last_slow < 60:
        return
    _last_slow = time.time()
    try:
        st = _load()
        first_setups = not st["setups"]
        for r in setup_rows or []:
            prev = st["setups"].get(r["key"])
            if prev is not None and prev != r["verdict"]:
                log("setup", f"Trade type '{r['label']}': {SETUP_WORDS.get(prev, prev)} → {SETUP_WORDS.get(r['verdict'], r['verdict'])}. "
                             f"{r.get('why', '')} ({r['n']} trades, net ${float(r['net']):+.2f}).", setup=r["key"])
            st["setups"][r["key"]] = r["verdict"]
        try:
            with open(os.path.join(LOG, "graduation.json"), encoding="utf-8") as f:
                grad = json.load(f).get("symbols") or {}
        except (OSError, ValueError):
            grad = {}
        for sym, v in grad.items():
            prev, cur = st["grad"].get(sym), v.get("status")
            if prev is not None and prev != cur:
                log("training", f"{NAMES.get(sym, sym)} AI: {prev.replace('_', ' ')} → {str(cur).replace('_', ' ')}. {v.get('why', '')}",
                    symbol=sym)
            st["grad"][sym] = cur
        if first_setups or setup_rows or grad:
            _save()
    except Exception:
        pass


def recent(n: int = 40) -> list[dict]:
    try:
        with open(OUT, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return [json.loads(x) for x in reversed(lines)]
    except (OSError, ValueError):
        return []
