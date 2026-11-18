"""Who called it: for every trade, which voters were on the right side and what they said at entry.

At entry (trade_open) the bot's latest council votes, each voter's reading/reason, the external models'
reasons and the LSTM/PPO view for that market are saved with the ticket. When the trade closes, each voter
that had a view is marked RIGHT or WRONG:
    agreed with the trade and it made money after costs  -> right
    agreed with the trade and it lost                    -> wrong
    opposed the trade and it lost                        -> right (it warned against it)
    opposed the trade and it made money                  -> wrong
Abstaining voters are listed as "no view". Files:
    logs/trade_entries.jsonl         one line per opened trade, with the full ballot at entry
    logs/voter_scorecard.jsonl       one line per closed trade: outcome + every voter's call and reason
    logs/voter_scorecard_summary.json running tally per voter: calls, right, wrong, hit rate (overall and
                                      per market), so the war room can show who has really been calling it.
Read-only for trading: nothing here changes a decision.
"""
from __future__ import annotations

import json
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")
ENTRIES = os.path.join(LOG, "trade_entries.jsonl")
CARD = os.path.join(LOG, "voter_scorecard.jsonl")
SUMMARY = os.path.join(LOG, "voter_scorecard_summary.json")
AUDIT = os.path.join(LOG, "audit_events.jsonl")
VIEW = 0.05


def _tail_events(path: str, n_bytes: int = 600_000) -> list[dict]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n_bytes))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def _json(path: str) -> dict:
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def _name(key: str) -> str:
    try:
        from Python.indicator_council import describe
        return describe(key)
    except Exception:
        return key


def ballot(symbol: str) -> dict:
    """The latest view of every voter for `symbol`, with the reason each gave."""
    council = signal = None
    for e in reversed(_tail_events(AUDIT)):
        p = e.get("payload") or {}
        if p.get("symbol") != symbol:
            continue
        if council is None and e.get("event") == "council":
            council = p
        if signal is None and e.get("event") == "signal":
            signal = p
        if council and signal:
            break
    readings = ((_json(os.path.join(LOG, "council_readings.json")).get(symbol) or {}).get("readings") or {})
    details = _json(os.path.join(LOG, "external_voters.json")).get("details") or {}
    voters = {}
    for k, v in ((council or {}).get("votes") or {}).items():
        if v is None:
            continue
        det = details.get(f"{symbol}:{k}") or {}
        rd = readings.get(k) or {}
        voters[k] = {"name": _name(k), "vote": round(float(v), 3),
                     "reason": det.get("reason") or rd.get("text") or "",
                     "reading": rd.get("short"), "ignored": k in ((council or {}).get("ignored") or [])}
    if signal:
        lstm = float(signal.get("agi_counted", signal.get("agi_bias", 0)) or 0)
        raw = float(signal.get("agi_bias", 0) or 0)
        if raw:
            voters["lstm"] = {"name": "LSTM pattern model", "vote": round(raw, 3), "ignored": not lstm,
                              "reason": f"pattern model {'counted' if lstm else 'not counted (below its confidence bar)'}, "
                                        f"{float(signal.get('agi_confidence', signal.get('confidence', 0)) or 0):.0%} sure"}
        ppo = float(signal.get("ppo_exposure", 0) or 0)
        if ppo:
            voters["ppo"] = {"name": "PPO trader model", "vote": round(ppo, 3), "ignored": False,
                             "reason": "trader model (exploration: it has not passed its exam)"}
        voters["council"] = {"name": "Council (all indicators + your models)",
                             "vote": round(float(signal.get("council", 0) or 0), 3), "ignored": False,
                             "reason": f"{(council or {}).get('up', 0)} voters up, {(council or {}).get('down', 0)} down"}
    return {"context": (council or {}).get("context"), "signal": (signal or {}).get("exposure"), "voters": voters}


def on_open(trade: dict) -> None:
    try:
        rec = {"t": time.time(), **trade, **ballot(str(trade.get("symbol")))}
        with open(ENTRIES, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, default=str) + "\n")
    except Exception:
        pass


def _entry(ticket: int) -> dict | None:
    try:
        with open(ENTRIES, encoding="utf-8") as f:
            for ln in f:
                if f'"ticket": {ticket},' in ln or f'"ticket": {ticket}}}' in ln:
                    rec = json.loads(ln)
                    if int(rec.get("ticket", -1)) == int(ticket):
                        return rec
    except (OSError, ValueError):
        pass
    return None


def judge(entry: dict, profit: float, exit_price: float | None = None) -> dict:
    side = 1 if entry.get("side") == "BUY" else -1
    won = profit > 0
    calls = []
    for k, v in (entry.get("voters") or {}).items():
        vote = float(v.get("vote", 0) or 0)
        if abs(vote) <= VIEW:
            calls.append({"key": k, "name": v.get("name", k), "call": "no view", "vote": vote, "reason": v.get("reason")})
            continue
        agreed = (vote > 0) == (side > 0)
        right = agreed == won
        calls.append({"key": k, "name": v.get("name", k), "vote": vote, "agreed": agreed,
                      "call": "right" if right else "wrong", "ignored": bool(v.get("ignored")),
                      "reading": v.get("reading"), "reason": v.get("reason"),
                      "why": (("backed it and it made money" if won else "backed it and it lost") if agreed else
                              ("warned against it and it lost" if not won else "warned against it but it made money"))})
    move = None
    if exit_price and entry.get("open_price"):
        move = round((float(exit_price) / float(entry["open_price"]) - 1) * 1e4 * side, 2)
    return {"outcome": "win" if won else "loss", "move_bps": move, "calls": calls}


def _update_summary(sym: str, calls: list[dict]) -> None:
    s = _json(SUMMARY) or {"voters": {}}
    for c in calls:
        if c["call"] == "no view":
            continue
        v = s["voters"].setdefault(c["key"], {"name": c["name"], "calls": 0, "right": 0, "markets": {}})
        v["calls"] += 1
        v["right"] += c["call"] == "right"
        m = v["markets"].setdefault(sym, {"calls": 0, "right": 0})
        m["calls"] += 1
        m["right"] += c["call"] == "right"
    for v in s["voters"].values():
        v["hit"] = round(v["right"] / v["calls"], 3) if v["calls"] else None
    s["updated"] = time.time()
    tmp = SUMMARY + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(s, f, indent=1)
    os.replace(tmp, SUMMARY)


def on_close(closed: dict) -> dict | None:
    """Score one closing deal. Returns the scorecard line (or None if the entry ballot is unknown)."""
    try:
        entry = _entry(int(closed.get("ticket", 0) or 0))
        if not entry:
            return None
        res = judge(entry, float(closed.get("profit", 0) or 0), closed.get("price"))
        line = {"t": time.time(), "ticket": entry["ticket"], "symbol": entry.get("symbol"), "side": entry.get("side"),
                "open_price": entry.get("open_price"), "close_price": closed.get("price"),
                "profit": round(float(closed.get("profit", 0) or 0), 2), "context": entry.get("context"), **res}
        with open(CARD, "a", encoding="utf-8") as f:
            f.write(json.dumps(line, default=str) + "\n")
        _update_summary(str(entry.get("symbol")), res["calls"])
        return line
    except Exception:
        return None


def recent(n: int = 30) -> list[dict]:
    try:
        with open(CARD, encoding="utf-8") as f:
            lines = f.readlines()[-n:]
        return [json.loads(x) for x in reversed(lines)]
    except (OSError, ValueError):
        return []
