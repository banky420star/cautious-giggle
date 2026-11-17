"""Plain-English operations log: turns the bot's raw server.log lines into explained events.

Each event says what the bot did and why, e.g.
  "Planned a sell on gold: waiting up to 15 min for 3,867.48 (a 0.35 ATR pullback) instead of
   selling at 3,869.10 now, so the entry costs less."
"""
from __future__ import annotations

import datetime as dt
import os
import re

TS = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\.\d+ ([+-]\d{2}:\d{2})")

PLAN = re.compile(r"ENTRY PLAN (\S+) \| waiting up to ([\d.]+) min for ([\d.]+) \(now ([\d.]+), ([\d.]+) ATR pullback\)")
ENTRY = re.compile(r"ENTRY (\S+) \| (pullback|timeout_market) after (\d+)s \| (buy|sell) ([\d.]+) @ ([\d.]+) vs ([\d.]+) at signal "
                   r"\(([-+\d.]+) bps better\)")
STACK = re.compile(r"STACK (\S+) \| (\d+) trade\(s\) in profit .* adding trade (\d+) of (\d+)")
ACTION = re.compile(r"ACTION (\S+) \| req=(close|reduce|open) side=(\w+) volume=([\d.]+) target=([-\d.]+)")
MANAGE = re.compile(r"ORDER_SEND (\S+) \| action=manage .* sl=([\d.]+|None) tp=([\d.]+|None) .* retcode=10009")
MINLOT = re.compile(r"(\S+): using minimum lot ([\d.]+) \(stop risk ([\d.]+)% <= ceiling ([\d.]+)%\)")
SIZEBLOCK = re.compile(r"SIZE BLOCK (\S+) \| minimum lot ([\d.]+) would risk ([\d.]+)% of \$([\d.]+) at the stop \(cap ([\d.]+)%\); needs about \$(\d+)")
LOST = re.compile(r"MT5 connection lost")
HALT = re.compile(r"(risk halt|HALT|halted)[^\n]*", re.I)
NEWS = re.compile(r"NEWS (\S+) \| (.*)")


def _px(v: str) -> str:
    x = float(v)
    return f"{x:,.2f}" if x >= 100 else f"{x:,.4f}" if x < 10 else f"{x:,.3f}"


def _t(line: str) -> float | None:
    m = TS.match(line)
    if not m:
        return None
    try:
        return dt.datetime.fromisoformat(f"{m.group(1)}{m.group(2)}").timestamp()
    except ValueError:
        return None


def _tail(path: str, max_bytes: int) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - max_bytes))
            return f.read().decode("utf-8", "replace").splitlines()
    except OSError:
        return []


def parse(line: str, names: dict) -> dict | None:
    nm = lambda s: names.get(s, s)  # noqa: E731
    if m := PLAN.search(line):
        s, wait, lim, now, k = m.groups()
        side = "buy" if float(lim) < float(now) else "sell"
        return {"sym": s, "kind": "plan", "title": f"Planned a {side} on {nm(s)}",
                "text": f"The signal crossed the trade line. Instead of {side}ing at {_px(now)} right away, it waits up to "
                        f"{float(wait):.0f} min for {_px(lim)} — a {float(k):.2f}× ATR pullback (ATR = the typical bar range) — "
                        f"so the entry costs less. If the price never comes back it enters at market when time runs out."}
    if m := ENTRY.search(line):
        s, how, secs, side, vol, px, ref, bps = m.groups()
        good = float(bps) >= 0
        why = (f"the price pulled back after {int(secs) // 60} min {int(secs) % 60} s" if how == "pullback"
               else f"the pullback didn't come within {int(secs) // 60} min, so it entered at market to keep the trade")
        return {"sym": s, "kind": "entry", "title": f"{'Bought' if side == 'buy' else 'Sold'} {vol} lots of {nm(s)} @ {_px(px)}",
                "text": f"Entered because {why}. That is {abs(float(bps)):.1f} basis points "
                        f"{'better' if good else 'worse'} than the price when the signal fired ({_px(ref)}). "
                        f"A stop-loss and take-profit were set from the recent ATR.", "tone": "pos" if good else "neg"}
    if m := STACK.search(line):
        s, n, k, mx = m.groups()
        return {"sym": s, "kind": "stack", "title": f"Stacking {nm(s)}: trade {k} of {mx}",
                "text": f"All {n} open trade(s) on {nm(s)} are in profit and the signal still points the same way, so it adds "
                        f"one more, sized and stopped like a single trade. If the signal flips it goes back to one.",
                "tone": "pos"}
    if m := ACTION.search(line):
        s, req, side, vol, tgt = m.groups()
        t = float(tgt)
        if req == "close":
            why = ("the combined signal went back inside the trade line, so there is no longer an edge worth the risk"
                   if t == 0 else f"the signal flipped to the other side ({t:+.2f}), so the old trade had to go first")
            return {"sym": s, "kind": "close", "title": f"Closed the {side.lower()} on {nm(s)} ({vol} lots)",
                    "text": f"Closed because {why}."}
        if req == "reduce":
            return {"sym": s, "kind": "reduce", "title": f"Trimmed {nm(s)} by {vol} lots",
                    "text": f"The signal weakened (now {t:+.2f}), so the position was cut down to match how sure the bot is."}
        return None
    if m := MANAGE.search(line):
        s, sl, tp = m.groups()
        return {"sym": s, "kind": "manage", "title": f"Moved the stop on {nm(s)}" + (f" to {_px(sl)}" if sl != "None" else ""),
                "text": "The trade moved far enough in its favour to lock some of it in: the stop-loss follows the price "
                        "(breakeven, then trailing), so a reversal can no longer turn it into a full loss."}
    if m := MINLOT.search(line):
        s, lot, r, ceil = m.groups()
        return {"sym": s, "kind": "size", "title": f"Sized {nm(s)} at the minimum {lot} lots",
                "text": f"The size the signal asked for is below the broker's smallest lot. The minimum lot risks "
                        f"{float(r):.2f}% of the account if the stop is hit (allowed up to {float(ceil):.0f}%), so it trades the minimum."}
    if m := SIZEBLOCK.search(line):
        s, lot, r, eq, cap, need = m.groups()
        return {"sym": s, "kind": "blocked", "title": f"Skipped {nm(s)}: account too small",
                "text": f"The signal wants a trade, but the smallest lot Exness allows ({lot}) would lose {float(r):.1f}% "
                        f"of the ${float(eq):.2f} account if the stop is hit, over the {float(cap):.0f}% safety cap. "
                        f"It trades this market once the account is about ${int(need):,}.", "tone": "neg", "need": int(need),
                "risk": float(r)}
    if LOST.search(line):
        return {"sym": None, "kind": "warn", "title": "Broker connection dropped",
                "text": "Lost contact with Exness for a moment and reconnected. Open trades keep their stops at the broker.",
                "tone": "neg"}
    if m := NEWS.search(line):
        s, msg = m.groups()
        return {"sym": s, "kind": "news", "title": f"News play on {nm(s)}", "text": msg}
    return None


def events(path: str, names: dict, limit: int = 60, max_bytes: int = 600_000) -> list[dict]:
    out = []
    for line in _tail(path, max_bytes):
        if "| DEBUG" in line:
            continue
        ev = parse(line, names)
        if ev is None:
            continue
        ev["t"] = _t(line)
        ev["name"] = names.get(ev.get("sym"), ev.get("sym"))
        out.append(ev)
    # collapse identical stop moves fired every loop, and sizing notes repeated within 20 min
    dedup, last_size = [], {}
    for ev in out:
        if ev["kind"] in ("size", "blocked"):
            if ev["t"] and ev["t"] - last_size.get(ev["sym"], 0) < 1200:
                continue
            last_size[ev["sym"]] = ev["t"] or 0
        if dedup and ev["kind"] == "manage" and dedup[-1]["kind"] == "manage" and dedup[-1]["sym"] == ev["sym"]:
            dedup[-1] = ev
            continue
        dedup.append(ev)
    return dedup[-limit:][::-1]


if __name__ == "__main__":
    import json
    import sys
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for e in events(os.path.join(root, "logs", "server.log"), {}, int(sys.argv[1]) if len(sys.argv) > 1 else 10):
        print(json.dumps(e)[:300])
