"""Adds one new market a day, if it earns its place, until the bot trades MAX_MARKETS.

    .venv/bin/python tools/market_scout.py            # the daily run (started by the dashboard service at 07:15 UTC)
    .venv/bin/python tools/market_scout.py --dry-run  # test the next candidates, change nothing
    .venv/bin/python tools/market_scout.py --test US500m   # test one market, change nothing

For the next untested candidate (CANDIDATES order; a market that failed is retried after RETRY_DAYS):
  1. spread screen    today's spread against the market's M5 ATR (too costly: skip without downloading)
  2. history          about 17 months of M5 bars into data/exness_cache (same cache the training uses)
  3. gate test        the council's live setup: fresh council signal with the 4h 200-EMA trend, 8 / 12.8 ATR
                      stop / target, 24 h limit, today's spread charged (tools/stop_rules_backtest.py --wide).
                      Passes only if the average is positive in EACH of 3 time periods, with enough trades.
  4. added            appended to news_trading.symbols (the list every bot trader uses) and to graduation's
                      candidates (tools/graduation.py trains its own LSTM / PPO models); the bot restarts.
Up to MAX_TRIES candidates are tested per run until one passes. Results: logs/market_scout.json.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import signal
import sys
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

MAX_MARKETS = 12           # user's cap, 2026-11-24
MAX_TRIES = 4              # candidates tested per daily run
RETRY_DAYS = 30            # a failed market is tested again after this
MIN_BARS = 60_000          # ~ 8 months of M5 (17 months asked for)
WANT_BARS = 101_000
MAX_COST_ATR = 0.35        # spread / M5 ATR above this: not worth downloading
MIN_TRADES = 120
MIN_AVG_R = 0.02
SL, TP, HORIZON, GAP, ENTRY, FOLDS = 8.0, 12.8, 288, 6, 0.5, 3
STATE = os.path.join(ROOT, "logs", "market_scout.json")
CONFIG = os.path.join(ROOT, "config.yaml")

# Most liquid first: stock indices, other metals, big crypto, major FX, energy, then the largest US stocks.
CANDIDATES = ["US500m", "USTECm", "US30m", "DE30m", "JP225m", "UK100m", "XPTUSDm", "XRPUSDm", "GBPUSDm",
              "AUDUSDm", "USDCADm", "UKOILm", "STOXX50m", "LTCUSDm", "USDCHFm", "NZDUSDm", "GBPJPYm", "EURJPYm",
              "FR40m", "AUS200m", "HK50m", "XPDUSDm", "XNGUSDm", "NVDAm", "AAPLm", "MSFTm", "TSLAm", "AMZNm"]


def now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {"tested": {}, "added": [], "runs": []}


def save_state(st: dict) -> None:
    st["runs"] = st.get("runs", [])[-60:]
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


# ------------------------------------------------------------------------------- config
def current_markets(text: str | None = None) -> list[str]:
    text = text if text is not None else open(CONFIG, encoding="utf-8").read()
    m = re.search(r"^  news_trading:.*?\n((?:    .*\n)*?)    symbols: \[([^\]]*)\]", text, re.M)
    return [s.strip() for s in m.group(2).split(",") if s.strip()] if m else []


def add_market(sym: str) -> list[str]:
    """Append to news_trading.symbols in config.yaml, keeping the file's comments and layout."""
    text = open(CONFIG, encoding="utf-8").read()
    have = current_markets(text)
    if sym in have:
        return have
    new = have + [sym]
    m = re.search(r"^  news_trading:.*?\n((?:    .*\n)*?)    symbols: \[([^\]]*)\]", text, re.M)
    if not m:
        raise RuntimeError("news_trading.symbols not found in config.yaml")
    shutil.copy2(CONFIG, CONFIG + ".bak-scout")
    text = text[:m.start(2)] + ", ".join(new) + text[m.end(2):]
    # and to graduation's candidates, which trains its own LSTM + PPO models (tools/graduation.py)
    g = re.search(r"^  graduation:.*?\n((?:    .*\n)*?)    candidates: \[([^\]]*)\]", text, re.M)
    if g and sym not in [x.strip() for x in g.group(2).split(",")]:
        have_g = [x.strip() for x in g.group(2).split(",") if x.strip()]
        text = text[:g.start(2)] + ", ".join(have_g + [sym]) + text[g.end(2):]
    with open(CONFIG + ".tmp", "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(CONFIG + ".tmp", CONFIG)
    return new


def restart_bot() -> bool:
    """The supervisor starts it again (~30 s) with the new market list; its autonomy loop then trains it."""
    try:
        with open(os.path.join(ROOT, ".tmp", "server_agi.lock"), encoding="utf-8") as f:
            os.kill(int(f.read().strip()), signal.SIGTERM)
        return True
    except (OSError, ValueError):
        return False


# ------------------------------------------------------------------------------- the tests
def atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()


def spread_screen(mt5, sym: str) -> dict:
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 1500)
    if r is None or len(r) < 300:
        return {"ok": False, "unavailable": True, "why": "not offered on this account (no price history)"}
    tick = mt5.symbol_info_tick(sym)
    if tick is None:
        return {"ok": None, "why": "no live price now (market closed?)"}
    bid, ask = float(tick.bid), float(tick.ask)
    if not (bid > 0 and ask > bid):
        return {"ok": None, "why": "no two-sided price now (market closed?)"}
    h, l, c = (np.asarray([x[k] for x in r], dtype=float) for k in ("high", "low", "close"))
    atr = float(np.median(atr14(h, l, c)[-1200:]))
    if not atr > 0:
        return {"ok": None, "why": "flat price history"}
    spread_bps = (ask - bid) / bid * 1e4
    cost_atr = (ask - bid) / atr
    return {"ok": cost_atr <= MAX_COST_ATR, "spread_bps": round(spread_bps, 3), "cost_atr": round(cost_atr, 3),
            "why": "" if cost_atr <= MAX_COST_ATR else f"spread is {cost_atr:.2f} of a 5-min ATR (max {MAX_COST_ATR})"}


def history(mt5, sym: str) -> pd.DataFrame | None:
    r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 0, WANT_BARS)   # cached in data/exness_cache
    if r is None or len(r) < MIN_BARS:
        return None
    d = pd.DataFrame({k: np.asarray(r[k], dtype=float) for k in ("open", "high", "low", "close")},
                     index=pd.to_datetime(np.asarray(r["time"]), unit="s", utc=True))
    return d[~d.index.duplicated()].sort_index()


def gate_test(m5: pd.DataFrame, spread_bps: float) -> dict:
    """The council's live setup on this market's history, result per trade in R after the spread."""
    from Python.indicator_council import votes_mtf
    v = votes_mtf(m5)
    council = np.tanh(2.0 * v.mean(axis=1)).to_numpy()
    h4 = v["ema200@H4"].to_numpy() if "ema200@H4" in v.columns else np.zeros(len(v))
    o, h, l, c = (m5[k].to_numpy() for k in ("open", "high", "low", "close"))
    atr = atr14(h, l, c)
    res, busy, prev = [], -1, 0.0
    for b in range(2000, len(c) - HORIZON - 2):
        x = council[b]
        fresh = abs(x) >= ENTRY and abs(prev) < ENTRY and np.sign(h4[b]) == np.sign(x)
        prev = x
        if not fresh or b <= busy:
            continue
        d, e, R = (1.0 if x > 0 else -1.0), o[b + 1], SL * atr[b]
        if not R > 0:
            continue
        seg = slice(b + 1, b + 1 + HORIZON)
        up, dn = d * ((h if d > 0 else l)[seg] - e) / R, d * ((l if d > 0 else h)[seg] - e) / R
        op = d * (o[seg] - e) / R
        out = None
        for k in range(len(up)):
            if dn[k] <= -1.0:
                out = min(-1.0, op[k]); break
            if up[k] >= TP / SL:
                out = TP / SL; break
        if out is None:
            out = d * (c[b + HORIZON] - e) / R
        res.append(out - spread_bps / 1e4 * e / R)
        busy = b + GAP
    r = np.asarray(res)
    if len(r) < MIN_TRADES:
        return {"ok": False, "trades": int(len(r)), "why": f"only {len(r)} trades in the history (need {MIN_TRADES})"}
    per = [round(float(x.mean()), 4) for x in np.array_split(r, FOLDS)]
    avg = float(r.mean())
    ok = min(per) > 0 and avg >= MIN_AVG_R
    why = "" if ok else (f"lost money in period {per.index(min(per)) + 1} of 3" if min(per) <= 0
                         else f"average {avg:+.3f}R is below {MIN_AVG_R}R")
    return {"ok": ok, "trades": int(len(r)), "avg_r": round(avg, 4), "periods": per,
            "win_rate": round(float((r > 0).mean()), 3), "why": why,
            "from": str(m5.index[0].date()), "to": str(m5.index[-1].date())}


def test_market(mt5, sym: str, log=print) -> dict:
    row = {"at": now_iso()}
    sc = spread_screen(mt5, sym)
    row["screen"] = sc
    if sc["ok"] is None:
        return {**row, "result": "deferred", "why": sc["why"]}
    if not sc["ok"]:
        return {**row, "result": "unavailable" if sc.get("unavailable") else "failed", "why": sc["why"]}
    log(f"  {sym}: spread {sc['spread_bps']} bps = {sc['cost_atr']} ATR, downloading history...")
    m5 = history(mt5, sym)
    if m5 is None:
        return {**row, "result": "failed", "why": f"less than {MIN_BARS:,} bars of history"}
    log(f"  {sym}: {len(m5):,} bars, testing the council setup...")
    g = gate_test(m5, sc["spread_bps"])
    row["gate"] = g
    return {**row, "result": "passed" if g["ok"] else "failed", "why": g.get("why", "")}


def next_candidates(st: dict, markets: list[str]) -> list[str]:
    out = []
    for s in CANDIDATES:
        if s in markets:
            continue
        t = (st.get("tested") or {}).get(s)
        if t and t.get("result") in ("failed", "unavailable"):
            try:
                age = time.time() - datetime.strptime(t["at"], "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=timezone.utc).timestamp()
            except (KeyError, ValueError):
                age = 1e12
            if age < RETRY_DAYS * 86400:
                continue
        out.append(s)
    return out


def main() -> int:
    dry = "--dry-run" in sys.argv
    only = sys.argv[sys.argv.index("--test") + 1] if "--test" in sys.argv else None
    from Python.mt5_compat import mt5
    if not mt5.initialize():
        print("no connection to the broker")
        return 1
    st = load_state()
    if not (dry or only or "--force" in sys.argv) and str((st.get("last_run") or {}).get("at", "")).startswith(now_iso()[:10]):
        print("already ran today")
        return 0
    markets = current_markets()
    run = {"at": now_iso(), "markets_before": len(markets), "tested": [], "added": None}
    if only:
        r = test_market(mt5, only)
        print(json.dumps(r, indent=1))
        return 0
    if len(markets) >= MAX_MARKETS:
        run["note"] = f"full: {len(markets)} of {MAX_MARKETS} markets"
        print(run["note"])
    else:
        tries = 0
        for sym in next_candidates(st, markets):
            if tries >= MAX_TRIES:
                break
            print(f"testing {sym}")
            r = test_market(mt5, sym)
            if not (r.get("screen") or {}).get("unavailable"):
                tries += 1                                 # markets the account doesn't offer cost nothing
            print(f"  {sym}: {r['result']}" + (f" ({r['why']})" if r.get("why") else ""))
            run["tested"].append({"symbol": sym, "result": r["result"], "why": r.get("why", "")})
            if dry:
                continue
            st.setdefault("tested", {})[sym] = r
            if r["result"] == "passed":
                new = add_market(sym)
                st.setdefault("added", []).append({"symbol": sym, "at": now_iso(), "gate": r.get("gate")})
                run["added"] = sym
                run["restarted"] = restart_bot()
                print(f"ADDED {sym}: now {len(new)} markets; bot restarting to train and trade it")
                break
    st["max_markets"] = MAX_MARKETS
    st["markets"] = current_markets()
    st["queue"] = next_candidates(st, st["markets"])[:8]
    st["last_run"] = run
    st.setdefault("runs", []).append(run)
    if not dry:
        save_state(st)
    return 0


if __name__ == "__main__":
    sys.exit(main())
