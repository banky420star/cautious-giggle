"""Seconds and up: does anything predict price after the REAL spread, from 1-second bars to 15-minute bars?

Uses Exness's public tick archive (every bid/ask, data/ticks/Exness_<SYM>_<YYYY>_<MM>.zip, from
https://ticks.ex2archive.com). For each bar size (1 s, 5 s, 15 s, 30 s, 1 min, 5 min, 15 min) it builds
mid-price bars, runs the 12 council indicators and the council on them, and scores holding for 2 and
6 bars. Cost = the actual spread at the moment of entry (buy at the ask, sell at the bid).

Also prints the "physics" of each market: the typical price move over the hold vs the spread paid.
If the typical move is not several times the spread, no signal of realistic accuracy can pay for itself.

Read-only research. PYTHONPATH=. .venv/bin/python tools/tick_test.py [--month 2026_10] [--markets ...]
Writes logs/tick_test.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.indicator_council import STRATEGIES, votes  # noqa: E402

TICKS = os.path.join(ROOT, "data", "ticks")
MARKETS = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
SCALES = {"1 s": "1s", "5 s": "5s", "15 s": "15s", "30 s": "30s", "1 min": "1min", "5 min": "5min", "15 min": "15min"}
HOLDS = (2, 6)


def load(sym: str, month: str) -> pd.DataFrame:
    zp = os.path.join(TICKS, f"Exness_{sym}_{month}.zip")
    with zipfile.ZipFile(zp) as z:
        name = next(n for n in z.namelist() if n.endswith(".csv"))
        with z.open(name) as f:
            df = pd.read_csv(f, usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="ISO8601")
    df = df.set_index("Timestamp").sort_index()
    df = df[(df["Ask"] >= df["Bid"]) & (df["Bid"] > 0)]
    df["mid"] = (df["Bid"] + df["Ask"]) / 2
    df["spr"] = (df["Ask"] - df["Bid"]) / df["mid"] * 1e4
    return df


def bars(t: pd.DataFrame, rule: str) -> pd.DataFrame:
    g = t.resample(rule, label="left", closed="left")
    b = pd.DataFrame({"open": g["mid"].first(), "high": g["mid"].max(), "low": g["mid"].min(),
                      "close": g["mid"].last(), "spr": g["spr"].last()}).dropna()
    return b


def score(v, fwd, cost, ok, h, half) -> dict | None:
    act = ok & (np.abs(v) > 0.05)
    n = int(act.sum())
    if n < 100:
        return None
    gross = np.sign(v[act]) * fwd[act]
    net = gross - cost[act]
    t = float(net.mean()) / ((float(net.std()) or 1e-9) / np.sqrt(max(2.0, n / h)))
    a1, a2 = act & ~half, act & half
    m1 = float((np.sign(v[a1]) * fwd[a1] - cost[a1]).mean()) if a1.sum() > 30 else float("nan")
    m2 = float((np.sign(v[a2]) * fwd[a2] - cost[a2]).mean()) if a2.sum() > 30 else float("nan")
    return {"trades": n, "net_bps": round(float(net.mean()), 3), "gross_bps": round(float(gross.mean()), 3),
            "cost_bps": round(float(cost[act].mean()), 3), "hit_gross": round(float((gross > 0).mean()), 3),
            "t": round(t, 2), "halves": [round(m1, 3), round(m2, 3)],
            "holds_up": bool(t >= 3 and m1 > 0 and m2 > 0)}


def run(markets: list[str], month: str) -> dict:
    out = {"t": time.time(), "month": month, "markets": {}}
    for sym in markets:
        t0 = time.time()
        try:
            tk = load(sym, month)
        except (OSError, StopIteration, KeyError) as exc:
            print(f"{sym}: no tick file ({exc})")
            continue
        res = {"ticks": len(tk), "median_spread_bps": round(float(tk["spr"].median()), 3), "scales": {}}
        for sname, rule in SCALES.items():
            b = bars(tk, rule)
            if len(b) < 1000:
                continue
            v = votes(b[["open", "high", "low", "close"]])
            council = np.tanh(2 * v.to_numpy(float).mean(axis=1))
            close = b["close"].to_numpy(float)
            cost = b["spr"].to_numpy(float)
            tsec = pd.DatetimeIndex(b.index).as_unit("s").asi8.astype(float)
            step = pd.Timedelta(rule).total_seconds()
            half = np.arange(len(b)) >= len(b) // 2
            row = {"bars": len(b)}
            for h in HOLDS:
                fwd = np.full(len(close), np.nan)
                fwd[:-h] = np.log(close[h:] / close[:-h]) * 1e4
                el = np.full(len(close), np.inf)
                el[:-h] = tsec[h:] - tsec[:-h]
                ok = np.isfinite(fwd) & (el <= h * step * 3)       # skip holds across gaps (closed market)
                ok[:300] = False
                mv = np.abs(fwd[ok])
                hold = {"hold_seconds": int(h * step), "typical_move_bps": round(float(np.median(mv)), 3),
                        "move_to_spread": round(float(np.median(mv) / max(1e-9, np.median(cost[ok]))), 2),
                        "voters": {}}
                c = score(council, fwd, cost, ok, h, half)
                if c:
                    hold["voters"]["council"] = c
                for k in STRATEGIES:
                    s = score(v[k].to_numpy(float), fwd, cost, ok, h, half)
                    if s:
                        hold["voters"][k] = s
                row[f"hold {h} bars"] = hold
            res["scales"][sname] = row
        out["markets"][sym] = res
        print(f"{sym}: {len(tk):,} ticks, median spread {res['median_spread_bps']} bps, {time.time() - t0:.0f}s", flush=True)
        with open(os.path.join(ROOT, "logs", "tick_test.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    out["done"] = True
    with open(os.path.join(ROOT, "logs", "tick_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def summary(out: dict) -> None:
    for sym, r in out["markets"].items():
        print(f"\n{sym}  ({r['ticks']:,} ticks, median spread {r['median_spread_bps']} bps)")
        print(f"  {'bars':>6} {'hold':>7} {'move':>7} {'move/spr':>8} | {'council net':>11} {'gross':>7} {'hit':>5} | best indicator")
        for sname, row in r["scales"].items():
            for hk in (k for k in row if k.startswith("hold")):
                hd = row[hk]
                c = hd["voters"].get("council") or {}
                best = max(((k, s) for k, s in hd["voters"].items() if k != "council"),
                           key=lambda kv: kv[1]["net_bps"], default=(None, None))
                hs = hd["hold_seconds"]
                hold = f"{hs}s" if hs < 60 else f"{hs // 60}m" if hs < 3600 else f"{hs / 3600:.1f}h"
                bs = (f"{best[0]:11s} {best[1]['net_bps']:+.2f} t={best[1]['t']:+.1f}"
                      + (" HOLDS UP" if best[1]["holds_up"] else "")) if best[0] else ""
                print(f"  {sname:>6} {hold:>7} {hd['typical_move_bps']:7.2f} {hd['move_to_spread']:8.2f} | "
                      f"{c.get('net_bps', float('nan')):+11.2f} {c.get('gross_bps', float('nan')):+7.2f} "
                      f"{c.get('hit_gross', float('nan')):5.2f} | {bs}")
    hits = [(s, sc, hk, k, v) for s, r in out["markets"].items() for sc, row in r["scales"].items()
            for hk, hd in row.items() if hk.startswith("hold") for k, v in hd["voters"].items() if v["holds_up"]]
    print("\n=== Holds up (t >= 3, positive after the real spread in both halves of the month) ===")
    for s, sc, hk, k, v in hits:
        print(f"  {s} {sc} bars, {hk}: {k} net {v['net_bps']:+.3f} bps t={v['t']} halves {v['halves']} trades {v['trades']}")
    if not hits:
        print("  nothing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026_10")
    ap.add_argument("--markets", nargs="*", default=MARKETS)
    a = ap.parse_args()
    summary(run(a.markets, a.month))
