"""Which voters predict price after costs, at which holding time?

For every market in the M5 cache (about a year of history each), every indicator x chart-timeframe voter
(48) and the council (all, and per chart timeframe) is scored at five holding times: 10 min, 30 min,
1 h, 4 h and 1 day. A "trade" = follow the voter's side whenever it has a view (|vote| > 0.05), hold for
the horizon, pay the spread once. Results are per trade in basis points (1 bp = 0.01%) AFTER the spread.

Guards against fooling ourselves:
  * overlapping holds are not independent: the t-statistic uses n / horizon effective samples;
  * the year is split in two halves; a voter only counts as "holds up" if it is positive after costs
    in BOTH halves and t >= 3 over the whole year (with ~1,900 tests, t >= 2 would give dozens of flukes);
  * cost = the spread recorded in the history, or today's live spread when the history has none.

    PYTHONPATH=. .venv/bin/python tools/horizon_test.py [--markets BTCUSDm XAUUSDm]
Writes logs/horizon_test.json and prints a summary. Read-only: it changes nothing in trading.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.indicator_council import TIMEFRAMES, describe, votes_mtf  # noqa: E402

CACHE = os.path.join(ROOT, "data", "exness_cache")
INST = {"BTCUSDm": "BTCUSD", "XAUUSDm": "XAUUSD", "EURUSDm": "EURUSD", "USDJPYm": "USDJPY",
        "XAGUSDm": "XAGUSD", "ETHUSDm": "ETHUSD", "USOILm": "USOIL"}
HORIZONS = {"10 min": 2, "30 min": 6, "1 hour": 12, "4 hours": 48, "1 day": 288}


def _spread_bps(sym: str, arr, close: np.ndarray) -> tuple[np.ndarray | None, str]:
    from tools.warm_memory import _live_spread_bps, _point
    pt = _point(sym)
    if pt:
        rec = arr["spread"].astype(float) * pt / close * 1e4
        if np.median(rec) > 0.01:
            return rec, f"recorded spread, median {np.median(rec):.2f} bps"
    live = _live_spread_bps(sym)
    if live is None:
        return None, "no spread known"
    return np.full(len(close), live), f"today's live spread {live:.2f} bps"


def _stats(v: np.ndarray, fwd: np.ndarray, cost: np.ndarray, ok: np.ndarray, h: int, half: np.ndarray) -> dict | None:
    act = ok & (np.abs(v) > 0.05)
    n = int(act.sum())
    if n < 50:
        return None
    gross = np.sign(v[act]) * fwd[act]
    net = gross - cost[act]
    n_eff = max(2.0, n / h)
    sd = float(net.std()) or 1e-9
    h1, h2 = act & ~half, act & half
    net1 = (np.sign(v[h1]) * fwd[h1] - cost[h1]).mean() if h1.sum() > 20 else np.nan
    net2 = (np.sign(v[h2]) * fwd[h2] - cost[h2]).mean() if h2.sum() > 20 else np.nan
    t = float(net.mean()) / (sd / np.sqrt(n_eff))
    return {"trades": n, "net_bps": round(float(net.mean()), 2), "gross_bps": round(float(gross.mean()), 2),
            "cost_bps": round(float(cost[act].mean()), 2), "hit": round(float((net > 0).mean()), 3),
            "t": round(t, 2), "first_half": None if np.isnan(net1) else round(float(net1), 2),
            "second_half": None if np.isnan(net2) else round(float(net2), 2),
            "holds_up": bool(t >= 3 and net1 > 0 and net2 > 0)}


def run(markets: list[str]) -> dict:
    out = {"t": time.time(), "horizons": list(HORIZONS), "markets": {}}
    for sym in markets:
        t0 = time.time()
        path = os.path.join(CACHE, f"{INST[sym]}_M5.npy")
        if not os.path.exists(path):
            print(f"{sym}: no cached history")
            continue
        arr = np.load(path)
        df = pd.DataFrame({k: arr[k] for k in ("open", "high", "low", "close")},
                          index=pd.to_datetime(arr["time"], unit="s", utc=True))
        close = df["close"].to_numpy(float)
        cost, how = _spread_bps(sym, arr, close)
        if cost is None:
            print(f"{sym}: {how}, skipped (every result here is after costs)")
            continue
        v = votes_mtf(df)
        cols = list(v.columns)
        V = v.to_numpy(float)
        councils = {"council (all 48)": np.tanh(2 * V.mean(axis=1))}
        for tf in TIMEFRAMES:
            idx = [i for i, c in enumerate(cols) if (c.split("@")[1] if "@" in c else "M5") == tf]
            if idx:
                councils[f"council ({tf} charts)"] = np.tanh(2 * V[:, idx].mean(axis=1))
        t_ns = df.index.asi8
        half = np.arange(len(df)) >= len(df) // 2
        res = {"from": str(df.index[0]), "to": str(df.index[-1]), "bars": len(df), "cost": how, "horizons": {}}
        for hname, h in HORIZONS.items():
            fwd = np.full(len(close), np.nan)
            fwd[:-h] = np.log(close[h:] / close[:-h]) * 1e4
            span = np.full(len(close), np.inf)
            span[:-h] = (t_ns[h:] - t_ns[:-h]) / 6e10                      # minutes actually elapsed
            ok = np.isfinite(fwd) & (span <= h * 5 + 3 * 24 * 60)          # no holds across long outages
            ok[:1500] = False                                              # indicator warm-up (H4 needs history)
            rows = {}
            for name, vv in councils.items():
                st = _stats(vv, fwd, cost, ok, h, half)
                if st:
                    rows[name] = st
            for i, c in enumerate(cols):
                st = _stats(V[:, i], fwd, cost, ok, h, half)
                if st:
                    tf = c.split("@")[1] if "@" in c else "M5"
                    rows[f"{describe(c.split('@')[0])} [{tf}]"] = st
            res["horizons"][hname] = rows
        out["markets"][sym] = res
        print(f"{sym}: done in {time.time() - t0:.0f}s ({how})", flush=True)
        with open(os.path.join(ROOT, "logs", "horizon_test.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    out["done"] = True
    with open(os.path.join(ROOT, "logs", "horizon_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def summary(out: dict) -> None:
    print("\n=== Best voter per market and holding time (net of spread, bps per trade; t = reliability) ===")
    for sym, r in out["markets"].items():
        print(f"\n{sym}  ({r['from'][:10]} → {r['to'][:10]}, {r['cost']})")
        for hname, rows in r["horizons"].items():
            if not rows:
                continue
            best = max(rows.items(), key=lambda kv: kv[1]["net_bps"])
            c = rows.get("council (all 48)") or {}
            ok = [k for k, s in rows.items() if s["holds_up"]]
            print(f"  {hname:>8}: council {c.get('net_bps', float('nan')):+7.2f} (gross {c.get('gross_bps', float('nan')):+.2f}) | "
                  f"best {best[0][:44]:44s} {best[1]['net_bps']:+7.2f} t={best[1]['t']:+.1f} "
                  f"halves {best[1]['first_half']}/{best[1]['second_half']} | holds up: {len(ok)}")
    print("\n=== Everything that holds up (t >= 3, positive after costs in both halves) ===")
    any_ = False
    for sym, r in out["markets"].items():
        for hname, rows in r["horizons"].items():
            for k, s in sorted(rows.items(), key=lambda kv: -kv[1]["net_bps"]):
                if s["holds_up"]:
                    any_ = True
                    print(f"  {sym} {hname:>8} {k[:50]:50s} net {s['net_bps']:+.2f} bps  t={s['t']:.1f}  "
                          f"trades {s['trades']}  halves {s['first_half']}/{s['second_half']}")
    if not any_:
        print("  nothing")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--markets", nargs="*", default=list(INST))
    a = ap.parse_args()
    summary(run(a.markets))
