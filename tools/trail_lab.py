"""Trail lab: per market, search entry methods x trailing-stop settings for small, frequent wins.

Real Exness ticks, second by second, one position at a time per market, re-entering whenever the entry
method fires again. Costs are real: buy at the ask, sell at the bid, stops filled at that second's price
(so gaps slip). Entries are skipped while the spread is wider than 1.5x its median (rollover etc).

Entry methods ("votes", computed fast):
  random      coin flip every 60 s (the baseline: what the exit alone is worth)
  council15   the 12-indicator council on 15-second bars, when it leans |c| >= 0.2
  council30   the same on 30-second bars
  burst_go    price moved >= K spreads in the last 10 s: follow it
  burst_fade  the same: bet against it
Exit (all in multiples of the spread, net of costs):
  SL     initial stop
  BE     once the trade could be closed at this profit, the stop jumps to break-even
  TRAIL  then trails this far behind the best profit
  TP     optional hard take-profit cap
  HOLD   max seconds open
Method: tune on the first half of the month (TRAIN), report the chosen setting once on the second half (TEST).
A result only counts if TEST is profitable with t >= 2 and it later holds on a different month.

The maths (printed per market): every trade pays about one spread. With entries that know nothing, every
exit setting averages about -1 spread per trade. A setting makes money only if the entries' average
directional move during the hold beats the cost: gross edge per trade > spread paid.

    PYTHONPATH=. .venv/bin/python tools/trail_lab.py [--month 2026_10] [--markets XAUUSDm ...]
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import sys
import time
import zipfile

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.indicator_council import votes  # noqa: E402

TICKS = os.path.join(ROOT, "data", "ticks")
MARKETS = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
CORE = list(itertools.product((3, 6, 12), (0.5, 1, 2), (0.5, 1, 2)))      # SL, BE, TRAIL
LEDGER = os.path.join(ROOT, "logs", "trail_lab_ledger.jsonl")


def load(sym: str, month: str):
    with zipfile.ZipFile(os.path.join(TICKS, f"Exness_{sym}_{month}.zip")) as z:
        n = next(x for x in z.namelist() if x.endswith(".csv"))
        df = pd.read_csv(z.open(n), usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    df = df[(df.Ask >= df.Bid) & (df.Bid > 0)]
    s = df.set_index("Timestamp").sort_index().resample("1s").last()
    live = s["Bid"].notna().to_numpy()                     # a real quote arrived this second
    s = s.ffill().dropna()
    live = live[-len(s):]
    return s.index, s["Bid"].to_numpy(), s["Ask"].to_numpy(), live


def signals(idx, bid, ask, live, sp_med, rng) -> dict[str, np.ndarray]:
    mid = (bid + ask) / 2
    n = len(mid)
    out = {}
    r = np.zeros(n, np.int8)
    pick = np.arange(0, n, 60)
    r[pick] = np.where(rng.random(len(pick)) < 0.5, 1, -1)
    out["random"] = r
    ser = pd.Series(mid, index=idx)
    for name, rule in (("council15", "15s"), ("council30", "30s")):
        g = ser.resample(rule, label="right", closed="left")       # bar is known at its right edge
        b = pd.DataFrame({"open": g.first(), "high": g.max(), "low": g.min(), "close": g.last()}).dropna()
        c = np.tanh(2 * votes(b).to_numpy(float).mean(axis=1))
        sig = np.zeros(n, np.int8)
        pos = idx.searchsorted(b.index)                          # first second at/after the bar's close
        ok = (pos < n) & (np.abs(c) >= 0.2)
        sig[pos[ok]] = np.sign(c[ok]).astype(np.int8)
        out[name] = sig
    mv = np.zeros(n)
    mv[10:] = (mid[10:] - mid[:-10]) / mid[:-10] * 1e4
    k = 3 * sp_med
    go = np.where(mv >= k, 1, np.where(mv <= -k, -1, 0)).astype(np.int8)
    out["burst_go"] = go
    out["burst_fade"] = (-go).astype(np.int8)
    return out


def trade(bid, ask, i, side, sp, SL, BE, TR, TP, HOLD):
    end = min(len(bid), i + 1 + HOLD)
    if end <= i + 1:
        return len(bid), 0.0
    if side == 1:
        net = (bid[i + 1:end] - ask[i]) / ask[i] * 1e4
    else:
        net = (bid[i] - ask[i + 1:end]) / bid[i] * 1e4
    best = np.maximum.accumulate(np.concatenate([[-sp], net]))[:-1]
    stop = np.where(best >= BE * sp, np.maximum(0.0, best - TR * sp), -SL * sp)
    hit = net <= stop
    if TP:
        hit |= net >= TP * sp
    if hit.any():
        j = int(np.argmax(hit))
        return i + 1 + j, float(net[j])
    return end - 1, float(net[-1])


def simulate(bid, ask, sig, okspr, sp, lo, hi, SL, BE, TR, TP, HOLD):
    """Sequential trades between seconds lo..hi; returns per-trade net bps."""
    cand = np.flatnonzero(sig[lo:hi] != 0) + lo
    cand = cand[okspr[cand]]
    res, i, k = [], lo, 0
    while k < len(cand):
        if cand[k] < i:
            k = np.searchsorted(cand, i)
            continue
        e = cand[k]
        j, r = trade(bid, ask, e, int(sig[e]), sp, SL, BE, TR, TP, min(HOLD, hi - e))
        res.append(r)
        i = j + 1
        k += 1
    return np.array(res)


def summ(r: np.ndarray, sp: float) -> dict:
    if len(r) < 2:
        return {"trades": len(r), "per_trade_bps": 0.0, "t": 0.0, "total_bps": 0.0, "win_pct": 0.0}
    return {"trades": int(len(r)), "per_trade_bps": round(float(r.mean()), 3),
            "per_trade_spreads": round(float(r.mean()) / sp, 2), "win_pct": round(100 * float((r > 0).mean()), 1),
            "avg_win": round(float(r[r > 0].mean()), 2) if (r > 0).any() else 0.0,
            "avg_loss": round(float(r[r <= 0].mean()), 2) if (r <= 0).any() else 0.0,
            "t": round(float(r.mean()) / (float(r.std()) / np.sqrt(len(r)) or 1e-9), 2),
            "total_bps": round(float(r.sum()), 1)}


def run(month: str, markets: list[str]) -> dict:
    rng = np.random.default_rng(11)
    out = {}
    for sym in markets:
        t0 = time.time()
        idx, bid, ask, live = load(sym, month)
        sprd = (ask - bid) / ((ask + bid) / 2) * 1e4
        sp = float(np.median(sprd[live]))
        okspr = (sprd <= 1.5 * sp) & live
        sigs = signals(idx, bid, ask, live, sp, rng)
        n = len(bid)
        half = n // 2
        res = {"spread_bps": round(sp, 3), "methods": {}}
        for mname, sig in sigs.items():
            best = None
            for SL, BE, TR in CORE:
                r = simulate(bid, ask, sig, okspr, sp, 0, half, SL, BE, TR, None, 900)
                if len(r) < 200:
                    continue
                s = summ(r, sp)
                if best is None or s["per_trade_bps"] > best[1]["per_trade_bps"]:
                    best = ((SL, BE, TR, None, 900), s)
            if best is None:
                continue
            SL, BE, TR, _, _ = best[0]
            for TP, HOLD in ((None, 120), (None, 3600), (BE + TR + 1, 900), (2 * BE + 2, 900)):   # refine exits
                r = simulate(bid, ask, sig, okspr, sp, 0, half, SL, BE, TR, TP, HOLD)
                if len(r) >= 200:
                    s = summ(r, sp)
                    if s["per_trade_bps"] > best[1]["per_trade_bps"]:
                        best = ((SL, BE, TR, TP, HOLD), s)
            cfg = best[0]
            test = summ(simulate(bid, ask, sig, okspr, sp, half, n, *cfg), sp)
            gross = round(test["per_trade_bps"] + sp, 3)
            row = {"config": dict(zip(("SL", "BE", "TRAIL", "TP", "HOLD_s"), cfg)), "train": best[1], "test": test,
                   "test_gross_edge_bps": gross, "passes": bool(test["t"] >= 2 and best[1]["per_trade_bps"] > 0
                                                               and test["per_trade_bps"] > 0)}
            res["methods"][mname] = row
            with open(LEDGER, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t": time.time(), "month": month, "market": sym, "method": mname, **row}) + "\n")
            c = row["config"]
            print(f"{sym:8} {mname:10} SL {c['SL']:>4}x BE {c['BE']:>3}x trail {c['TRAIL']:>3}x TP {str(c['TP']):>4} hold {c['HOLD_s']:>4}s | "
                  f"TRAIN {best[1]['per_trade_spreads']:+.2f} spr/trade ({best[1]['trades']} tr, win {best[1]['win_pct']}%) | "
                  f"TEST {test.get('per_trade_spreads', 0):+.2f} spr/trade ({test['trades']} tr, win {test['win_pct']}%, t={test['t']:+.1f})"
                  f"{'  PASS' if row['passes'] else ''}", flush=True)
        out[sym] = res
        print(f"  [{sym}: spread {sp:.2f} bps, {time.time() - t0:.0f}s]", flush=True)
        with open(os.path.join(ROOT, "logs", f"trail_lab_{month}.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, indent=1)
    return out


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--month", default="2026_10")
    ap.add_argument("--markets", nargs="*", default=MARKETS)
    a = ap.parse_args()
    run(a.month, a.markets)
