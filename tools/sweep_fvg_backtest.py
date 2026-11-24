"""Backtest the user's "Sweep + FVG Reversal [Strategy]" with ITS OWN rules on the cached Exness bars.

    .venv/bin/python tools/sweep_fvg_backtest.py                 # M5 and M15, 7 markets, target 2R
    .venv/bin/python tools/sweep_fvg_backtest.py --target 1.5 --tf 15

Setup detection: Python/meta_brain.py sweep_fvg (liquidity sweep -> structure break within 10 bars -> FVG).
Trade: limit at the FVG near edge, valid `window` bars from the next bar; stop 0.1 ATR beyond the sweep
extreme; fixed-R target; one setup at a time per market; a bar that touches stop and target counts as the
stop. Costs: live spread (bps), plus --commission per side (Pine default 0.02%; Exness standard accounts: none). Results in R and as
% of equity at the Pine default 0.5% risk. Writes logs/sweep_fvg_backtest.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python import meta_brain as mb  # noqa: E402
from Python.indicator_council import resample_ohlc  # noqa: E402


def load(sym, tf):
    d = pd.DataFrame(np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy")))
    d.index = pd.to_datetime(d["time"], unit="s", utc=True)
    d = d[["open", "high", "low", "close"]].astype(float)
    return d if tf == 5 else resample_ohlc(d, tf)


def run(df, sym, target_r=2.0, window=20, commission=0.0002, min_stop_spreads=3.0):
    s = mb.sweep_fvg(df, window=window)
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    fresh, ent, stp = s["sfr_fresh"].to_numpy(), s["_sfr_entry_px"].to_numpy(), s["_sfr_stop_px"].to_numpy()
    spread = mb.SPREAD_BPS.get(sym, 1.0) / 1e4
    trades, n, t = [], len(c), 0
    unfilled = 0
    while t < n:
        if fresh[t] == 0:
            t += 1
            continue
        d, e, sp = fresh[t], ent[t], stp[t]
        risk = abs(e - sp)
        if risk < min_stop_spreads * spread * e:   # stop inside a few spreads: not tradeable at a real broker
            t += 1
            continue
        tp = e + d * target_r * risk
        fill = None
        for j in range(t + 1, min(n, t + 1 + window)):
            if fresh[j] != 0 and j > t:       # a new setup replaces the pending one (Pine: one setup at a time)
                break
            if (d > 0 and l[j] <= e) or (d < 0 and h[j] >= e):
                fill = j
                break
        if fill is None:
            unfilled += 1
            t += 1
            continue
        res, k = None, fill
        while k < n:
            hit_s = (l[k] <= sp) if d > 0 else (h[k] >= sp)
            hit_t = (h[k] >= tp) if d > 0 else (l[k] <= tp)
            if k == fill:                      # fill bar: only count the target if price got there after the fill
                hit_t = hit_t and ((c[k] >= tp) if d > 0 else (c[k] <= tp))
            if hit_s:
                res = -1.0
                break
            if hit_t:
                res = target_r
                break
            k += 1
        if res is None:
            res = d * (c[-1] - e) / risk
        cost_r = (spread + 2 * commission) * e / risk
        trades.append({"t": str(df.index[fill]), "dir": int(d), "R": res - cost_r, "stop_atr": None, "bars": k - fill})
        t = k + 1
    return trades, unfilled


def summ(rs):
    rs = np.array(rs)
    if not len(rs):
        return {"n": 0}
    eq = np.cumsum(rs)
    return {"n": int(len(rs)), "win": round(float((rs > 0).mean()), 3), "avgR": round(float(rs.mean()), 3),
            "totalR": round(float(rs.sum()), 1), "pf": round(float(rs[rs > 0].sum() / max(1e-9, -rs[rs < 0].sum())), 2),
            "maxDD_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1),
            "t": round(float(rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)) + 1e-12)), 2) if len(rs) > 2 else None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tf", default="5,15")
    ap.add_argument("--target", type=float, default=2.0)
    ap.add_argument("--window", type=int, default=20)
    ap.add_argument("--commission", type=float, default=0.0, help="per side, fraction (Pine default 0.0002; Exness standard: 0)")
    a = ap.parse_args()
    out = {}
    for tf in [int(x) for x in a.tf.split(",")]:
        allr, first, second = [], [], []
        out[f"M{tf}"] = {}
        for sym in mb.SYMBOLS:
            df = load(sym, tf)
            tr, unf = run(df, sym, a.target, a.window, a.commission)
            rs = [x["R"] for x in tr]
            mid = df.index[len(df) // 2]
            first += [x["R"] for x in tr if pd.Timestamp(x["t"]) < mid]
            second += [x["R"] for x in tr if pd.Timestamp(x["t"]) >= mid]
            res = summ(rs)
            res["unfilled_setups"] = unf
            res["days"] = (df.index[-1] - df.index[0]).days
            out[f"M{tf}"][sym] = res
            allr += rs
            print(f"M{tf} {sym:7s} {res}")
        out[f"M{tf}"]["ALL"] = summ(allr)
        out[f"M{tf}"]["first_half"] = summ(first)
        out[f"M{tf}"]["second_half"] = summ(second)
        print(f"M{tf} ALL  {out[f'M{tf}']['ALL']}\n   first half {out[f'M{tf}']['first_half']}\n   second half {out[f'M{tf}']['second_half']}")
    json.dump(out, open(os.path.join(ROOT, "logs", "sweep_fvg_backtest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
