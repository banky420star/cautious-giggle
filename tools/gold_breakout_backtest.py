"""Backtest "Gold Breakout 15M Strategy" (WillyAlgoTrader, as read from the user's TradingView page) on Exness bars.

Rules: the high/low from 00:00 to 07:00 broker time (GMT+3) is the range. From 07:00 a buy stop sits at the
range high and a sell stop at the range low; the first to fill cancels the other (one trade a day). Stop loss
1% of the entry price, no take-profit; anything still open is closed at 20:00 broker time.

    .venv/bin/python tools/gold_breakout_backtest.py                  # gold, M5 bars, GMT+3
    .venv/bin/python tools/gold_breakout_backtest.py --stop-pct 0.5 --symbols XAUUSD,XAGUSD

A bar that touches both the entry and the stop counts as a loss; a bar that touches both range levels takes
the side whose level is closer to the bar's open. Results in R (1R = the 1% stop) after the live spread,
plus % growth at --risk per trade. Writes logs/gold_breakout_backtest.json.
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


def load(sym, tz_hours):
    d = pd.DataFrame(np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy")))
    d.index = pd.to_datetime(d["time"], unit="s", utc=True) + pd.Timedelta(hours=tz_hours)  # broker clock
    return d[["open", "high", "low", "close"]].astype(float)


def run(df, sym, stop_pct=1.0, range_end=7, close_hour=20):
    spread = mb.SPREAD_BPS.get(sym, 1.0) / 1e4
    out = []
    for day, g in df.groupby(df.index.date):
        if pd.Timestamp(day).dayofweek >= 5:
            continue
        rng = g[g.index.hour < range_end]
        sess = g[(g.index.hour >= range_end) & (g.index.hour < close_hour)]
        if len(rng) < 60 or len(sess) < 24:          # need most of the range window and of the session
            continue
        hi, lo = rng["high"].max(), rng["low"].min()
        side = entry = stop = None
        o, h, l, c = (sess[x].to_numpy() for x in ("open", "high", "low", "close"))
        res = None
        for i in range(len(sess)):
            if side is None:
                up, dn = h[i] >= hi, l[i] <= lo
                if not (up or dn):
                    continue
                if up and dn:
                    up = abs(hi - o[i]) <= abs(o[i] - lo)
                    dn = not up
                side = 1 if up else -1
                entry = max(hi, o[i]) if side == 1 else min(lo, o[i])     # gap through the level fills at the open
                stop = entry * (1 - side * stop_pct / 100)
                # same bar: stop also touched after the fill -> loss (conservative)
                if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
                    res = -1.0
                    break
                continue
            if (side == 1 and l[i] <= stop) or (side == -1 and h[i] >= stop):
                res = -1.0
                break
        if side is None:
            continue
        if res is None:
            res = side * (c[-1] - entry) / abs(entry - stop)
        cost = spread * entry / abs(entry - stop)
        out.append({"day": str(day), "side": side, "R": float(res - cost), "range_pct": float((hi - lo) / lo * 100)})
    return out


def summ(rs, risk):
    rs = np.array(rs)
    if not len(rs):
        return {"n": 0}
    eq = np.cumsum(rs)
    grow = np.cumprod(1 + rs * risk / 100)
    return {"n": int(len(rs)), "win": round(float((rs > 0).mean()), 3), "avgR": round(float(rs.mean()), 3),
            "totalR": round(float(rs.sum()), 1), "pf": round(float(rs[rs > 0].sum() / max(1e-9, -rs[rs < 0].sum())), 2),
            "maxDD_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1),
            "t": round(float(rs.mean() / (rs.std(ddof=1) / np.sqrt(len(rs)) + 1e-12)), 2) if len(rs) > 2 else None,
            f"growth_at_{risk}pct": round(float(grow[-1] * 100 - 100), 1),
            f"maxDD_pct_at_{risk}pct": round(float(((np.maximum.accumulate(grow) - grow) / np.maximum.accumulate(grow)).max() * 100), 1)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="XAUUSD")
    ap.add_argument("--tz", type=int, default=3, help="broker clock offset from UTC (TradingView page: GMT+3)")
    ap.add_argument("--stop-pct", type=float, default=1.0)
    ap.add_argument("--risk", type=float, default=1.0, help="%% of equity per trade for the growth column")
    a = ap.parse_args()
    report = {}
    for sym in a.symbols.split(","):
        tr = run(load(sym, a.tz), sym, a.stop_pct)
        rs = [x["R"] for x in tr]
        months = pd.Series(rs, index=pd.to_datetime([x["day"] for x in tr])).resample("ME").sum().round(2)
        res = {"all": summ(rs, a.risk), "first_half": summ(rs[:len(rs) // 2], a.risk),
               "second_half": summ(rs[len(rs) // 2:], a.risk),
               "last_month": summ([x["R"] for x in tr if x["day"] >= "2026-10-27"], a.risk),
               "by_month_R": {str(k.date())[:7]: v for k, v in months.items()}}
        report[sym] = res
        print(sym, json.dumps({k: v for k, v in res.items() if k != "by_month_R"}, indent=1))
        print(" by month (R):", res["by_month_R"])
    json.dump(report, open(os.path.join(ROOT, "logs", "gold_breakout_backtest.json"), "w"), indent=1)


if __name__ == "__main__":
    main()
