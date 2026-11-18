"""Slow trend-following on years of daily data: does it make money after costs, and does it hold up over time?

Rules tested (each decided on the daily close, traded the next day, no look-ahead):
  * momentum L: long if the last L days rose, short if they fell (L = 20, 60, 120, 250 trading days)
  * blend: the average of the four momentum signals (from -1 to +1)
  * ema 50/200: long while the 50-day EMA is above the 200-day, short below
Each market's position is scaled to the same risk (10% yearly volatility, from the last 60 days), so a
calm currency and a wild coin count equally. The portfolio averages all markets that have data that day.

Costs:
  * spread: the market's median real bid/ask spread, charged every time the position changes
  * financing: CFDs charge overnight swap on open positions. Exness rates vary by market and side, so a flat
    3% a year on the position held is charged as an assumption (the report shows results with and without).
Checks: results per period (2003-09, 2010-14, 2015-19, 2020-26) and vs simply buying and holding.

    PYTHONPATH=. .venv/bin/python tools/trend_test.py      -> logs/trend_test.json
"""
from __future__ import annotations

import glob
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "data", "daily")
BOT = ["XAUUSD", "BTCUSD", "EURUSD", "USDJPY", "XAGUSD", "ETHUSD", "USOIL"]
VOL_TARGET, FIN = 0.10, 0.03
PERIODS = [("2003-09", "2003", "2009"), ("2010-14", "2010", "2014"), ("2015-19", "2015", "2019"), ("2020-26", "2020", "2026")]


def load() -> dict[str, pd.DataFrame]:
    out = {}
    for p in sorted(glob.glob(os.path.join(DAILY, "*_D1.csv"))):
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        df.index = df.index.tz_convert(None) if df.index.tz is not None else df.index
        df = df[df.index.day_of_week < 5]                                    # drop weekend stubs (crypto keeps weekdays)
        out[os.path.basename(p).split("_")[0]] = df
    return out


def signals(c: pd.Series) -> dict[str, pd.Series]:
    s = {f"momentum {L}d": np.sign(c / c.shift(L) - 1) for L in (20, 60, 120, 250)}
    s["blend"] = sum(s.values()) / 4
    e50, e200 = c.ewm(span=50, adjust=False).mean(), c.ewm(span=200, adjust=False).mean()
    s["ema 50/200"] = np.sign(e50 - e200)
    s["buy and hold"] = pd.Series(1.0, index=c.index)
    return s


def market_returns(df: pd.DataFrame, sig: pd.Series, fin: float) -> pd.Series:
    c = df["close"]
    r = c.pct_change()
    vol = r.rolling(60).std() * np.sqrt(252)
    pos = (sig * (VOL_TARGET / vol)).clip(-3, 3).shift(1)                   # decided at close, held next day
    spread = ((df["ask_close"] - df["close"]) / df["close"]).clip(lower=0)
    sp = float(spread[spread > 0].median()) if (spread > 0).any() else 0.0003
    cost = pos.diff().abs() * sp + pos.abs() * fin / 252
    return (pos * r - cost).dropna()


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    if len(r) < 120:
        return {}
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    dd = float((eq / eq.cummax() - 1).min())
    t = r.mean() / (r.std() / np.sqrt(len(r))) if r.std() > 0 else 0.0
    return {"years": round(len(r) / 252, 1), "return_pct_yr": round(ann * 100, 2), "vol_pct": round(vol * 100, 1),
            "sharpe": round(ann / vol, 2) if vol > 0 else 0.0, "max_dd_pct": round(dd * 100, 1), "t": round(float(t), 2)}


def run() -> dict:
    data = load()
    out = {"markets": list(data), "assumed_financing_pct_yr": FIN * 100, "vol_target_pct": VOL_TARGET * 100,
           "rules": {}}
    rules = list(signals(next(iter(data.values()))["close"]))
    for rule in rules:
        res = {}
        for fin_name, fin in (("after spread + financing", FIN), ("after spread only", 0.0)):
            per = {name: market_returns(df, signals(df["close"])[rule], fin) for name, df in data.items()}
            port = pd.concat(per, axis=1, sort=True).mean(axis=1, skipna=True)
            bot = pd.concat({k: v for k, v in per.items() if k in BOT}, axis=1, sort=True).mean(axis=1, skipna=True)
            res[fin_name] = {
                "all 21 markets": stats(port), "your 7 markets": stats(bot),
                "by period (all markets)": {p: stats(port.loc[a:b]) for p, a, b in PERIODS},
                "by market": {k: stats(v) for k, v in per.items()},
            }
        out["rules"][rule] = res
    with open(os.path.join(ROOT, "logs", "trend_test.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)
    return out


def summary(out: dict) -> None:
    k = "after spread + financing"
    print(f"Risk-matched to {out['vol_target_pct']:.0f}% yearly volatility per market; costs = real spread + "
          f"{out['assumed_financing_pct_yr']:.0f}%/yr assumed financing\n")
    print(f"{'rule':14} | {'ALL 21: %/yr':>12} {'Sharpe':>6} {'maxDD':>6} {'t':>5} | {'YOUR 7: %/yr':>12} {'Sharpe':>6} | "
          + " ".join(f"{p:>8}" for p, _, _ in PERIODS) + "  (Sharpe by period)")
    for rule, r in out["rules"].items():
        a, b = r[k]["all 21 markets"], r[k]["your 7 markets"]
        per = " ".join(f"{r[k]['by period (all markets)'][p].get('sharpe', float('nan')):8.2f}" for p, _, _ in PERIODS)
        print(f"{rule:14} | {a['return_pct_yr']:12.2f} {a['sharpe']:6.2f} {a['max_dd_pct']:6.1f} {a['t']:5.1f} | "
              f"{b['return_pct_yr']:12.2f} {b['sharpe']:6.2f} | {per}")
    print("\nBlend rule, per market (Sharpe after costs; buy-and-hold for comparison):")
    bl, bh = out["rules"]["blend"][k]["by market"], out["rules"]["buy and hold"][k]["by market"]
    for m in sorted(bl, key=lambda m: -bl[m].get("sharpe", -9)):
        print(f"  {m:7} trend {bl[m].get('sharpe', float('nan')):5.2f} ({bl[m].get('return_pct_yr', float('nan')):+6.2f}%/yr, "
              f"maxDD {bl[m].get('max_dd_pct', float('nan')):6.1f}%) | buy&hold {bh[m].get('sharpe', float('nan')):5.2f} "
              f"| {bl[m].get('years')} yrs")


if __name__ == "__main__":
    summary(run())
