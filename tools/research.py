"""Research harness: test trading ideas honestly on 23 years of daily data, and keep a ledger of every attempt.

Rules (so that "profitable" means real, not lucky):
  * DEV = history up to 2024-11-17. HOLDOUT = 2024-11-18 onward (about 2 years), never used to design ideas.
  * Every idea ever tested is recorded in logs/research_ledger.jsonl. The bar on DEV rises with the number
    of ideas tried (Bonferroni, two-sided 5%): with N ideas, an idea needs t >= z(1 - 0.025 / N).
  * Only ideas that clear the DEV bar get ONE look at the HOLDOUT; there they must make money (Sharpe > 0.3).
  * Costs: real median bid/ask spread on every position change, plus a financing scenario:
    "swap-free" (0) and "3%/yr" on the position held. An idea must clear the bar swap-free AND stay
    positive with 3%/yr.
  * Each market is scaled to the same risk (10% yearly volatility, 60-day estimate), capped at 3x.

    PYTHONPATH=. .venv/bin/python tools/research.py [--only NAME ...]
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DAILY = os.path.join(ROOT, "data", "daily")
LEDGER = os.path.join(ROOT, "logs", "research_ledger.jsonl")
SPLIT = pd.Timestamp("2024-11-18")
VOL_T = 0.10
COST_MODEL = "real_swap_v2"     # v1 = flat 3%/yr; v2 = the account's real swap per market and side
SWAPS = os.path.join(DAILY, "swaps.json")
UNVERIFIED = {"XCUUSD", "XNGUSD", "USOIL", "UKOIL"}   # swap units could not be verified: not traded in research
INDICES = ["US500", "USTEC", "US30", "DE40"]


# ---------------------------------------------------------------------------- data
def load() -> tuple[pd.DataFrame, pd.DataFrame]:
    closes, spreads = {}, {}
    for p in sorted(glob.glob(os.path.join(DAILY, "*_D1.csv"))):
        n = os.path.basename(p).split("_")[0]
        df = pd.read_csv(p, parse_dates=["date"]).set_index("date")
        if df.index.tz is not None:
            df.index = df.index.tz_convert(None)
        df = df[df.index.day_of_week < 5]
        closes[n] = df["close"]
        s = ((df["ask_close"] - df["close"]) / df["close"]).clip(lower=0)
        spreads[n] = float(s[s > 0].median()) if (s > 0).any() else 3e-4
    return pd.DataFrame(closes).sort_index(), pd.Series(spreads)


def risk_scale(ret: pd.DataFrame) -> pd.DataFrame:
    vol = ret.rolling(60, min_periods=40).std() * np.sqrt(252)
    return (VOL_T / vol).clip(upper=3.0)


def real_swaps(cols) -> tuple[pd.Series, pd.Series]:
    """Yearly cost of holding long / short, as a positive fraction of position value (from the account)."""
    with open(SWAPS, encoding="utf-8") as f:
        m = json.load(f)["markets"]
    lo = pd.Series({c: max(0.0, -m.get(c, {}).get("swap_long_pct_yr", -5.0) / 100) for c in cols})
    sh = pd.Series({c: max(0.0, -m.get(c, {}).get("swap_short_pct_yr", -5.0) / 100) for c in cols})
    return lo, sh


def backtest(pos: pd.DataFrame, ret: pd.DataFrame, spread: pd.Series, fin) -> pd.Series:
    """pos = target position decided at day t's close (in risk-scaled units); earns day t+1's return.
    fin = flat yearly financing on the position, or "real" for the account's swap per market and side."""
    if fin == "real":
        pos = pos.drop(columns=[c for c in UNVERIFIED if c in pos.columns]).reindex(columns=pos.columns, fill_value=0.0)
    p = pos.shift(1).fillna(0.0)
    if fin == "real":
        lo, sh = real_swaps(p.columns)
        carry = p.clip(lower=0).mul(lo, axis=1) + (-p.clip(upper=0)).mul(sh, axis=1)
        cost = p.diff().abs().mul(spread, axis=1) + carry / 252
    else:
        cost = p.diff().abs().mul(spread, axis=1) + p.abs() * fin / 252
    pnl = (p * ret - cost)
    active = (p.abs() > 0).sum(axis=1)
    return (pnl.sum(axis=1) / active.clip(lower=1)).where(active > 0, 0.0)


def stats(r: pd.Series) -> dict:
    r = r.dropna()
    r = r[r.index >= r.ne(0).idxmax()] if r.ne(0).any() else r
    if len(r) < 120 or r.std() == 0:
        return {"days": len(r), "sharpe": 0.0, "t": 0.0, "ret_pct_yr": 0.0, "max_dd_pct": 0.0}
    ann, vol = r.mean() * 252, r.std() * np.sqrt(252)
    eq = (1 + r).cumprod()
    return {"days": len(r), "sharpe": round(ann / vol, 2), "t": round(r.mean() / (r.std() / math.sqrt(len(r))), 2),
            "ret_pct_yr": round(ann * 100, 2), "vol_pct": round(vol * 100, 1),
            "max_dd_pct": round(float((eq / eq.cummax() - 1).min()) * 100, 1)}


# ---------------------------------------------------------------------------- ideas
def _weekly(sig: pd.DataFrame) -> pd.DataFrame:
    """Only change positions on the last trading day of each week (cuts trading costs)."""
    wk = sig.index.to_series().dt.isocalendar().week.astype(str) + sig.index.to_series().dt.year.astype(str)
    last = wk != wk.shift(-1)
    out = sig.copy()
    out.loc[~last.values] = np.nan
    return out.ffill().fillna(0.0)


def xs_momentum(c, ret, scale):
    """Cross-sectional momentum: each week, long the 5 strongest markets over 120 days (risk-adjusted),
    short the 5 weakest."""
    mom = c.pct_change(120) / (ret.rolling(120).std() * np.sqrt(120))
    rk = mom.rank(axis=1)
    n = mom.notna().sum(axis=1)
    sig = rk.gt(n - 5, axis=0).astype(float) - (rk <= 5).astype(float)
    sig = sig.mul((n >= 10).astype(float), axis=0)
    return _weekly(sig) * scale


def xs_reversal(c, ret, scale):
    """Short-term reversal: each week, long the 5 biggest losers of the past week, short the 5 biggest winners."""
    m = c.pct_change(5) / (ret.rolling(60).std() * np.sqrt(5))
    rk = m.rank(axis=1)
    n = m.notna().sum(axis=1)
    sig = (rk <= 5).astype(float) - rk.gt(n - 5, axis=0).astype(float)
    sig = sig.mul((n >= 10).astype(float), axis=0)
    return _weekly(sig) * scale


def turn_of_month(c, ret, scale):
    """Stock indices tend to rise around month end: long the 4 indices on the last trading day of the month
    and the first 3 of the next, flat otherwise."""
    idx = c.index.to_series()
    ym = idx.dt.year * 100 + idx.dt.month
    pos_in_m = idx.groupby(ym).cumcount()
    from_end = idx.groupby(ym).cumcount(ascending=False)
    # decided at the close of day t for day t+1: hold on (last day) and (first 3 days)
    hold_next = (from_end.shift(-1) == 0) | (pos_in_m.shift(-1) <= 2)
    sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for m in INDICES:
        if m in sig:
            sig[m] = hold_next.astype(float).values
    return sig * scale


def volmanaged_long(c, ret, scale):
    """Hold indices and gold long, sized to constant risk (cuts exposure when markets get wild)."""
    sig = pd.DataFrame(0.0, index=c.index, columns=c.columns)
    for m in INDICES + ["XAUUSD"]:
        if m in sig:
            sig[m] = 1.0
    return sig * scale


def tsmom_weekly_blend(c, ret, scale):
    """Trend blend (20/60/120/250-day momentum averaged), traded weekly instead of daily."""
    s = sum(np.sign(c.pct_change(L)) for L in (20, 60, 120, 250)) / 4
    return _weekly(s) * scale


IDEAS = {"xs_momentum": xs_momentum, "xs_reversal": xs_reversal, "turn_of_month": turn_of_month,
         "volmanaged_long": volmanaged_long, "tsmom_weekly_blend": tsmom_weekly_blend}
BENCH = "volmanaged_long"   # long-only market exposure: a benchmark, not an "edge"


# ---------------------------------------------------------------------------- ledger
def ledger() -> list[dict]:
    try:
        with open(LEDGER, encoding="utf-8") as f:
            return [json.loads(x) for x in f if x.strip()]
    except OSError:
        return []


def bar(n_ideas: int) -> float:
    from statistics import NormalDist
    return round(NormalDist().inv_cdf(1 - 0.025 / max(1, n_ideas)), 2)


def run(only: list[str] | None = None) -> list[dict]:
    c, spread = load()
    ret = c.pct_change()
    scale = risk_scale(ret)
    led = ledger()
    done = {r["idea"] for r in led if r.get("cost_model") == COST_MODEL}
    todo = [k for k in IDEAS if (not only or k in only) and k not in done]
    n_total = len({r["idea"] for r in led} | set(todo))      # every idea ever tried counts toward the bar
    need = bar(n_total)
    out = []
    for name in todo:
        pos = IDEAS[name](c, ret, scale)
        res = {"t": time.time(), "idea": name, "cost_model": COST_MODEL, "about": (IDEAS[name].__doc__ or "").strip().split("\n")[0],
               "n_ideas_so_far": n_total, "dev_bar_t": need}
        for fname, fin in (("swapfree", 0.0), ("realswap", "real")):
            r = backtest(pos, ret, spread, fin)
            res[fname] = {"dev": stats(r[r.index < SPLIT]),
                          "dev_by_period": {p: stats(r[(r.index >= a) & (r.index < b)]) for p, a, b in
                                            (("2003-09", "2003", "2010"), ("2010-14", "2010", "2015"),
                                             ("2015-19", "2015", "2020"), ("2020-24", "2020", "2024-11-18"))}}
        dev = res["realswap"]["dev"]      # the account's real costs decide
        res["passes_dev"] = bool(dev["t"] >= need and name != BENCH)
        if res["passes_dev"] or name == BENCH:     # one look at the hold-out
            for fname, fin in (("swapfree", 0.0), ("realswap", "real")):
                r = backtest(pos, ret, spread, fin)
                res[fname]["holdout"] = stats(r[r.index >= SPLIT])
            res["passes_holdout"] = bool(res["passes_dev"] and res["realswap"]["holdout"]["sharpe"] > 0.3)
        with open(LEDGER, "a", encoding="utf-8") as f:
            f.write(json.dumps(res) + "\n")
        out.append(res)
    return out


def show(rows: list[dict]) -> None:
    for r in rows:
        d, d3 = r["swapfree"]["dev"], r["realswap"]["dev"]
        per = " ".join(f"{k}:{v['sharpe']:+.2f}" for k, v in r["realswap"]["dev_by_period"].items())
        print(f"{r['idea']:20} DEV real-swap Sharpe {d3['sharpe']:+.2f} t={d3['t']:+.2f} (bar {r['dev_bar_t']}) "
              f"{d3['ret_pct_yr']:+.2f}%/yr DD {d3['max_dd_pct']}% | swap-free {d['sharpe']:+.2f} | {per}")
        if "holdout" in r["swapfree"]:
            h, h3 = r["swapfree"]["holdout"], r["realswap"]["holdout"]
            print(f"{'':20} HOLDOUT real-swap Sharpe {h3['sharpe']:+.2f} ({h3['ret_pct_yr']:+.2f}%/yr) | swap-free {h['sharpe']:+.2f}"
                  f" -> {'PASS' if r.get('passes_holdout') else 'benchmark' if r['idea'] == BENCH else 'fail'}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", nargs="*")
    a = ap.parse_args()
    show(run(a.only))
