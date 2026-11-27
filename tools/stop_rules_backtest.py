"""Which stop management locks in profit best, per market?

    .venv/bin/python tools/stop_rules_backtest.py          # writes logs/stop_rules_report.json
    .venv/bin/python tools/stop_rules_backtest.py --wide   # the council's setup since 2026-11-24: 8 / 12.8 ATR,
                                                           # 24 h, only with the 4h 200-EMA trend (logs/stop_rules_report_wide.json)

Entries: the indicator council's fresh signals (|council| crosses 0.5, its side), one trade at a time per market,
entered at the next M5 bar's open, 17 months of Exness M5 history (data/exness_cache, data/meta_brain).
Every trade starts with the bot's stop / target (Python/meta_brain.STOPS, M5 ATR(14)) and an 8 h limit.

Rules compared (R = the starting stop distance):
  fixed          never move the stop
  current        the executor's rule today: breakeven at 1/3 R, then trail 1/8 R behind the high from 1/2 R
  be X lock L    once the trade has been X R in profit, stop to entry + L R
  trail A / D    once A R in profit, stop trails D ATR behind the best price
  be+trail       breakeven first, trail later
Each rule is also tried without the fixed target (let the trail take the exit).

Simulation is conservative: inside a bar the stop is checked before the target, the stop only moves after the bar
closes (using that bar's best price), a gap through the stop fills at the bar's open, and the spread is charged.
The history is split into 3 time periods; the best rule per market is the one whose WORST period is best (it has
to hold up in every period, not just on average), and its gain over never moving the stop is reported with a t-stat.
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.meta_brain import HORIZON, SPREAD_BPS, STOPS  # noqa: E402

SYMS = ["XAUUSD", "BTCUSD", "EURUSD", "USDJPY", "XAGUSD", "ETHUSD", "USOIL"]
ENTRY = 0.5
GAP = 6          # bars between trades on one market (the bot takes several per day)
FOLDS = 3        # time periods: a rule must hold up in each


def atr14(h, l, c):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / 14, adjust=False).mean().to_numpy()


WIDE = "--wide" in sys.argv
if WIDE:
    STOPS = {**{k: (8.0, 12.8) for k in SYMS}, "default": (8.0, 12.8)}  # noqa: F811
    HORIZON = 288  # noqa: F811
    SYMS = ["XAUUSD", "BTCUSD", "XAGUSD", "ETHUSD"]      # the council's markets now


def rules():
    out = {"fixed": dict(kind="fixed"),
           "current": dict(kind="current")}
    for x in ((0.25, 0.5, 0.75, 1.0) if WIDE else (0.5, 0.75, 1.0, 1.5)):
        for lock in (0.0, 0.1, 0.25, 0.5):
            if lock < x:
                out[f"be {x}R lock {lock}R"] = dict(kind="be", x=x, lock=lock)
    for a in ((0.25, 0.5, 0.75, 1.0) if WIDE else (0.5, 1.0, 1.5)):
        for d in ((1.0, 2.0, 3.0, 4.0, 6.0) if WIDE else (0.5, 1.0, 1.5, 2.0)):
            out[f"trail from {a}R, {d} ATR"] = dict(kind="trail", a=a, d=d)
    for x in ((0.25, 0.5) if WIDE else (0.5, 1.0)):
        for a, d in (((0.5, 3.0), (0.75, 3.0), (0.75, 4.0)) if WIDE else ((1.0, 1.0), (1.5, 1.0), (1.5, 1.5))):
            if a > x:
                out[f"be {x}R + trail from {a}R, {d} ATR"] = dict(kind="betrail", x=x, a=a, d=d)
    full = {}
    for k, v in out.items():
        full[k] = {**v, "tp": True}
        if v["kind"] in ("trail", "betrail"):
            full[k + ", no target"] = {**v, "tp": False}
    return full


def trades(sym):
    a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"))
    feat = pd.read_pickle(os.path.join(ROOT, "data", "meta_brain", f"{sym}.pkl"))
    o, h, l, c = (a[k].astype(float) for k in ("open", "high", "low", "close"))
    atr = atr14(h, l, c)
    sl_m, tp_m = STOPS.get(sym, STOPS["default"])
    cs, bars = feat["council"].to_numpy(), feat["bar"].to_numpy().astype(int)
    h4 = feat["ema200@H4"].to_numpy()
    rows, busy_until = [], -1
    prev = 0.0
    for i_, (v, b) in enumerate(zip(cs, bars)):
        fresh = abs(v) >= ENTRY and abs(prev) < ENTRY
        prev = v
        if WIDE and fresh and np.sign(h4[i_]) != np.sign(v):
            fresh = False
        if not fresh or b <= busy_until or b + 1 + HORIZON >= len(c):
            continue
        d = 1.0 if v > 0 else -1.0
        e = o[b + 1]
        R = sl_m * atr[b]
        if not R > 0:
            continue
        seg = slice(b + 1, b + 1 + HORIZON)
        best = (h[seg] if d > 0 else l[seg])
        worst = (l[seg] if d > 0 else h[seg])
        rows.append({"bar": b, "t": int(a["time"][b]), "op": d * (o[seg] - e) / R, "up": d * (best - e) / R,
                     "dn": d * (worst - e) / R, "cl": d * (c[seg] - e) / R, "atr_r": atr[b] / R,
                     "cost": SPREAD_BPS[sym] / 1e4 * e / R, "tp": tp_m / sl_m})
        busy_until = b + GAP        # a new signal may open another trade after GAP bars (more samples)
    return rows, (sl_m, tp_m)


def simulate(T, rule):
    """R result per trade (after spread) and how it ended."""
    n = len(T)
    op = np.stack([t["op"] for t in T]); up = np.stack([t["up"] for t in T])
    dn = np.stack([t["dn"] for t in T]); cl = np.stack([t["cl"] for t in T])
    atr_r = np.array([t["atr_r"] for t in T]); cost = np.array([t["cost"] for t in T])
    tp = np.array([t["tp"] for t in T]) if rule["tp"] else np.full(n, np.inf)
    stop = np.full(n, -1.0); mfe = np.zeros(n)
    res = np.full(n, np.nan); how = np.zeros(n, dtype=int)   # 1 stop, 2 target, 3 time
    live = np.ones(n, dtype=bool)
    for k in range(op.shape[1]):
        hit_s = live & (dn[:, k] <= stop)
        res[hit_s] = np.minimum(stop[hit_s], op[hit_s, k]); how[hit_s] = 1; live &= ~hit_s
        hit_t = live & (up[:, k] >= tp)
        res[hit_t] = tp[hit_t]; how[hit_t] = 2; live &= ~hit_t
        mfe = np.where(live, np.maximum(mfe, up[:, k]), mfe)
        kind = rule["kind"]
        new = stop.copy()
        if kind == "current":
            new = np.where(mfe >= 1 / 3, np.maximum(new, 0.02), new)
            new = np.where(mfe >= 0.5, np.maximum(new, mfe - 1 / 8), new)
        if kind in ("be", "betrail"):
            new = np.where(mfe >= rule["x"], np.maximum(new, rule.get("lock", 0.0)), new)
        if kind in ("trail", "betrail"):
            new = np.where(mfe >= rule["a"], np.maximum(new, mfe - rule["d"] * atr_r), new)
        stop = np.where(live, new, stop)
    res[live] = cl[live, -1]; how[live] = 3
    return res - cost, how


def summary(r, how):
    w = r > 0
    return {"trades": int(len(r)), "avg_r": round(float(r.mean()), 4), "win_rate": round(float(w.mean()), 3),
            "avg_win": round(float(r[w].mean()), 3) if w.any() else 0.0,
            "avg_loss": round(float(r[~w].mean()), 3) if (~w).any() else 0.0,
            "stopped": round(float((how == 1).mean()), 3), "target": round(float((how == 2).mean()), 3),
            "time": round(float((how == 3).mean()), 3)}


def main():
    R = rules()
    report = {"entry": f"council fresh signal |council| >= {ENTRY}, >= {GAP} bars apart",
              "method": f"{FOLDS} consecutive time periods; rule chosen by its WORST period (must hold up in each)",
              "markets": {}}
    for sym in SYMS:
        T, (sl_m, tp_m) = trades(sym)
        folds = np.array_split(np.arange(len(T)), FOLDS)
        rows = {}
        for name, rule in R.items():
            r, how = simulate(T, rule)
            per = [float(r[f].mean()) for f in folds]
            rows[name] = {"rule": rule, "all": summary(r, how), "periods": [round(x, 4) for x in per],
                          "worst": round(min(per), 4), "_r": r}
        best = max(rows, key=lambda k: (rows[k]["worst"], rows[k]["all"]["avg_r"]))
        fx, cur, b = rows["fixed"], rows["current"], rows[best]
        diff = b["_r"] - fx["_r"]
        se = float(diff.std(ddof=1) / np.sqrt(len(diff)))
        rank = sorted(rows, key=lambda k: (-rows[k]["worst"], -rows[k]["all"]["avg_r"]))
        report["markets"][sym] = {"stops_atr": [sl_m, tp_m], "trades": len(T), "best": best, "best_rule": b["rule"],
                                  "best": best, "best_all": b["all"], "best_periods": b["periods"],
                                  "fixed_all": fx["all"], "fixed_periods": fx["periods"],
                                  "current_all": cur["all"], "current_periods": cur["periods"],
                                  "gain_vs_fixed_r": round(float(diff.mean()), 4), "gain_t": round(float(diff.mean()) / se, 2) if se else None,
                                  "top8": [{"rule": k, "worst_period_r": rows[k]["worst"], "avg_r": rows[k]["all"]["avg_r"],
                                            "win_rate": rows[k]["all"]["win_rate"]} for k in rank[:8]]}
        print(f"{sym:7s} n={len(T):5d} fixed {fx['all']['avg_r']:+.3f} {fx['periods']}  current {cur['all']['avg_r']:+.3f} {cur['periods']}\n"
              f"        best [{best}] {b['all']['avg_r']:+.3f} {b['periods']} win {b['all']['win_rate']:.0%}  "
              f"gain vs fixed {diff.mean():+.3f}R (t={diff.mean() / se:.1f})", flush=True)
    with open(os.path.join(ROOT, "logs", "stop_rules_report_wide.json" if WIDE else "stop_rules_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=1)


if __name__ == "__main__":
    main()
