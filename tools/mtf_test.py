"""Multi-timeframe test: big timeframes set the bias, small timeframes time the entry.

Rules fixed in advance (nothing fitted), so both halves of the year are honest out-of-sample checks:
  BIAS   4-hour AND daily trend agree: close above (below) its 50-bar EMA on both -> only longs (shorts).
         Uses completed 4h / daily bars only (no peeking at the bar still forming).
  ENTRY  on 5-minute bars, in the bias direction:
           rsi_pullback  RSI(14) comes back above 40 after dipping below it (longs; mirror for shorts)
           bb_bounce     low touched the lower Bollinger band (20, 2) and the bar closed back inside, up (longs)
         one trade per 60 minutes at most
  CONTROL bias only, entering at a random 5-minute bar instead (same number of trades)
  EXIT   fixed holds of 1 h and 4 h; result = move in the trade's direction minus the real spread
Reported per market and per half of the year (first / second), in bps per trade, with t-statistics.
    PYTHONPATH=. .venv/bin/python tools/mtf_test.py
"""
import json
import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
HOLDS = {"1h": 12, "4h": 48}


def htf_trend(c5: pd.Series, rule: str) -> pd.Series:
    b = c5.resample(rule, label="right", closed="left").last().dropna()     # bar known at its right edge
    tr = np.sign(b - b.ewm(span=50, adjust=False).mean())
    return tr.reindex(c5.index, method="ffill")


def rsi(c, n=14):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def stat(r):
    r = np.asarray(r)
    if len(r) < 20:
        return {"n": len(r), "bps": None, "t": None}
    return {"n": int(len(r)), "bps": round(float(r.mean()), 2), "t": round(float(r.mean() / (r.std() / np.sqrt(len(r)))), 2),
            "win": round(100 * float((r > 0).mean()), 1)}


def run():
    rng = np.random.default_rng(3)
    out = {}
    print(f"{'market':7} {'entry':13} {'hold':4} | {'first half':>24} | {'second half':>24}")
    for sym, cost in COST.items():
        a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        df = pd.DataFrame(a)
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[~df.index.duplicated()].sort_index()
        c, l, h = df["close"], df["low"], df["high"]
        bias = htf_trend(c, "4h")
        bias = bias.where(bias == htf_trend(c, "1D"), 0).fillna(0)
        r14 = rsi(c)
        ma, sd = c.rolling(20).mean(), c.rolling(20).std()
        long_rsi = (r14 > 40) & (r14.shift(1) <= 40)
        short_rsi = (r14 < 60) & (r14.shift(1) >= 60)
        long_bb = (l <= ma - 2 * sd) & (c > ma - 2 * sd) & (c > df["open"])
        short_bb = (h >= ma + 2 * sd) & (c < ma + 2 * sd) & (c < df["open"])
        entries = {"rsi_pullback": np.where((bias > 0) & long_rsi, 1, np.where((bias < 0) & short_rsi, -1, 0)),
                   "bb_bounce": np.where((bias > 0) & long_bb, 1, np.where((bias < 0) & short_bb, -1, 0))}
        t = df["time"].to_numpy()
        cv = c.to_numpy()
        half = len(cv) // 2
        res = {}
        for ename, sig in entries.items():
            sig = sig.copy()
            sig[:300] = 0
            idx, last = [], -10**9
            for i in np.flatnonzero(sig):
                if i - last >= 12:
                    idx.append(i)
                    last = i
            idx = np.array(idx)
            bias_idx = np.flatnonzero(bias.to_numpy() != 0)
            bias_idx = bias_idx[bias_idx > 300]
            ctrl = np.sort(rng.choice(bias_idx, size=min(len(bias_idx), len(idx)), replace=False)) if len(idx) else idx
            for hname, hb in HOLDS.items():
                for label, ii, side in ((ename, idx, sig), ("bias+random", ctrl, np.sign(bias.to_numpy()))):
                    if label == "bias+random" and ename != "rsi_pullback":
                        continue
                    ok = ii[(ii + hb < len(cv))]
                    ok = ok[(t[ok + hb] - t[ok]) <= hb * 300 * 1.5]
                    r = side[ok] * (cv[ok + hb] / cv[ok] - 1) * 1e4 - cost
                    first, second = stat(r[ok < half]), stat(r[ok >= half])
                    res[f"{label}|{hname}"] = {"first": first, "second": second}
                    f = lambda s: f"{s['n']:5} tr {s['bps']:+7.2f} bps t={s['t']:+5.1f}" if s["bps"] is not None else f"{s['n']:5} tr      n/a"  # noqa: E731
                    print(f"{sym:7} {label:13} {hname:4} | {f(first):>24} | {f(second):>24}", flush=True)
        out[sym] = res
    with open(os.path.join(ROOT, "logs", "mtf_test.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh, indent=1)


if __name__ == "__main__":
    run()
