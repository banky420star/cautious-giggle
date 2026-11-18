"""Indicator settings sweep: tune every indicator's settings per market, then check on unseen data.

Families and settings tried (on 5-minute, 15-minute and 1-hour bars):
  rsi_revert   period 5/7/9/14/21/28 x levels 20-80 / 25-75 / 30-70 / 35-65: buy when RSI crosses back up
               through the low level, sell when it crosses back down through the high level
  rsi_trend    period 5..28: follow RSI crossing 50
  bb_revert    length 10/20/30/50 x width 1.5/2/2.5/3: close back inside the band after closing outside -> fade
  bb_break     same settings: close outside the band -> follow
  ema_cross    fast 5/8/10/12/20/30 x slow 20/30/50/100/200 (fast < slow): follow the cross
  macd         (fast, slow, signal) 8-21-5 / 12-26-9 / 5-35-5 / 19-39-9: follow, and fade, the signal-line cross
  stoch        %K 5/9/14/21 x levels 20-80 / 10-90: fade crossing back from the extremes
  donchian     channel 10/20/55/100: follow, and fade, a new N-bar high/low
Each signal is a trade at the bar's close, held 1 hour or 4 hours, minus the market's real spread.
TRAIN = first 70% of the year, TEST = last 30% (never used to choose).
  * strict bar: with ~N settings per market tried, a setting must clear z >= Bonferroni(N x 7 markets) on TRAIN
  * the best setting per family (chosen on TRAIN) is reported on TEST, too, so you can see what tuning buys
    PYTHONPATH=. .venv/bin/python tools/param_sweep.py
"""
import itertools
import json
import math
import os
from statistics import NormalDist

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
TFS = {"5m": 5, "15m": 15, "1h": 60}
HOLDS_MIN = {"1h": 60, "4h": 240}


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def rsi(c, n):
    d = c.diff()
    up = d.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
    dn = (-d.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
    return 100 - 100 / (1 + up / dn)


def xup(a, b):
    return (a > b) & (a.shift(1) <= b.shift(1) if isinstance(b, pd.Series) else a.shift(1) <= b)


def xdn(a, b):
    return (a < b) & (a.shift(1) >= b.shift(1) if isinstance(b, pd.Series) else a.shift(1) >= b)


def signals(b: pd.DataFrame):
    c, h, l = b["close"], b["high"], b["low"]
    out = {}
    for n in (5, 7, 9, 14, 21, 28):
        r = rsi(c, n)
        for lo, hi in ((20, 80), (25, 75), (30, 70), (35, 65)):
            out[f"rsi_revert n={n} {lo}/{hi}"] = np.where(xup(r, lo), 1, np.where(xdn(r, hi), -1, 0))
        out[f"rsi_trend n={n}"] = np.where(xup(r, 50), 1, np.where(xdn(r, 50), -1, 0))
    for n in (10, 20, 30, 50):
        ma, sd = c.rolling(n).mean(), c.rolling(n).std()
        for k in (1.5, 2.0, 2.5, 3.0):
            lo_b, hi_b = ma - k * sd, ma + k * sd
            back_in_lo = (c > lo_b) & (c.shift(1) <= lo_b.shift(1))
            back_in_hi = (c < hi_b) & (c.shift(1) >= hi_b.shift(1))
            out[f"bb_revert n={n} k={k}"] = np.where(back_in_lo, 1, np.where(back_in_hi, -1, 0))
            out[f"bb_break n={n} k={k}"] = np.where(xup(c, hi_b), 1, np.where(xdn(c, lo_b), -1, 0))
    for f, s in itertools.product((5, 8, 10, 12, 20, 30), (20, 30, 50, 100, 200)):
        if f < s:
            ef, es = ema(c, f), ema(c, s)
            out[f"ema_cross {f}/{s}"] = np.where(xup(ef, es), 1, np.where(xdn(ef, es), -1, 0))
    for fa, sl, sg in ((8, 21, 5), (12, 26, 9), (5, 35, 5), (19, 39, 9)):
        m = ema(c, fa) - ema(c, sl)
        sgl = ema(m, sg)
        x = np.where(xup(m, sgl), 1, np.where(xdn(m, sgl), -1, 0))
        out[f"macd {fa}-{sl}-{sg} follow"] = x
        out[f"macd {fa}-{sl}-{sg} fade"] = -x
    for k in (5, 9, 14, 21):
        ll, hh = l.rolling(k).min(), h.rolling(k).max()
        st = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
        for lo, hi in ((20, 80), (10, 90)):
            out[f"stoch k={k} {lo}/{hi}"] = np.where(xup(st, lo), 1, np.where(xdn(st, hi), -1, 0))
    for n in (10, 20, 55, 100):
        x = np.where(c > h.shift(1).rolling(n).max(), 1, np.where(c < l.shift(1).rolling(n).min(), -1, 0))
        out[f"donchian n={n} follow"] = x
        out[f"donchian n={n} fade"] = -x
    return out


def run():
    rows, per_market = [], {}
    n_settings = None
    for sym, cost in COST.items():
        a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        m5 = pd.DataFrame(a)
        m5.index = pd.to_datetime(m5["time"], unit="s", utc=True)
        m5 = m5[~m5.index.duplicated()].sort_index()[["open", "high", "low", "close"]]
        split = m5.index[int(len(m5) * 0.7)]
        res = []
        for tfn, mins in TFS.items():
            b = m5 if mins == 5 else m5.resample(f"{mins}min", label="right", closed="left").agg(
                {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
            c = b["close"].to_numpy()
            ts = b.index.asi8 // 10**9
            sig = signals(b)
            for hn, hm in HOLDS_MIN.items():
                hb = max(1, hm // mins)
                fwd = np.full(len(c), np.nan)
                fwd[:-hb] = (c[hb:] / c[:-hb] - 1) * 1e4
                gap = np.full(len(c), False)
                gap[:-hb] = (ts[hb:] - ts[:-hb]) <= hm * 60 * 1.5
                tr_mask = (b.index < split) & gap & np.isfinite(fwd)
                te_mask = (b.index >= split + pd.Timedelta(minutes=hm)) & gap & np.isfinite(fwd)
                for name, s in sig.items():
                    s = np.asarray(s)
                    s[:250] = 0
                    out = {}
                    for part, mk in (("train", tr_mask), ("test", te_mask)):
                        ii = np.flatnonzero((s != 0) & mk)
                        r = s[ii] * fwd[ii] - cost
                        # trades overlapping the same hold window are not independent: count them per window
                        n_eff = max(1.0, min(len(ii), len(np.unique(ii // hb))))
                        out[part] = (len(ii), float(r.mean()) if len(ii) else 0.0,
                                     float(r.mean() / (r.std() / math.sqrt(n_eff))) if len(ii) > 5 and r.std() > 0 else 0.0)
                    res.append({"market": sym, "tf": tfn, "hold": hn, "setting": name, "family": name.split()[0],
                                "n_train": out["train"][0], "train_bps": round(out["train"][1], 2), "z_train": round(out["train"][2], 2),
                                "n_test": out["test"][0], "test_bps": round(out["test"][1], 2), "z_test": round(out["test"][2], 2)})
        n_settings = len(res)
        per_market[sym] = res
        rows += res
        print(f"{sym}: {len(res)} settings x timeframes x holds tested", flush=True)
    zbar = abs(NormalDist().inv_cdf(0.025 / (n_settings * len(COST))))
    print(f"\nstrict bar on TRAIN: z >= {zbar:.2f} ({n_settings} per market x {len(COST)} markets)\n")
    summary = {}
    for sym, res in per_market.items():
        ok = [r for r in res if r["n_train"] >= 100 and r["n_test"] >= 40]
        passed = [r for r in ok if r["z_train"] >= zbar]
        held = [r for r in passed if r["test_bps"] > 0]
        best_fam = {}
        for r in ok:
            k = r["family"]
            if k not in best_fam or r["train_bps"] > best_fam[k]["train_bps"]:
                best_fam[k] = r
        tuned = sorted(best_fam.values(), key=lambda r: -r["train_bps"])
        summary[sym] = {"tested": len(ok), "passed_train": len(passed), "held_test": len(held),
                        "tuned_best_per_family": tuned, "passed": passed[:20]}
        avg_test = np.mean([r["test_bps"] for r in tuned])
        print(f"{sym:7} cost {COST[sym]:.2f} | {len(ok)} usable settings | pass strict bar on train: {len(passed)} | of those positive on test: {len(held)}"
              f" | best-per-family (tuned on train) average on test: {avg_test:+.2f} bps")
        for r in tuned[:4]:
            print(f"        {r['family']:11} best = {r['setting']:28} {r['tf']:>3} hold {r['hold']}: train {r['train_bps']:+6.2f} bps "
                  f"(z {r['z_train']:+.1f}, {r['n_train']}) -> test {r['test_bps']:+6.2f} bps (z {r['z_test']:+.1f}, {r['n_test']})")
    with open(os.path.join(ROOT, "logs", "param_sweep.json"), "w", encoding="utf-8") as fh:
        json.dump({"z_bar": zbar, "summary": summary}, fh, indent=1)


if __name__ == "__main__":
    run()
