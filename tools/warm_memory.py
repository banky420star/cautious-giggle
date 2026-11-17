"""Warm up the model memory by replaying cached Exness history through the indicator council.

Each indicator strategy's vote on bar t is scored the same way the live bot scores it: against
the move over the next `horizon_bars`, minus the spread paid that bar (|vote| x spread), filed
under the market context of bar t. Run once (or after clearing logs/memory) so the council starts with
experience instead of a blank slate:

    .venv/bin/python tools/warm_memory.py
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Python.indicator_council import context_series, describe, votes_mtf
from Python.model_memory import ModelMemory
from Python.config_utils import load_project_config

CACHE = os.path.join(ROOT, "data", "exness_cache")
INST = {"BTCUSDm": "BTCUSD", "XAUUSDm": "XAUUSD"}


def _point(sym):
    try:
        from Python.mt5_compat import mt5
        mt5.initialize()
        info = mt5.symbol_info(sym)
        return float(info.point) if info is not None and float(info.point) > 0 else None
    except Exception:
        return None


def main():
    cfg = load_project_config(ROOT)
    mem = ModelMemory.from_cfg(cfg, os.path.join(ROOT, "logs"))
    h = mem.horizon
    for sym, inst in INST.items():
        path = os.path.join(CACHE, f"{inst}_M5.npy")
        if not os.path.exists(path):
            print(f"{sym}: no cached history ({path}); run training first")
            continue
        arr = np.load(path)
        df = pd.DataFrame({k: arr[k] for k in ("open", "high", "low", "close")},
                          index=pd.to_datetime(arr["time"], unit="s", utc=True))
        v = votes_mtf(df)
        ctx = context_series(df)
        ret_bps = (np.log(df["close"]).shift(-h) - np.log(df["close"])) * 1e4   # move from t to t+h
        gap_ok = (df.index.to_series().shift(-h) - df.index.to_series() <= pd.Timedelta(hours=4))
        pt = _point(sym)
        if pt is None:
            print(f"{sym}: point size unknown (MT5 not reachable) — scoring BEFORE costs")
            cost = np.zeros(len(df))
        else:
            cost = arr["spread"].astype(float) * pt / df["close"].to_numpy() * 1e4
            print(f"{sym}: charging the recorded spread, median {np.median(cost):.2f} bps")
        ok = (ret_bps.notna() & gap_ok).to_numpy().copy()
        ok[:300] = False                                               # indicator warm-up
        for name in v.columns:
            vv = v[name].to_numpy()
            score = vv * ret_bps.to_numpy() - np.abs(vv) * cost
            mem.bulk_learn(sym, f"ind:{name}", ctx.to_numpy()[ok], score[ok], (np.abs(vv) > 0.05)[ok])
        mem.save()
        print(f"{sym}: replayed {int(ok.sum()):,} bars x {v.shape[1]} indicator/timeframe voters "
              f"({df.index[0].date()} .. {df.index[-1].date()})")
        cur = ctx.iloc[-1]
        print(f"  current context {cur}:")
        rows = []
        for name in v.columns:
            a = mem.assess(sym, f"ind:{name}", cur)
            rows.append((a["mult"], name, a))
        ranked = sorted(rows, reverse=True)
        for mult, name, a in ranked[:6] + [(None, "...", None)] + ranked[-4:]:
            if a is None:
                print("    ...")
                continue
            hr = "-" if a["hit_rate"] is None else f"{a['hit_rate'] * 100:.0f}%"
            print(f"    {describe(name):<44} x{mult:.2f}  net edge {a['edge_bps']:+.3f} bps  hit {hr:>4}  {a['verdict']}")


if __name__ == "__main__":
    main()
