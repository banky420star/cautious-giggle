"""Warm up the model memory by replaying cached Exness history through the indicator council.

Each indicator strategy's vote on bar t is scored against the move from t to t+1, filed under
the market context of bar t. Run once (or after clearing logs/memory) so the council starts with
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


def main():
    cfg = load_project_config(ROOT)
    mem = ModelMemory.from_cfg(cfg, os.path.join(ROOT, "logs"))
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
        ret_bps = np.log(df["close"]).diff().shift(-1) * 1e4          # move from t to t+1
        gap_ok = (df.index.to_series().diff().shift(-1) <= pd.Timedelta(hours=4))
        ok = (ret_bps.notna() & gap_ok).to_numpy().copy()
        ok[:300] = False                                               # indicator warm-up
        for name in v.columns:
            vv = v[name].to_numpy()
            score = vv * ret_bps.to_numpy()
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
            print(f"    {describe(name):<44} x{mult:.2f}  edge {a['edge_bps']:+.3f} bps/bar  hit {hr:>4}  {a['verdict']}")


if __name__ == "__main__":
    main()
