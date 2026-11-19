"""Honest big-move filter readings for PPO training: data/move_filter_oof/<SYM>.npz (time -> chance of a big move).

The live filter (models/move_filter/<SYM>.pkl) learned from the first 70% of the history. PPO trains on that same
stretch, where the filter's readings are in-sample and look better than they really are. So for every bar in the
filter's training stretch the reading comes from a filter trained WITHOUT that part of the data (5 time blocks,
each predicted by a model fitted on the other 4, with a 1-hour purge on both sides so no label leaks across).
After the training stretch, the live filter itself is used: those readings are what it really saw.

Run BEFORE torch is imported anywhere in the process (LightGBM segfaults after torch on macOS):
    PYTHONPATH=. .venv/bin/python tools/move_filter_oof.py [SYM ...]
"""
import lightgbm as lgb  # noqa: I001  (must load before torch)
import os
import pickle
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.agi_brain import build_feature_frame  # noqa: E402
from tools.train_move_filter import COST, H, dataset  # noqa: E402

OUT = os.path.join(ROOT, "data", "move_filter_oof")
FOLDS = 5


def run(symbols):
    os.makedirs(OUT, exist_ok=True)
    for sym in symbols:
        prod = pickle.load(open(os.path.join(ROOT, "models", "move_filter", f"{sym}.pkl"), "rb"))
        m_prod, feats, thr = prod["model"], prod["features"], float(prod["threshold"])
        f, fwd, typ = dataset(sym)
        y = (fwd.abs() > 2 * typ).to_numpy().astype(int)
        n = len(f)
        cut = int(n * 0.7)
        cut_time = f.index[cut]
        # every bar (labelled or not) gets a reading
        a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        df = pd.DataFrame(a)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        fa = build_feature_frame(df)
        fa.index = df["time"].iloc[-len(fa):].to_numpy()
        fa = fa.reindex(columns=feats).replace([np.inf, -np.inf], np.nan).astype("float32")
        p = np.full(len(fa), np.nan)
        after = fa.index >= cut_time
        p[after] = m_prod.predict_proba(fa[after])[:, 1]
        n_trees = int(getattr(m_prod, "best_iteration_", 0) or m_prod.n_estimators)
        bounds = np.linspace(0, cut, FOLDS + 1).astype(int)
        X, Y = f.reindex(columns=feats), y
        for k in range(FOLDS):
            lo, hi = bounds[k], bounds[k + 1]
            keep = np.ones(cut, bool)
            keep[max(0, lo - H):min(cut, hi + H)] = False           # the block itself plus a 1-hour purge
            m = lgb.LGBMClassifier(n_estimators=n_trees, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                                   subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)
            m.fit(X.iloc[:cut][keep], Y[:cut][keep])
            t0 = f.index[lo] if k else fa.index[0]
            t1 = f.index[hi] if hi < cut else cut_time
            sel = (fa.index >= t0) & (fa.index < t1)
            p[sel] = m.predict_proba(fa[sel])[:, 1]
        ts = (pd.DatetimeIndex(fa.index).as_unit("s").asi8).astype(np.int64)
        ok = np.isfinite(p)
        np.savez(os.path.join(OUT, f"{sym}.npz"), t=ts[ok], p=p[ok].astype(np.float32), thr=np.float32(thr),
                 cut=np.int64(cut_time.value // 10**9))
        tr, te = ok & ~after, ok & after
        print(f"{sym:7} {ok.sum():6} readings | held-out-block share GO on train stretch {np.mean(p[tr] >= thr):.0%} "
              f"| live filter share GO after {str(cut_time)[:10]}: {np.mean(p[te] >= thr):.0%}", flush=True)


if __name__ == "__main__":
    run(sys.argv[1:] or list(COST))
