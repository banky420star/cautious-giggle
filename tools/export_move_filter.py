"""Export each trained move filter (LightGBM) to plain arrays so the bot can run it with NumPy only.
LightGBM can't be loaded in the bot's process: it segfaults next to PyTorch on macOS.
Checks the NumPy result equals LightGBM's own on 5,000 real rows before saving.
    PYTHONPATH=. .venv/bin/python tools/export_move_filter.py      (run in a process WITHOUT torch loaded first)
"""
import glob
import os
import pickle
import sys

import lightgbm  # noqa: F401  (must load before torch)
import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def flatten(tree):
    feat, thr, left, right, dleft, miss, leaf = [], [], [], [], [], [], []

    def add(node):
        i = len(feat)
        for lst in (feat, thr, left, right, dleft, miss, leaf):
            lst.append(0)
        if "leaf_value" in node:
            feat[i], leaf[i], left[i] = -1, node["leaf_value"], -1
            return i
        if node.get("decision_type", "<=") != "<=":
            raise ValueError("categorical split not supported")
        feat[i], thr[i] = node["split_feature"], node["threshold"]
        dleft[i] = 1 if node.get("default_left", True) else 0
        miss[i] = {"None": 0, "Zero": 1, "NaN": 2}[node.get("missing_type", "None")]
        left[i] = add(node["left_child"])
        right[i] = add(node["right_child"])
        return i

    add(tree["tree_structure"])
    return {k: np.asarray(v, dtype=t) for k, v, t in (("feat", feat, np.int32), ("thr", thr, np.float64), ("left", left, np.int32),
                                                     ("right", right, np.int32), ("dleft", dleft, np.int8), ("miss", miss, np.int8),
                                                     ("leaf", leaf, np.float64))}


def main():
    from Python.move_filter import predict_trees
    from tools.train_move_filter import dataset
    for p in sorted(x for x in glob.glob(os.path.join(ROOT, "models", "move_filter", "*.pkl")) if not x.endswith(".npz.pkl")):
        sym = os.path.basename(p)[:-4]
        b = pickle.load(open(p, "rb"))
        dump = b["model"].booster_.dump_model()
        trees = [flatten(t) for t in dump["tree_info"]]
        f, _, _ = dataset(sym)
        x = f[b["features"]].iloc[-5000:].to_numpy(np.float64)
        ref = b["model"].predict_proba(x.astype("float32"))[:, 1]
        got = predict_trees(trees, x)
        err = float(np.abs(ref - got).max())
        assert err < 1e-6, (sym, err)
        out = {"trees": trees, "features": b["features"], "threshold": b["threshold"], "horizon_bars": b["horizon_bars"],
               "report": b["report"]}
        with open(os.path.join(ROOT, "models", "move_filter", f"{sym}.npz.pkl"), "wb") as fh:
            pickle.dump(out, fh)
        print(f"{sym}: {len(trees)} trees exported, max difference vs LightGBM {err:.1e}", flush=True)


if __name__ == "__main__":
    main()
