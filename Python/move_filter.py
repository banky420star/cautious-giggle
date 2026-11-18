"""Per-market "big move coming" filter (trained by tools/train_move_filter.py).

prob(symbol, df) -> {"p": chance the next hour moves > 2x the usual hour, "go": p >= trained threshold} or None.
WATCH-ONLY: the bot logs this next to every decision; it does not block or size trades.
Needs enough M5 history for the daily inputs: HISTORY_BARS are fetched separately and cached per symbol.
"""
import os
import pickle
import threading

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_DIR = os.path.join(ROOT, "models", "move_filter")
_models, _lock = {}, threading.Lock()
HISTORY_BARS = int(os.environ.get("MOVE_FILTER_BARS", "6000"))


def base_symbol(sym: str) -> str:
    """XAUUSDm / USOILm / XAUUSDc -> the trained market name (exact name first, then without the account suffix)."""
    s = str(sym)
    for cand in (s.upper(), s[:-1].upper() if s[-1:] in "mcMC" else None):
        if cand and os.path.exists(os.path.join(_DIR, f"{cand}.npz.pkl")):
            return cand
    return s.upper()


def _load(sym):
    b = base_symbol(sym)
    with _lock:
        if b not in _models:
            p = os.path.join(_DIR, f"{b}.npz.pkl")          # NumPy export: LightGBM must not load next to torch
            _models[b] = pickle.load(open(p, "rb")) if os.path.exists(p) else None
        return _models[b]


def predict_trees(trees, x):
    """Sum of LightGBM trees evaluated with NumPy (binary objective -> sigmoid). x: (n, n_features) float64."""
    x = np.asarray(x, dtype=np.float64)
    n = len(x)
    raw = np.zeros(n)
    rows = np.arange(n)
    for t in trees:
        node = np.zeros(n, dtype=np.int64)
        active = t["feat"][node] >= 0
        while active.any():
            r, nd = rows[active], node[active]
            v = x[r, t["feat"][nd]]
            miss = t["miss"][nd]
            nan = np.isnan(v)
            v = np.where(nan & (miss != 2), 0.0, v)                     # missing_type None/Zero: NaN counts as 0
            is_missing = (np.isnan(v) & (miss == 2)) | ((miss == 1) & (np.abs(v) <= 1e-35))
            go_left = np.where(is_missing, t["dleft"][nd] == 1, v <= t["thr"][nd])
            node[active] = np.where(go_left, t["left"][nd], t["right"][nd])
            active = t["feat"][node] >= 0
        raw += t["leaf"][node]
    return 1.0 / (1.0 + np.exp(-raw))


def prob(sym, df):
    m = _load(sym)
    if m is None or df is None or len(df) < 300:
        return None
    from Python.agi_brain import build_feature_frame
    f = build_feature_frame(df)
    if f is None or len(f) == 0:
        return None
    x = f.reindex(columns=m["features"]).iloc[[-1]].replace([np.inf, -np.inf], np.nan).astype("float32").to_numpy(np.float64)
    p = float(predict_trees(m["trees"], x)[0])
    return {"p": round(p, 4), "go": bool(p >= m["threshold"]), "threshold": round(float(m["threshold"]), 4)}
