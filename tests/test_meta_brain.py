import numpy as np
import pandas as pd

from Python import meta_brain as mb


def _m5(n=3000, seed=5):
    r = np.random.default_rng(seed)
    c = 2000 + np.cumsum(r.normal(0, 1.5, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + r.uniform(0, 1.5, n)
    l = np.minimum(o, c) - r.uniform(0, 1.5, n)
    idx = pd.date_range("2026-10-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c}, index=idx)


def test_labels_target_stop_and_cost():
    d = _m5(400)
    y = mb.labels(d, "XAUUSDm", horizon=48)
    sl, tp = mb.STOPS["XAUUSD"]
    ok = y["r_long"].iloc[50:].dropna()                         # after the ATR has warmed up
    assert ok.max() <= tp / sl and ok.min() >= -1.0 - 0.1      # target, stop, minus a small spread
    assert (y["end_long"].dropna().iloc[:-48] > np.arange(len(d) - 48)).all()  # exits after entry
    assert np.isnan(y["r_long"].iloc[-1])                      # no label without any future bar


def test_labels_known_path():
    idx = pd.date_range("2026-10-01", periods=60, freq="5min", tz="UTC")
    c = np.full(60, 100.0)
    d = pd.DataFrame({"open": c, "high": c + 0.5, "low": c - 0.5, "close": c}, index=idx)
    d.iloc[30, d.columns.get_loc("high")] = 200.0              # one huge spike up: long hits target first
    y = mb.labels(d, "BTCUSDm", horizon=20)
    assert y["r_long"].iloc[25] > 1.5 and y["r_short"].iloc[25] < -0.99


def test_features_no_lookahead():
    d = _m5(2500)
    f1 = mb.features(d.iloc[:2000], "XAUUSDm")
    d2 = d.copy()
    d2.iloc[2000:] *= 1.3
    f2 = mb.features(d2.iloc[:2000], "XAUUSDm")
    pd.testing.assert_frame_equal(f1, f2)
    cols = ["council", "bank_gap@M15", "ict_sig", "smc_rules", "cost_atr"]
    assert all(c in f1.columns for c in cols)


def test_bankbot_and_ict_shapes():
    d = _m5(1500)
    b = mb.bankbot(d)
    assert set(b["state"].unique()) <= {-1.0, 0.0, 1.0}
    i = mb.ict(d)
    assert set(np.unique(i["ict_fresh"])) <= {-1.0, 0.0, 1.0}
    assert len(i) == len(d)


def test_sweep_fvg_shapes_and_no_lookahead():
    d = _m5(3000, seed=11)
    s1 = mb.sweep_fvg(d.iloc[:2500])
    d2 = d.copy()
    d2.iloc[2500:] *= 0.7
    s2 = mb.sweep_fvg(d2.iloc[:2500])
    pd.testing.assert_frame_equal(s1, s2)
    assert set(np.unique(s1["sfr_fresh"])) <= {-1.0, 0.0, 1.0}
    st = s1["sfr_stop_atr"].dropna()
    assert (st > 0).all() and (st <= 3.0).all()


def test_live_model_never_imports_lightgbm_and_matches_it():
    """lightgbm + torch in one process segfaults on macOS: the live path must use the numpy trees."""
    import json
    import os
    import subprocess
    import sys
    if not os.path.exists(os.path.join(mb.MODEL_DIR, "long.json")):
        return
    code = ("import sys; from Python.meta_brain import MetaBrain; MetaBrain(); "
            "assert 'lightgbm' not in sys.modules, 'live path imported lightgbm'")
    subprocess.run([sys.executable, "-c", code], check=True, cwd=mb.ROOT, env={**os.environ, "PYTHONPATH": mb.ROOT})
    try:
        import lightgbm as lgb
    except ImportError:
        return
    tm = mb.TreeModel(json.load(open(os.path.join(mb.MODEL_DIR, "long.json"))))
    ref = lgb.Booster(model_file=os.path.join(mb.MODEL_DIR, "long.txt"))
    x = np.random.default_rng(0).normal(0, 1, (50, ref.num_feature()))
    x[::5, 2] = np.nan
    assert np.allclose(tm.predict(x), ref.predict(x), atol=1e-9)
