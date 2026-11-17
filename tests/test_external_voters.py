import json
import numpy as np
import pandas as pd

from Python import external_voters as xv
from Python.indicator_council import council_vote


def _bars(n=600, seed=3, step="15min"):
    r = np.random.default_rng(seed)
    c = 2000 + np.cumsum(r.normal(0, 2, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + r.uniform(0, 2, n)
    l = np.minimum(o, c) - r.uniform(0, 2, n)
    t = pd.date_range("2026-11-01", periods=n, freq=step, tz="UTC")
    return pd.DataFrame({"time": t, "open": o, "high": h, "low": l, "close": c, "tick_volume": r.integers(50, 500, n)})


def test_ob_vote_range_and_state():
    d = _bars()
    v = xv.ob_vote(d)
    assert v is not None and -1.0 <= v <= 1.0
    st = xv.ob_state(*(d[k].to_numpy(float) for k in ("open", "close", "high", "low")))
    assert set(st) == {"open", "armed", "closed"}
    assert xv.ob_vote(d.head(50)) is None


def test_catboost_feature_names_match_training():
    meta_files = __import__("glob").glob(
        "/Volumes/AI_DRIVE/Trading_stack.bank/user_data/models/bank-freqai-spot-v1/sub-train-BTC_*/*_metadata.json")
    if not meta_files:
        return
    want = json.load(open(sorted(meta_files)[-1]))["training_features_list"]
    f = xv._CatBoost.features(_bars(200), "BTC/USDT")
    assert set(want) <= set(f.columns)


def test_council_weights():
    base = council_vote({"a": 1.0, "b": -1.0}, {})
    assert abs(base["raw"]) < 1e-9
    w = council_vote({"a": 1.0, "b": -1.0}, {}, weights={"a": 3.0})
    assert w["raw"] > 0.4


def test_disabled_runner_returns_nothing():
    r = xv.ExternalVoters({"enabled": False})
    assert r.votes("XAUUSDm", _bars(step="5min"), _bars()) == {}
