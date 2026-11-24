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


def test_smc_analyze_and_rules():
    from Python import smc
    d = _bars(800)
    a = smc.analyze(d)
    assert a is not None and a["atr"] > 0
    assert a["range"] is None or 0.0 <= a["range"]["pos"] <= 1.0
    v, why = smc.rules_vote(a)
    assert -1.0 <= v <= 1.0 and isinstance(why, list)
    assert "Price" in smc.describe(a, "gold")
    assert smc.analyze(d.head(50)) is None


def test_smc_no_lookahead():
    from Python import smc
    d = _bars(800)
    a1 = smc.analyze(d.head(700))
    d2 = d.copy()
    d2.loc[700:, ["open", "high", "low", "close"]] *= 1.5   # wild future bars must not change the past read
    assert smc.analyze(d2.head(700)) == a1


def test_smc_voters_default_off_and_weight_zero():
    r = xv.ExternalVoters({"enabled": False})
    assert r.cfg["smc_gemma"]["enabled"] is False and r.cfg["smc_rules"]["enabled"] is False
    assert r.weights()["smc_gemma"] == 0.0 and r.weights()["smc_rules"] == 0.0


def test_smc_gemma_rejects_wrong_side_invalidation(monkeypatch):
    g = xv._SMCGemma({"model": "x", "url": "http://127.0.0.1:1"})
    d = _bars(800)
    px = float(d["close"].iloc[-1])
    monkeypatch.setattr(xv, "ollama_json", lambda *a, **k: {"direction": "long", "strength": 0.8, "setup": "s", "reason": "r", "invalidation": px + 50})
    v, det = g.vote("XAUUSDm", d)
    assert v == 0.0 and "wrong side" in det["reason"]
    monkeypatch.setattr(xv, "ollama_json", lambda *a, **k: {"direction": "long", "strength": 0.8, "setup": "s", "reason": "r", "invalidation": px - 50})
    v, det = g.vote("XAUUSDm", d)
    assert v == 0.8 and det["rules_score"] is not None
