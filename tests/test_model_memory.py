import datetime as dt

import numpy as np
import pandas as pd

from Python.model_memory import ModelMemory, market_context


def test_memory_rewards_right_model_and_penalizes_wrong(tmp_path):
    mem = ModelMemory(str(tmp_path / "m.json"), half_life_bars=2000, prior_bars=50, min_samples=30)
    rng = np.random.default_rng(0)
    price, t0 = 100.0, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    nxt = rng.normal(0, 0.001)
    ctx = {"key": "up|normal|us"}
    for i in range(600):
        good = np.sign(nxt) if rng.random() < 0.65 else -np.sign(nxt)   # right 65% of the time
        targets = {"ppo": float(good), "lstm": float(-good), "dreamer": None, "blend": 0.0}
        mem.observe("BTCUSDm", t0 + dt.timedelta(minutes=5 * i), price, ctx, targets)
        price *= np.exp(nxt)
        nxt = rng.normal(0, 0.001)
    m = mem.multipliers("BTCUSDm", "up|normal|us")
    assert m["ppo"]["mult"] > 1.1 and m["ppo"]["verdict"] == "works"
    assert m["lstm"]["mult"] < 0.9 and m["lstm"]["verdict"] == "doesn't work"
    assert m["dreamer"]["mult"] == 1.0  # never called -> no opinion
    # persisted and reloaded
    again = ModelMemory(str(tmp_path / "m.json"), half_life_bars=2000, prior_bars=50, min_samples=30)
    assert again.multipliers("BTCUSDm", "up|normal|us")["ppo"]["mult"] == m["ppo"]["mult"]


def test_no_evidence_means_no_change(tmp_path):
    mem = ModelMemory(str(tmp_path / "m.json"))
    assert all(v["mult"] == 1.0 for v in mem.multipliers("XAUUSDm", "range|low|asia").values())


def test_market_context_shape():
    idx = pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC")
    close = np.linspace(100, 130, 300)
    df = pd.DataFrame({"open": close, "high": close + 0.1, "low": close - 0.1, "close": close}, index=idx)
    ctx = market_context(df)
    assert ctx["trend"] == "up" and ctx["key"].count("|") == 2
