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


def _feed(mem, moves, targets_fn, cost):
    price, t0 = 100.0, dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    for i, mv in enumerate(moves):
        mem.observe("BTCUSDm", t0 + dt.timedelta(minutes=5 * i), price, {"key": "c"}, targets_fn(i), cost_bps=cost)
        price *= np.exp(mv / 1e4)


def test_right_direction_but_smaller_than_spread_scores_negative(tmp_path):
    # always right about direction, but every move (0.5 bp) is below the 1 bp spread
    mem = ModelMemory(str(tmp_path / "m.json"), min_samples=30, horizon_bars=1)
    moves = [0.5] * 400
    _feed(mem, moves, lambda i: {"ppo": 1.0, "blend": 1.0}, cost=1.0)
    a = mem.assess("BTCUSDm", "ppo", "*")
    assert a["edge_bps"] < 0 and a["hit_rate"] == 0.0
    assert mem.assess("BTCUSDm", "trade", "*")["edge_bps"] < 0


def test_scored_over_the_horizon_not_the_next_bar(tmp_path):
    # moves alternate +3/-1: the next bar is a coin flip for a long, the next 2 bars are always +2
    moves = [3.0, -1.0] * 300
    mem2 = ModelMemory(str(tmp_path / "a.json"), min_samples=30, horizon_bars=2)
    _feed(mem2, moves, lambda i: {"ppo": 1.0}, cost=0.0)
    assert abs(mem2.assess("BTCUSDm", "ppo", "*")["edge_bps"] - 2.0) < 0.2


def test_old_gross_records_are_not_mixed_with_net_ones(tmp_path):
    import json
    p = tmp_path / "m.json"
    p.write_text(json.dumps({"stats": {"BTCUSDm|ppo|*": {"w": 500, "m": 5.0, "s": 30.0}}, "pending": {}}))
    assert ModelMemory(str(p)).assess("BTCUSDm", "ppo", "*")["n_all"] == 0


def test_gap_longer_than_horizon_is_dropped_not_scored(tmp_path):
    mem = ModelMemory(str(tmp_path / "m.json"), min_samples=1, horizon_bars=2)
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    mem.observe("BTCUSDm", t0, 100.0, {"key": "c"}, {"ppo": 1.0}, cost_bps=0.0)
    # outage: next bar seen 45 min later -> the 10-min call can't be judged on a 45-min move
    rows = mem.observe("BTCUSDm", t0 + dt.timedelta(minutes=45), 110.0, {"key": "c"}, {"ppo": 1.0}, cost_bps=0.0)
    assert rows == [] and mem.assess("BTCUSDm", "ppo", "*")["n_all"] == 0


def test_unknown_cost_is_never_scored_as_free(tmp_path):
    mem = ModelMemory(str(tmp_path / "m.json"), min_samples=1, horizon_bars=1)
    t0 = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    mem.observe("BTCUSDm", t0, 100.0, {"key": "c"}, {"ppo": 1.0}, cost_bps=None)
    rows = mem.observe("BTCUSDm", t0 + dt.timedelta(minutes=5), 101.0, {"key": "c"}, {"ppo": 1.0}, cost_bps=1.0)
    assert rows == []
