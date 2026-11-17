import numpy as np
import pandas as pd

from Python.indicator_council import STRATEGIES, context_series, council_vote, votes


def _df(n=800, drift=0.0005, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(drift + rng.normal(0, 0.002, n)))
    idx = pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC")
    return pd.DataFrame({"open": c, "high": c * 1.001, "low": c * 0.999, "close": c}, index=idx)


def test_votes_bounded_and_trend_following_agree_in_uptrend():
    v = votes(_df())
    assert set(STRATEGIES) <= set(v.columns)
    assert v.abs().max().max() <= 1.0 + 1e-9
    last = v.iloc[-50:].mean()
    assert last["ema200"] > 0 and last["ema_cross"] > 0


def test_votes_use_no_future_data():
    df = _df()
    full = votes(df)
    cut = votes(df.iloc[:600])
    pd.testing.assert_frame_equal(full.iloc[:600], cut, check_exact=False, atol=1e-9)


def test_council_ignores_strategies_that_dont_work():
    out = council_vote({"a": 1.0, "b": -1.0}, {"a": {"mult": 1.4, "verdict": "works"},
                                               "b": {"mult": 0.5, "verdict": "doesn't work"}})
    assert out["target"] > 0.9 and out["ignored"] == ["b"]


def test_context_series_buckets():
    c = context_series(_df())
    assert c.iloc[-1].count("|") == 2
