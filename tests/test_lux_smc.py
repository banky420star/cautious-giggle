import numpy as np
import pandas as pd

from Python.lux_smc import lux_smc


def _walk(n=3000, seed=1):
    r = np.random.default_rng(seed)
    c = 100 + np.cumsum(r.normal(0, 0.3, n))
    o = np.r_[c[0], c[:-1]]
    h = np.maximum(o, c) + r.random(n) * 0.2
    l = np.minimum(o, c) - r.random(n) * 0.2
    return pd.DataFrame({"open": o, "high": h, "low": l, "close": c})


def test_no_lookahead_and_ranges():
    d = _walk()
    f = lux_smc(d)
    g = lux_smc(d.iloc[:2000])
    assert (f.iloc[:2000] - g).abs().max().max() == 0
    assert set(np.unique(f["lux_swing_trend"])) <= {-1, 0, 1}
    assert f["lux_pd"].between(0, 1).all()
    assert (f["lux_int_fresh"] != 0).sum() > 0


def test_bullish_break_sets_trend_and_order_block():
    # down leg to a low, bounce to a swing high, pull back, then break above it
    seg = np.r_[np.linspace(110, 100, 40), np.linspace(100, 105, 20), np.linspace(105, 102, 20), np.linspace(102, 108, 20)]
    c = seg
    o = np.r_[c[0], c[:-1]]
    d = pd.DataFrame({"open": o, "high": np.maximum(o, c) + 0.05, "low": np.minimum(o, c) - 0.05, "close": c})
    f = lux_smc(d, swing=10, internal=5)
    assert f["lux_int_trend"].iloc[-1] == 1
    assert (f["lux_int_ev"] > 0).any()
