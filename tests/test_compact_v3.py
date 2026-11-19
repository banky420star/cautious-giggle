import numpy as np
import pandas as pd
import pytest

from Python.compact_features import CORE, LAGS, build_compact_matrix, htf_trend, trend_columns
from Python.feature_pipeline import feature_count_for_version


def _bars(n=3000, drift=0.0, seed=1):
    rng = np.random.default_rng(seed)
    c = 100 * np.exp(np.cumsum(rng.normal(drift, 0.001, n)))
    t = pd.date_range("2026-01-05", periods=n, freq="5min", tz="UTC")
    o = np.concatenate([[c[0]], c[:-1]])
    return pd.DataFrame({"time": t, "open": o, "high": np.maximum(o, c) * 1.0005, "low": np.minimum(o, c) * 0.9995,
                         "close": c, "volume": rng.integers(100, 1000, n).astype(float)})


def test_shape_is_about_205_not_16400():
    n = feature_count_for_version("compact_v3")
    assert n == feature_count_for_version("ultimate_150") + len(CORE) * LAGS + 2 + 3
    assert 190 <= n <= 215


def test_trend_uses_completed_bars_only_no_future():
    df = _bars(6000, drift=0.0002)
    full = build_compact_matrix(df)
    c4, c1 = trend_columns(full.shape[1])
    for k in (4900, 5001, 5555, 5999):
        part = build_compact_matrix(df.iloc[: k + 1])
        assert part[-1, c4] == full[k, c4] and part[-1, c1] == full[k, c1]
    assert set(np.unique(full[:, c4])) <= {-1.0, 0.0, 1.0}
    assert (full[-500:, c4] == 1.0).mean() > 0.8                     # steady uptrend reads up


def test_lags_and_filter_columns_have_no_future():
    df = _bars(1500)
    full = build_compact_matrix(df)
    base = feature_count_for_version("ultimate_150")
    k = 1200
    part = build_compact_matrix(df.iloc[: k + 1])
    np.testing.assert_allclose(part[-1, base: base + len(CORE) * LAGS + 2], full[k, base: base + len(CORE) * LAGS + 2],
                               rtol=1e-4, atol=1e-4)


def test_live_filter_reading_lands_on_last_bar_only():
    df = _bars(600)
    a = build_compact_matrix(df, mf_last=0.9)
    b = build_compact_matrix(df)
    base = feature_count_for_version("ultimate_150") + len(CORE) * LAGS
    assert a[-1, base] == pytest.approx(0.9) and b[-1, base] == pytest.approx(0.5)
    np.testing.assert_array_equal(a[:-1], b[:-1])


def test_htf_trend_flat_history_is_no_view():
    s = _bars(300)["close"]
    s.index = _bars(300)["time"]
    assert (htf_trend(s, "4h") == 0).all()                           # not enough 4h bars yet


def test_env_trade_state_and_trend_gate():
    from drl.trading_env import TradingEnv
    df = _bars(6000, drift=-0.0002)                                    # steady downtrend
    env = TradingEnv(df, feature_version="compact_v3", execution_mode="target", trend_gate=True,
                     sl_atr_mult=2.0, tp_atr_mult=3.2)
    obs, _ = env.reset()
    assert env.window_size == 1 and obs.shape == (env.n_features + 5,)
    env.current_step = 5600
    obs, *_ = env.step(np.array([1.0, 1.0, -1, 0, 0, 0], dtype=np.float32))   # full buy against the downtrend
    assert env.position == 0.0                                                 # blocked, like live
    obs, *_ = env.step(np.array([-1.0, 1.0, -1, 0, 0, 0], dtype=np.float32))  # sell with the trend
    if env.position != 0.0:                                                    # (unless stopped out in the bar)
        ts = obs[-5:]
        assert ts[0] == env.position and 0 < ts[2] <= 3 and ts[3] > 0 and ts[4] > 0
    flat = TradingEnv(df, feature_version="compact_v3", execution_mode="target")
    o, _ = flat.reset()
    assert np.all(o[-5:] == 0.0)


def test_live_trade_state_matches_training_layout():
    from drl.trading_env import TradingEnv
    env = TradingEnv(_bars(1200), feature_version="compact_v3", execution_mode="target", sl_atr_mult=2.0)
    env.reset()
    env.current_step = len(env.feature_data)
    env.position = -0.5
    px = float(env.prices[-1])
    unit = 2.0 * float(env.atr[-1])
    env.set_live_trade(entry=px + 0.5 * unit, sl=px + 1.5 * unit, tp=px - 2.7 * unit, bars_in_trade=24)
    ts = env._get_obs()[-5:]
    np.testing.assert_allclose(ts, [-0.5, 0.5, 0.5, 1.5, 2.7], atol=1e-4)
