import pandas as pd

from drl.trading_env import TradingEnv


def test_trading_env_exposes_reward_components():
    n = 250
    df = pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(n)],
            "high": [1.1002 + i * 0.0001 for i in range(n)],
            "low": [1.0998 + i * 0.0001 for i in range(n)],
            "close": [1.1 + i * 0.0001 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )

    env = TradingEnv(df, window_size=50)
    obs, _ = env.reset()
    assert obs.shape[0] == 50 * env.n_features + env.portfolio_feature_count
    assert env.n_features >= 15
    assert env.portfolio_feature_count == 3

    obs, reward, terminated, truncated, info = env.step([0.2])
    assert isinstance(reward, float)
    assert "reward_components" in info
    assert "growth" in info["reward_components"]
    assert "loss_streak_penalty" in info["reward_components"]
    assert "memory_expectancy_norm" in info["reward_components"]
    assert info.get("feature_version") == "engineered_v2"


def test_trading_env_decodes_legacy_three_dim_action():
    meta = TradingEnv.decode_action([0.8, 0.2, -0.4])
    assert meta["entry_mode"] == "market"
    assert meta["legacy"] is True
    assert "tp_offset_pct" in meta
    assert "sl_offset_pct" in meta
