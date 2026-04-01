import shutil
import uuid
from pathlib import Path

import numpy as np
import pandas as pd

from Python import backtester


class _FakeModel:
    def predict(self, obs, deterministic=True):
        return np.array([[0.1]], dtype=np.float32), None


class _FakeVecEnv:
    def __init__(self):
        self.training = False
        self.norm_reward = False
        self._step = 0
        self._equity = 10000.0
        self._position = 0.0

    def reset(self):
        self._step = 0
        self._equity = 10000.0
        self._position = 0.0
        return np.zeros((1, 8), dtype=np.float32)

    def step(self, action):
        self._step += 1
        self._position = float(action[0][0])
        self._equity += 5.0
        done = self._step >= 12
        info = [
            {
                "equity": self._equity,
                "cost": 0.1,
                "position": self._position,
                "reward_components": {
                    "growth": 0.001,
                    "payoff": 0.001,
                    "sharpe_bonus": 0.001,
                    "drawdown_penalty": 0.0,
                    "cost_penalty": 0.0,
                    "churn_penalty": 0.0,
                },
            }
        ]
        return np.zeros((1, 8), dtype=np.float32), 0.1, done, info


class _FakeArrayVecEnv(_FakeVecEnv):
    def step(self, action):
        obs, reward, done, info = super().step(action)
        payload = info[0]
        payload["equity"] = np.array([payload["equity"]], dtype=np.float64)
        payload["cost"] = np.array([payload["cost"]], dtype=np.float64)
        payload["position"] = np.array([payload["position"]], dtype=np.float64)
        payload["reward_components"] = {
            key: np.array([value], dtype=np.float64)
            for key, value in payload["reward_components"].items()
        }
        return obs, np.array([reward], dtype=np.float32), np.array([done]), info


def _frame(n=500):
    return pd.DataFrame(
        {
            "open": [1.1 + i * 0.0001 for i in range(n)],
            "high": [1.1002 + i * 0.0001 for i in range(n)],
            "low": [1.0998 + i * 0.0001 for i in range(n)],
            "close": [1.1 + i * 0.0001 for i in range(n)],
            "volume": [1000 + i for i in range(n)],
        }
    )


def _run(monkeypatch, env_factory):
    monkeypatch.setattr(backtester, "fetch_training_data", lambda *_a, **_k: _frame())
    monkeypatch.setattr(backtester, "_make_env", lambda *_a, **_k: env_factory())
    monkeypatch.setattr(backtester.VecNormalize, "load", lambda *_a, **_k: env_factory())
    monkeypatch.setattr(backtester.PPO, "load", lambda *_a, **_k: _FakeModel())

    root = Path(".tmp") / f"backtest_smoke_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    try:
        model = root / "ppo_trading.zip"
        vec = root / "vec_normalize.pkl"
        model.write_text("x", encoding="utf-8")
        vec.write_text("x", encoding="utf-8")

        out = backtester.run_ppo_backtest(
            symbol="EURUSDm",
            model_path=str(model),
            vecnorm_path=str(vec),
            period="30d",
            interval="5m",
            max_steps=10,
        )
        assert isinstance(out, dict)
        assert out["symbol"] == "EURUSDm"
        assert out["steps"] > 0
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_backtester_smoke(monkeypatch):
    _run(monkeypatch, _FakeVecEnv)


def test_backtester_handles_array_wrapped_vec_values(monkeypatch):
    _run(monkeypatch, _FakeArrayVecEnv)
