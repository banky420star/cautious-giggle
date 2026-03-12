import argparse
import json
import os
import sys

import numpy as np
import torch
import yaml
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from drl.dreamer_agent import DreamerV3Agent
from Python.data_feed import fetch_training_data
from Python.feature_pipeline import ENGINEERED_V2, ULTIMATE_150, build_env_feature_matrix


class DreamerTradingEnvironment:
    def __init__(self, features: np.ndarray, returns: np.ndarray, window: int = 64, cost_per_trade: float = 0.0001):
        self.X = features.astype(np.float32)
        self.r = returns.astype(np.float32)
        self.window = int(window)
        self.cost = float(cost_per_trade)
        self.T = len(self.r)
        self.reset()

    def reset(self):
        self.t = self.window
        self.pos = 0
        self.equity = 1.0
        return self._get_obs()

    def _get_obs(self):
        w = self.X[self.t - self.window : self.t]
        obs = np.concatenate([w.reshape(-1), np.array([self.pos], dtype=np.float32)])
        return obs.astype(np.float32)

    def step(self, action_onehot):
        action_idx = int(np.argmax(action_onehot))
        new_pos = 0 if action_idx == 0 else (1 if action_idx == 1 else -1)
        delta = abs(new_pos - self.pos)
        trade_cost = self.cost * delta
        pnl = self.pos * self.r[self.t]
        reward = pnl - trade_cost
        self.equity *= 1.0 + reward
        self.pos = new_pos
        self.t += 1
        done = self.t >= self.T
        obs = np.zeros_like(self._get_obs()) if done else self._get_obs()
        return obs, float(reward), done, {"equity": float(self.equity), "pos": int(self.pos)}


def _load_cfg() -> dict:
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    if not os.path.exists(cfg_path):
        return {}
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _prepare_training_arrays(symbol: str, period: str, interval: str, candles: int, feature_version: str):
    df = fetch_training_data(symbol, period=period, interval=interval, strict=True, bars=candles, min_bars=min(5000, candles))
    features = build_env_feature_matrix(df.reset_index(drop=False), feature_version=feature_version)
    close = df["close"].to_numpy(dtype=np.float32)
    returns = np.zeros_like(close)
    returns[1:] = (close[1:] - close[:-1]) / (np.abs(close[:-1]) + 1e-8)
    return features, returns


def main():
    parser = argparse.ArgumentParser(description="Train experimental DreamerV3 policy on current trading features.")
    parser.add_argument("--symbol", default=os.environ.get("AGI_DREAMER_SYMBOL", "BTCUSDm"))
    parser.add_argument("--steps", type=int, default=int(os.environ.get("AGI_DREAMER_STEPS", "5000")))
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--window", type=int, default=64)
    parser.add_argument("--feature-version", default=os.environ.get("AGI_DREAMER_FEATURE_VERSION", ULTIMATE_150))
    args = parser.parse_args()

    cfg = _load_cfg()
    drl_cfg = cfg.get("drl", {}) if isinstance(cfg, dict) else {}
    period = str(drl_cfg.get("period", "90d"))
    interval = str(drl_cfg.get("interval", "M5"))
    candles = int(drl_cfg.get("candles_per_symbol", 100000))
    feature_version = str(args.feature_version or ULTIMATE_150)

    if feature_version not in {ENGINEERED_V2, ULTIMATE_150}:
        raise ValueError(f"unsupported feature version: {feature_version}")

    features, returns = _prepare_training_arrays(args.symbol, period, interval, candles, feature_version)
    env = DreamerTradingEnvironment(features, returns, window=args.window)
    device = "cuda" if torch.cuda.is_available() else ("mps" if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else "cpu")
    obs_dim = env.reset().shape[0]
    agent = DreamerV3Agent(obs_dim=obs_dim, action_dim=3, device=device)

    logger.info(
        f"Dreamer training start | symbol={args.symbol} | steps={args.steps} | window={args.window} | obs_dim={obs_dim} | device={device} | features={feature_version}"
    )

    obs = env.reset()
    h, z = None, None
    warmup = max(1000, args.window * 50)
    for _ in range(warmup):
        action_onehot = np.zeros(3, dtype=np.float32)
        action_onehot[np.random.randint(0, 3)] = 1.0
        next_obs, reward, done, _ = env.step(action_onehot)
        agent.replay_buffer.add(obs, action_onehot, reward, done)
        obs = env.reset() if done else next_obs

    obs = env.reset()
    h, z = None, None
    for step in range(args.steps):
        action_onehot, (h, z) = agent.act(obs, h, z, deterministic=False)
        next_obs, reward, done, _ = env.step(action_onehot)
        agent.replay_buffer.add(obs, action_onehot, reward, done)
        if step % 4 == 0:
            agent.train_step(batch_size=args.batch_size)
        obs = env.reset() if done else next_obs
        if done:
            h, z = None, None

    out_dir = os.path.join(PROJECT_ROOT, "models", "dreamer")
    os.makedirs(out_dir, exist_ok=True)
    model_path = os.path.join(out_dir, f"dreamer_{args.symbol}.pt")
    meta_path = os.path.join(out_dir, f"dreamer_{args.symbol}.json")
    agent.save(model_path)
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "symbol": args.symbol,
                "feature_version": feature_version,
                "window_size": args.window,
                "obs_dim": obs_dim,
                "steps": args.steps,
                "period": period,
                "interval": interval,
            },
            f,
            indent=2,
        )
    logger.success(f"Dreamer artifact saved: {model_path}")


if __name__ == "__main__":
    main()
