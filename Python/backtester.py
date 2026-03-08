"""
Backtester using the same TradingEnv profile as training.
"""
import json
import os
import sys

import numpy as np
import pandas as pd
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.data_feed import fetch_training_data
from drl.trading_env import TradingEnv

LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "backtester.log"), rotation="10 MB", level="INFO")


def _normalize_interval(interval: str | None) -> str:
    if not interval:
        return "5m"
    m = str(interval).strip().lower()
    if m.startswith("m") and m[1:].isdigit():
        return f"{m[1:]}m"
    if m.startswith("h") and m[1:].isdigit():
        return f"{m[1:]}h"
    return m


def _make_env(df_pd: pd.DataFrame, initial_balance: float = 10000.0, reward_weights: dict | None = None):
    def _init():
        return TradingEnv(df_pd, initial_balance=initial_balance, reward_weights=reward_weights)

    return DummyVecEnv([_init])


def run_ppo_backtest(
    symbol: str,
    model_path: str,
    vecnorm_path: str,
    period: str = "120d",
    interval: str = "5m",
    initial_balance: float = 10000.0,
    max_steps: int | None = None,
    reward_weights: dict | None = None,
) -> dict | None:
    df = fetch_training_data(symbol, period=period, interval=interval)
    if df is None or df.empty or len(df) < 400:
        logger.error(f"Insufficient data for {symbol} (len={0 if df is None else len(df)})")
        return None

    env = _make_env(df, initial_balance=initial_balance, reward_weights=reward_weights)
    if not os.path.exists(vecnorm_path):
        logger.error(f"Missing vecnorm file: {vecnorm_path}")
        return None

    env = VecNormalize.load(vecnorm_path, env)
    env.training = False
    env.norm_reward = False

    if not os.path.exists(model_path):
        logger.error(f"Missing model file: {model_path}")
        return None

    model = PPO.load(model_path, device="cpu")

    obs = env.reset()
    equities, costs, positions, rewards, step_rets = [], [], [], [], []
    reward_component_sums = {
        "growth": 0.0,
        "payoff": 0.0,
        "sharpe_bonus": 0.0,
        "drawdown_penalty": 0.0,
        "cost_penalty": 0.0,
        "churn_penalty": 0.0,
    }

    steps = 0
    prev_eq = None

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, info = env.step(action)

        info0 = info[0] if isinstance(info, (list, tuple)) else info
        eq = float(info0.get("equity", np.nan))
        cost = float(info0.get("cost", 0.0))
        pos = float(info0.get("position", 0.0))

        rc = info0.get("reward_components", {}) if isinstance(info0, dict) else {}
        for k in reward_component_sums:
            reward_component_sums[k] += float(rc.get(k, 0.0))

        equities.append(eq)
        costs.append(cost)
        positions.append(pos)
        rewards.append(float(reward))

        if prev_eq is not None and prev_eq > 0:
            step_rets.append((eq - prev_eq) / prev_eq)
        prev_eq = eq

        steps += 1
        if max_steps and steps >= max_steps:
            break
        if bool(done):
            break

    equity = np.array(equities, dtype=np.float64)
    if len(equity) < 3:
        return None

    rets = np.array(step_rets, dtype=np.float64) if step_rets else np.zeros(1)
    vol = float(np.std(rets) + 1e-12)
    sharpe = float(np.mean(rets) / vol) if vol > 0 else 0.0

    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / (peak + 1e-12)
    max_dd = float(np.max(dd))

    total_return = float((equity[-1] / (equity[0] + 1e-12)) - 1.0)
    avg_cost = float(np.mean(costs)) if costs else 0.0
    pos_arr = np.array(positions, dtype=np.float64)
    turnover = float(np.mean(np.abs(np.diff(pos_arr)))) if len(pos_arr) > 2 else 0.0

    score = (total_return * 100.0) - (max_dd * 100.0 * 1.8) + (sharpe * 6.0) - (turnover * 2.0)

    n = max(1, steps)
    result = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "candles": int(len(df)),
        "total_return": float(total_return),
        "sharpe": float(sharpe),
        "max_drawdown": float(max_dd),
        "avg_cost": float(avg_cost),
        "turnover": float(turnover),
        "steps": int(steps),
        "final_equity": float(equity[-1]),
        "score": float(score),
        "reward_component_avg": {k: float(v / n) for k, v in reward_component_sums.items()},
    }

    logger.info(
        f"BACKTEST {symbol} | tf={interval} ret={total_return:.2%} sharpe={sharpe:.2f} "
        f"maxDD={max_dd:.2%} score={score:.2f} steps={steps}"
    )
    return result


def run_multi(
    symbols: list[str],
    model_dir: str,
    period: str = "120d",
    interval: str = "5m",
    reward_weights: dict | None = None,
) -> dict:
    model_path = os.path.join(model_dir, "ppo_trading.zip")
    vec_path = os.path.join(model_dir, "vec_normalize.pkl")

    per_symbol = []
    tf = _normalize_interval(interval)
    for sym in symbols:
        r = run_ppo_backtest(
            sym,
            model_path,
            vec_path,
            period=period,
            interval=tf,
            reward_weights=reward_weights,
        )
        if r:
            per_symbol.append(r)

    if not per_symbol:
        return {"error": "No valid backtests"}

    scores = [x["score"] for x in per_symbol]
    rets = [x["total_return"] for x in per_symbol]
    dds = [x["max_drawdown"] for x in per_symbol]
    sharpes = [x["sharpe"] for x in per_symbol]

    agg = {
        "symbols": [x["symbol"] for x in per_symbol],
        "period": period,
        "interval": tf,
        "avg_score": float(np.mean(scores)),
        "avg_return": float(np.mean(rets)),
        "worst_drawdown": float(np.max(dds)),
        "avg_sharpe": float(np.mean(sharpes)),
        "per_symbol": per_symbol,
    }
    return agg


if __name__ == "__main__":
    symbols = ["EURUSDm", "GBPUSDm", "XAUUSDm"]
    md = sys.argv[1] if len(sys.argv) > 1 else os.path.join("models", "registry", "champion")
    report = run_multi(symbols, md, period="120d", interval="5m")
    print(json.dumps(report, indent=2))
