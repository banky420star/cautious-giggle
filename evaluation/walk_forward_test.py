import os
import sys
import numpy as np
import polars as pl
import pandas as pd
from loguru import logger
from datetime import datetime
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from drl.trading_env import TradingEnv
from Python.data_feed import fetch_training_data
from Python.model_registry import ModelRegistry

def run_walk_forward_evaluation(symbol: str = "EURUSD", period: str = "300d"):
    """
    Runs an Out-Of-Sample evaluation on the specified symbol using the active Champion/Canary model.
    """
    logger.info(f"🚀 Evaluation: Walking forward on {symbol} (period={period})")

    # 1. Load active model from registry
    registry = ModelRegistry()
    active_dir = registry.load_active_model(prefer_canary=True)
    
    if not active_dir:
        logger.error("No active model found in registry candidates or champion folders.")
        return None
        
    model_path = os.path.join(active_dir, "ppo_trading.zip")
    vec_path = os.path.join(active_dir, "vec_normalize.pkl")
    
    if not os.path.exists(model_path):
        logger.error(f"Missing model file: {model_path}")
        return None

    # 2. Fetch market data
    df_pd = fetch_training_data(symbol, period=period)
    if df_pd.empty or len(df_pd) < 200:
        logger.error(f"Insufficient data for {symbol} evaluation.")
        return None
        
    df = pl.from_pandas(df_pd)
    logger.info(f"Test Data: {len(df)} candles for {symbol}")

    # 3. Reconstruct environment with normalization
    def _make_env():
        return TradingEnv(df, initial_balance=10000.0)
        
    env = DummyVecEnv([_make_env])
    if os.path.exists(vec_path):
        env = VecNormalize.load(vec_path, env)
        env.training = False
        env.norm_reward = False
        
    # 4. Load PPO
    try:
        model = PPO.load(model_path, env=env, device="auto")
    except Exception as e:
        logger.error(f"Failed to load PPO model from {active_dir}: {e}")
        return None
    
    # 5. Execute Episode
    obs = env.reset()
    done = False
    equities = [10000.0]
    trades = 0
    
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = env.step(action)
        done = dones[0]
        
        # VecEnv info is a list of dicts
        info = infos[0]
        equity = float(info["equity"])
        equities.append(equity)
        
        pos = info.get("position", 0.0)
        if abs(pos) > 0.01:
            trades += 1
    
    # 6. Compute Results
    equities = np.array(equities)
    returns = np.diff(equities) / equities[:-1]
    total_return = (equities[-1] / equities[0]) - 1
    max_dd = 1.0 - (equities / np.maximum.accumulate(equities)).min()
    
    # Volatility and Sharpe
    vol = np.std(returns) + 1e-12
    sharpe = (np.mean(returns) / vol) * np.sqrt(252 * 24) # Approx hourly annualization
    
    result = {
        "symbol": symbol,
        "return_pct": total_return * 100.0,
        "max_drawdown_pct": max_dd * 100.0,
        "sharpe": sharpe,
        "trades": trades,
        "final_equity": equities[-1],
        "model_dir": active_dir
    }
    
    logger.success(
        f"✅ {symbol} Results: | Return: {result['return_pct']:.1f}% | "
        f"MaxDD: {result['max_drawdown_pct']:.1f}% | Sharpe: {sharpe:.2f} | "
        f"Trades: {trades}"
    )
    
    # Save a small report
    os.makedirs("logs", exist_ok=True)
    report_file = f"logs/eval_{symbol}_{datetime.now().strftime('%Y%m%d_%H%M')}.json"
    import json
    with open(report_file, "w") as f:
        json.dump(result, f, indent=4)
        
    return result

if __name__ == "__main__":
    # Test on primary symbols
    run_walk_forward_evaluation("EURUSD")
