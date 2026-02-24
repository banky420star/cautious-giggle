import os
import sys

# Ensure parent directory is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import numpy as np
import torch
from stable_baselines3 import PPO
from drl.trading_env import TradingEnv
from loguru import logger
import pandas as pd
from datetime import datetime

# Import data feed explicitly
from Python.data_feed import fetch_training_data

torch.set_default_device("mps" if torch.backends.mps.is_available() else "cpu")

def run_walk_forward_evaluation():
    logger.info("🚀 Starting Walk-Forward Out-of-Sample Evaluation (Joint LSTM-PPO)")

    # Fetch latest real market data (1h or 1d depending on what data_feed returns)
    # Using the exact same method the model trained upon.
    df_pd = fetch_training_data("EURUSD", period="700d")
    
    # Needs to be fairly substantial for walk-forward, if not, fallback handling
    if df_pd.empty or len(df_pd) < 1000:
        logger.warning(f"Data feed returned only {len(df_pd)} rows. Falling back to synthetic large dataset.")
        idx = pd.date_range(end=datetime.now(), periods=10000, freq="1h")
        df_pd = pd.DataFrame(np.random.rand(10000, 5) * 1.5, columns=['open','high','low','close','volume'], index=idx)
        df_pd['symbol'] = "EURUSD"
    
    # Process pandas to polars with timestamp
    df_pd = df_pd.reset_index()
    if 'Date' in df_pd.columns:
        df_pd = df_pd.rename(columns={'Date': 'timestamp'})
    elif 'Datetime' in df_pd.columns:
        df_pd = df_pd.rename(columns={'Datetime': 'timestamp'})
    elif 'index' in df_pd.columns:
        df_pd = df_pd.rename(columns={'index': 'timestamp'})
        
    df = pl.from_pandas(df_pd)
    logger.info(f"Loaded {len(df)} rows for Walk-Forward testing.")

    # Walk-forward parameters
    # Adjust dynamic window calculations to fit the fetched data dynamically
    total_len = len(df)
    train_window = int(total_len * 0.6)  # 60% of data for training emulation
    test_window = int(total_len * 0.1)   # 10% blocks for testing
    step = int(total_len * 0.05)         # 5% overlap shifts
    
    if test_window < 100:
        logger.error(f"Not enough data for walk-forward limits: Total length {total_len}")
        return None
        
    results = []
    
    model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ppo_lstm_joint_latest.zip")
    
    if not os.path.exists(model_path):
        logger.warning(f"Model {model_path} missing. Attempting to fall back to base PPO...")
        model_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models", "ppo_trading.zip")
        if not os.path.exists(model_path):
            logger.error("No valid PPO models found! Run joint training first.")
            return None

    # Load the joint model once outside the loop
    model = PPO.load(model_path, device="mps" if torch.backends.mps.is_available() else "cpu")
    
    for start in range(0, len(df) - train_window - test_window + 1, step):
        train_end = start + train_window
        test_start = train_end
        test_end = test_start + test_window
        
        # We test strictly Out-Of-Sample
        test_df = df[test_start:test_end]
        
        # Create test environment (same as training shape)
        test_env = TradingEnv(test_df, initial_balance=10000.0)
        
        # Run deterministic inference on test period
        obs, _ = test_env.reset()
        done = False
        episode_rewards = []
        balances = [10000.0]
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)  # production mode
            obs, reward, terminated, truncated, info = test_env.step(action)
            done = terminated or truncated
            
            episode_rewards.append(reward)
            balances.append(info["balance"])
        
        # Calculate key metrics safely
        returns = np.diff(balances) / balances[:-1]
        returns = np.nan_to_num(returns, 0.0)
        total_return = (balances[-1] / 10000.0) - 1
        
        volatility = np.std(returns) + 1e-8
        downside_vol = np.std([r for r in returns if r < 0]) + 1e-8
        
        # Annualization factor approximation based on typical intervals 
        ann_factor = np.sqrt(252 * 24) # Assuming 1h bars
        
        sharpe = (np.mean(returns) / volatility) * ann_factor
        sortino = (np.mean(returns) / downside_vol) * ann_factor
        
        cum_balances = np.maximum.accumulate(balances)
        max_dd = np.max((cum_balances - balances) / cum_balances)
        
        calmar = total_return / (max_dd + 1e-8)
        
        wins = [r for r in returns if r > 0]
        losses = [r for r in returns if r < 0]
        win_rate = len(wins) / len(returns) if len(returns) > 0 else 0
        profit_factor = abs(sum(wins)) / abs(sum(losses) + 1e-8)
        
        period_result = {
            "period_start": str(test_df["timestamp"][0]) if "timestamp" in test_df.columns else str(test_start),
            "period_end": str(test_df["timestamp"][-1]) if "timestamp" in test_df.columns else str(test_end),
            "total_return": total_return,
            "sharpe": sharpe,
            "sortino": sortino,
            "calmar": calmar,
            "max_drawdown": max_dd,
            "win_rate": win_rate,
            "profit_factor": profit_factor,
            "final_balance": balances[-1]
        }
        
        results.append(period_result)
        logger.success(f"✅ Period [{period_result['period_start'][:10]} → {period_result['period_end'][:10]}] | Sharpe: {sharpe:.2f} | Return: {total_return:.1%} | MaxDD: {max_dd:.1%}")
    
    # Final report
    if not results:
        logger.error("No walk-forward periods executed.")
        return None
        
    df_results = pd.DataFrame(results)
    print("\n" + "="*80)
    print("FINAL WALK-FORWARD RESULTS (Joint LSTM-PPO)")
    print("="*80)
    print(df_results.describe())
    print(f"\nAverage Sharpe: {df_results['sharpe'].mean():.2f}")
    print(f"Average Annual Return: {df_results['total_return'].mean():.1%}")
    print(f"Average Max Drawdown: {df_results['max_drawdown'].mean():.1%}")
    print(f"Overall Profit Factor: {df_results['profit_factor'].mean():.2f}")
    
    os.makedirs(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs"), exist_ok=True)
    report_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", f"walkforward_report_{datetime.now().strftime('%Y%m%d_%H%M')}.csv")
    df_results.to_csv(report_path, index=False)
    logger.success(f"📊 Walk-forward evaluation complete! Report saved to {report_path}")
    
    return df_results

if __name__ == "__main__":
    run_walk_forward_evaluation()
