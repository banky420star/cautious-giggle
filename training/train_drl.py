import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import pandas as pd
import numpy as np
import torch
import yaml
from loguru import logger
from torch.optim.lr_scheduler import ReduceLROnPlateau
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold
from stable_baselines3.common.utils import set_random_seed
from drl.trading_env import TradingEnv
from Python.data_feed import fetch_training_data
from drl.lstm_feature_extractor import LSTMFeatureExtractor
from analysis.gradient_flow_analyzer import LSTMGradientDiagnostics

# Log to /tmp to avoid sandbox permission issues
os.makedirs("/tmp/logs", exist_ok=True)
logger.add("/tmp/logs/ppo_training.log", rotation="10 MB", level="INFO")

def make_env(df, seed: int = 0):
    def _init():
        set_random_seed(seed)
        return TradingEnv(df, initial_balance=10000.0)
    return _init

def train_drl():
    with open("config.yaml") as f:
        cfg = yaml.safe_load(f)
    symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
    total_timesteps = cfg.get("drl", {}).get("total_timesteps", 100_000)
    
    logger.info(f"DRL Training (Joint LSTM-PPO 2026) — symbols: {symbols} | timesteps: {total_timesteps:,}")

    all_dfs = []
    for sym in symbols:
        df_pd = fetch_training_data(sym, period="60d")
        if not df_pd.empty and len(df_pd) > 200:
            all_dfs.append(pl.from_pandas(df_pd))
            
    if not all_dfs:
        logger.error("No valid training data found.")
        return
        
    df = all_dfs[0]
    
    # Curriculum: Easy phase mapping
    vols = pd.Series(df["close"].to_numpy()).pct_change().rolling(20).std().to_numpy()
    easy_threshold = np.nanquantile(vols, 0.4) if not np.isnan(vols).all() else 1.0
    easy_mask = vols < easy_threshold
    easy_mask = np.nan_to_num(easy_mask, nan=True).astype(bool)
    
    train_df_easy = df.filter(pl.Series(easy_mask))
    if len(train_df_easy) < 300:
        train_df_easy = df
        
    train_df_full = df
    n_envs = 4
    
    # ── Stage 1: Easy curriculum ──────────────────────────────────
    env = DummyVecEnv([make_env(train_df_easy, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Hybrid LSTM-PPO policy
    policy_kwargs = dict(
        features_extractor_class=LSTMFeatureExtractor,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=[512, 256],
        activation_fn=torch.nn.ReLU
    )
    
    model = PPO(
        "MlpPolicy",
        env,
        policy_kwargs=policy_kwargs,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=128,
        n_epochs=10,
        gamma=0.99,
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        use_sde=True,
        sde_sample_freq=4,
        tensorboard_log="/tmp/logs/drl_joint/",
        device="mps" if torch.backends.mps.is_available() else "cpu",
        verbose=1,
    )
    
    # === DIFFERENTIAL LEARNING RATES ===
    lstm_params = [p for p in model.policy.features_extractor.lstm_brain.model.parameters() if p.requires_grad]
    ppo_params = [p for name, p in model.policy.named_parameters() 
                  if "lstm_brain" not in name and p.requires_grad]
    
    optimizer = torch.optim.Adam([
        {'params': lstm_params, 'lr': 5e-5},
        {'params': ppo_params,  'lr': 3e-4},
    ])
    model.policy.optimizer = optimizer
    
    # Adaptive LR Scheduler
    scheduler = ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8, 
        threshold=1e-4, min_lr=1e-6
    )
    
    # Eval callback (higher reward threshold to avoid premature stopping)
    eval_env = DummyVecEnv([make_env(train_df_full, 99)])
    eval_env = VecNormalize(eval_env, training=False, norm_obs=True, norm_reward=False)
    
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/",
        log_path="/tmp/logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    
    # Gradient flow analyzer (proper SB3 BaseCallback)
    grad_callback = LSTMGradientDiagnostics()
    
    # ── Train Stage 1 ──
    logger.info("Stage 1: Curriculum training (Low Volatility Segments)")
    model.learn(
        total_timesteps=total_timesteps // 3,
        callback=[eval_callback, grad_callback],
        progress_bar=True
    )
    
    # Update scheduler based on best eval reward tracking from Phase 1
    scheduler.step(eval_callback.best_mean_reward)
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Current LSTM LR post-Stage-1: {current_lr:.2e}")
    
    # ── Train Stage 2 ──
    logger.info("Stage 2: Full history training (Joint Learning)")
    env = DummyVecEnv([make_env(train_df_full, i) for i in range(n_envs)])
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    model.set_env(env)
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, grad_callback],
        progress_bar=True
    )
    
    # Final updates tracking
    scheduler.step(eval_callback.best_mean_reward)
    current_lr = optimizer.param_groups[0]['lr']
    logger.info(f"Current LSTM LR post-Stage-2: {current_lr:.2e}")
    
    # Save
    model.save("models/ppo_trading.zip")
    env.save("models/vec_normalize.pkl")
    logger.success("✅ Joint LSTM-PPO training complete! Model & normalizer saved.")

if __name__ == "__main__":
    train_drl()
