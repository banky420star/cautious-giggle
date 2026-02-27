import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import pandas as pd
import numpy as np
import torch
import yaml
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
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
        env = TradingEnv(df, initial_balance=10000.0)
        env = Monitor(env)
        return env
    return _init

def linear_schedule(initial_value: float):
    def func(progress_remaining: float) -> float:
        return progress_remaining * initial_value
    return func

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
    
    # Curriculum: Easy phase mapping (safely handling NaNs)
    vols = pd.Series(df["close"].to_numpy()).pct_change().rolling(20).std()
    easy_threshold = np.nanquantile(vols.to_numpy(), 0.4) if not np.isnan(vols).all() else 1.0
    easy_mask = (vols < easy_threshold).fillna(False).to_numpy()
    
    train_df_easy = df.filter(pl.Series(easy_mask))
    if len(train_df_easy) < 300:
        train_df_easy = df
        
    train_df_full = df
    n_envs = 4
    
    # ── Stage 1: Easy curriculum ──────────────────────────────────
    env = DummyVecEnv([make_env(train_df_easy, i) for i in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    eval_env = DummyVecEnv([make_env(train_df_full, 99)])
    eval_env = VecMonitor(eval_env)
    eval_env = VecNormalize(eval_env, norm_obs=True, norm_reward=False, clip_obs=10.0)
    
    # Critical: Lock eval Normalization to training Normalization stats
    eval_env.obs_rms = env.obs_rms
    eval_env.training = False
    eval_env.norm_reward = False

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
        learning_rate=linear_schedule(3e-4),
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
        device='cuda' if torch.cuda.is_available() else ('mps' if getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available() else 'cpu'),
        verbose=1,
    )
    
    # Eval callback 
    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path="./models/best_eval_models/",
        log_path="/tmp/logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
    )
    
    grad_callback = LSTMGradientDiagnostics()
    
    # ── Train Stage 1 ──
    logger.info("Stage 1: Curriculum training (Low Volatility Segments)")
    model.learn(
        total_timesteps=total_timesteps // 3,
        callback=[eval_callback, grad_callback],
        progress_bar=True
    )
    
    # ── Train Stage 2 ──
    logger.info("Stage 2: Full history training (Joint Learning)")
    base_env2 = DummyVecEnv([make_env(train_df_full, i) for i in range(n_envs)])
    base_env2 = VecMonitor(base_env2)
    
    # Re-use the SAME VecNormalize instance so RMS stats track continuously
    env.venv = base_env2
    model.set_env(env)
    
    # Re-align Eval RMS tracking
    eval_env.obs_rms = env.obs_rms
    
    model.learn(
        total_timesteps=total_timesteps,
        callback=[eval_callback, grad_callback],
        progress_bar=True
    )
    
    # Save into registry as candidate
    logger.info("Building new PPO candidate via ModelRegistry...")
    try:
        from Python.model_registry import ModelRegistry
        registry = ModelRegistry()
        
        import datetime, json
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = os.path.join(registry.candidates_dir, timestamp)
        os.makedirs(candidate_path, exist_ok=True)
        
        # Save SB3 Model and Normalizer to Candidate Path
        model_path = os.path.join(candidate_path, "ppo_trading.zip")
        vec_path = os.path.join(candidate_path, "vec_normalize.pkl")
        
        # NOTE: We currently save the LATEST model at the end of training, not the best eval!
        model.save(model_path)
        env.save(vec_path)
        
        # Stage Metadata
        metrics = {
            "type": "ppo",
            "symbols": symbols,
            "timesteps": total_timesteps,
            "loss": 0.0, 
            "win_rate": 0.0, 
            "date": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(candidate_path, "scorecard.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.success(f"✅ Joint LSTM-PPO Candidate saved to: {candidate_path}")
        
    except Exception as e:
        logger.error(f"Failed to register PPO candidate model: {e}")

if __name__ == "__main__":
    train_drl()
