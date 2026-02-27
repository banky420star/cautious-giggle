import sys, os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import polars as pl
import pandas as pd
import numpy as np
import torch
import yaml
import shutil
from loguru import logger
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize, VecMonitor
from stable_baselines3.common.callbacks import EvalCallback
from stable_baselines3.common.utils import set_random_seed
from drl.trading_env import TradingEnv
from Python.data_feed import fetch_training_data, get_combined_training_df
from drl.lstm_feature_extractor import LSTMFeatureExtractor
from analysis.gradient_flow_analyzer import LSTMGradientDiagnostics

# Log to /tmp to avoid sandbox permission issues
os.makedirs("/tmp/logs", exist_ok=True)
logger.add("/tmp/logs/ppo_training.log", rotation="10 MB", level="INFO")

class EvalCallbackSaveVec(EvalCallback):
    """
    Extends EvalCallback:
    - when a new best model is found, also saves VecNormalize stats.
    """
    def __init__(self, *args, vec_env=None, vec_save_path=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.vec_env = vec_env
        self.vec_save_path = vec_save_path

    def _on_step(self) -> bool:
        # Default to -np.inf to avoid type comparisons failing if None
        old_best = self.best_mean_reward if self.best_mean_reward is not None else -np.inf
        cont = super()._on_step()

        # If best improved, save VecNormalize simultaneously!
        if self.best_mean_reward is not None and self.best_mean_reward > old_best:
            if self.vec_env is not None and self.vec_save_path:
                os.makedirs(os.path.dirname(self.vec_save_path), exist_ok=True)
                self.vec_env.save(self.vec_save_path)
                logger.success(f"✅ Saved VecNormalize with new best model → {self.vec_save_path}")

        return cont

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

    df_pd = get_combined_training_df(symbols, period="60d")
    if df_pd.empty:
        logger.error("No valid training data found.")
        return
        
    df = pl.from_pandas(df_pd)
    
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
    
    # Eval callback setup
    best_dir = os.path.join("models", "best_eval_models")
    os.makedirs(best_dir, exist_ok=True)
    best_vec_path = os.path.join(best_dir, "vec_normalize.pkl")
    
    eval_callback = EvalCallbackSaveVec(
        eval_env=eval_env,
        best_model_save_path=best_dir,
        log_path="/tmp/logs/",
        eval_freq=10_000,
        deterministic=True,
        render=False,
        vec_env=env,
        vec_save_path=best_vec_path
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
    
    # Save into registry as candidate using EXACTLY the best evaluation model
    logger.info("Building new PPO candidate via ModelRegistry using best_model.zip...")
    try:
        from Python.model_registry import ModelRegistry
        registry = ModelRegistry()
        
        import datetime, json
        src_model = os.path.join(best_dir, "best_model.zip")
        src_vec   = os.path.join(best_dir, "vec_normalize.pkl")
        
        if not os.path.exists(src_model) or not os.path.exists(src_vec):
            logger.error("Could not find best_model.zip or vec_normalize.pkl. Did training actually step?")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        candidate_path = os.path.join(registry.candidates_dir, timestamp)
        os.makedirs(candidate_path, exist_ok=True)
        
        # Copy the cleanly evaluated BEST models
        shutil.copy2(src_model, os.path.join(candidate_path, "ppo_trading.zip"))
        shutil.copy2(src_vec, os.path.join(candidate_path, "vec_normalize.pkl"))
        
        # Stage Metadata
        metrics = {
            "type": "ppo",
            "symbols": symbols,
            "timesteps": total_timesteps,
            "source": "EvalCallback best_model.zip + matching VecNormalize",
            "loss": 0.0, 
            "win_rate": 0.0, 
            "date": datetime.datetime.now().isoformat()
        }
        
        with open(os.path.join(candidate_path, "scorecard.json"), "w") as f:
            json.dump(metrics, f, indent=4)
            
        logger.success(f"✅ Optimal Joint LSTM-PPO Candidate staged to: {candidate_path}")
        
    except Exception as e:
        logger.error(f"Failed to register PPO candidate model: {e}")

if __name__ == "__main__":
    train_drl()
