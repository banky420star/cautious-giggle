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

# Local log path for Windows/Mac compatibility
LOG_DIR = os.path.join(os.getcwd(), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "ppo_training.log"), rotation="10 MB", level="INFO")

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
        
        # Ensure TradingEnv gets a clean dataframe with proper index
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
            if "time" in pdf.columns:
                pdf["time"] = pd.to_datetime(pdf["time"])
                pdf = pdf.sort_values("time").set_index("time")
            env = TradingEnv(pdf, initial_balance=10000.0)
        else:
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
        
    # ---- sanitize pandas frame for polars (belt and suspenders) ----
    if isinstance(df_pd, pd.Series):
        df_pd = df_pd.to_frame()

    # Flatten MultiIndex columns (common with yfinance)
    if isinstance(df_pd.columns, pd.MultiIndex):
        df_pd.columns = [
            "_".join([str(x) for x in col if x is not None and str(x) != ""])
            for col in df_pd.columns.to_list()
        ]

    # Force string column names
    df_pd.columns = [str(c) for c in df_pd.columns]

    # Drop duplicate columns (Polars hates them)
    if df_pd.columns.duplicated().any():
        df_pd = df_pd.loc[:, ~df_pd.columns.duplicated(keep="last")]

    # Drop duplicate index values + sort (Polars also hates non-unique index)
    df_pd = df_pd.loc[~df_pd.index.duplicated(keep="last")].sort_index()
    # Resetting index to drop the non-unique timestamp column from polars conversion
    df_pd = df_pd.reset_index(drop=True)
    
    # ── NaN Defense ──
    if df_pd.isna().any().any():
        logger.warning("⚠️ NaNs detected in historical data. Cleaning via ffill/bfill.")
        df_pd = df_pd.ffill().bfill()
        assert not df_pd.isna().any().any(), "❌ CRITICAL: Failed to clean all NaNs in training data"
    # ---------------------------------------------------------------
    
    df = pl.from_pandas(df_pd)
    
    # We remove the disjoint "easy mask" logic because it shreds time-series continuity
    # and causes exactly 12-step truncation episodes! We train on the full continuous series.
    n_envs = 4
    
    # ── Stage 1: Continuous Full Training ──
    env = DummyVecEnv([make_env(df, i) for i in range(n_envs)])
    env = VecMonitor(env)
    env = VecNormalize(env, norm_obs=True, norm_reward=True, clip_obs=10.0)
    
    # Eval must use the exact same logic but without reward normalization
    eval_env = DummyVecEnv([make_env(df, 99)])
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
        learning_rate=linear_schedule(1e-4), # Lowered 3x for stability
        n_steps=4096,                        # Increased steps per rollout
        batch_size=512,                      # Larger batch for smoother gradients
        n_epochs=10,
        gamma=0.995,                         # Longer horizon discounting
        gae_lambda=0.95,
        clip_range=0.2,
        ent_coef=0.005,
        vf_coef=0.5,
        max_grad_norm=0.5,
        target_kl=0.01,                      # ADDED target_kl to prevent policy collapse
        use_sde=True,
        sde_sample_freq=4,
        tensorboard_log=os.path.join(LOG_DIR, "drl_joint"),
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
        log_path=LOG_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        vec_env=env,
        vec_save_path=best_vec_path
    )
    
    grad_callback = LSTMGradientDiagnostics()
    
    # ── Train ──
    logger.info("Starting Stable Training Protocol (Single Stage)")
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
