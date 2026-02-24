"""
PPO Agent — Trains and loads a Stable-Baselines3 PPO model
on the TradingEnv (with real or synthetic market data) using
VecNormalize for distribution normalization.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
from stable_baselines3.common.callbacks import EvalCallback
from drl.trading_env import TradingEnv
from loguru import logger

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading.zip")
VEC_NORM_PATH = os.path.join(MODEL_DIR, "vec_normalize.pkl")
LOG_DIR = os.path.join(ROOT, "logs", "drl")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

try:
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"

def make_env(df=None):
    def _init():
        return TradingEnv(df=df)
    return _init

def load_model():
    """Load the trained PPO model & VecNormalize for inference."""
    if os.path.exists(MODEL_PATH):
        model = PPO.load(MODEL_PATH, device=DEVICE)
        
        vec_env = None
        if os.path.exists(VEC_NORM_PATH):
            dummy = DummyVecEnv([make_env()])
            vec_env = VecNormalize.load(VEC_NORM_PATH, dummy)
            vec_env.training = False
            vec_env.norm_reward = False
            logger.success("PPO VecNormalize parameters loaded.")
            
        logger.success(f"PPO Base Model loaded from {MODEL_PATH}")
        return model, vec_env
    else:
        logger.warning("No trained PPO model found — run training first!")
        return None, None

def predict(obs, model=None, vec_env=None):
    """Get a continuous action from the PPO model + VecNormalizer."""
    if model is None:
        model, vec_env = load_model()
    if model is None:
        return 0.0  # default HOLD (0 leverage)
        
    if vec_env is not None:
        obs = vec_env.normalize_obs(obs)
        
    action, _ = model.predict(obs, deterministic=True)
    return action[0]
