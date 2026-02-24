"""
PPO Agent — trains and loads a Stable-Baselines3 PPO model
on the TradingEnv (with real or synthetic market data).
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.callbacks import EvalCallback
from drl.trading_env import TradingEnv
from loguru import logger

# Paths
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models")
MODEL_PATH = os.path.join(MODEL_DIR, "ppo_trading")
LOG_DIR = os.path.join(ROOT, "logs", "drl")
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(LOG_DIR, exist_ok=True)

# Detect device
try:
    import torch
    DEVICE = "mps" if torch.backends.mps.is_available() else "cpu"
except Exception:
    DEVICE = "cpu"


def make_env(df=None):
    """Factory that returns a TradingEnv (optionally with real data)."""
    def _init():
        env = TradingEnv(df=df)
        return env
    return _init


def train(steps: int = 300_000, df=None):
    """Train PPO from scratch or continue from existing checkpoint."""
    logger.info(f"PPO training starting | steps={steps:,} | device={DEVICE}")

    env = DummyVecEnv([make_env(df)])

    # Resume from checkpoint if one exists
    if os.path.exists(MODEL_PATH + ".zip"):
        logger.info(f"Resuming from checkpoint: {MODEL_PATH}.zip")
        model = PPO.load(MODEL_PATH, env=env, device=DEVICE)
    else:
        model = PPO(
            "MlpPolicy",
            env,
            verbose=1,
            device=DEVICE,
            learning_rate=3e-4,
            n_steps=2048,
            batch_size=64,
            n_epochs=10,
            gamma=0.99,
            gae_lambda=0.95,
            clip_range=0.2,
            ent_coef=0.01,           # encourage exploration
            tensorboard_log=LOG_DIR,
        )

    # Evaluation callback
    eval_env = DummyVecEnv([make_env(df)])
    eval_cb = EvalCallback(
        eval_env,
        best_model_save_path=os.path.join(MODEL_DIR, "best_ppo"),
        log_path=LOG_DIR,
        eval_freq=max(steps // 20, 1000),
        n_eval_episodes=5,
        deterministic=True,
    )

    model.learn(total_timesteps=steps, callback=eval_cb, progress_bar=True)
    model.save(MODEL_PATH)
    logger.success(f"PPO model saved → {MODEL_PATH}.zip ({os.path.getsize(MODEL_PATH + '.zip') / 1024:.1f} KB)")

    env.close()
    eval_env.close()
    return model


def load_model():
    """Load the trained PPO model for inference."""
    if os.path.exists(MODEL_PATH + ".zip"):
        model = PPO.load(MODEL_PATH, device=DEVICE)
        logger.success(f"PPO model loaded from {MODEL_PATH}.zip")
        return model
    else:
        logger.warning("No trained PPO model found — train first!")
        return None


def predict(obs, model=None):
    """Get a single action from the PPO model."""
    if model is None:
        model = load_model()
    if model is None:
        return 0  # default to HOLD
    action, _ = model.predict(obs, deterministic=True)
    return int(action)


if __name__ == "__main__":
    logger.info("Running standalone PPO training with synthetic data...")
    train(steps=100_000)
