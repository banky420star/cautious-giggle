"""
Hybrid Brain — Combines LSTM (PyTorch) and PPO (Stable-Baselines3).
LSTM determines direction (BUY/SELL) via argmax in production.
PPO agent is queried for confirmation and dynamic confidence multiplier.
"""
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

# Add parent to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.agi_brain import SmartAGI
from drl.ppo_agent import load_model, make_env


class HybridBrain:
    def __init__(self):
        logger.info("Initializing Hybrid Brain (LSTM + PPO)...")
        # 1. Load LSTM
        self.lstm = SmartAGI()
        
        # 2. Load PPO & Normalizer
        self.ppo_model, self.vec_env = load_model()
        if self.ppo_model:
            logger.success("Hybrid Brain successfully loaded PPO Agent & Environments.")
        else:
            logger.warning("Hybrid Brain running in LSTM-only mode (PPO missing).")

    def get_ppo_action(self, df: pd.DataFrame):
        """Mock the env to get the exact observation formatting."""
        if self.ppo_model is None:
            return 0  # HOLD

        # The env expects window_size (100)
        if len(df) < 100:
            return 0 
            
        from stable_baselines3.common.vec_env import DummyVecEnv
        dummy = DummyVecEnv([make_env(df)])
        env = dummy.envs[0]
        env.current_step = len(df) - 1
        obs = env._get_obs()
        
        if self.vec_env:
            obs = self.vec_env.normalize_obs(obs)
        
        # Predict continuous action
        action, _ = self.ppo_model.predict(obs, deterministic=True)
        action_val = action[0]
        
        # Discretize continuous leverage (-1.0 to 1.0) into discrete vote
        if action_val > 0.3:
            return 1 # BUY
        elif action_val < -0.3:
            return 2 # SELL
        return 0 # HOLD

    def predict(self, df: pd.DataFrame, production: bool = True) -> dict:
        """
        Combines LSTM and PPO.
        LSTM gives the base signal and confidence probability.
        PPO gives a secondary action. If they agree, confidence is boosted.
        """
        # 1. Get LSTM Base Prediction
        lstm_result = self.lstm.predict(df, production=production)
        base_signal = lstm_result["signal"]
        base_conf = lstm_result["confidence"]
        
        # 2. Get PPO Action (0=HOLD, 1=BUY, 2=SELL)
        ppo_action_int = self.get_ppo_action(df)
        ppo_signal = ["HOLD", "BUY", "SELL"][ppo_action_int]
        
        # 3. Hybrid Voting Logic
        final_signal = base_signal
        final_conf = base_conf

        if base_signal != "HOLD" and ppo_signal == base_signal:
            # Full agreement! Boost confidence
            final_conf = min(base_conf * 1.2, 0.9999)
            logger.success(f"🧠 HYBRID AGREEMENT: Both LSTM & PPO say {base_signal} for {df['symbol'].iloc[0]}. Conf boosted {base_conf:.2f} -> {final_conf:.2f}")
        
        elif base_signal != "HOLD" and ppo_signal == "HOLD":
            # PPO is cautious, reduce confidence
            final_conf = base_conf * 0.8
            logger.warning(f"🧠 HYBRID DISAGREEMENT: LSTM={base_signal} but PPO=HOLD for {df['symbol'].iloc[0]}. Conf penalised {base_conf:.2f} -> {final_conf:.2f}")
            
        elif base_signal != "HOLD" and ppo_signal != "HOLD" and ppo_signal != base_signal:
            # Complete conflict. Veto the trade.
            logger.error(f"🧠 HYBRID VETO: LSTM={base_signal} but PPO={ppo_signal}. Forcing HOLD.")
            final_signal = "HOLD"
            final_conf = 0.0

        return {
            "signal": final_signal,
            "confidence": round(final_conf, 4),
            "symbol": df['symbol'].iloc[0]
        }
