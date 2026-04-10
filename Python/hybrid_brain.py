"""
Hybrid Brain — PPO + LSTM joint inference engine.

Decision flow:
  1. LSTM (SmartAGI) classifies volatility regime → LOW / MED / HIGH
  2. PPO determines position sizing and direction → continuous action in [-1, 1]
  3. Deadzone logic: if LSTM says LOW_VOLATILITY and confidence > threshold → HOLD
  4. Canary scaling: reduce position size when running a canary model
  5. Final signal passed to executor for trade reconciliation
"""
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize


class HybridBrain:
    """
    RL Executor — PPO-first policy with LSTM volatility gating,
    deadzones, and Canary risk scaling.
    """

    def __init__(self, risk, executor, confidence_threshold: float = 0.85):
        self.risk = risk
        self.executor = executor
        self.confidence_threshold = confidence_threshold

        # Canary lot multiplier (reduce risk for unproven models)
        self.canary_lot_mult = float(os.environ.get("CANARY_LOT_MULT", "0.25"))

        # Device
        try:
            if torch.cuda.is_available():
                self.device = "cuda"
            elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
                self.device = "mps"
            else:
                self.device = "cpu"
        except Exception:
            self.device = "cpu"

        # Load models from registry
        self.ppo_model = None
        self.vec_env = None
        self.lstm_brain = None
        self._is_canary = False

        self._load_ppo_from_registry()
        self._load_lstm()

        logger.success(f"HybridBrain initialized on {self.device.upper()} | canary={self._is_canary}")

    def _load_ppo_from_registry(self):
        """Load PPO model + VecNormalize from the model registry (champion or canary)."""
        try:
            from Python.model_registry import ModelRegistry
            registry = ModelRegistry()
            active_dir = registry.load_active_model(prefer_canary=True)

            if active_dir:
                model_path = os.path.join(active_dir, "ppo_trading.zip")
                vec_path = os.path.join(active_dir, "vec_normalize.pkl")

                # Check if this is a canary
                active = registry._read_active()
                self._is_canary = (active.get("canary") is not None and
                                   active_dir == active.get("canary"))

                if os.path.exists(model_path):
                    self.ppo_model = PPO.load(model_path, device=self.device)
                    logger.success(f"PPO loaded from registry: {active_dir}")

                    if os.path.exists(vec_path):
                        # Build a dummy env for VecNormalize
                        from drl.trading_env import TradingEnv
                        dummy = DummyVecEnv([lambda: TradingEnv()])
                        self.vec_env = VecNormalize.load(vec_path, dummy)
                        self.vec_env.training = False
                        self.vec_env.norm_reward = False
                        logger.success("VecNormalize loaded from registry")
                else:
                    logger.warning(f"No ppo_trading.zip in {active_dir}")

        except Exception as e:
            logger.warning(f"Failed to load PPO from registry: {e}")

        # Fallback to base model directory
        if self.ppo_model is None:
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            model_path = os.path.join(base, "models", "ppo_trading.zip")
            vec_path = os.path.join(base, "models", "vec_normalize.pkl")

            if os.path.exists(model_path):
                try:
                    self.ppo_model = PPO.load(model_path, device=self.device)
                    logger.success(f"PPO loaded from fallback: {model_path}")

                    if os.path.exists(vec_path):
                        from drl.trading_env import TradingEnv
                        dummy = DummyVecEnv([lambda: TradingEnv()])
                        self.vec_env = VecNormalize.load(vec_path, dummy)
                        self.vec_env.training = False
                        self.vec_env.norm_reward = False
                except Exception as e:
                    logger.error(f"Failed to load fallback PPO: {e}")
            else:
                logger.warning("No PPO model found anywhere — brain will use LSTM-only mode")

    def _load_lstm(self):
        """Load the LSTM SmartAGI brain for volatility classification."""
        try:
            from Python.agi_brain import SmartAGI
            self.lstm_brain = SmartAGI()
            logger.success("LSTM SmartAGI brain loaded for volatility gating")
        except Exception as e:
            logger.warning(f"Could not load LSTM brain: {e}")
            self.lstm_brain = None

    def decide(self, symbol: str, df: pd.DataFrame) -> dict:
        """
        Full hybrid inference pipeline.

        Args:
            symbol: Trading symbol (e.g. "EURUSD")
            df: DataFrame with columns [open, high, low, close, volume]

        Returns:
            dict with keys: action, exposure, confidence, volatility, reason
        """
        result = {
            "action": "HOLD",
            "exposure": 0.0,
            "confidence": 0.0,
            "volatility": "UNKNOWN",
            "reason": "no_signal",
            "symbol": symbol,
        }

        # ── Step 1: LSTM Volatility Classification ──
        lstm_signal = None
        lstm_confidence = 0.0

        if self.lstm_brain is not None and len(df) >= 60:
            try:
                df_with_sym = df.copy()
                if "symbol" not in df_with_sym.columns:
                    df_with_sym["symbol"] = symbol
                lstm_result = self.lstm_brain.predict(df_with_sym, production=True)
                lstm_signal = lstm_result.get("signal", "LOW_VOLATILITY")
                lstm_confidence = lstm_result.get("confidence", 0.0)
                result["volatility"] = lstm_signal
                result["confidence"] = lstm_confidence
            except Exception as e:
                logger.warning(f"LSTM prediction failed: {e}")

        # ── Step 2: Deadzone Gate ──
        # If LSTM says LOW_VOLATILITY with high confidence, don't trade
        if lstm_signal == "LOW_VOLATILITY" and lstm_confidence > self.confidence_threshold:
            result["action"] = "HOLD"
            result["reason"] = f"deadzone (low_vol conf={lstm_confidence:.2%})"
            logger.debug(f"{symbol}: DEADZONE — low volatility, holding")
            return result

        # ── Step 3: PPO Position Sizing ──
        ppo_action = 0.0
        if self.ppo_model is not None and len(df) >= 100:
            try:
                obs = self._build_observation(df)
                if obs is not None:
                    # Apply VecNormalize if available
                    if self.vec_env is not None:
                        obs = self.vec_env.normalize_obs(obs)

                    action, _ = self.ppo_model.predict(obs, deterministic=True)
                    ppo_action = float(action[0]) if hasattr(action, '__len__') else float(action)
            except Exception as e:
                logger.warning(f"PPO prediction failed: {e}")

        # ── Step 4: Volatility-Scaled Exposure ──
        # Scale PPO action by volatility regime
        vol_scale = 1.0
        if lstm_signal == "MED_VOLATILITY":
            vol_scale = 0.7  # Moderate confidence → moderate position
        elif lstm_signal == "HIGH_VOLATILITY":
            vol_scale = 1.0  # Full confidence → full PPO sizing
        else:
            vol_scale = 0.3  # Unknown or low → conservative

        exposure = ppo_action * vol_scale

        # ── Step 5: Canary Risk Scaling ──
        if self._is_canary:
            exposure *= self.canary_lot_mult
            result["reason"] = f"canary_scaled (×{self.canary_lot_mult})"

        # ── Step 6: Determine Action ──
        if abs(exposure) < 0.05:
            result["action"] = "HOLD"
            result["exposure"] = 0.0
            result["reason"] = "sub_threshold"
        elif exposure > 0:
            result["action"] = "BUY"
            result["exposure"] = round(float(exposure), 4)
            result["reason"] = f"ppo={ppo_action:.3f} vol={lstm_signal} scale={vol_scale}"
        else:
            result["action"] = "SELL"
            result["exposure"] = round(float(exposure), 4)
            result["reason"] = f"ppo={ppo_action:.3f} vol={lstm_signal} scale={vol_scale}"

        logger.info(
            f"[HybridBrain] {symbol}: {result['action']} | "
            f"exposure={result['exposure']:.4f} | vol={lstm_signal} | "
            f"conf={lstm_confidence:.2%} | canary={self._is_canary}"
        )

        return result

    def _build_observation(self, df: pd.DataFrame) -> np.ndarray | None:
        """
        Build the observation vector matching TradingEnv format:
        [window_size * 5 OHLCV features] + [3 portfolio state features]
        """
        try:
            window_size = 100
            cols = ["open", "high", "low", "close", "volume"]

            for c in cols:
                if c not in df.columns:
                    logger.error(f"Missing column '{c}' in data for observation")
                    return None

            data = df[cols].values.astype(np.float32)

            if len(data) < window_size:
                logger.warning(f"Not enough data for observation: {len(data)} < {window_size}")
                return None

            window = data[-window_size:].copy()

            # Normalize OHLC by last close (same as TradingEnv._get_obs)
            last_close = float(window[-1, 3]) + 1e-12
            window[:, 0:4] = (window[:, 0:4] / last_close) - 1.0
            window[:, 4] = np.log1p(np.maximum(window[:, 4], 0.0))

            obs_window = window.flatten()

            # Portfolio state: [equity_ratio, position, avg_return]
            # In live mode we use neutral defaults
            portfolio_state = np.array([1.0, 0.0, 0.0], dtype=np.float32)

            obs = np.concatenate([obs_window, portfolio_state]).astype(np.float32)
            return obs

        except Exception as e:
            logger.error(f"Failed to build observation: {e}")
            return None

    def live_trade(self, symbol: str, df: pd.DataFrame, max_lots: float = None):
        """
        Full live trading loop: decide → execute.
        """
        if not self.risk.can_trade():
            logger.debug(f"{symbol}: Risk engine blocked trading")
            return

        if max_lots is None:
            max_lots = self.risk.max_lots

        decision = self.decide(symbol, df)

        if decision["action"] == "HOLD":
            return decision

        # Execute via MT5 or dry-run executor
        try:
            self.executor.reconcile_exposure(symbol, decision["exposure"], max_lots)
        except Exception as e:
            logger.error(f"Execution error for {symbol}: {e}")
            self.risk.record_error()

        return decision
