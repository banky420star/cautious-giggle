import asyncio
import os
import threading
from typing import Optional

import numpy as np

from loguru import logger


class HybridBrain:
    def __init__(self, risk, executor):
        self.risk = risk
        self.risk_engine = risk
        self.executor = executor
        self._autonomy_thread = None

        self.ppo_model = None
        self.vec_norm = None
        self.ppo_enabled = os.environ.get("AGI_PPO_ENABLED", "true").lower() == "true"
        self.ppo_blend = float(os.environ.get("AGI_PPO_BLEND", "0.55"))
        self.ppo_min_abs = float(os.environ.get("AGI_PPO_MIN_ABS", "0.03"))
        self.window_size = int(os.environ.get("AGI_PPO_WINDOW", "100"))

        self._ppo_error_count = 0
        self._vecnorm_disabled = False

        self._load_ppo_from_registry()
        self._start_autonomy_if_enabled()

    def _start_autonomy_if_enabled(self):
        if os.environ.get("AGI_AUTONOMY_ENABLED", "true").lower() != "true":
            return

        if self._autonomy_thread and self._autonomy_thread.is_alive():
            return

        from Python.autonomy_loop import AutonomyLoop

        loop = AutonomyLoop(self)

        def _runner():
            try:
                asyncio.run(loop.start())
            except Exception as exc:
                logger.warning(f"AutonomyLoop thread stopped: {exc}")

        self._autonomy_thread = threading.Thread(target=_runner, name="autonomy-loop", daemon=True)
        self._autonomy_thread.start()
        logger.info("AutonomyLoop background thread started")

    def _candidate_model_paths(self):
        base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        out = []

        try:
            from Python.model_registry import ModelRegistry

            reg = ModelRegistry()
            active = reg._read_active()
            for k in ("canary", "champion"):
                d = active.get(k)
                if d:
                    out.append((os.path.join(d, "ppo_trading.zip"), os.path.join(d, "vec_normalize.pkl"), f"registry:{k}"))
        except Exception:
            pass

        out.append((os.path.join(base, "models", "best_eval_models", "best_model.zip"), os.path.join(base, "models", "best_eval_models", "vec_normalize.pkl"), "best_eval"))
        out.append((os.path.join(base, "models", "ppo_trading.zip"), os.path.join(base, "models", "vec_normalize.pkl"), "models_root"))
        return out

    def _load_ppo_from_registry(self):
        if not self.ppo_enabled:
            return

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize
            from drl.trading_env import TradingEnv

            for model_path, vec_path, source in self._candidate_model_paths():
                if not os.path.exists(model_path):
                    continue

                self.ppo_model = PPO.load(model_path)
                self.vec_norm = None
                self._vecnorm_disabled = False

                if os.path.exists(vec_path):
                    dummy = DummyVecEnv([lambda: TradingEnv()])
                    self.vec_norm = VecNormalize.load(vec_path, dummy)
                    self.vec_norm.training = False
                    self.vec_norm.norm_reward = False

                if not self._validate_loaded_ppo():
                    logger.warning(f"Skipping incompatible PPO artifact from {source}: {model_path}")
                    self.ppo_model = None
                    self.vec_norm = None
                    continue

                logger.success(f"Loaded PPO model from {source}: {model_path}")
                return

            logger.warning("No PPO model found for live inference; using SmartAGI-only exposure")
        except Exception as exc:
            self.ppo_model = None
            self.vec_norm = None
            logger.warning(f"PPO load failed: {exc}")


    def _validate_loaded_ppo(self) -> bool:
        if self.ppo_model is None:
            return False

        try:
            # Validate exact runtime path: optional VecNormalize + PPO.predict.
            obs = np.zeros(self.window_size * 5 + 3, dtype=np.float32)
            if self.vec_norm is not None:
                obs = self.vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
            self.ppo_model.predict(obs, deterministic=True)
            return True
        except Exception as exc:
            logger.warning(f"PPO compatibility check failed: {exc}")
            return False
    def _build_ppo_observation(self, df) -> Optional[np.ndarray]:
        req = ["open", "high", "low", "close", "volume"]
        if df is None or any(c not in df.columns for c in req):
            return None
        if len(df) < self.window_size:
            return None

        window = df[req].tail(self.window_size).copy()
        arr = window.to_numpy(dtype=np.float32)

        last_close = float(arr[-1, 3]) + 1e-12
        arr[:, 0:4] = (arr[:, 0:4] / last_close) - 1.0
        arr[:, 4] = np.log1p(np.maximum(arr[:, 4], 0.0))

        closes = window["close"].to_numpy(dtype=np.float64)
        rets = np.diff(closes) / (closes[:-1] + 1e-12)
        mean_ret = float(np.mean(rets[-50:])) if rets.size else 0.0

        portfolio_state = np.array([1.0, 0.0, mean_ret], dtype=np.float32)
        obs = np.concatenate([arr.flatten().astype(np.float32), portfolio_state]).astype(np.float32)
        return obs

    def _normalize_obs_safe(self, obs: np.ndarray) -> np.ndarray:
        if self.vec_norm is None:
            return obs

        try:
            return self.vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
        except Exception as exc:
            self.vec_norm = None
            if not self._vecnorm_disabled:
                logger.warning(f"VecNormalize disabled due to shape mismatch/incompatibility: {exc}")
                self._vecnorm_disabled = True
            return obs

    def predict_ppo_exposure(self, symbol: str, df) -> Optional[float]:
        if self.ppo_model is None:
            return None

        try:
            obs = self._build_ppo_observation(df)
            if obs is None:
                return None

            obs = self._normalize_obs_safe(obs)
            action, _ = self.ppo_model.predict(obs, deterministic=True)

            if isinstance(action, np.ndarray):
                action_val = float(action[0])
            else:
                action_val = float(action)

            action_val = float(np.clip(action_val, -1.0, 1.0))
            if abs(action_val) < self.ppo_min_abs:
                return 0.0

            self._ppo_error_count = 0
            return action_val
        except Exception as exc:
            self._ppo_error_count += 1
            if self._ppo_error_count <= 3:
                logger.warning(f"PPO inference failed for {symbol}: {exc}")
            elif self._ppo_error_count == 4:
                logger.warning("PPO inference continues failing; suppressing further per-tick warnings")
            elif self._ppo_error_count >= 10:
                logger.warning("Disabling PPO for this runtime due to repeated inference failures; AGI-only fallback active")
                self.ppo_model = None
                self.vec_norm = None
            return None

    def blend_exposure(self, agi_exposure: float, ppo_exposure: Optional[float], confidence: float = 1.0) -> float:
        if ppo_exposure is None:
            return float(agi_exposure)

        conf = float(np.clip(confidence, 0.0, 1.0))
        # Increase PPO influence when AGI confidence is low.
        ppo_w = float(np.clip(self.ppo_blend + (1.0 - conf) * 0.25, 0.0, 0.9))
        agi_w = 1.0 - ppo_w
        mixed = agi_w * float(agi_exposure) + ppo_w * float(ppo_exposure)
        return float(np.clip(mixed, -1.0, 1.0))

    def live_trade(self, symbol, exposure, max_lots):
        if not self.risk.can_trade():
            return
        self.executor.reconcile_exposure(symbol, exposure, max_lots)


