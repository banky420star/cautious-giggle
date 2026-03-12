import asyncio
import json
import os
import threading
from typing import Optional

import numpy as np
import yaml
from loguru import logger

from Python.action_translator import translate_trade_action
from Python.feature_pipeline import ENGINEERED_V2, feature_count_for_version


class HybridBrain:
    def __init__(self, risk, executor):
        self.risk = risk
        self.risk_engine = risk
        self.executor = executor
        self._autonomy_thread = None

        self.ppo_model = None
        self.vec_norm = None
        self.ppo_bundles = []
        self.ppo_metadata = {}
        self.dreamer_enabled = os.environ.get("AGI_DREAMER_ENABLED", "false").lower() == "true"
        self.dreamer_blend = float(os.environ.get("AGI_DREAMER_BLEND", "0.15"))
        self.dreamer_policies = {}
        self._last_action_meta = None

        cfg = self._load_cfg()
        drl_cfg = (cfg.get("drl", {}) or {}) if isinstance(cfg, dict) else {}
        ensemble_cfg = (drl_cfg.get("ensemble", {}) or {}) if isinstance(drl_cfg.get("ensemble", {}), dict) else {}

        cfg_blend = float(drl_cfg.get("ppo_blend", 0.55))
        self.ppo_enabled = os.environ.get("AGI_PPO_ENABLED", "true").lower() == "true"
        self.ppo_blend = float(os.environ.get("AGI_PPO_BLEND", str(cfg_blend)))
        self.ppo_min_abs = float(os.environ.get("AGI_PPO_MIN_ABS", "0.03"))
        self.window_size = int(os.environ.get("AGI_PPO_WINDOW", str(drl_cfg.get("window_size", 100) or 100)))
        self.ppo_ensemble_enabled = os.environ.get("AGI_PPO_ENSEMBLE", str(bool(ensemble_cfg.get("enabled", False)))).lower() == "true"
        self.ppo_ensemble_min_votes = int(os.environ.get("AGI_PPO_ENSEMBLE_MIN_VOTES", str(ensemble_cfg.get("min_votes", 2) or 2)))
        self.ppo_ensemble_threshold = float(os.environ.get("AGI_PPO_ENSEMBLE_THRESHOLD", str(ensemble_cfg.get("agreement_threshold", 0.5) or 0.5)))

        self._ppo_error_count = 0
        self._vecnorm_disabled = False

        self._load_ppo_from_registry()
        self._load_dreamer_policies()
        self._start_autonomy_if_enabled()

    def _load_cfg(self) -> dict:
        cfg_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        if os.path.exists(cfg_path):
            try:
                with open(cfg_path, "r", encoding="utf-8") as f:
                    return yaml.safe_load(f) or {}
            except Exception:
                return {}
        return {}

    @staticmethod
    def _infer_portfolio_feature_count(obs_dim: Optional[int], feature_version: str = ENGINEERED_V2) -> int:
        from drl.trading_env import TradingEnv

        n_features = feature_count_for_version(feature_version)
        return TradingEnv.infer_portfolio_feature_count(obs_dim, n_features=n_features)

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
        seen = set()

        try:
            from Python.model_registry import ModelRegistry

            reg = ModelRegistry()
            active = reg._read_active()
            for role in ("canary", "champion"):
                candidate_dir = active.get(role)
                if candidate_dir and candidate_dir not in seen:
                    seen.add(candidate_dir)
                    out.append((candidate_dir, f"registry:{role}"))
            if self.ppo_ensemble_enabled:
                for item in reg.get_recent_champions():
                    candidate_dir = str((item or {}).get("path") or "")
                    if candidate_dir and candidate_dir not in seen:
                        seen.add(candidate_dir)
                        out.append((candidate_dir, "registry:history"))
        except Exception:
            pass

        fallbacks = [
            (os.path.join(base, "models", "best_eval_models"), "best_eval"),
            (os.path.join(base, "models"), "models_root"),
        ]
        for candidate_dir, source in fallbacks:
            if candidate_dir not in seen:
                out.append((candidate_dir, source))
        return out

    def _load_candidate_metadata(self, candidate_dir: str) -> dict:
        meta_path = os.path.join(candidate_dir, "metadata.json")
        if not os.path.exists(meta_path):
            return {}
        try:
            with open(meta_path, "r", encoding="utf-8") as f:
                payload = json.load(f) or {}
            return payload if isinstance(payload, dict) else {}
        except Exception:
            return {}

    def _build_dummy_env(self, feature_version: str, portfolio_feature_count: int):
        from stable_baselines3.common.vec_env import DummyVecEnv
        from drl.trading_env import TradingEnv

        return DummyVecEnv(
            [lambda fv=feature_version, pfc=portfolio_feature_count: TradingEnv(feature_version=fv, portfolio_feature_count=pfc)]
        )

    def _load_ppo_from_registry(self):
        if not self.ppo_enabled:
            return

        try:
            from stable_baselines3 import PPO
            from stable_baselines3.common.vec_env import VecNormalize

            bundles = []
            for candidate_dir, source in self._candidate_model_paths():
                model_name = "ppo_trading.zip" if source.startswith("registry") else ("best_model.zip" if source == "best_eval" else "ppo_trading.zip")
                vec_name = "vec_normalize.pkl"
                model_path = os.path.join(candidate_dir, model_name)
                vec_path = os.path.join(candidate_dir, vec_name)
                if not os.path.exists(model_path):
                    continue

                meta = self._load_candidate_metadata(candidate_dir)
                feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
                try:
                    model = PPO.load(model_path)
                    vec_norm = None
                    if os.path.exists(vec_path):
                        obs_dim = int(np.prod(model.observation_space.shape))
                        portfolio_feature_count = self._infer_portfolio_feature_count(obs_dim, feature_version=feature_version)
                        dummy = self._build_dummy_env(feature_version, portfolio_feature_count)
                        vec_norm = VecNormalize.load(vec_path, dummy)
                        vec_norm.training = False
                        vec_norm.norm_reward = False
                except Exception as exc:
                    logger.warning(f"Skipping PPO artifact from {source} due to load error: {exc}")
                    continue

                bundle = {
                    "model": model,
                    "vec_norm": vec_norm,
                    "meta": meta,
                    "source": source,
                    "candidate_dir": candidate_dir,
                }
                if not self._validate_loaded_ppo(bundle):
                    logger.warning(f"Skipping incompatible PPO artifact from {source}: {model_path}")
                    continue
                bundles.append(bundle)
                logger.success(f"Loaded PPO model from {source}: {model_path}")
                if not self.ppo_ensemble_enabled:
                    break

            self.ppo_bundles = bundles
            if bundles:
                self.ppo_model = bundles[0]["model"]
                self.vec_norm = bundles[0]["vec_norm"]
                self.ppo_metadata = bundles[0]["meta"]
            else:
                self.ppo_model = None
                self.vec_norm = None
                self.ppo_metadata = {}
                logger.warning("No PPO model found for live inference; using SmartAGI-only exposure")
        except Exception as exc:
            self.ppo_model = None
            self.vec_norm = None
            self.ppo_bundles = []
            self.ppo_metadata = {}
            logger.warning(f"PPO load failed: {exc}")

    def _load_dreamer_policies(self):
        if not self.dreamer_enabled:
            return
        try:
            from Python.dreamer_policy import DreamerPolicy

            cfg = self._load_cfg()
            symbols = list((cfg.get("trading", {}) or {}).get("symbols", []) or [])
            for symbol in symbols:
                policy = DreamerPolicy.load_symbol(symbol)
                if policy is not None:
                    self.dreamer_policies[str(symbol)] = policy
                    logger.success(f"Loaded Dreamer policy for {symbol}")
        except Exception as exc:
            logger.warning(f"Dreamer load failed: {exc}")

    def _validate_loaded_ppo(self, bundle: dict) -> bool:
        model = bundle.get("model")
        vec_norm = bundle.get("vec_norm")
        if model is None:
            return False

        try:
            obs_dim = int(np.prod(model.observation_space.shape))
            obs = np.zeros(obs_dim, dtype=np.float32)
            if vec_norm is not None:
                obs = vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
            model.predict(obs, deterministic=True)
            return True
        except Exception as exc:
            logger.warning(f"PPO compatibility check failed: {exc}")
            return False

    def _expected_obs_dim(self, bundle: dict | None = None) -> Optional[int]:
        target = bundle or (self.ppo_bundles[0] if self.ppo_bundles else None)
        if not target:
            return None
        try:
            return int(np.prod(target["model"].observation_space.shape))
        except Exception:
            return None

    def _build_ppo_observation(self, df, bundle: dict) -> Optional[np.ndarray]:
        req = ["open", "high", "low", "close", "volume"]
        if df is None or any(c not in df.columns for c in req):
            return None

        from drl.trading_env import TradingEnv

        meta = bundle.get("meta", {}) or {}
        feature_version = str(meta.get("feature_set_version", ENGINEERED_V2) or ENGINEERED_V2)
        obs_dim = self._expected_obs_dim(bundle)
        inferred_window = int(meta.get("window_size", self.window_size) or self.window_size)
        n_features = feature_count_for_version(feature_version)
        portfolio_feature_count = self._infer_portfolio_feature_count(obs_dim, feature_version=feature_version)
        if obs_dim is not None and obs_dim > portfolio_feature_count:
            if (obs_dim - portfolio_feature_count) % n_features == 0:
                inferred_window = max(10, int((obs_dim - portfolio_feature_count) / n_features))

        if len(df) < inferred_window:
            return None

        env = TradingEnv(
            df=df.copy(),
            window_size=inferred_window,
            portfolio_feature_count=portfolio_feature_count,
            feature_version=feature_version,
        )
        obs, _ = env.reset()
        obs = np.asarray(obs, dtype=np.float32).reshape(-1)
        if obs_dim is not None and obs.shape[0] != obs_dim:
            return None
        return obs

    def _normalize_obs_safe(self, bundle: dict, obs: np.ndarray) -> np.ndarray:
        vec_norm = bundle.get("vec_norm")
        if vec_norm is None:
            return obs

        try:
            return vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
        except Exception as exc:
            bundle["vec_norm"] = None
            if not self._vecnorm_disabled:
                logger.warning(f"VecNormalize disabled due to shape mismatch/incompatibility: {exc}")
                self._vecnorm_disabled = True
            return obs

    def _predict_bundle_action(self, symbol: str, df, bundle: dict) -> Optional[dict]:
        obs = self._build_ppo_observation(df, bundle)
        if obs is None:
            return None
        obs = self._normalize_obs_safe(bundle, obs)
        action, _ = bundle["model"].predict(obs, deterministic=True)
        from drl.trading_env import TradingEnv

        action_meta = TradingEnv.decode_action(action, max_leverage=1.0)
        action_val = float(np.clip(action_meta["target"], -1.0, 1.0))
        if abs(action_val) < self.ppo_min_abs:
            return None
        return action_meta

    def predict_ppo_action(self, symbol: str, df) -> Optional[dict]:
        if not self.ppo_bundles:
            self._last_action_meta = None
            return None

        try:
            metas = []
            for bundle in self.ppo_bundles:
                action_meta = self._predict_bundle_action(symbol, df, bundle)
                if action_meta is not None:
                    metas.append(action_meta)

            if not metas:
                self._ppo_error_count = 0
                self._last_action_meta = None
                return None

            if self.ppo_ensemble_enabled and len(metas) > 1:
                votes = [float(np.sign(meta.get("target", 0.0))) for meta in metas if abs(float(meta.get("target", 0.0))) >= self.ppo_min_abs]
                if votes:
                    non_zero = [v for v in votes if v != 0.0]
                    agreement = abs(sum(non_zero)) / max(1, len(non_zero)) if non_zero else 0.0
                    same_side_votes = int(abs(sum(non_zero))) if non_zero else 0
                    if same_side_votes < self.ppo_ensemble_min_votes or agreement < self.ppo_ensemble_threshold:
                        self._last_action_meta = None
                        return None
                blended = dict(metas[0])
                blended["target"] = float(np.mean([float(meta.get("target", 0.0)) for meta in metas]))
                blended["size"] = float(np.mean([float(meta.get("size", 0.0)) for meta in metas]))
                self._last_action_meta = blended
            else:
                self._last_action_meta = metas[0]

            self._ppo_error_count = 0
            return self._last_action_meta
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
                self.ppo_bundles = []
            self._last_action_meta = None
            return None

    def predict_ppo_exposure(self, symbol: str, df) -> Optional[float]:
        action_meta = self.predict_ppo_action(symbol, df)
        if action_meta is None:
            return None
        return float(np.clip(action_meta["target"], -1.0, 1.0))

    def predict_dreamer_exposure(self, symbol: str, df) -> Optional[float]:
        if not self.dreamer_enabled:
            return None
        policy = self.dreamer_policies.get(str(symbol))
        if policy is None:
            return None
        try:
            exposure = policy.predict_exposure(df)
            if exposure is None or abs(float(exposure)) < self.ppo_min_abs:
                return None
            return float(np.clip(exposure, -1.0, 1.0))
        except Exception as exc:
            logger.warning(f"Dreamer inference failed for {symbol}: {exc}")
            return None

    def get_last_action_meta(self) -> Optional[dict]:
        return self._last_action_meta

    def blend_exposure(
        self,
        agi_exposure: float,
        ppo_exposure: Optional[float],
        confidence: float = 1.0,
        dreamer_exposure: Optional[float] = None,
    ) -> float:
        if ppo_exposure is None and dreamer_exposure is None:
            return float(agi_exposure)

        conf = float(np.clip(confidence, 0.0, 1.0))
        ppo_w = float(np.clip(self.ppo_blend + (1.0 - conf) * 0.25, 0.0, 0.9))
        dreamer_w = float(np.clip(self.dreamer_blend if dreamer_exposure is not None else 0.0, 0.0, 0.4))
        agi_w = max(0.0, 1.0 - ppo_w - dreamer_w)
        mixed = agi_w * float(agi_exposure)
        if ppo_exposure is not None:
            mixed += ppo_w * float(ppo_exposure)
        if dreamer_exposure is not None:
            mixed += dreamer_w * float(dreamer_exposure)
        return float(np.clip(mixed, -1.0, 1.0))

    def live_trade(self, symbol, exposure, max_lots, action_meta=None):
        if not self.risk.can_trade(symbol):
            return
        meta = action_meta or self.get_last_action_meta()
        tick = self.executor.get_tick(symbol)
        order_meta = translate_trade_action(symbol, meta, exposure, max_lots, tick) if meta else None
        self.executor.reconcile_exposure(symbol, exposure, max_lots)
        return order_meta
