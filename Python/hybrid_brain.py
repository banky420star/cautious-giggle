import os
import numpy as np
import pandas as pd
import MetaTrader5 as mt5
from loguru import logger

from Python.agi_brain import SmartAGI
from Python.model_registry import ModelRegistry
from drl.ppo_agent import load_model, predict
from drl.trading_env import TradingEnv


class HybridBrain:
    def __init__(self, risk, executor):
        self.risk = risk
        self.executor = executor
        self.risk_engine = risk  # compatibility with AutonomyLoop

        self.agi = SmartAGI()
        self.ppo_model = None
        self.ppo_vec_env = None

        self.registry = ModelRegistry()
        self._active_model_dir = None
        self._load_ppo_from_registry(initial=True)

        self.deadzone = float(os.environ.get("AGI_PPO_DEADZONE", "0.08"))

    def _load_ppo_from_registry(self, initial: bool = False):
        active_dir = self.registry.load_active_model(prefer_canary=True)
        if not active_dir:
            self.ppo_model, self.ppo_vec_env = load_model()
            if initial:
                logger.info("Using base PPO model path (registry has no active model).")
            return

        if active_dir == self._active_model_dir:
            return

        model_path = os.path.join(active_dir, "ppo_trading.zip")
        vec_path = os.path.join(active_dir, "vec_normalize.pkl")

        try:
            if os.path.exists(model_path):
                from stable_baselines3 import PPO
                from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

                self.ppo_model = PPO.load(model_path)
                if os.path.exists(vec_path):
                    dummy = DummyVecEnv([lambda: TradingEnv(df=None)])
                    self.ppo_vec_env = VecNormalize.load(vec_path, dummy)
                    self.ppo_vec_env.training = False
                    self.ppo_vec_env.norm_reward = False
                self._active_model_dir = active_dir
                logger.success(f"HybridBrain swapped PPO model from registry: {active_dir}")
            else:
                logger.warning(f"Active registry dir has no ppo_trading.zip: {active_dir}")
                self.ppo_model, self.ppo_vec_env = load_model()
        except Exception as e:
            logger.warning(f"Failed loading registry PPO model: {e}")
            self.ppo_model, self.ppo_vec_env = load_model()

    def _build_df_from_mt5(self, symbol: str, timeframe: int, bars: int = 220) -> pd.DataFrame:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
        if rates is None or len(rates) < 120:
            return pd.DataFrame()

        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df.rename(columns={"tick_volume": "volume"}, inplace=True)
        keep = ["time", "open", "high", "low", "close", "volume"]
        df = df[keep].copy()
        df["symbol"] = symbol
        return df

    def _ppo_exposure(self, df: pd.DataFrame) -> float:
        if self.ppo_model is None:
            self.ppo_model, self.ppo_vec_env = load_model()

        if self.ppo_model is None or len(df) < 110:
            return 0.0

        env = TradingEnv(df=df[["open", "high", "low", "close", "volume"]], initial_balance=10000.0)
        obs, _ = env.reset()
        action = float(predict(obs, model=self.ppo_model, vec_env=self.ppo_vec_env))

        if abs(action) < self.deadzone:
            return 0.0
        return float(np.clip(action, -1.0, 1.0))

    def _dynamic_risk_controls(self, df: pd.DataFrame, signal: str, confidence: float):
        returns = df["close"].pct_change().dropna()
        vol = float(returns.tail(60).std()) if not returns.empty else 0.0

        # Conservative defaults; expand only when confidence and vol regime support it
        sl_points = 180
        tp_points = 360

        if signal == "HIGH_VOLATILITY":
            sl_points, tp_points = 300, 600
        elif signal == "MED_VOLATILITY":
            sl_points, tp_points = 220, 440

        lot_mult = 0.35 + min(max(confidence, 0.0), 1.0) * 0.65
        if vol > 0.002:
            lot_mult *= 0.7  # reduce risk in very noisy periods

        return lot_mult, sl_points, tp_points

    def trade_cycle(self, symbol: str, timeframe: int = mt5.TIMEFRAME_M5, execute: bool = True):
        if not self.risk.can_trade():
            return {"action": "HALT", "reason": "risk_engine"}

        self._load_ppo_from_registry()
        df = self._build_df_from_mt5(symbol, timeframe)
        if df.empty:
            return {"action": "HOLD", "reason": "insufficient_data", "symbol": symbol}

        pred = self.agi.predict(df, production=True)
        signal = pred["signal"]
        confidence = float(pred["confidence"])

        exposure = self._ppo_exposure(df)

        # Volatility gate: low-vol regime forces reduced exposure / hold
        if signal == "LOW_VOLATILITY":
            exposure *= 0.25
        elif signal == "HIGH_VOLATILITY":
            exposure *= 0.75

        lot_mult, sl_points, tp_points = self._dynamic_risk_controls(df, signal, confidence)
        max_lots = max(0.01, float(self.risk.max_lots) * lot_mult)

        force_limits = os.environ.get("AGI_USE_LIMIT_ORDERS", "false").lower() == "true"
        auto_dynamic_entry = os.environ.get("AGI_AUTO_DYNAMIC_ENTRY", "true").lower() == "true"
        limit_offset_points = float(os.environ.get("AGI_LIMIT_OFFSET_POINTS", "30"))

        # Dynamic entry strategy: in high-vol or low-confidence zones, prefer passive entry.
        use_limits = force_limits or (auto_dynamic_entry and (signal == "HIGH_VOLATILITY" or confidence < 0.55))

        if execute:
            self.executor.reconcile_exposure(
                symbol=symbol,
                target_exposure=exposure,
                max_lots=max_lots,
                sl_points=sl_points,
                tp_points=tp_points,
                use_limit_orders=use_limits,
                limit_offset_points=limit_offset_points,
            )

        action = "BUY" if exposure > 0 else ("SELL" if exposure < 0 else "HOLD")
        return {
            "action": action,
            "symbol": symbol,
            "exposure": round(float(exposure), 4),
            "max_lots": round(float(max_lots), 2),
            "signal": signal,
            "confidence": confidence,
            "sl_points": sl_points,
            "tp_points": tp_points,
            "executed": bool(execute),
            "use_limit_orders": bool(use_limits),
            "limit_offset_points": float(limit_offset_points),
        }

    def live_trade(self, symbol, exposure, max_lots):
        if not self.risk.can_trade():
            return
        self.executor.reconcile_exposure(symbol, exposure, max_lots)
