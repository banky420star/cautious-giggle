import os
import time
import numpy as np
import pandas as pd
from loguru import logger

from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.model_registry import ModelRegistry

# Optional MT5
try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None


class HybridBrain:
    """
    PPO-first autonomous brain with safety guardrails:
      - Registry hot-swap (canary/champion)
      - Spread-aware deadzone
      - Cooldown + min-hold to reduce thrash
      - Optional LSTM volatility veto
      - Canary lot reduction
    """

    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.risk_engine = RiskEngine()
        self.executor = MT5Executor(paper_mode=paper_mode)

        logger.info("Initializing Hybrid Brain (PPO primary + guardrails)...")

        # Optional LSTM volatility classifier
        self.lstm = None
        try:
            from Python.agi_brain import SmartAGI
            self.lstm = SmartAGI()
            logger.success("✅ LSTM volatility brain loaded")
        except Exception as e:
            logger.warning(f"LSTM brain unavailable (PPO-only mode): {e}")

        # Registry-driven PPO
        self.registry = ModelRegistry()
        self._active_dir_cached = None
        self.ppo_model = None
        self.vecnorm_path = None

        # Per-symbol decision state
        # { "EURUSD": {"action":"BUY","ts":12345.0,"pos":0.33} }
        self._state = {}

        # Guardrail config (env overrides)
        self.cooldown_sec = int(os.environ.get("AGI_COOLDOWN_SEC", "45"))         # ignore flip attempts within this window
        self.min_hold_sec = int(os.environ.get("AGI_MIN_HOLD_SEC", "120"))        # minimum time to keep a non-HOLD position
        self.base_deadzone = float(os.environ.get("AGI_BASE_DEADZONE", "0.18"))   # fallback if no symbol override
        self.deadzone_aggressive = float(os.environ.get("AGI_DZ_AGGR", "0.12"))
        self.deadzone_moderate = float(os.environ.get("AGI_DZ_MOD", "0.18"))
        self.deadzone_conservative = float(os.environ.get("AGI_DZ_CONS", "0.26"))

        # Symbol deadzone overrides (spread-aware tuning)
        # You can override with env: AGI_DZ_EURUSD=0.18, AGI_DZ_XAUUSD=0.22, etc.
        self.symbol_deadzone = {
            "EURUSD": float(os.environ.get("AGI_DZ_EURUSD", "0.18")),
            "GBPUSD": float(os.environ.get("AGI_DZ_GBPUSD", "0.20")),
            "USDJPY": float(os.environ.get("AGI_DZ_USDJPY", "0.20")),
            "XAUUSD": float(os.environ.get("AGI_DZ_XAUUSD", "0.22")),  # futures proxy is jumpy
            "BTCUSD": float(os.environ.get("AGI_DZ_BTCUSD", "0.28")),
        }

    # ----------------------------
    # Registry-driven PPO loader
    # ----------------------------
    def _load_ppo_from_registry(self):
        active_dir = self.registry.load_active_model(prefer_canary=True)
        if not active_dir:
            self.ppo_model = None
            self.vecnorm_path = None
            return None, None, None

        if active_dir == self._active_dir_cached and self.ppo_model is not None:
            return self.ppo_model, self.vecnorm_path, active_dir

        ppo_path = os.path.join(active_dir, "ppo_trading.zip")
        vec_path = os.path.join(active_dir, "vec_normalize.pkl")

        if not os.path.exists(ppo_path) or not os.path.exists(vec_path):
            logger.warning(f"Active registry dir missing PPO/VecNormalize: {active_dir}")
            self.ppo_model = None
            self.vecnorm_path = None
            self._active_dir_cached = active_dir
            return None, None, active_dir

        try:
            from stable_baselines3 import PPO
            self.ppo_model = PPO.load(ppo_path, device="auto")
            self.vecnorm_path = vec_path
            self._active_dir_cached = active_dir
            logger.success(f"✅ PPO hot-loaded from registry: {active_dir}")
            return self.ppo_model, self.vecnorm_path, active_dir
        except Exception as e:
            logger.error(f"Failed to load PPO from {active_dir}: {e}")
            self.ppo_model = None
            self.vecnorm_path = None
            return None, None, active_dir

    def _is_canary_active(self, active_dir: str | None) -> bool:
        if not active_dir:
            return False
        active = self.registry._read_active()
        return active.get("canary") == active_dir

    # ----------------------------
    # Market price (live prefers MT5)
    # ----------------------------
    def get_current_price(self, symbol: str) -> float:
        if not self.paper_mode and mt5 is not None:
            try:
                if mt5.initialize():
                    mt5.symbol_select(symbol, True)
                    tick = mt5.symbol_info_tick(symbol)
                    if tick is not None and tick.ask and tick.bid:
                        return float((tick.ask + tick.bid) / 2.0)
            except Exception:
                pass

        import yfinance as yf
        from Python.data_feed import SYMBOL_MAP
        try:
            mapped = SYMBOL_MAP.get(symbol, symbol if "USD" not in symbol else symbol + "=X")
            ticker = yf.Ticker(mapped)
            return float(ticker.history(period="1d")["Close"].iloc[-1])
        except Exception:
            return 1.0850

    # ----------------------------
    # PPO decision (obs via TradingEnv + VecNormalize)
    # ----------------------------
    def _ppo_position(self, df: pd.DataFrame, vecnorm_path: str) -> float:
        """
        Returns PPO continuous action in [-1, 1], deterministic.
        Uses the same TradingEnv obs pipeline and VecNormalize stats as training.
        """
        from drl.trading_env import TradingEnv
        from stable_baselines3.common.vec_env import DummyVecEnv, VecNormalize

        if df is None or df.empty or len(df) < 200:
            return 0.0

        def _init():
            return TradingEnv(df, initial_balance=10000.0)

        env = DummyVecEnv([_init])
        env = VecNormalize.load(vecnorm_path, env)
        env.training = False
        env.norm_reward = False

        obs = env.reset()
        
        if np.isnan(obs).any():
            logger.error("🛑 CRITICAL: NaNs detected in observation vector. Rejecting inference.")
            return 0.0
            
        action, _ = self.ppo_model.predict(obs, deterministic=True)

        try:
            a = float(action[0][0])
        except Exception:
            a = float(np.array(action).reshape(-1)[0])

        return float(np.clip(a, -1.0, 1.0))

    # ----------------------------
    # Guardrails
    # ----------------------------
    def _deadzone_for(self, symbol: str, aggression: str) -> float:
        sym = symbol.replace("m", "").replace("M", "")
        dz_sym = self.symbol_deadzone.get(sym, self.base_deadzone)
        dz_agg = {
            "aggressive": self.deadzone_aggressive,
            "moderate": self.deadzone_moderate,
            "conservative": self.deadzone_conservative,
        }.get(aggression, self.deadzone_moderate)

        # take the larger of the two (spread-aware floor)
        return float(max(dz_sym, dz_agg))

    def _now(self) -> float:
        return time.time()

    def _last(self, symbol: str) -> dict:
        return self._state.get(symbol, {"action": "HOLD", "ts": 0.0, "pos": 0.0})

    def _set_last(self, symbol: str, action: str, pos: float):
        self._state[symbol] = {"action": action, "ts": self._now(), "pos": float(pos)}

    def _apply_cooldown_minhold(self, symbol: str, proposed: str, pos: float) -> str:
        last = self._last(symbol)
        last_action = last["action"]
        dt = self._now() - float(last["ts"])

        # If last action was BUY/SELL, enforce min-hold unless proposing HOLD (risk-off) or same action
        if last_action in ("BUY", "SELL") and proposed != last_action:
            if proposed == "HOLD":
                # Always allow going flat (risk-off)
                return proposed

            if dt < self.min_hold_sec:
                logger.info(f"{symbol}: min-hold active ({int(dt)}s<{self.min_hold_sec}s). Keep {last_action}.")
                return last_action

        # Cooldown: block rapid flips between BUY/SELL
        if last_action in ("BUY", "SELL") and proposed in ("BUY", "SELL") and proposed != last_action:
            if dt < self.cooldown_sec:
                logger.info(f"{symbol}: cooldown active ({int(dt)}s<{self.cooldown_sec}s). Keep {last_action}.")
                return last_action

        return proposed

    # ----------------------------
    # Main entry
    # ----------------------------
    async def live_trade(self, symbol: str, direction: str = "AUTO",
                         confidence: float = 0.0, aggression: str = "moderate"):
        try:
            aggression = (aggression or "moderate").lower().strip()

            # hot-load PPO
            model, vec_path, active_dir = self._load_ppo_from_registry()

            from Python.data_feed import fetch_training_data
            df = fetch_training_data(symbol, period="120d")
            if df is None or df.empty:
                return {"status": "error", "error": "No data", "symbol": symbol}

            # optional LSTM vol filter
            volatility_class = None
            vol_conf = None
            if self.lstm is not None:
                try:
                    pred = self.lstm.predict(df, production=True)
                    volatility_class = pred.get("signal")
                    vol_conf = float(pred.get("confidence", 0.0))
                except Exception as e:
                    logger.debug(f"LSTM predict skipped: {e}")

            final_action = direction

            # Decide automatically
            pos = 0.0
            if not final_action or final_action == "AUTO":
                if model is None or vec_path is None:
                    # Without PPO, safest autonomous behavior is HOLD
                    final_action = "HOLD"
                else:
                    pos = self._ppo_position(df, vec_path)
                    dz = self._deadzone_for(symbol, aggression)

                    # Convert position -> action (deadzone)
                    if abs(pos) < dz:
                        final_action = "HOLD"
                    else:
                        final_action = "BUY" if pos > 0 else "SELL"

                    # LSTM veto: block low volatility trades for moderate/conservative
                    if volatility_class == "LOW_VOLATILITY" and aggression in ("moderate", "conservative"):
                        final_action = "HOLD"

                    # set confidence as |pos| (bounded)
                    confidence = float(min(1.0, abs(pos)))

                    # Apply cooldown/min-hold against thrash
                    final_action = self._apply_cooldown_minhold(symbol, final_action, pos)

            # If HOLD, record state and exit
            if final_action == "HOLD":
                self._set_last(symbol, "HOLD", pos)
                return {
                    "status": "skipped",
                    "action": "HOLD",
                    "symbol": symbol,
                    "reason": f"Policy hold (pos={pos:.3f}, vol={volatility_class}, agg={aggression})",
                    "confidence": confidence,
                    "active_model_dir": active_dir,
                }

            # Price
            current_price = self.get_current_price(symbol)

            # Lot sizing (executor does live balance checks, but we compute a base lot here)
            base_balance = 10000.0
            lot = self.risk_engine.lot_size(balance=base_balance, price=current_price)

            # Canary risk reduction
            if self._is_canary_active(active_dir):
                mult = float(os.environ.get("CANARY_LOT_MULT", "0.25"))
                lot = max(0.01, round(lot * mult, 2))
                logger.warning(f"🟡 Canary risk reduction applied. Lot={lot:.2f} (x{mult})")

            # Stops
            sl_tp = self.risk_engine.compute_sl_tp(final_action, current_price)
            sl = sl_tp.get("sl")
            tp = sl_tp.get("tp")

            # Execute
            result = await self.executor.execute_order(symbol, final_action, lot, sl, tp)

            # Record state only if we actually executed or paper-executed
            if result.get("status") in ("paper_executed", "live_executed"):
                self._set_last(symbol, final_action, pos)

            # enrich response
            result.update({
                "action": final_action,
                "symbol": symbol,
                "price": current_price,
                "lot": lot,
                "sl": sl,
                "tp": tp,
                "confidence": confidence,
                "ppo_pos": float(pos),
                "volatility_class": volatility_class,
                "volatility_confidence": vol_conf,
                "active_model_dir": active_dir,
                "paper_mode": self.paper_mode
            })

            # Telegram (optional)
            try:
                from alerts.telegram_alerts import send_trade_alert
                send_trade_alert(final_action, symbol, current_price, confidence, lots=lot, sl_pct=2.0, tp_pct=4.0)
            except Exception as e:
                logger.debug(f"Telegram alert skipped: {e}")

            return result

        except Exception as e:
            logger.error(f"Live trade error: {e}")
            return {"status": "error", "error": str(e), "symbol": symbol}
