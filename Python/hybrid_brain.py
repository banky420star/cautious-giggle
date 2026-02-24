import torch
import numpy as np
from loguru import logger
from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
import asyncio

class HybridBrain:
    def __init__(self, paper_mode: bool = True):
        self.paper_mode = paper_mode
        self.risk_engine = RiskEngine()
        self.executor = MT5Executor(paper_mode=paper_mode)

        logger.info("Initializing Hybrid Brain (LSTM + PPO)...")

        # Load LSTM Brain
        try:
            from Python.agi_brain import SmartAGI
            self.lstm = SmartAGI()
            # self.lstm.load_model("models/lstm_brain.pth")
            # self.lstm.model.to("mps")
            logger.success("LSTM Brain loaded on MPS")
        except Exception as e:
            logger.warning(f"Failed to load full LSTM: {e}")

        # Try to load PPO (graceful fallback)
        self.ppo_model = None
        try:
            from stable_baselines3 import PPO
            self.ppo_model = PPO.load("models/ppo_lstm_joint_latest", device="mps")
            logger.success("✅ PPO model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load PPO: {e}. Running in LSTM-only mode.")
            self.ppo_model = None

    async def live_trade(self, symbol: str, direction: str, confidence: float = 0.85):
        """Main live trading entry point — called from n8n"""
        try:
            # Get current price (real-time)
            current_price = self.get_current_price(symbol)

            # RiskEngine + position sizing
            lot = self.risk_engine.lot_size(balance=10000, price=current_price)
            sl_tp = self.risk_engine.compute_sl_tp(direction, current_price)

            sl = sl_tp.get("sl")
            tp = sl_tp.get("tp")

            # Execute (paper or live)
            result = await self.executor.execute_order(symbol, direction, lot, sl, tp)
            result["lot"] = lot
            result["sl"] = sl
            result["tp"] = tp

            # Telegram alert
            try:
                from alerts.telegram_alerts import send_trade_alert
                send_trade_alert(direction, symbol, current_price, confidence, lots=lot, sl_pct=2.0, tp_pct=4.0)
            except Exception as alert_error:
                logger.debug(f"Telegram alert skipped: {alert_error}")

            return result

        except Exception as e:
            logger.error(f"Live trade error: {e}")
            return {"status": "error", "error": str(e)}

    def get_current_price(self, symbol: str):
        """Real-time price via yfinance (fallback for Mac)"""
        import yfinance as yf
        try:
            ticker = yf.Ticker(symbol if "USD" not in symbol else symbol + "=X")
            return ticker.history(period="1d")["Close"].iloc[-1]
        except:
            return 1.0850  # safe fallback price
