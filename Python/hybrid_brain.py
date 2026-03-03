import numpy as np
import pandas as pd

from Python.agi_brain import SmartAGI


class HybridBrain:
    def __init__(self, risk, executor):
        self.risk = risk
        self.executor = executor
        self.agi = SmartAGI()

    def _signal_to_exposure(self, signal: str, confidence: float) -> float:
        # Volatility-aware exposure scaling (conservative by default)
        base = {
            "LOW_VOLATILITY": 0.20,
            "MED_VOLATILITY": 0.50,
            "HIGH_VOLATILITY": 0.85,
        }.get(signal, 0.0)

        # Confidence can scale up/down but keep hard cap
        scaled = float(np.clip(base * (0.5 + confidence), 0.0, 1.0))
        return scaled

    def predict(self, symbol: str, df: pd.DataFrame) -> dict:
        if df is None or df.empty or len(df) < 60:
            return {"action": "HOLD", "confidence": 0.0, "signal": "LOW_VOLATILITY", "symbol": symbol}

        if "symbol" not in df.columns:
            df = df.copy()
            df["symbol"] = symbol

        pred = self.agi.predict(df, production=True)
        exposure = self._signal_to_exposure(pred["signal"], pred["confidence"])

        # This stack is long-only until PPO routing is wired here.
        action = "BUY" if exposure >= 0.25 else "HOLD"
        return {
            "action": action,
            "signal": pred["signal"],
            "confidence": float(pred["confidence"]),
            "exposure": float(exposure),
            "symbol": symbol,
        }

    def live_trade(self, symbol, exposure, max_lots):
        if not self.risk.can_trade():
            return False
        self.executor.reconcile_exposure(symbol, exposure, max_lots)
        return True

    def decide_and_trade(self, symbol: str, df: pd.DataFrame):
        decision = self.predict(symbol, df)
        traded = False
        if decision["action"] != "HOLD":
            traded = self.live_trade(symbol, decision["exposure"], self.risk.max_lots)
        decision["traded"] = traded
        return decision
