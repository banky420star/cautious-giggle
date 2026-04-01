import json
import os
import time
from datetime import datetime
from loguru import logger

try:
    import pandas as pd
except Exception:
    pd = None


class PatternRecognitionSystem:
    """Lightweight pattern recognition for market regimes based on price/volume history."""

    def __init__(self, log_dir: str = None):
        self.log_dir = log_dir or os.path.join(os.getcwd(), "logs")
        os.makedirs(self.log_dir, exist_ok=True)
        self.events_path = os.path.join(self.log_dir, "pattern_events.jsonl")
        self.library_path = os.path.join(self.log_dir, "pattern_library.json")
        self._library = {}
        self._load_library()

    def _load_library(self):
        if os.path.exists(self.library_path):
            try:
                with open(self.library_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    self._library = data
            except Exception:
                self._library = {}

    def _save_library(self):
        try:
            with open(self.library_path, "w", encoding="utf-8") as f:
                json.dump(self._library, f, indent=2)
        except Exception:
            pass

    def log_event(self, event: dict):
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(event) + "\n")
        except Exception:
            pass

    def _detect_regime(self, df) -> str:
        if df is None or df.empty:
            return "neutral_low_vol"
        try:
            s = df["close"].astype(float)
            if len(s) < 2:
                return "neutral_low_vol"
            first = float(s.iloc[0])
            last = float(s.iloc[-1])
            price_change = (last - first) / (max(abs(first), 1e-8))
            # compute simple volatility proxy from percent changes
            returns = s.pct_change().fillna(0.0)
            vol = float(returns.std()) if len(returns) > 1 else 0.0
            if vol > 0.01 and price_change > 0.02:
                return "bull_high_vol"
            if vol > 0.01 and price_change < -0.02:
                return "bear_high_vol"
            return "neutral_low_vol"
        except Exception:
            return "neutral_low_vol"

    def detect_and_log(self, symbol: str, df) -> list:
        """Detect current regime and log a discovery event if new."""
        if df is None or df.empty:
            return []
        regime = self._detect_regime(df)
        pattern_name = f"{symbol}_regime_{regime}"
        # naive update to library
        now = datetime.utcnow().isoformat() + "Z"
        if pattern_name not in self._library:
            self._library[pattern_name] = {"symbol": symbol, "regime": regime, "discovered_at": now, "count": 1}
            self._save_library()
            event = {
                "ts": now,
                "event": "pattern_discovery",
                "payload": {
                    "symbol": symbol,
                    "pattern": pattern_name,
                    "regime": regime,
                },
            }
            self.log_event(event)
            logger.info(f"Discovered pattern: {pattern_name} for {symbol} (regime={regime})")
            return [event]
        else:
            # update count
            if isinstance(self._library.get(pattern_name), dict):
                self._library[pattern_name]["count"] = self._library[pattern_name].get("count", 0) + 1
                self._library[pattern_name]["discovered_at"] = now
                self._save_library()
            return []

def get_pattern_library() -> dict:
    """Expose the current in-memory pattern library for UI consumption."""
    try:
        prs = PatternRecognitionSystem()
        return dict(prs._library)
    except Exception:
        return {}
