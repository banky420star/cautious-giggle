"""
Risk Engine — enforces position sizing, max drawdown limits,
stop-loss / take-profit, and daily trade caps.
"""
import os
import yaml
from datetime import datetime, date
from loguru import logger


class RiskEngine:
    def __init__(self):
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml"
        )
        with open(config_path) as f:
            self.cfg = yaml.safe_load(f)

        trading = self.cfg.get("trading", {})
        self.risk_pct = trading.get("risk_percent", 1.0)
        self.max_dd = trading.get("max_drawdown", 10.0)
        self.confidence_threshold = trading.get("confidence_threshold", 0.85)

        # Runtime state
        self.peak_balance = 0.0
        self.current_dd = 0.0
        self.daily_trades = 0
        self.max_daily_trades = 20
        self.last_trade_date = None
        self.open_positions: list[dict] = []

        logger.success(
            f"Risk Engine active — {self.risk_pct}% risk/trade | "
            f"max DD {self.max_dd}% | conf threshold {self.confidence_threshold}"
        )

    # ── Core gate ────────────────────────────────────────────────────
    def can_trade(self, balance: float, signal: str, price: float,
                  confidence: float = 1.0) -> bool:
        """Return True if the trade passes all risk checks."""
        # 1. Never trade HOLD signals
        if signal == "HOLD":
            logger.debug("Risk: HOLD signal — skipping")
            return False

        # 2. Confidence filter
        if confidence < self.confidence_threshold:
            logger.info(f"Risk: confidence {confidence:.2%} < threshold {self.confidence_threshold:.2%} — blocked")
            return False

        # 3. Drawdown check
        if self.peak_balance == 0.0:
            self.peak_balance = balance
        if balance > self.peak_balance:
            self.peak_balance = balance
        self.current_dd = ((self.peak_balance - balance) / self.peak_balance) * 100
        if self.current_dd >= self.max_dd:
            logger.warning(f"Risk: drawdown {self.current_dd:.1f}% >= limit {self.max_dd}% — BLOCKED")
            return False

        # 4. Daily trade limit
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.last_trade_date = today
        if self.daily_trades >= self.max_daily_trades:
            logger.warning(f"Risk: daily trade limit ({self.max_daily_trades}) reached — BLOCKED")
            return False

        # 5. All checks passed
        risk_amount = balance * (self.risk_pct / 100)
        self.daily_trades += 1
        logger.info(
            f"Risk OK — ${risk_amount:.2f} at risk | "
            f"DD={self.current_dd:.1f}% | "
            f"trades today={self.daily_trades}/{self.max_daily_trades}"
        )
        return True

    # ── Position sizing ──────────────────────────────────────────────
    def lot_size(self, balance: float, price: float, sl_pips: float = 50.0) -> float:
        """Calculate position size based on fixed-fraction risk."""
        risk_amount = balance * (self.risk_pct / 100)
        pip_value = 10.0  # standard lot pip value for major FX (approximate)
        lots = risk_amount / (sl_pips * pip_value)
        lots = round(max(0.01, min(lots, 10.0)), 2)  # clamp 0.01–10.0
        logger.debug(f"Position size: {lots} lots (risk ${risk_amount:.2f}, SL {sl_pips} pips)")
        return lots

    # ── Stop-loss / Take-profit levels ───────────────────────────────
    @staticmethod
    def compute_sl_tp(signal: str, price: float,
                      sl_pips: float = 50.0, rr_ratio: float = 2.0) -> dict:
        """Return stop-loss and take-profit prices."""
        pip = 0.0001 if price < 50 else 0.01  # forex vs gold/indices heuristic
        if signal == "BUY":
            sl = price - sl_pips * pip
            tp = price + sl_pips * rr_ratio * pip
        elif signal == "SELL":
            sl = price + sl_pips * pip
            tp = price - sl_pips * rr_ratio * pip
        else:
            sl = tp = price

        return {"sl": round(sl, 5), "tp": round(tp, 5), "rr_ratio": rr_ratio}

    # ── Summary ──────────────────────────────────────────────────────
    def status(self) -> dict:
        return {
            "peak_balance": self.peak_balance,
            "current_drawdown_pct": round(self.current_dd, 2),
            "daily_trades": self.daily_trades,
            "max_daily_trades": self.max_daily_trades,
            "risk_per_trade_pct": self.risk_pct,
            "max_drawdown_pct": self.max_dd,
        }
