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
        self.max_daily_trades = 15
        
        # Kill Switch Mechanics
        self.daily_max_loss_dollars = 150.0  # HARD STOP if lost more than $150 Today
        self.max_consecutive_losses = 4      # HARD STOP after 4 losses in a row
        self.max_concurrent_positions = 2    # Never hold more than 2 trades at once
        
        self.start_of_day_balance = 0.0
        self.consecutive_loss_count = 0
        self.realized_pnl_today = 0.0
        self.last_trade_date = None

        logger.success(
            f"Risk Engine KILL SWITCH active — Max Daily Loss: ${self.daily_max_loss_dollars} | "
            f"Max {self.max_concurrent_positions} concurrent | max DD {self.max_dd}%"
        )

    def record_closed_trade(self, pnl: float):
        """Call this when a trade closes (from MT5 deals poller)."""
        today = date.today()
        if self.last_trade_date != today:
            # reset day
            self.daily_trades = 0
            self.start_of_day_balance = self.start_of_day_balance or 0.0
            self.consecutive_loss_count = 0
            self.realized_pnl_today = 0.0
            self.last_trade_date = today

        self.realized_pnl_today += float(pnl)

        if pnl < 0:
            self.consecutive_loss_count += 1
        else:
            self.consecutive_loss_count = 0

    # ── Core gate ────────────────────────────────────────────────────
    def can_trade(self, balance: float, signal: str, price: float,
                  confidence: float = 1.0, current_open_positions: int = 0,
                  realized_pnl: float = 0.0) -> bool:
        """Return True if the trade passes all risk checks."""
        # Date tracker for daily resets
        today = date.today()
        if self.last_trade_date != today:
            self.daily_trades = 0
            self.start_of_day_balance = balance
            self.consecutive_loss_count = 0
            self.realized_pnl_today = 0.0
            self.last_trade_date = today

        # 1. Never trade HOLD/LOW VOLATILITY signals
        if signal in ["HOLD", "LOW_VOLATILITY"]:
            return False

        # 2. Daily Loss Kill Switch (HARD STOP via Native MT5 Ledger)
        # Instead of guessing PnL from memory, we now enforce limits strictly against the broker's realized PnL.
        if realized_pnl <= -self.daily_max_loss_dollars:
            logger.error(f"💀 RISK KILL SWITCH TRIGGERED: Daily loss ${abs(realized_pnl):.2f} exceeds ${self.daily_max_loss_dollars}. Trading halted until tomorrow.")
            return False

        # 3. Consecutive Loss Cooldown
        if self.consecutive_loss_count >= self.max_consecutive_losses:
            logger.error(f"💀 RISK KILL SWITCH TRIGGERED: {self.max_consecutive_losses} consecutive losses. Trading halted for safety.")
            return False

        # 4. Max Concurrent Positions
        if current_open_positions >= self.max_concurrent_positions:
            logger.warning(f"Risk: Max concurrent positions ({self.max_concurrent_positions}) reached. Blocked.")
            return False

        # 5. Drawdown check
        if self.peak_balance == 0.0 or balance > self.peak_balance:
            self.peak_balance = balance
        self.current_dd = ((self.peak_balance - balance) / self.peak_balance) * 100
        if self.current_dd >= self.max_dd:
            logger.error(f"💀 RISK KILL SWITCH TRIGGERED: Drawdown {self.current_dd:.1f}% >= limit {self.max_dd}%.")
            return False

        # 6. All checks passed
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
