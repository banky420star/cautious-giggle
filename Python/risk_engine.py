"""
Risk Engine — Portfolio risk management with safe config defaults.
"""
import os
import sys
import yaml
from datetime import datetime
from loguru import logger


class RiskEngine:
    def __init__(self):
        cfg = {}
        config_path = os.path.join(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
            "config.yaml"
        )
        try:
            with open(config_path, "r") as f:
                cfg = yaml.safe_load(f) or {}
        except FileNotFoundError:
            logger.warning(f"config.yaml not found at {config_path} — using defaults.")
        except Exception as e:
            logger.warning(f"Failed to read config.yaml: {e} — using defaults.")

        risk_cfg = cfg.get("risk", {})
        self.max_daily_loss = float(risk_cfg.get("max_daily_loss", 500))
        self.max_daily_trades = int(risk_cfg.get("max_daily_trades", 20))
        self.max_lots = float(risk_cfg.get("max_lots", 1.0))

        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.halt = False
        self.error_count = 0

        # ── Bootstrap equity from MT5 or AGI_START_EQUITY ─────────────
        initial_equity = self._bootstrap_equity()
        self._peak_equity = initial_equity
        self._current_equity = initial_equity

        logger.info(
            f"RiskEngine initialized: max_loss=${self.max_daily_loss} "
            f"max_trades={self.max_daily_trades} max_lots={self.max_lots} "
            f"initial_equity=${initial_equity:.2f}"
        )

    @staticmethod
    def _bootstrap_equity() -> float:
        """Read initial equity from MT5 account (Windows) or AGI_START_EQUITY env var."""
        # Try MT5 first on Windows
        if sys.platform == "win32":
            try:
                import MetaTrader5 as mt5
                if mt5.initialize():
                    info = mt5.account_info()
                    if info is not None and info.equity > 0:
                        logger.info(f"RiskEngine: initial equity from MT5 = ${info.equity:.2f}")
                        return float(info.equity)
            except Exception as e:
                logger.debug(f"RiskEngine: MT5 equity read failed ({e}), falling back to env var")

        # Fallback: env var
        env_eq = os.environ.get("AGI_START_EQUITY", "0")
        try:
            val = float(env_eq)
            if val > 0:
                logger.info(f"RiskEngine: initial equity from AGI_START_EQUITY = ${val:.2f}")
                return val
        except ValueError:
            pass

        logger.warning("RiskEngine: no initial equity source — defaulting to 0.0")
        return 0.0

    @property
    def current_dd(self) -> float:
        """Current drawdown as a percentage (0-100)."""
        if self._peak_equity <= 0:
            return 0.0
        return max(0.0, (self._peak_equity - self._current_equity) / self._peak_equity * 100.0)

    def update_equity(self, equity: float):
        """Update equity tracking for drawdown calculation."""
        self._current_equity = equity
        if equity > self._peak_equity:
            self._peak_equity = equity

    def reset_daily(self):
        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.error_count = 0
        self.halt = False

    def record_trade(self):
        self.daily_trades += 1

    def record_pnl(self, pnl: float):
        self.realized_pnl_today += pnl
        if self.realized_pnl_today <= -abs(self.max_daily_loss):
            self.halt = True
            logger.error(f"🛑 KILL SWITCH: Daily loss ${self.realized_pnl_today:.2f} exceeded limit ${self.max_daily_loss}")

    def record_error(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.halt = True
            logger.error(f"🛑 KILL SWITCH: {self.error_count} consecutive errors")

    def can_trade(self) -> bool:
        if self.halt:
            return False
        if self.daily_trades >= self.max_daily_trades:
            return False
        return True
