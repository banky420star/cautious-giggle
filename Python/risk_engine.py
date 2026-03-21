import os
from datetime import datetime, timezone

import yaml

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


class RiskEngine:
    def __init__(self):
        with open(_CFG_PATH, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}

        risk_cfg = cfg.get("risk", {})
        trading_cfg = cfg.get("trading", {})

        self.max_daily_loss = float(risk_cfg.get("max_daily_loss", 1000))
        self.max_daily_trades = int(risk_cfg.get("max_daily_trades", 50))
        self.max_daily_trades_per_symbol = int(risk_cfg.get("max_daily_trades_per_symbol", 50))
        self.max_daily_losing_trades_per_symbol = int(risk_cfg.get("max_daily_losing_trades_per_symbol", 10))
        self.max_lots = float(risk_cfg.get("max_lots", 1.0))

        # Default symbol profile used when a symbol-specific profile does not exist.
        self.default_symbol_profile = {
            "entry_deviation": int(trading_cfg.get("entry_deviation", 20)),
            "sl_points": int(trading_cfg.get("sl_points", 250)),
            "tp_points": int(trading_cfg.get("tp_points", 450)),
        }

        self.symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}

        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.daily_trades_by_symbol = {}
        self.daily_losing_trades_by_symbol = {}
        self.halt = False
        self.error_halt = False  # True when halt was triggered by consecutive order errors (requires restart)
        self.error_count = 0
        self.current_dd = 0.0
        self.peak_equity = None
        self.last_reset_day = datetime.now(timezone.utc).date()

    def reset_daily(self):
        self.realized_pnl_today = 0.0
        self.daily_trades = 0
        self.daily_trades_by_symbol = {}
        self.daily_losing_trades_by_symbol = {}
        self.error_count = 0
        # Only auto-clear P&L-triggered halts on day roll; error halts require restart.
        if not self.error_halt:
            self.halt = False
        self.last_reset_day = datetime.now(timezone.utc).date()

    def maybe_roll_day(self):
        today = datetime.now(timezone.utc).date()
        if today != self.last_reset_day:
            self.reset_daily()

    def record_trade(self, symbol=None):
        self.maybe_roll_day()
        self.daily_trades += 1
        if symbol:
            key = str(symbol)
            self.daily_trades_by_symbol[key] = int(self.daily_trades_by_symbol.get(key, 0)) + 1

    def record_pnl(self, pnl):
        self.maybe_roll_day()
        self.realized_pnl_today += float(pnl)
        if self.realized_pnl_today <= -abs(self.max_daily_loss):
            self.halt = True

    def record_trade_result(self, symbol, pnl):
        self.maybe_roll_day()
        self.record_pnl(pnl)
        if symbol is None:
            return
        if float(pnl) < 0.0:
            key = str(symbol)
            self.daily_losing_trades_by_symbol[key] = int(self.daily_losing_trades_by_symbol.get(key, 0)) + 1

    def update_equity(self, equity: float):
        eq = float(equity)
        if self.peak_equity is None:
            self.peak_equity = eq
            self.current_dd = 0.0
            return
        self.peak_equity = max(self.peak_equity, eq)
        if self.peak_equity > 0:
            self.current_dd = (self.peak_equity - eq) / self.peak_equity * 100.0

    def record_error(self):
        self.error_count += 1
        if self.error_count >= 3:
            self.halt = True
            self.error_halt = True  # Requires manual restart to clear

    def can_trade(self, symbol=None):
        self.maybe_roll_day()
        if self.halt:
            return False
        if self.daily_trades >= self.max_daily_trades:
            return False
        if symbol:
            key = str(symbol)
            if int(self.daily_trades_by_symbol.get(key, 0)) >= self.max_daily_trades_per_symbol:
                return False
            if int(self.daily_losing_trades_by_symbol.get(key, 0)) >= self.max_daily_losing_trades_per_symbol:
                return False
        return True

    def get_symbol_profile(self, symbol: str) -> dict:
        prof = self.default_symbol_profile.copy()
        sym_prof = self.symbol_profiles.get(symbol, {})
        if isinstance(sym_prof, dict):
            prof.update(sym_prof)
        return prof
