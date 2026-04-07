import os
from datetime import datetime, timezone

import yaml
from loguru import logger

_CFG_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")


class RiskEngine:
    def __init__(self, cfg: dict | None = None, cfg_path: str | None = None):
        if cfg is None:
            path = cfg_path or _CFG_PATH
            with open(path, "r", encoding="utf-8") as f:
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
            logger.info("RISK_HALT_CLEARED reason=day_roll")
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
        self.save_state()

    def record_pnl(self, pnl):
        self.maybe_roll_day()
        self.realized_pnl_today += float(pnl)
        if self.realized_pnl_today <= -abs(self.max_daily_loss):
            self.halt = True
            logger.warning("RISK_HALT_SET reason=daily_loss pnl={:.2f} limit={:.2f}", self.realized_pnl_today, self.max_daily_loss)

    def record_trade_result(self, symbol, pnl):
        self.maybe_roll_day()
        self.record_pnl(pnl)
        if symbol is None:
            return
        if float(pnl) < 0.0:
            key = str(symbol)
            self.daily_losing_trades_by_symbol[key] = int(self.daily_losing_trades_by_symbol.get(key, 0)) + 1
        self.save_state()

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
            logger.warning("RISK_HALT_SET reason=consecutive_errors count={}", self.error_count)
        self.save_state()

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

    def to_dict(self) -> dict:
        return {
            "realized_pnl_today": self.realized_pnl_today,
            "daily_trades": self.daily_trades,
            "daily_trades_by_symbol": dict(self.daily_trades_by_symbol),
            "daily_losing_trades_by_symbol": dict(self.daily_losing_trades_by_symbol),
            "halt": self.halt,
            "error_halt": self.error_halt,
            "error_count": self.error_count,
            "current_dd": self.current_dd,
            "peak_equity": self.peak_equity,
            "last_reset_day": self.last_reset_day.isoformat(),
        }

    @classmethod
    def from_dict(cls, data: dict, cfg: dict | None = None, cfg_path: str | None = None) -> "RiskEngine":
        engine = cls(cfg=cfg, cfg_path=cfg_path)
        engine.realized_pnl_today = float(data.get("realized_pnl_today", 0.0))
        engine.daily_trades = int(data.get("daily_trades", 0))
        engine.daily_trades_by_symbol = dict(data.get("daily_trades_by_symbol", {}))
        engine.daily_losing_trades_by_symbol = dict(data.get("daily_losing_trades_by_symbol", {}))
        engine.halt = bool(data.get("halt", False))
        engine.error_halt = bool(data.get("error_halt", False))
        engine.error_count = int(data.get("error_count", 0))
        engine.current_dd = float(data.get("current_dd", 0.0))
        engine.peak_equity = data.get("peak_equity")
        if engine.peak_equity is not None:
            engine.peak_equity = float(engine.peak_equity)
        try:
            engine.last_reset_day = datetime.strptime(data["last_reset_day"], "%Y-%m-%d").date()
        except Exception:
            engine.last_reset_day = datetime.now(timezone.utc).date()
        engine.maybe_roll_day()
        return engine

    def save_state(self, path: str | None = None):
        import json
        state_path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "risk_engine_state.json")
        os.makedirs(os.path.dirname(state_path), exist_ok=True)
        tmp_path = state_path + ".tmp"
        data = json.dumps(self.to_dict(), indent=2)
        try:
            with open(tmp_path, "w", encoding="utf-8") as f:
                f.write(data)
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp_path, state_path)
        except OSError:
            # Fallback: on Windows os.replace can fail if another process
            # briefly locks the file.  Write directly instead.
            try:
                with open(state_path, "w", encoding="utf-8") as f:
                    f.write(data)
                    f.flush()
                    os.fsync(f.fileno())
            except Exception as exc:
                logger.error("RISK_STATE_SAVE_FAILED path={} error={}", state_path, exc)

    @classmethod
    def load_state(
        cls,
        path: str | None = None,
        cfg: dict | None = None,
        cfg_path: str | None = None,
    ) -> "RiskEngine | None":
        import json
        state_path = path or os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs", "risk_engine_state.json")
        if not os.path.exists(state_path):
            return None
        try:
            with open(state_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            return cls.from_dict(data, cfg=cfg, cfg_path=cfg_path)
        except Exception:
            return None
