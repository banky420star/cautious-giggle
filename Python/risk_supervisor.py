from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
from dataclasses import dataclass


@dataclass
class RiskDecision:
    allowed: bool
    reason: str


class RiskSupervisor:
    """
    Deterministic live-trade circuit breaker layer. This complements RiskEngine
    with portfolio and market-state checks before a new exposure is sent.
    """

    def __init__(self, cfg: dict | None = None):
        cfg = cfg or {}
        risk_cfg = cfg.get("risk", {}) if isinstance(cfg, dict) else {}
        trading_cfg = cfg.get("trading", {}) if isinstance(cfg, dict) else {}
        supervisor_cfg = risk_cfg.get("supervisor", {}) if isinstance(risk_cfg.get("supervisor", {}), dict) else {}
        self.symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}

        self.enabled = bool(supervisor_cfg.get("enabled", True))
        self.max_daily_loss = float(supervisor_cfg.get("max_daily_loss", risk_cfg.get("max_daily_loss", 100.0)))
        self.max_drawdown_pct = float(
            supervisor_cfg.get("max_drawdown_pct", risk_cfg.get("max_drawdown_pct_guard", trading_cfg.get("max_drawdown", 8.0)))
        )
        self.max_symbol_exposure = float(supervisor_cfg.get("max_symbol_exposure", risk_cfg.get("max_symbol_exposure", 0.35)))
        self.max_total_exposure = float(supervisor_cfg.get("max_total_exposure", risk_cfg.get("max_total_exposure", 1.2)))
        self.max_open_positions = int(supervisor_cfg.get("max_open_positions", risk_cfg.get("max_open_positions", 6)))
        self.max_positions_per_symbol = int(
            supervisor_cfg.get("max_positions_per_symbol", risk_cfg.get("max_positions_per_symbol", 3))
        )
        self.min_trade_interval_sec = int(supervisor_cfg.get("min_trade_interval_sec", 45))
        self.max_spread_bps = float(supervisor_cfg.get("max_spread_bps", trading_cfg.get("max_spread_bps", 25.0)))
        self.max_confidence_gap = float(supervisor_cfg.get("max_confidence_gap", 1.0))

        self.last_trade_at_by_symbol: dict[str, dt.datetime] = {}
        self.halt_until: dt.datetime | None = None
        self._load_state()

    def _state_path(self) -> str:
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(project_root, "logs", "risk_supervisor_state.json")

    def _save_state(self) -> None:
        try:
            state_path = self._state_path()
            os.makedirs(os.path.dirname(state_path), exist_ok=True)
            state = {
                "halt_until": self.halt_until.isoformat() if self.halt_until else None,
                "last_trade_at": {k: v.isoformat() for k, v in self.last_trade_at_by_symbol.items()},
            }
            dir_name = os.path.dirname(state_path)
            with tempfile.NamedTemporaryFile("w", dir=dir_name, delete=False, suffix=".tmp") as tmp:
                json.dump(state, tmp)
                tmp_path = tmp.name
            os.replace(tmp_path, state_path)
        except Exception:
            pass

    def _load_state(self) -> None:
        try:
            state_path = self._state_path()
            if not os.path.exists(state_path):
                return
            with open(state_path, "r", encoding="utf-8") as f:
                state = json.load(f)
            halt_until_str = state.get("halt_until")
            if halt_until_str:
                self.halt_until = dt.datetime.fromisoformat(halt_until_str)
            for k, v in (state.get("last_trade_at") or {}).items():
                self.last_trade_at_by_symbol[str(k)] = dt.datetime.fromisoformat(v)
        except Exception:
            pass

    def _symbol_profile(self, symbol: str) -> dict:
        profile = self.symbol_profiles.get(str(symbol), {})
        return profile if isinstance(profile, dict) else {}

    def _now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def mark_trade(self, symbol: str):
        self.last_trade_at_by_symbol[str(symbol)] = self._now()
        self._save_state()

    def enforce_halt(self, minutes: int, reason: str) -> RiskDecision:
        self.halt_until = self._now() + dt.timedelta(minutes=max(1, int(minutes)))
        self._save_state()
        return RiskDecision(False, reason)

    def allow_trade(
        self,
        *,
        symbol: str,
        target_exposure: float,
        confidence: float,
        spread_bps: float | None,
        snapshot: dict,
        symbol_positions: int,
        total_positions: int,
        current_symbol_exposure: float,
        total_exposure: float,
        drawdown_pct: float,
    ) -> RiskDecision:
        if not self.enabled:
            return RiskDecision(True, "disabled")

        if abs(float(target_exposure)) <= abs(float(current_symbol_exposure)):
            return RiskDecision(True, "risk_reduction")

        symbol_profile = self._symbol_profile(symbol)
        max_positions_per_symbol = int(symbol_profile.get("max_positions_per_symbol", self.max_positions_per_symbol))
        max_symbol_exposure = float(symbol_profile.get("max_symbol_exposure", self.max_symbol_exposure))
        max_spread_bps = float(symbol_profile.get("max_spread_bps", self.max_spread_bps))
        min_trade_interval_sec = int(symbol_profile.get("min_trade_interval_sec", self.min_trade_interval_sec))

        now = self._now()
        if self.halt_until and now < self.halt_until:
            return RiskDecision(False, f"halt_until {self.halt_until.isoformat()}")

        pnl_today = float(snapshot.get("pnl_today", 0.0) or 0.0)
        if pnl_today <= -abs(self.max_daily_loss):
            return self.enforce_halt(24 * 60, f"daily_loss {pnl_today:.2f} <= -{abs(self.max_daily_loss):.2f}")

        if drawdown_pct >= self.max_drawdown_pct:
            return self.enforce_halt(24 * 60, f"drawdown_pct {drawdown_pct:.2f} >= {self.max_drawdown_pct:.2f}")

        if total_positions >= self.max_open_positions and abs(target_exposure) > 0.0:
            return RiskDecision(False, f"max_open_positions {total_positions} >= {self.max_open_positions}")

        if symbol_positions >= max_positions_per_symbol and abs(target_exposure) > abs(current_symbol_exposure):
            return RiskDecision(False, f"max_positions_per_symbol {symbol_positions} >= {max_positions_per_symbol}")

        projected_symbol_exposure = max(abs(current_symbol_exposure), abs(target_exposure))
        if projected_symbol_exposure > max_symbol_exposure:
            return RiskDecision(False, f"symbol_exposure {projected_symbol_exposure:.3f} > {max_symbol_exposure:.3f}")

        projected_total_exposure = total_exposure - abs(current_symbol_exposure) + abs(target_exposure)
        if projected_total_exposure > self.max_total_exposure:
            return RiskDecision(False, f"total_exposure {projected_total_exposure:.3f} > {self.max_total_exposure:.3f}")

        if spread_bps is not None and float(spread_bps) > max_spread_bps:
            return RiskDecision(False, f"spread_bps {float(spread_bps):.2f} > {max_spread_bps:.2f}")

        conf = float(confidence)
        if conf < 0.0 or conf > self.max_confidence_gap:
            return RiskDecision(False, f"confidence {conf:.3f} outside range")

        last_trade_at = self.last_trade_at_by_symbol.get(str(symbol))
        if last_trade_at is not None:
            elapsed = (now - last_trade_at).total_seconds()
            if elapsed < min_trade_interval_sec:
                return RiskDecision(False, f"cooldown {elapsed:.0f}s < {min_trade_interval_sec}s")

        return RiskDecision(True, "ok")
