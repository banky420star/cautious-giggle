from __future__ import annotations

import datetime as dt
from collections import deque
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

        # --- Losing-streak throttle ---
        self.loss_streak_threshold = int(supervisor_cfg.get("loss_streak_threshold", 5))
        self.loss_streak_cooldown_minutes = int(supervisor_cfg.get("loss_streak_cooldown_minutes", 10))

        # --- Drawdown-based size reduction ---
        default_drawdown_tiers = [
            {"pct": 5.0, "factor": 0.50},
            {"pct": 10.0, "factor": 0.25},
            {"pct": 15.0, "factor": 0.0},
        ]
        self.drawdown_reduction_tiers: list[dict] = sorted(
            supervisor_cfg.get("drawdown_reduction_tiers", default_drawdown_tiers),
            key=lambda t: float(t["pct"]),
            reverse=True,
        )

        # --- Max adjustments per symbol per hour ---
        self.max_adjustments_per_hour = int(supervisor_cfg.get("max_adjustments_per_hour", 10))

        # --- In-memory tracking ---
        self.last_trade_at_by_symbol: dict[str, dt.datetime] = {}
        self.halt_until: dt.datetime | None = None
        self.consecutive_losses: dict[str, int] = {}
        self.symbol_cooldown_until: dict[str, dt.datetime] = {}
        self.adjustment_timestamps: dict[str, deque[dt.datetime]] = {}

    def clear_blocks(self):
        """Clear all halts, cooldowns, and tracking state for a fresh start."""
        self.halt_until = None
        self.consecutive_losses.clear()
        self.symbol_cooldown_until.clear()
        self.adjustment_timestamps.clear()
        self.last_trade_at_by_symbol.clear()

    def _symbol_profile(self, symbol: str) -> dict:
        profile = self.symbol_profiles.get(str(symbol), {})
        return profile if isinstance(profile, dict) else {}

    def _now(self) -> dt.datetime:
        return dt.datetime.now(dt.timezone.utc)

    def mark_trade(self, symbol: str):
        self.last_trade_at_by_symbol[str(symbol)] = self._now()

    def record_trade_result(self, symbol: str, *, is_loss: bool):
        """Call after a trade closes to update the consecutive-loss tracker."""
        sym = str(symbol)
        if is_loss:
            self.consecutive_losses[sym] = self.consecutive_losses.get(sym, 0) + 1
            if self.consecutive_losses[sym] >= self.loss_streak_threshold:
                self.symbol_cooldown_until[sym] = self._now() + dt.timedelta(
                    minutes=self.loss_streak_cooldown_minutes
                )
        else:
            self.consecutive_losses[sym] = 0

    def _record_adjustment(self, symbol: str) -> None:
        sym = str(symbol)
        now = self._now()
        if sym not in self.adjustment_timestamps:
            self.adjustment_timestamps[sym] = deque()
        self.adjustment_timestamps[sym].append(now)

    def _adjustments_in_last_hour(self, symbol: str) -> int:
        sym = str(symbol)
        if sym not in self.adjustment_timestamps:
            return 0
        now = self._now()
        cutoff = now - dt.timedelta(hours=1)
        q = self.adjustment_timestamps[sym]
        while q and q[0] < cutoff:
            q.popleft()
        return len(q)

    def _drawdown_exposure_factor(self, drawdown_pct: float) -> float:
        """Return the multiplier to apply to exposure limits based on drawdown tiers.

        Returns 1.0 when no tier is hit. Returns 0.0 when trading should halt.
        """
        for tier in self.drawdown_reduction_tiers:
            if drawdown_pct >= float(tier["pct"]):
                return float(tier["factor"])
        return 1.0

    def enforce_halt(self, minutes: int, reason: str) -> RiskDecision:
        self.halt_until = self._now() + dt.timedelta(minutes=max(1, int(minutes)))
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

        # --- Losing-streak cooldown ---
        cooldown_until = self.symbol_cooldown_until.get(str(symbol))
        if cooldown_until and now < cooldown_until:
            return RiskDecision(
                False,
                f"loss_streak_cooldown {str(symbol)} until {cooldown_until.isoformat()}"
            )

        # --- Max adjustments per symbol per hour ---
        adj_count = self._adjustments_in_last_hour(symbol)
        if adj_count >= self.max_adjustments_per_hour:
            return RiskDecision(
                False,
                f"max_adjustments_per_hour {adj_count} >= {self.max_adjustments_per_hour} for {str(symbol)}"
            )

        # --- Drawdown-based size reduction ---
        dd_factor = self._drawdown_exposure_factor(drawdown_pct)
        if dd_factor <= 0.0:
            return RiskDecision(False, f"drawdown_halt drawdown_pct={drawdown_pct:.2f}")

        pnl_today = float(snapshot.get("pnl_today", 0.0) or 0.0)
        if pnl_today <= -abs(self.max_daily_loss):
            return self.enforce_halt(24 * 60, f"daily_loss {pnl_today:.2f} <= -{abs(self.max_daily_loss):.2f}")

        if drawdown_pct >= self.max_drawdown_pct:
            return self.enforce_halt(24 * 60, f"drawdown_pct {drawdown_pct:.2f} >= {self.max_drawdown_pct:.2f}")

        if total_positions >= self.max_open_positions and abs(target_exposure) > 0.0:
            return RiskDecision(False, f"max_open_positions {total_positions} >= {self.max_open_positions}")

        if symbol_positions >= max_positions_per_symbol and abs(target_exposure) > abs(current_symbol_exposure):
            return RiskDecision(False, f"max_positions_per_symbol {symbol_positions} >= {max_positions_per_symbol}")

        # Apply drawdown reduction factor to exposure limits
        effective_max_symbol_exposure = max_symbol_exposure * dd_factor
        effective_max_total_exposure = self.max_total_exposure * dd_factor

        projected_symbol_exposure = max(abs(current_symbol_exposure), abs(target_exposure))
        if projected_symbol_exposure > effective_max_symbol_exposure:
            return RiskDecision(False, f"symbol_exposure {projected_symbol_exposure:.3f} > {effective_max_symbol_exposure:.3f} (dd_factor={dd_factor})")

        projected_total_exposure = max(abs(total_exposure), abs(total_exposure - current_symbol_exposure + target_exposure))
        if projected_total_exposure > effective_max_total_exposure:
            return RiskDecision(False, f"total_exposure {projected_total_exposure:.3f} > {effective_max_total_exposure:.3f} (dd_factor={dd_factor})")

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

        self._record_adjustment(symbol)
        return RiskDecision(True, "ok")
