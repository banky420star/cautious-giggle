import datetime as dt
import json
import os

import pytest

from Python.risk_supervisor import RiskSupervisor


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_supervisor(extra_risk: dict | None = None, extra_supervisor: dict | None = None) -> RiskSupervisor:
    """Build a RiskSupervisor with sensible defaults; extra dicts are merged in."""
    risk = {
        "max_daily_loss": 1_000_000.0,
        "max_symbol_exposure": 0.9,       # high – don't block on symbol exposure
        "max_total_exposure": 1.2,
        "supervisor": {
            "enabled": True,
            "max_drawdown_pct": 99.0,      # high – don't block on drawdown
            "max_spread_bps": 9999.0,      # high – don't block on spread
            "max_confidence_gap": 1.0,
            "min_trade_interval_sec": 0,   # no cooldown
            "max_open_positions": 999,
        },
    }
    if extra_risk:
        risk.update(extra_risk)
    if extra_supervisor:
        risk["supervisor"].update(extra_supervisor)
    return RiskSupervisor({"risk": risk})


def _allow(sup: RiskSupervisor, **kwargs):
    """Thin wrapper around allow_trade with safe defaults."""
    defaults = dict(
        symbol="BTCUSDm",
        target_exposure=0.1,
        confidence=0.8,
        spread_bps=1.0,
        snapshot={"pnl_today": 0.0},
        symbol_positions=0,
        total_positions=0,
        current_symbol_exposure=0.0,
        total_exposure=0.0,
        drawdown_pct=0.0,
    )
    defaults.update(kwargs)
    return sup.allow_trade(**defaults)


# ---------------------------------------------------------------------------
# Exposure tests
# ---------------------------------------------------------------------------

def test_projected_total_exposure_short_positions():
    """
    Short-position exposure fix: projected_total should use abs() of both
    current and target exposures.

    total_exposure=0.6, current=-0.3, target=-0.5
    projected = 0.6 - abs(-0.3) + abs(-0.5) = 0.8  >  max_total=0.7 → blocked
    """
    sup = _make_supervisor(
        extra_risk={"max_total_exposure": 0.7},
        extra_supervisor={"max_total_exposure": 0.7, "max_symbol_exposure": 0.9},
    )
    # Clear any halt loaded from a persisted state file so only the exposure check fires.
    sup.halt_until = None
    decision = _allow(
        sup,
        current_symbol_exposure=-0.3,
        target_exposure=-0.5,
        total_exposure=0.6,
    )
    assert not decision.allowed
    assert "total_exposure" in decision.reason


def test_projected_total_exposure_long_positions():
    """
    Long-position exposure fix: projected_total correctly computed for longs.

    total=0.6, current=0.1, target=0.4
    projected = 0.6 - 0.1 + 0.4 = 0.9  >  max_total=0.85 → blocked
    """
    sup = _make_supervisor(
        extra_risk={"max_total_exposure": 0.85},
        extra_supervisor={"max_total_exposure": 0.85, "max_symbol_exposure": 0.9},
    )
    sup.halt_until = None
    decision = _allow(
        sup,
        current_symbol_exposure=0.1,
        target_exposure=0.4,
        total_exposure=0.6,
    )
    assert not decision.allowed
    assert "total_exposure" in decision.reason


# ---------------------------------------------------------------------------
# Persistence: mark_trade
# ---------------------------------------------------------------------------

def test_state_persisted_and_loaded_on_mark_trade(tmp_path, monkeypatch):
    """mark_trade() saves state; a fresh RiskSupervisor loaded from the same path
    should see the symbol in last_trade_at_by_symbol."""
    state_file = str(tmp_path / "risk_state.json")
    cfg = {"risk": {"supervisor": {"enabled": True, "max_daily_loss": 1_000_000}}}

    sup = RiskSupervisor(cfg)
    monkeypatch.setattr(sup, "_state_path", lambda: state_file)
    sup.mark_trade("BTC")

    # New instance pointing at the same state file.
    sup2 = RiskSupervisor(cfg)
    monkeypatch.setattr(sup2, "_state_path", lambda: state_file)
    sup2._load_state()

    assert "BTC" in sup2.last_trade_at_by_symbol, (
        "symbol should appear in last_trade_at_by_symbol after loading persisted state"
    )


# ---------------------------------------------------------------------------
# Persistence: enforce_halt
# ---------------------------------------------------------------------------

def test_halt_until_persisted_and_loaded(tmp_path, monkeypatch):
    """enforce_halt() saves halt_until; a fresh instance reading the same state
    should report halt_until in the future."""
    state_file = str(tmp_path / "risk_state.json")
    cfg = {"risk": {"supervisor": {"enabled": True, "max_daily_loss": 1_000_000}}}

    sup = RiskSupervisor(cfg)
    monkeypatch.setattr(sup, "_state_path", lambda: state_file)
    sup.enforce_halt(60, "test_reason")

    sup2 = RiskSupervisor(cfg)
    monkeypatch.setattr(sup2, "_state_path", lambda: state_file)
    sup2._load_state()

    assert sup2.halt_until is not None, "halt_until should be set after loading persisted state"
    assert sup2.halt_until > dt.datetime.now(dt.timezone.utc), (
        "halt_until should be in the future after a 60-minute enforce_halt"
    )


# ---------------------------------------------------------------------------
# Robustness: missing state file
# ---------------------------------------------------------------------------

def test_load_state_handles_missing_file(tmp_path, monkeypatch):
    """_load_state() must not raise when the state file does not exist."""
    nonexistent = str(tmp_path / "does_not_exist.json")
    cfg = {"risk": {"supervisor": {"enabled": True}}}

    sup = RiskSupervisor(cfg)
    monkeypatch.setattr(sup, "_state_path", lambda: nonexistent)

    # Reset any state loaded during __init__ from the real state file.
    sup.halt_until = None
    sup.last_trade_at_by_symbol = {}

    # Should not raise.
    sup._load_state()

    assert sup.halt_until is None
    assert sup.last_trade_at_by_symbol == {}


# ---------------------------------------------------------------------------
# Robustness: corrupt state file
# ---------------------------------------------------------------------------

def test_load_state_handles_corrupt_file(tmp_path, monkeypatch):
    """_load_state() must not raise when the state file contains garbage JSON."""
    state_file = tmp_path / "corrupt_state.json"
    state_file.write_text("NOT VALID JSON{{{{{{ ]]]", encoding="utf-8")

    cfg = {"risk": {"supervisor": {"enabled": True}}}
    sup = RiskSupervisor(cfg)
    monkeypatch.setattr(sup, "_state_path", lambda: str(state_file))

    # Reset any state loaded during __init__ from the real state file.
    sup.halt_until = None
    sup.last_trade_at_by_symbol = {}

    # Should not raise.
    sup._load_state()

    assert sup.halt_until is None
    assert sup.last_trade_at_by_symbol == {}
