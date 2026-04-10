import builtins
import io
import os
from datetime import datetime, timedelta, timezone

import pytest
import yaml

from Python.risk_engine import RiskEngine, _CFG_PATH


@pytest.fixture
def engine(monkeypatch):
    """Return a RiskEngine constructed without touching the real config.yaml."""
    monkeypatch.setattr(builtins, "open", lambda *a, **kw: io.StringIO(""))
    monkeypatch.setattr(yaml, "safe_load", lambda f: {})
    return RiskEngine()


# ---------------------------------------------------------------------------
# 1. Error halt persists across day roll
# ---------------------------------------------------------------------------

def test_error_halt_persists_across_day_roll(engine):
    """record_error() x3 sets error_halt; reset_daily() must NOT clear halt."""
    engine.record_error()
    engine.record_error()
    engine.record_error()
    assert engine.halt is True
    assert engine.error_halt is True

    engine.reset_daily()

    assert engine.halt is True, "halt should remain True after reset_daily when error_halt is set"


# ---------------------------------------------------------------------------
# 2. PnL halt clears on day roll
# ---------------------------------------------------------------------------

def test_pnl_halt_clears_on_day_roll(engine):
    """A PnL-triggered halt (error_halt=False) is cleared when the day rolls."""
    engine.record_pnl(-9999.0)
    assert engine.halt is True
    assert engine.error_halt is False

    # Simulate that the last reset happened yesterday so maybe_roll_day() fires.
    yesterday = (datetime.now(timezone.utc) - timedelta(days=1)).date()
    engine.last_reset_day = yesterday

    engine.maybe_roll_day()

    assert engine.halt is False, "halt should be cleared after day roll for a PnL-only halt"


# ---------------------------------------------------------------------------
# 3. error_halt flag is set after three consecutive errors
# ---------------------------------------------------------------------------

def test_error_halt_flag_set_on_record_error(engine):
    """Three calls to record_error() must set both halt and error_halt."""
    assert engine.error_halt is False

    engine.record_error()
    engine.record_error()
    assert engine.error_halt is False, "error_halt should not be set after only two errors"

    engine.record_error()
    assert engine.error_halt is True
    assert engine.halt is True


# ---------------------------------------------------------------------------
# 4. PnL halt does not set error_halt
# ---------------------------------------------------------------------------

def test_pnl_halt_does_not_set_error_halt(engine):
    """Exceeding the daily loss limit sets halt but must NOT set error_halt."""
    engine.record_pnl(-9999.0)

    assert engine.halt is True
    assert engine.error_halt is False


# ---------------------------------------------------------------------------
# 5. _CFG_PATH is an absolute path ending with config.yaml
# ---------------------------------------------------------------------------

def test_config_path_is_absolute():
    """The module-level _CFG_PATH constant must be an absolute path ending with config.yaml."""
    assert os.path.isabs(_CFG_PATH), f"_CFG_PATH is not absolute: {_CFG_PATH}"
    assert _CFG_PATH.endswith("config.yaml"), f"_CFG_PATH does not end with 'config.yaml': {_CFG_PATH}"
