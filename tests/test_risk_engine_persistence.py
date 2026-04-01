import json
import os
import shutil
import tempfile
from datetime import datetime, timedelta, timezone
from pathlib import Path
from unittest.mock import patch

import pytest

from Python.risk_engine import RiskEngine


@pytest.fixture
def temp_dir():
    """Create a temporary directory for state files and config."""
    tmpdir = Path(tempfile.mkdtemp())
    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def engine(temp_dir):
    """Return a RiskEngine constructed with a minimal temp config.yaml."""
    cfg_path = temp_dir / "config.yaml"
    cfg_path.write_text("risk: {}\ntrading: {}\n", encoding="utf-8")
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        return RiskEngine()


# ---------------------------------------------------------------------------
# 1. to_dict() roundtrip tests
# ---------------------------------------------------------------------------


def test_to_dict_roundtrip(engine, temp_dir):
    """Create RiskEngine, set state, call to_dict(), then from_dict(); verify all fields match."""
    engine.realized_pnl_today = 123.45
    engine.daily_trades = 5
    engine.daily_trades_by_symbol = {"BTCUSDm": 3, "XAUUSDm": 2}
    engine.daily_losing_trades_by_symbol = {"BTCUSDm": 1}
    engine.halt = True
    engine.error_halt = False
    engine.error_count = 2
    engine.current_dd = 5.5
    engine.peak_equity = 10000.0
    engine.last_reset_day = datetime.now(timezone.utc).date()

    state_dict = engine.to_dict()
    cfg_path = temp_dir / "config.yaml"
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.from_dict(state_dict)

    assert engine2.realized_pnl_today == engine.realized_pnl_today
    assert engine2.daily_trades == engine.daily_trades
    assert engine2.daily_trades_by_symbol == engine.daily_trades_by_symbol
    assert engine2.daily_losing_trades_by_symbol == engine.daily_losing_trades_by_symbol
    assert engine2.halt == engine.halt
    assert engine2.error_halt == engine.error_halt
    assert engine2.error_count == engine.error_count
    assert engine2.current_dd == engine.current_dd
    assert engine2.peak_equity == engine.peak_equity
    assert engine2.last_reset_day == engine.last_reset_day


def test_to_dict_includes_all_required_fields(engine):
    """Verify to_dict() returns all necessary fields."""
    state_dict = engine.to_dict()
    required_fields = [
        "realized_pnl_today", "daily_trades", "daily_trades_by_symbol",
        "daily_losing_trades_by_symbol", "halt", "error_halt", "error_count",
        "current_dd", "peak_equity", "last_reset_day",
    ]
    for field in required_fields:
        assert field in state_dict, f"Missing field: {field}"


# ---------------------------------------------------------------------------
# 2. save_state() and load_state() tests
# ---------------------------------------------------------------------------


def test_save_and_load_state(engine, temp_dir):
    """Save state to a temp file, load it back, verify counters are preserved."""
    state_path = temp_dir / "risk_state.json"

    engine.realized_pnl_today = 500.0
    engine.daily_trades = 10
    engine.daily_trades_by_symbol = {"BTCUSDm": 6, "XAUUSDm": 4}
    engine.error_count = 1

    engine.save_state(str(state_path))
    assert state_path.exists(), "State file was not created"

    cfg_path = temp_dir / "config.yaml"
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.load_state(str(state_path))
    assert engine2 is not None
    assert engine2.realized_pnl_today == 500.0
    assert engine2.daily_trades == 10
    assert engine2.daily_trades_by_symbol == {"BTCUSDm": 6, "XAUUSDm": 4}
    assert engine2.error_count == 1


# ---------------------------------------------------------------------------
# 3. load_state_same_day tests
# ---------------------------------------------------------------------------


def test_load_state_same_day(engine, temp_dir):
    """Save state with today's date, load it, verify counters are NOT reset."""
    state_path = temp_dir / "risk_state.json"
    today = datetime.now(timezone.utc).date()

    engine.realized_pnl_today = 250.0
    engine.daily_trades = 3
    engine.last_reset_day = today
    engine.save_state(str(state_path))

    cfg_path = temp_dir / "config.yaml"
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.load_state(str(state_path))
    assert engine2.realized_pnl_today == 250.0, "P&L should not be reset on same day"
    assert engine2.daily_trades == 3, "Trade count should not be reset on same day"


# ---------------------------------------------------------------------------
# 4. load_state_previous_day tests
# ---------------------------------------------------------------------------


def test_load_state_previous_day(engine, temp_dir):
    """Save state with yesterday's date, load it, verify counters are reset."""
    state_path = temp_dir / "risk_state.json"
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    engine.realized_pnl_today = 500.0
    engine.daily_trades = 5
    engine.last_reset_day = yesterday
    engine.halt = False
    engine.error_halt = False
    engine.save_state(str(state_path))

    cfg_path = temp_dir / "config.yaml"
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.load_state(str(state_path))
    assert engine2.realized_pnl_today == 0.0, "P&L should be reset on day roll"
    assert engine2.daily_trades == 0, "Trade count should be reset on day roll"
    assert engine2.halt is False
    assert engine2.error_halt is False


# ---------------------------------------------------------------------------
# 5. error_halt_survives_day_roll tests
# ---------------------------------------------------------------------------


def test_error_halt_survives_day_roll(engine, temp_dir):
    """Set error_halt=True, save, load next day, verify error_halt persists."""
    state_path = temp_dir / "risk_state.json"
    yesterday = datetime.now(timezone.utc).date() - timedelta(days=1)

    engine.halt = True
    engine.error_halt = True
    engine.last_reset_day = yesterday
    engine.save_state(str(state_path))

    cfg_path = temp_dir / "config.yaml"
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.load_state(str(state_path))
    assert engine2.error_halt is True, "error_halt should persist across day roll"
    assert engine2.halt is True, "halt should remain True when error_halt is set"


# ---------------------------------------------------------------------------
# 6. load_state_missing_file tests
# ---------------------------------------------------------------------------


def test_load_state_missing_file():
    """Load from non-existent path returns None."""
    result = RiskEngine.load_state("/nonexistent/path/to/state.json")
    assert result is None, "Should return None for missing file"


# ---------------------------------------------------------------------------
# 7. load_state_corrupt_file tests
# ---------------------------------------------------------------------------


def test_load_state_corrupt_file(temp_dir):
    """Load from a file with invalid JSON returns None."""
    state_path = temp_dir / "corrupt.json"
    state_path.write_text("{ invalid json }", encoding="utf-8")

    result = RiskEngine.load_state(str(state_path))
    assert result is None, "Should return None for corrupt JSON file"


def test_load_state_empty_file(temp_dir):
    """Load from an empty file returns None."""
    state_path = temp_dir / "empty.json"
    state_path.write_text("", encoding="utf-8")

    result = RiskEngine.load_state(str(state_path))
    assert result is None, "Should return None for empty file"


# ---------------------------------------------------------------------------
# 8. save_state_atomic tests
# ---------------------------------------------------------------------------


def test_save_state_atomic(engine, temp_dir):
    """Verify that save_state uses a .tmp file atomically."""
    state_path = temp_dir / "risk_state.json"

    engine.realized_pnl_today = 100.0
    engine.save_state(str(state_path))

    assert state_path.exists()
    tmp_path = Path(str(state_path) + ".tmp")
    assert not tmp_path.exists(), "Temporary file should not remain after atomic write"

    content = state_path.read_text(encoding="utf-8")
    data = json.loads(content)
    assert data["realized_pnl_today"] == 100.0


def test_save_state_creates_directories(engine, temp_dir):
    """Verify that save_state creates missing directories."""
    nested_path = temp_dir / "a" / "b" / "c" / "risk_state.json"
    assert not nested_path.parent.exists()

    engine.save_state(str(nested_path))
    assert nested_path.exists(), "save_state should create parent directories"


# ---------------------------------------------------------------------------
# 9. from_dict with edge cases
# ---------------------------------------------------------------------------


def test_from_dict_missing_fields_uses_defaults(temp_dir):
    """from_dict should handle missing fields with sensible defaults."""
    cfg_path = temp_dir / "config.yaml"
    cfg_path.write_text("risk: {}\ntrading: {}\n", encoding="utf-8")

    incomplete_data = {"realized_pnl_today": 100.0}
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine = RiskEngine.from_dict(incomplete_data)
    assert engine.realized_pnl_today == 100.0
    assert engine.daily_trades == 0
    assert engine.halt is False
    assert engine.error_halt is False
    assert engine.error_count == 0


def test_from_dict_with_invalid_date_uses_today(temp_dir):
    """from_dict should use today's date if last_reset_day is invalid."""
    cfg_path = temp_dir / "config.yaml"
    cfg_path.write_text("risk: {}\ntrading: {}\n", encoding="utf-8")

    data = {"last_reset_day": "invalid-date"}
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine = RiskEngine.from_dict(data)
    assert engine.last_reset_day is not None


def test_from_dict_with_none_peak_equity(temp_dir):
    """from_dict should handle peak_equity=None correctly."""
    cfg_path = temp_dir / "config.yaml"
    cfg_path.write_text("risk: {}\ntrading: {}\n", encoding="utf-8")

    data = {"peak_equity": None, "current_dd": 0.0}
    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine = RiskEngine.from_dict(data)
    assert engine.peak_equity is None


# ---------------------------------------------------------------------------
# 10. Integration: full lifecycle test
# ---------------------------------------------------------------------------


def test_full_lifecycle_save_modify_load(engine, temp_dir):
    """Full lifecycle: create, save, load, modify, save again, load final."""
    state_path = temp_dir / "risk_state.json"
    cfg_path = temp_dir / "config.yaml"

    engine.realized_pnl_today = 100.0
    engine.daily_trades = 1
    engine.save_state(str(state_path))

    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine2 = RiskEngine.load_state(str(state_path))
    assert engine2.realized_pnl_today == 100.0
    engine2.realized_pnl_today = 200.0
    engine2.daily_trades = 2
    engine2.save_state(str(state_path))

    with patch("Python.risk_engine._CFG_PATH", str(cfg_path)):
        engine3 = RiskEngine.load_state(str(state_path))
    assert engine3.realized_pnl_today == 200.0
    assert engine3.daily_trades == 2
