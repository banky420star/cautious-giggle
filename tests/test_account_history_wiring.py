"""
Tests for _record_account_history() and _account_history_series()
in tools/project_status_ui.py.
"""

import json
import os
from datetime import datetime, timezone

import pytest

from tools import project_status_ui as ui


@pytest.fixture
def history_path(tmp_path, monkeypatch):
    path = str(tmp_path / "account_history.jsonl")
    monkeypatch.setattr(ui, "ACCOUNT_HISTORY_PATH", path)
    monkeypatch.setattr(ui, "_ACCOUNT_HISTORY_LAST_TS", None)
    monkeypatch.setattr(ui, "_ACCOUNT_HISTORY_LAST_SIG", None)
    return path


# ---------------------------------------------------------------------------
# _record_account_history tests
# ---------------------------------------------------------------------------


def test_record_writes_when_connected(history_path):
    """A connected account snapshot must be written as a valid JSON line."""
    ui._record_account_history(
        {
            "connected": True,
            "balance": 1000.0,
            "equity": 1000.0,
            "profit": 0.0,
            "free_margin": 500.0,
            "open_positions": 0,
        }
    )

    assert os.path.exists(history_path), "File must be created when account is connected"
    with open(history_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 1, "Exactly one line must be written"
    row = json.loads(lines[0])
    assert "ts" in row, "'ts' key must be present in written row"
    assert "balance" in row, "'balance' key must be present in written row"
    assert "equity" in row, "'equity' key must be present in written row"


def test_record_skips_when_not_connected(history_path):
    """A snapshot with connected=False must not create the history file."""
    ui._record_account_history(
        {
            "connected": False,
            "balance": 1000.0,
            "equity": 1000.0,
            "profit": 0.0,
            "free_margin": 500.0,
            "open_positions": 0,
        }
    )

    assert not os.path.exists(history_path), (
        "File must NOT be created when account is not connected"
    )


def test_record_deduplicates_same_sig(history_path, monkeypatch):
    """A second call with identical values within the interval must not write a new line."""
    account = {
        "connected": True,
        "balance": 1000.0,
        "equity": 1000.0,
        "profit": 0.0,
        "free_margin": 500.0,
        "open_positions": 0,
    }

    # First call — should write one line.
    ui._record_account_history(account)

    assert os.path.exists(history_path)
    with open(history_path, "r", encoding="utf-8") as f:
        after_first = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(after_first) == 1

    # Manually set the dedup state to simulate that the interval has NOT elapsed
    # and the signature matches.
    expected_sig = (1000.0, 1000.0, 0.0, 500.0, 0)
    monkeypatch.setattr(ui, "_ACCOUNT_HISTORY_LAST_TS", datetime.now(timezone.utc))
    monkeypatch.setattr(ui, "_ACCOUNT_HISTORY_LAST_SIG", expected_sig)

    # Second call with the same values — must be suppressed.
    ui._record_account_history(account)

    with open(history_path, "r", encoding="utf-8") as f:
        after_second = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(after_second) == 1, "Duplicate entry within interval must not be written"


def test_record_writes_after_different_sig(history_path):
    """Two calls with different values must each write a line."""
    base = {
        "connected": True,
        "balance": 1000.0,
        "equity": 1000.0,
        "profit": 0.0,
        "free_margin": 500.0,
        "open_positions": 0,
    }
    ui._record_account_history(base)
    changed = {**base, "equity": 1005.0}
    ui._record_account_history(changed)

    with open(history_path, "r", encoding="utf-8") as f:
        lines = [ln for ln in f.read().splitlines() if ln.strip()]
    assert len(lines) == 2, "Two lines must be written when the signature changes"


# ---------------------------------------------------------------------------
# _account_history_series tests
# ---------------------------------------------------------------------------


def test_account_history_series_parses_file(history_path):
    """Series must parse valid JSONL entries and return source=='account_history'."""
    lines = [
        '{"ts":"2026-03-12T00:00:00+00:00","balance":1000.0,"equity":1000.0}',
        '{"ts":"2026-03-12T00:00:05+00:00","balance":1000.0,"equity":1002.0}',
    ]
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    result = ui._account_history_series(limit=10)

    assert result["source"] == "account_history"
    assert isinstance(result["equity"], list)
    assert len(result["equity"]) > 0, "equity list must be non-empty when file has valid entries"
    assert result["equity"] == [1000.0, 1002.0]


def test_account_history_series_missing_file(history_path):
    """When the history file does not exist the series must report source=='unavailable'."""
    # history_path fixture sets ACCOUNT_HISTORY_PATH but does not create the file.
    assert not os.path.exists(history_path)

    result = ui._account_history_series()

    assert result["source"] == "unavailable"


def test_account_history_series_limit(history_path):
    """limit parameter must cap the number of returned data points."""
    lines = [
        f'{{"ts":"2026-03-12T00:00:{i:02d}+00:00","balance":1000.0,"equity":{1000 + i}.0}}'
        for i in range(10)
    ]
    with open(history_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")

    result = ui._account_history_series(limit=3)

    assert len(result["equity"]) <= 3, (
        f"Expected at most 3 equity points with limit=3, got {len(result['equity'])}"
    )
