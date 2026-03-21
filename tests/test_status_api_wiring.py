"""
Tests that verify read_status() output shape matches what the frontend JavaScript expects.
"""

from datetime import datetime, timezone

import pytest

from tools import project_status_ui as ui


@pytest.fixture
def patched_status(monkeypatch):
    monkeypatch.setattr(ui, "_processes", lambda: [])
    monkeypatch.setattr(ui, "_server_state", lambda _p: {"running": False, "pids": []})
    monkeypatch.setattr(
        ui,
        "_training_state",
        lambda _p: {"drl_running": False, "lstm_running": False, "drl_pids": [], "lstm_pids": []},
    )
    monkeypatch.setattr(ui, "_active_models", lambda: {"champion": None, "canary": None})
    monkeypatch.setattr(
        ui,
        "_mt5_snapshot",
        lambda: {
            "connected": False,
            "balance": None,
            "equity": None,
            "profit": None,
            "free_margin": None,
            "open_positions": 0,
            "positions": [],
        },
    )
    monkeypatch.setattr(ui, "_mt5_symbol_perf", lambda _days=7: [])
    monkeypatch.setattr(
        ui,
        "_dashboard_charts",
        lambda _account, _perf: {
            "equity_curve": {
                "source": "test",
                "labels": [],
                "equity": [],
                "drawdown_pct": [],
            },
            "drawdown_curve": {"source": "test", "labels": [], "values": []},
            "symbol_pnl": {"source": "test", "labels": [], "values": []},
        },
    )
    monkeypatch.setattr(ui, "_tail", lambda _p, _n=60: [])
    # Patch helpers that _collect_status also calls so no I/O or registry access occurs.
    monkeypatch.setattr(ui, "_registry_summary", lambda _active: {})
    monkeypatch.setattr(ui, "_runtime_owner_health", lambda _p: {"ok": True, "issues": []})
    monkeypatch.setattr(ui, "_n8n_state", lambda: {"running": False})
    monkeypatch.setattr(ui, "_incident_feed", lambda _n=40: [])
    monkeypatch.setattr(ui, "_trade_learning_status", lambda: {})
    monkeypatch.setattr(ui, "_event_intel_status", lambda: {})
    monkeypatch.setattr(ui, "_source_health", lambda: {})
    monkeypatch.setattr(ui, "_telegram_status", lambda: {"enabled": False})
    monkeypatch.setattr(ui, "_symbol_stage_rows", lambda *_a, **_kw: [])
    monkeypatch.setattr(ui, "_symbol_pipeline_summary", lambda _rows: {})
    monkeypatch.setattr(ui, "_symbol_lane_rows", lambda *_a, **_kw: [])
    monkeypatch.setattr(ui, "_symbol_lane_summary", lambda _rows: {})
    monkeypatch.setattr(ui, "_record_account_history", lambda _account: None)
    # Force a fresh collection on the next read_status() call.
    monkeypatch.setattr(ui, "STATUS_CACHE", {"state": "booting"})
    return ui


def test_status_top_level_keys(patched_status):
    """read_status() must contain all keys the frontend JavaScript references."""
    out = patched_status.read_status()
    required = {
        "timestamp_utc",
        "active_models",
        "server",
        "training",
        "account",
        "symbol_perf",
        "logs",
        "canary_gate",
        "telegram",
        "charts",
        "runtime_owner",
    }
    assert required <= set(out.keys()), (
        f"Missing top-level keys: {required - set(out.keys())}"
    )


def test_account_sub_keys(patched_status):
    """out['account'] must contain every field the frontend reads."""
    out = patched_status.read_status()
    account = out["account"]
    assert isinstance(account, dict)
    required = {"connected", "balance", "equity", "profit", "free_margin", "open_positions", "positions"}
    assert required <= set(account.keys()), (
        f"Missing account keys: {required - set(account.keys())}"
    )


def test_charts_sub_keys(patched_status):
    """out['charts'] must provide the three chart series the dashboard renders."""
    out = patched_status.read_status()
    charts = out["charts"]
    assert isinstance(charts, dict)
    required = {"equity_curve", "drawdown_curve", "symbol_pnl"}
    assert required <= set(charts.keys()), (
        f"Missing charts keys: {required - set(charts.keys())}"
    )


def test_equity_curve_keys(patched_status):
    """equity_curve must have 'source', 'labels', and at least one data series key."""
    out = patched_status.read_status()
    ec = out["charts"]["equity_curve"]
    assert isinstance(ec, dict)
    assert "source" in ec, "'source' missing from equity_curve"
    assert "labels" in ec, "'labels' missing from equity_curve"
    data_series_keys = {"equity", "values", "drawdown_pct"}
    assert data_series_keys & set(ec.keys()), (
        f"equity_curve must contain at least one of {data_series_keys}; got {set(ec.keys())}"
    )


def test_canary_gate_exact_keys(patched_status):
    """canary_gate must have exactly the keys 'ready' and 'reason' — no extras."""
    out = patched_status.read_status()
    assert set(out["canary_gate"].keys()) == {"ready", "reason"}


def test_logs_sub_keys(patched_status):
    """out['logs'] must expose the standard log streams expected by the frontend."""
    out = patched_status.read_status()
    logs = out["logs"]
    assert isinstance(logs, dict)
    required = {"server", "lstm", "ppo", "dreamer", "audit"}
    assert required <= set(logs.keys()), (
        f"Missing log keys: {required - set(logs.keys())}"
    )


def test_timestamp_is_iso_string(patched_status):
    """timestamp_utc must be a non-empty string that parses as an ISO 8601 datetime."""
    out = patched_status.read_status()
    ts = out["timestamp_utc"]
    assert isinstance(ts, str) and ts, "timestamp_utc must be a non-empty string"
    # Must be parseable; datetime.fromisoformat raises ValueError if not.
    parsed = datetime.fromisoformat(ts)
    assert parsed.tzinfo is not None, "timestamp_utc must be timezone-aware"
