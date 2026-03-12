from tools import project_status_ui as ui


def test_status_payload_shape(monkeypatch):
    monkeypatch.setattr(ui, "_processes", lambda: [])
    monkeypatch.setattr(ui, "_server_state", lambda _p: {"running": True, "pids": [1234]})
    monkeypatch.setattr(
        ui,
        "_training_state",
        lambda _p: {"drl_running": False, "lstm_running": False, "drl_pids": [], "lstm_pids": []},
    )
    monkeypatch.setattr(ui, "_active_models", lambda: {"champion": "c1", "canary": "c2"})
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
            "equity_curve": {"source": "test", "labels": ["now"], "values": [1000.0]},
            "drawdown_curve": {"source": "test", "labels": ["now"], "values": [0.0]},
            "symbol_pnl": {"source": "test", "labels": [], "values": []},
        },
    )
    monkeypatch.setattr(ui, "_tail", lambda _p, _n=60: [])

    out = ui.read_status()
    assert isinstance(out, dict)
    assert "timestamp_utc" in out
    assert "active_models" in out
    assert "server" in out
    assert "training" in out
    assert "account" in out
    assert "symbol_perf" in out
    assert "logs" in out
    assert "canary_gate" in out
    assert "telegram" in out
    assert "charts" in out
    assert set(out["canary_gate"].keys()) == {"ready", "reason"}


def test_profitability_chart_series_parses_recent_entries(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    profit_log = log_dir / "profitability.jsonl"
    profit_log.write_text(
        "\n".join(
            [
                '{"ts":"2026-03-12T00:00:00+00:00","equity":1000.0}',
                '{"ts":"2026-03-12T00:05:00+00:00","equity":995.0}',
                '{"ts":"2026-03-12T00:10:00+00:00","equity":1005.0}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui, "LOG_DIR", str(log_dir))

    out = ui._profitability_chart_series(limit=8)

    assert out["source"] == "profitability_log"
    assert out["equity"] == [1000.0, 995.0, 1005.0]
    assert out["drawdown_pct"] == [0.0, 0.5, 0.0]
