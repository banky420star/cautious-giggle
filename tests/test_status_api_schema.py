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
    assert set(out["canary_gate"].keys()) == {"ready", "reason"}
