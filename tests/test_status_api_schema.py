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
    assert set(out["logs"].keys()) >= {"server", "lstm", "ppo", "dreamer", "audit"}


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


def test_account_history_series_and_preference(tmp_path, monkeypatch):
    log_dir = tmp_path / "logs"
    log_dir.mkdir()
    account_log = log_dir / "account_history.jsonl"
    account_log.write_text(
        "\n".join(
            [
                '{"ts":"2026-03-12T00:00:00+00:00","equity":1000.0}',
                '{"ts":"2026-03-12T00:00:05+00:00","equity":998.0}',
                '{"ts":"2026-03-12T00:00:10+00:00","equity":1003.0}',
            ]
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(ui, "ACCOUNT_HISTORY_PATH", str(account_log))

    series = ui._account_history_series(limit=8)
    charts = ui._dashboard_charts({"equity": 1003.0}, [])

    assert series["source"] == "account_history"
    assert series["equity"] == [1000.0, 998.0, 1003.0]
    assert charts["equity_curve"]["source"] == "account_history"
    assert charts["drawdown_curve"]["values"] == [0.0, 0.2, 0.0]


def test_symbol_cards_from_status_uses_positions_and_signals():
    status = {
        "account": {
            "positions": [
                {
                    "symbol": "BTCUSDm",
                    "type": "SELL",
                    "volume": 0.04,
                    "profit": 12.5,
                    "open_price": 70000.0,
                    "tp": 69000.0,
                    "sl": 70500.0,
                    "tp_value_usd": 40.0,
                    "sl_value_usd": -20.0,
                }
            ]
        },
        "incidents": [
            {
                "event": "signal",
                "symbol": "BTCUSDm",
                "payload": {
                    "symbol": "BTCUSDm",
                    "signal": "LOW_VOLATILITY",
                    "confidence": 0.99,
                    "agi_exposure": 0.1,
                    "ppo_exposure": -0.2,
                    "dreamer_exposure": -1.0,
                    "exposure": 0.0,
                },
            }
        ],
    }

    out = ui._symbol_cards_from_status(status)

    assert "BTCUSDm" in out
    assert out["BTCUSDm"]["position_side"] == "SELL"
    assert out["BTCUSDm"]["signal"] == "LOW_VOLATILITY"
    assert out["BTCUSDm"]["dreamer_exposure"] == -1.0


def test_dreamer_visual_parses_active_run():
    lines = [
        "2026-03-12 19:01:20.719 | INFO | __main__:_train_symbol:138 - Dreamer training start | symbol=XAUUSDm | steps=5000 | window=64 | obs_dim=10497 | device=cpu | features=ultimate_150",
        "2026-03-12 20:29:19.144 | SUCCESS | __main__:_train_symbol:179 - Dreamer artifact saved: C:\\repo\\models\\dreamer\\dreamer_XAUUSDm.pt",
        "2026-03-12 20:29:22.646 | INFO | __main__:_train_symbol:138 - Dreamer training start | symbol=BTCUSDm | steps=5000 | window=64 | obs_dim=10497 | device=cpu | features=ultimate_150",
    ]

    out = ui._build_dreamer_visual(lines, running=True)

    assert out["phase"] == "optimizing"
    assert out["current_symbol"] == "BTCUSDm"
    assert out["last_saved_symbol"] == "XAUUSDm"
    assert out["steps"] == 5000


def test_ppo_visual_parses_progress_line():
    lines = [
        "2026-03-13 00:42:46.371 | INFO | __main__:_train_once:400 - DRL Training | symbols=['BTCUSDm'] | timesteps=500,000 | period=90d | tf=5m | candles=100,000 | per_symbol=True | initial_balance=813.26 | features=ultimate_150 | source=mt5",
        "2026-03-13 00:43:00.968 | INFO | __main__:_train_once:455 - Starting PPO training",
        "2026-03-13 00:50:00.000 | INFO | __main__:_on_step:99 - PPO progress | symbols=['BTCUSDm'] | step=25,000/500,000 | pct=5.00 | elapsed_s=420 | eta_s=7980",
    ]

    out = ui._build_ppo_visual(lines, running=True)

    assert out["phase"] == "optimizing"
    assert out["current_symbol"] == "BTCUSDm"
    assert out["current_timesteps"] == 25000
    assert out["target_timesteps"] == 500000
    assert out["progress_pct"] == 5.0
    assert out["elapsed_seconds"] == 420
    assert out["eta_seconds"] == 7980
