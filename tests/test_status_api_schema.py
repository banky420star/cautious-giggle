from datetime import datetime, timezone

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
    assert out["summary"]["completed_symbols"] == 1
    assert out["summary"]["active_symbols"] == 1
    by_symbol = {row["symbol"]: row for row in out["queue"]}
    assert by_symbol["XAUUSDm"]["status"] == "done"
    assert by_symbol["BTCUSDm"]["status"] == "active"


def test_dreamer_visual_scales_eta_by_steps_and_window(monkeypatch):
    class _FixedDateTime(datetime):
        @classmethod
        def now(cls, tz=None):
            base = cls(2026, 3, 12, 2, 40, 0)
            return base.replace(tzinfo=tz or timezone.utc)

    monkeypatch.setattr(ui, "datetime", _FixedDateTime)
    monkeypatch.setattr(ui, "_configured_symbols", lambda: ["XAUUSDm", "BTCUSDm"])
    lines = [
        "2026-03-12 00:00:00.000 | INFO | __main__:_train_symbol:138 - Dreamer training start | symbol=XAUUSDm | steps=5000 | window=64 | obs_dim=10497 | device=cpu | features=ultimate_150",
        "2026-03-12 01:00:00.000 | SUCCESS | __main__:_train_symbol:179 - Dreamer artifact saved: C:\\repo\\models\\dreamer\\dreamer_XAUUSDm.pt",
        "2026-03-12 01:10:00.000 | INFO | __main__:_train_symbol:138 - Dreamer training start | symbol=BTCUSDm | steps=15000 | window=96 | obs_dim=15745 | device=cpu | features=ultimate_150",
    ]

    out = ui._build_dreamer_visual(lines, running=True)

    assert out["estimated_run_seconds"] == 16200.0
    by_symbol = {row["symbol"]: row for row in out["queue"]}
    assert by_symbol["BTCUSDm"]["status"] == "active"
    assert by_symbol["BTCUSDm"]["progress_pct"] == 55.56


def test_dreamer_visual_stalled_run_is_not_counted_active(monkeypatch):
    monkeypatch.setattr(ui, "_configured_symbols", lambda: ["BTCUSDm"])
    lines = [
        "2026-03-12 20:29:22.646 | INFO | __main__:_train_symbol:138 - Dreamer training start | symbol=BTCUSDm | steps=5000 | window=64 | obs_dim=10497 | device=cpu | features=ultimate_150",
    ]

    out = ui._build_dreamer_visual(lines, running=False)

    assert out["phase"] == "stalled"
    assert out["summary"]["active_symbols"] == 0
    assert out["queue"][0]["status"] == "partial"


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


def test_root_pids_collapse_windows_launcher_pair():
    rows = [
        {"pid": 2332, "ppid": 100, "cmd": '".venv312\\Scripts\\python.exe" tools/champion_cycle.py'},
        {"pid": 16212, "ppid": 2332, "cmd": '"C:\\Users\\Administrator\\Desktop\\python.exe" tools/champion_cycle.py'},
    ]

    assert ui._root_pids(rows) == [2332]


def test_latest_training_progress_uses_cycle_log_when_ppo_has_not_logged_start(monkeypatch):
    def fake_tail(path, _n=60):
        path = str(path)
        if path.endswith("ppo_training.log"):
            return []
        if path.endswith("lstm_training.log"):
            return []
        if path.endswith("champion_cycle_stderr.log"):
            return ["2026-03-13 14:39:02.495 | INFO | __main__:main:236 - Cycle step: train PPO candidate for EURUSDm"]
        return []

    monkeypatch.setattr(ui, "_tail", fake_tail)

    out = ui._latest_training_progress()

    assert out["drl_symbol"] is None
    assert out["cycle_ppo_symbol"] == "EURUSDm"


def test_training_state_keeps_cycle_symbol_visible_before_drl_process_starts(monkeypatch):
    monkeypatch.setattr(ui, "_configured_symbols", lambda: ["BTCUSDm", "XAUUSDm", "EURUSDm", "GBPUSDm"])
    monkeypatch.setattr(
        ui,
        "_latest_training_progress",
        lambda: {
            "drl_symbol": None,
            "drl_timesteps": None,
            "drl_candles": None,
            "lstm_symbol": None,
            "lstm_epoch": None,
            "lstm_epochs_total": None,
            "train_error": None,
            "cycle_ppo_symbol": "EURUSDm",
        },
    )
    monkeypatch.setattr(
        ui,
        "_build_training_visuals",
        lambda *_args, **_kwargs: {
            "active_stage": "idle",
            "active_label": "Training idle",
            "lstm": {"queue": [], "summary": {}},
            "ppo": {"current_symbol": None, "progress_pct": None, "current_timesteps": None, "target_timesteps": None},
            "dreamer": {"current_symbol": None, "last_saved_symbol": None},
        },
    )
    monkeypatch.setattr(ui, "_tail", lambda *_args, **_kwargs: [])

    procs = [{"pid": 2332, "ppid": 100, "cmd": '".venv312\\Scripts\\python.exe" tools/champion_cycle.py'}]
    out = ui._training_state(procs)

    assert out["cycle_running"] is True
    assert out["drl_running"] is False
    assert out["drl_symbol"] == "EURUSDm"
    assert out["visual"]["active_stage"] == "ppo"


def test_symbol_stage_rows_include_pipeline_states(monkeypatch):
    training = {
        "configured_symbols": ["BTCUSDm", "XAUUSDm", "EURUSDm", "GBPUSDm"],
        "drl_running": True,
        "dreamer_running": False,
        "cycle_running": True,
        "drl_symbol": "EURUSDm",
        "visual": {
            "lstm": {
                "queue": [
                    {"symbol": "BTCUSDm", "status": "done", "progress_pct": 100.0, "epoch": 20, "epochs_total": 20},
                    {"symbol": "XAUUSDm", "status": "done", "progress_pct": 100.0, "epoch": 20, "epochs_total": 20},
                    {"symbol": "EURUSDm", "status": "active", "progress_pct": 55.0, "epoch": 11, "epochs_total": 20},
                ]
            },
            "ppo": {"current_symbol": "EURUSDm", "progress_pct": 5.0, "current_timesteps": 25000, "target_timesteps": 500000},
            "dreamer": {"current_symbol": None, "last_saved_symbol": "XAUUSDm", "steps": 5000},
        },
    }
    active = {
        "champion": "C:\\repo\\models\\registry\\candidates\\20260308_073222",
        "symbols": {
            "EURUSDm": {
                "champion": "C:\\repo\\models\\registry\\candidates\\20260308_073222",
                "canary": "C:\\repo\\models\\registry\\candidates\\20260313_142633",
                "canary_state": {"passed": True},
            }
        },
    }
    monkeypatch.setattr(ui, "_has_lstm_artifact", lambda symbol: symbol in {"BTCUSDm", "XAUUSDm", "EURUSDm"})
    monkeypatch.setattr(ui, "_has_dreamer_artifact", lambda symbol: symbol in {"BTCUSDm", "XAUUSDm"})
    monkeypatch.setattr(
        ui,
        "_latest_candidates_by_symbol",
        lambda symbols: {
            "BTCUSDm": {"label": "20260313_062330", "gates_passed": False},
            "XAUUSDm": {"label": "20260313_142633", "gates_passed": False},
        },
    )

    rows = ui._symbol_stage_rows(
        training,
        active,
        account={"positions": [{"symbol": "EURUSDm", "profit": 12.5}]},
        server={"running": True},
    )
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["EURUSDm"]["lstm"]["state"] == "active"
    assert by_symbol["EURUSDm"]["ppo"]["state"] == "active"
    assert by_symbol["EURUSDm"]["canary"]["state"] == "ready"
    assert by_symbol["EURUSDm"]["champion"]["state"] == "live"
    assert by_symbol["EURUSDm"]["trading"]["state"] == "active"
    assert by_symbol["BTCUSDm"]["dreamer"]["state"] == "done"
    assert by_symbol["BTCUSDm"]["champion"]["state"] == "waiting"
    assert by_symbol["BTCUSDm"]["trading"]["state"] == "waiting"
    assert by_symbol["GBPUSDm"]["ppo"]["state"] == "queued"


def test_symbol_pipeline_summary_counts_training_and_trading():
    rows = [
        {
            "symbol": "BTCUSDm",
            "lstm": {"state": "done"},
            "dreamer": {"state": "done"},
            "ppo": {"state": "done"},
            "canary": {"state": "waiting"},
            "champion": {"state": "live"},
            "trading": {"state": "armed"},
        },
        {
            "symbol": "EURUSDm",
            "lstm": {"state": "active"},
            "dreamer": {"state": "queued"},
            "ppo": {"state": "queued"},
            "canary": {"state": "testing"},
            "champion": {"state": "live"},
            "trading": {"state": "active"},
        },
        {
            "symbol": "XAUUSDm",
            "lstm": {"state": "done"},
            "dreamer": {"state": "partial"},
            "ppo": {"state": "queued"},
            "canary": {"state": "waiting"},
            "champion": {"state": "waiting"},
            "trading": {"state": "waiting"},
        },
    ]

    out = ui._symbol_pipeline_summary(rows)

    assert out["symbols_total"] == 3
    assert out["training_active_symbols"] == 1
    assert out["canary_review_symbols"] == 1
    assert out["champion_live_symbols"] == 2
    assert out["trading_ready_symbols"] == 2
    assert out["trading_active_symbols"] == 1


def test_symbol_lane_rows_include_decision_and_execution_trace():
    training = {
        "configured_symbols": ["BTCUSDm", "XAUUSDm"],
        "symbol_stage_rows": [
            {
                "symbol": "BTCUSDm",
                "lstm": {"state": "done", "detail": "epoch 20/20"},
                "dreamer": {"state": "done", "detail": "artifact saved"},
                "ppo": {"state": "active", "detail": "12,000/120,000"},
                "canary": {"state": "waiting", "detail": "none staged"},
                "champion": {"state": "waiting", "detail": "not set"},
                "trading": {"state": "waiting", "detail": "no champion"},
            },
            {
                "symbol": "XAUUSDm",
                "lstm": {"state": "done", "detail": "epoch 20/20"},
                "dreamer": {"state": "done", "detail": "artifact saved"},
                "ppo": {"state": "done", "detail": "20260308_073222"},
                "canary": {"state": "testing", "detail": "20260308_073222"},
                "champion": {"state": "live", "detail": "20260308_073222"},
                "trading": {"state": "active", "detail": "1 open | pnl +12.50"},
            },
        ],
    }
    active = {
        "symbols": {
            "BTCUSDm": {},
            "XAUUSDm": {
                "champion": "C:\\repo\\models\\registry\\candidates\\20260308_073222",
                "canary": "C:\\repo\\models\\registry\\candidates\\20260308_073222",
                "canary_state": {"passed": False},
            },
        }
    }
    incidents = [
        {
            "ts": "2026-03-17T17:40:46.196856+00:00",
            "event": "trade_action",
            "symbol": "XAUUSDm",
            "payload": {
                "symbol": "XAUUSDm",
                "request_action": "open",
                "executed": True,
                "side": "SELL",
                "lane": "canary",
                "model_source": "registry:XAUUSDm:canary",
                "model_version": "20260308_073222",
                "magic": 52100,
                "comment": "AGI|XAU|CA|O|P073222|P-31",
                "retcode": 10009,
                "ticket": 2515768622,
            },
        },
        {
            "ts": "2026-03-17T17:40:45.834461+00:00",
            "event": "signal",
            "symbol": "XAUUSDm",
            "payload": {
                "symbol": "XAUUSDm",
                "regime": "LOW_VOLATILITY",
                "confidence": 1.0,
                "risk_scalar": 0.95,
                "agi_bias": -0.05,
                "ppo_exposure": -0.3138,
                "dreamer_exposure": 1.0,
                "raw_target": 0.2209,
                "exposure": 0.2099,
                "decision_profile": {"ppo_weight": 0.58, "dreamer_weight": 0.42, "agi_weight": 0.34},
            },
        },
        {
            "ts": "2026-03-17T17:40:45.193761+00:00",
            "event": "signal",
            "symbol": "BTCUSDm",
            "payload": {
                "symbol": "BTCUSDm",
                "regime": "LOW_VOLATILITY",
                "confidence": 1.0,
                "risk_scalar": 0.95,
                "agi_bias": -0.05,
                "ppo_exposure": 0.0173,
                "dreamer_exposure": 0.0,
                "raw_target": 0.0034,
                "exposure": 0.0173,
                "decision_profile": {"ppo_weight": 0.72, "dreamer_weight": 0.28, "agi_weight": 0.18},
            },
        },
    ]

    rows = ui._symbol_lane_rows(training, active, incidents, account={"positions": [{"symbol": "XAUUSDm", "profit": 12.5}]})
    by_symbol = {row["symbol"]: row for row in rows}

    assert by_symbol["BTCUSDm"]["decision"]["final_target"] == 0.0173
    assert by_symbol["BTCUSDm"]["execution"]["state"] == "armed"
    assert by_symbol["XAUUSDm"]["execution"]["state"] == "executed"
    assert by_symbol["XAUUSDm"]["execution"]["magic"] == 52100
    assert by_symbol["XAUUSDm"]["execution"]["comment"] == "AGI|XAU|CA|O|P073222|P-31"
    assert by_symbol["XAUUSDm"]["registry"]["champion_label"] == "20260308_073222"
    assert by_symbol["XAUUSDm"]["position"]["open_positions"] == 1


def test_symbol_lane_summary_counts_execution_states():
    rows = [
        {"decision": {"state": "armed"}, "execution": {"state": "armed"}, "position": {"open_positions": 0}},
        {"decision": {"state": "executed"}, "execution": {"state": "executed"}, "position": {"open_positions": 1}},
        {"decision": {"state": "blocked"}, "execution": {"state": "blocked"}, "position": {"open_positions": 0}},
        {"decision": {"state": "neutral"}, "execution": {"state": "neutral"}, "position": {"open_positions": 0}},
    ]

    out = ui._symbol_lane_summary(rows)

    assert out["symbols_total"] == 4
    assert out["actionable_symbols"] == 2
    assert out["executed_symbols"] == 1
    assert out["blocked_symbols"] == 1
    assert out["neutral_symbols"] == 1
    assert out["open_positions"] == 1
