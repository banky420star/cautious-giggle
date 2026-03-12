# Cautious Giggle

MT5-first autonomous trading stack with symbol-scoped LSTM context models, PPO policies, optional Dreamer overlays, a champion/canary registry, and a live operator console.

## What is in the repo

- `Python/Server_AGI.py` runs the live trading loop, risk supervision, MT5 execution, audit logging, and Telegram notifications.
- `training/train_lstm.py` trains per-symbol LSTM context models.
- `training/train_drl.py` trains PPO candidates against `drl/trading_env.py`.
- `training/train_dreamer.py` trains per-symbol Dreamer policies for optional live blending.
- `tools/champion_cycle.py` runs the full retrain/evaluate/stage flow across the configured symbol set.
- `tools/project_status_ui.py` serves the dashboard and control API on `127.0.0.1:8088`.
- `Python/model_registry.py` tracks champion/canary state, history, integrity metadata, and promotion policy.

## Current architecture

1. MT5 candles are loaded through `Python/data_feed.py`.
2. Feature generation is centralized in `Python/feature_pipeline.py`.
3. LSTM training writes symbol-scoped model bundles into `models/per_symbol/`.
4. PPO training stages candidates into `models/registry/candidates/`.
5. Optional Dreamer artifacts are written into `models/dreamer/`.
6. `Python/autonomy_loop.py` and `tools/champion_cycle.py` evaluate and stage canaries.
7. `Python/Server_AGI.py` blends SmartAGI, PPO, and optional Dreamer outputs under the risk engine and supervisor.
8. The dashboard and Telegram read the same runtime/log/registry state.

## Feature and model defaults

- New LSTM and PPO training defaults use `ultimate_150`.
- Legacy promoted champions keep their recorded feature metadata until replaced.
- Dreamer can be enabled per symbol through `drl.dreamer`.
- Registry canary policy supports global defaults and per-symbol overrides.
- Live exposure is synthesized in `Python/hybrid_brain.py` from SmartAGI, registry-loaded PPO, optional PPO history/ensemble voting, and optional Dreamer policies from `models/dreamer/`.
- The blend is controlled by `drl.ppo_blend`, `drl.dreamer.blend`, and the configured Dreamer symbol set.

## Repo layout

- `Python/` runtime, registry, risk, MT5, and evaluation code
- `training/` LSTM, PPO, Dreamer, and trade-memory builders
- `drl/` trading environment and policy feature extractors
- `tools/` operator tools, cycle orchestration, drift analysis, and release helpers
- `alerts/` Telegram alert transport
- `docs/` architecture, sync flow, metrics, and release evidence
- `tests/` regression coverage for runtime, registry, env, status API, and training helpers

## Prerequisites

- Windows host with MetaTrader 5 access
- Python 3.12-compatible environment (`.venv312` is the repo convention)
- Telegram bot token/chat if you want alert delivery

## Install

```powershell
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

## Configure

Copy `config.yaml.example` to `config.yaml` and set:

- `mt5.login`, `mt5.password`, `mt5.server`
- `telegram.token`, `telegram.chat_id`
- `trading.symbols`
- `risk.*` and `risk.supervisor.*`
- `training.feature_version`
- `drl.feature_version`
- `drl.dreamer.*`
- `registry.canary_policy.*`

`config.yaml` is gitignored and is expected to be machine-local.

Key runtime safety knobs live under `risk`:

- daily loss cap
- max open positions
- max total and per-symbol exposure
- spread/slippage tolerances
- symbol-specific execution profiles

`risk.supervisor` adds the hard pre-trade gate used by the live server:

- cooldown enforcement
- max drawdown halt
- spread cap
- position count cap
- exposure cap
- risk-reduction allowed even while halted

## Run

Live server:

```powershell
python -m Python.Server_AGI --live
```

Dashboard:

```powershell
python tools/project_status_ui.py
```

Full training cycle:

```powershell
python tools/champion_cycle.py
```

Individual training:

```powershell
python training/train_lstm.py
python training/train_drl.py
python training/train_dreamer.py
```

## Operator surface

- UI: `http://127.0.0.1:8088`
- Status API: `http://127.0.0.1:8088/api/status`
- WebSocket: `ws://127.0.0.1:8088/ws`

The dashboard exposes runtime health, registry state, incident feed, Telegram parity, training status, logs, and operator controls.

The runtime also maintains:

- `logs/event_intel_state.json` for news/calendar regime context
- `logs/audit_events.jsonl` for runtime/trade/risk audit events
- `logs/trade_events.jsonl` for trade history
- `logs/learning/trade_learning_latest.json` for rolling trade-memory metrics
- Telegram daily profitability summaries sourced from the trade-learning pass

## Validation

Run the full regression suite with the repo venv:

```powershell
.venv312\Scripts\python.exe -m pytest
```

Useful spot checks:

```powershell
.venv312\Scripts\python.exe -m compileall Python training tools drl
python tools/backtest_vs_live_drift.py
python tools/release_summary.py
```

## Evidence and reporting

- `tools/release_summary.py` writes `docs/results/release_summary.md`
- `tools/build_evidence_pack.py` rebuilds the public evidence bundle in `docs/results/`
- `tools/profit_sweep.py` records profitability sweep output in `logs/`
- `docs/metrics.md` documents the current trading/profitability story

## Notes

- The runtime is designed for one active owner per role; Windows venv redirector child processes are expected.
- Generated logs, local runtime locks, and machine-local config are intentionally not tracked.
- The promoted model set, not the code alone, determines whether the live bot actually trades.
