# CautiousGiggle Trade Engine

MT5-first autonomous trading research/runtime stack with symbol-scoped model lifecycle and operator telemetry.

## What It Does
- Ingests MT5 OHLCV bars per symbol.
- Trains LSTM (regime/pattern) and PPO (exposure policy).
- Evaluates candidate vs champion with hard gates and forward windows.
- Publishes runtime status to API/UI and optional Telegram alerts.

## What Is Live Now
- MT5-only data path for runtime and training.
- Symbol-scoped DRL training and model registry paths.
- Candidate/canary/champion registry with promotion gates.
- PPO backtest path that fails if artifacts are missing (no strategy fallback).

## What Is Experimental
- Reward-weight tuning (`drl.reward.weights`) and feature evolution.
- Forward-window gate tuning and canary promotion cadence.
- LSTM/PPO blend policy weight calibration.

## Architecture
- `Python/Server_AGI.py`: runtime loop, MT5 execution, telemetry.
- `training/train_lstm.py`, `training/train_drl.py`: training pipelines.
- `Python/model_registry.py`, `Python/model_evaluator.py`: registry + gates.
- `Python/backtester.py`: PPO fidelity backtesting.
- `tools/project_status_ui.py`: status UI/API.

## Data Path
1. MT5 bar fetch in `Python/data_feed.py`.
2. Normalize to OHLCV frame.
3. Engineered features in `drl/trading_env.py` (returns, microstructure, volatility, time context, trend/regime bucket).

## Training Path
1. Run LSTM per symbol (`training/train_lstm.py`).
2. Run PPO per symbol (default) or explicit combined mode (`training/train_drl.py`).
3. Stage candidate with metadata (`scorecard.json`, `metadata.json`) including feature/reward versions and windows.

## Promotion Path
1. Evaluate candidate vs champion with strict thresholds.
2. Require per-symbol pass rate and forward-window win rate.
3. Require champion beat on multiple metrics (score, return, Sharpe, drawdown).
4. Set canary on pass; retain champion on fail.
5. Promote canary only after survival checks pass (min trades, non-negative realized PnL, drawdown cap, minimum runtime).

## How To Run

### 1. Install
```powershell
cd C:\windows\system32\cautious-giggle
python -m venv .venv312
.\.venv312\Scripts\Activate.ps1
pip install -r requirements.txt
```

### 2. Configure
- Copy `config.yaml.example` to `config.yaml`.
- Keep secrets out of git.
- Set runtime secrets via env vars when possible:
```powershell
$env:MT5_LOGIN = "<login>"
$env:MT5_PASSWORD = "<password>"
$env:MT5_SERVER = "<server>"
$env:TELEGRAM_TOKEN = "<token>"
$env:TELEGRAM_CHAT_ID = "<chat_id>"
```

### 3. Start Runtime
```powershell
.\.venv312\Scripts\python.exe -m Python.Server_AGI --live
```

### 4. Train + Evaluate
```powershell
.\.venv312\Scripts\python.exe training\train_lstm.py
.\.venv312\Scripts\python.exe training\train_drl.py
.\.venv312\Scripts\python.exe tools\champion_cycle.py
.\.venv312\Scripts\python.exe tools\champion_cycle_loop.py --interval-minutes 30
```

### 5. One-Click Launcher + Desktop Icon
```powershell
powershell -ExecutionPolicy Bypass -File .\create_agi_trading_shortcut.ps1
.\launch_agi_trading.ps1
```
This starts server, UI, n8n, and continuous champion-cycle retraining/promotion checks, then opens the UI.

## Known Limitations
- Walk-forward orchestration is gate-driven but still file-registry based (no external experiment DB).
- Champion promotion is gate-enforced and can be operator-overridden, but governance remains file-registry based.
- MT5 availability/network state can block training/evaluation windows.

## Evidence / Results
- Runtime health: `http://127.0.0.1:8088/api/status`
- Logs: `logs/server.log`, `logs/audit_events.jsonl`, `logs/trade_events.jsonl`
- Cycle report: `logs/champion_cycle_last_report.json`
- Add dashboards/screenshots under `docs/screenshots/`

## Security Baseline
- `config.yaml` is gitignored.
- `config.yaml.example` contains placeholders only.
- Live startup now rejects placeholder Telegram config values.

## CI and Tests
- GitHub Actions workflow: `.github/workflows/build.yml`
- Test suite: `tests/`
- Local run:
```powershell
pytest -q
```

## Evidence Pack
Generate walk-forward proof artifacts:

```powershell
python tools\build_evidence_pack.py
```

Outputs:
- `docs/results/walk_forward_results.csv`
- `docs/results/walk_forward_summary.md`
- `docs/results/evidence_bundle.md`

## Risk Warning
Simulation/education tooling. Live trading carries financial risk.
