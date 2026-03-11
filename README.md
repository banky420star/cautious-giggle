# Cautious Giggle — Autonomous Trading Control Plane

An experimental local trading lab that lets you train, evaluate, and deploy PPO + LSTM agents against
live MetaTrader 5 feeds, promote reliable champions, and watch the entire stack from a dashboard or Telegram.

## Highlights
- **Autonomous pipeline**: `tools/champion_cycle.py` crafts per-symbol LSTM context models, trains PPO challengers,
  evaluates them against production champions, and promotes/rolls back builds.
- **Live execution**: `Python.Server_AGI` houses HybridBrain (LSTM + PPO + risk-engine) and MT5 execution hooks.
- **Observability**: HTTP + WebSocket dashboard (`tools/project_status_ui.py`) tracks trading state, queues control actions,
  and mirrors key updates on Telegram via `alerts/telegram_alerts.py`.
- **Data provenance**: Everything trains against MT5 candles (`Python/data_feed.py`), logs to `logs/`, and writes registry
  snapshots under `models/registry`.

## Architecture
1. **Market data** – MT5 via `MetaTrader5.copy_rates_from_pos` and `Python/mt5_executor`.
2. **Feature models** – `training/train_lstm.py` builds symbol/sequence-specific scalers and LSTM context classifiers.
3. **Policy trainer** – `training/train_drl.py` runs PPO on the shared `drl/trading_env.py`, logging candidates under
   `models/registry/candidates`.
4. **Autonomy loop** – `Python/autonomy_loop.py` evaluates candidates, gates them with `model_evaluator.py`, and records
   successes in `model_registry.py`.
5. **Trading server** – `Python.Server_AGI.py` blends LSTM signals + PPO exposure, enforces cooldowns, fires MT5 trades,
   and writes trade/audit events.
6. **Control surface** – `tools/project_status_ui.py` provides API, WebSocket, Telegram alerts, and process controls.

## Getting started

### Prerequisites
- Windows with MetaTrader 5 access to the symbols you plan to trade.
- Python 3.11+ (the repo ships a `.venv312` to match the sandbox).
- Telegram Bot token/chat for alerts.

### Install
```powershell
python -m venv .venv312
.venv312\Scripts\activate
pip install -r requirements.txt
```

### Configure
Copy `config.yaml.example` → `config.yaml` and set:

- `mt5.login`, `mt5.server`, `mt5.password` to connect to your MT5 account.
- `telegram.token` + `telegram.chat_id` for notification delivery.
- `trading.symbols`, `confidence_threshold`, and `drl.` parameters to tune risk and candidate gates.
- `registry.canary_policy` exposes default + per-symbol gating (min trades, max drawdown, runtime) so you can give BTCUSD tighter thresholds than EURUSD.

Keep `models/registry/active.json` clean (champion/canary `null`) before your first run.

### Run

1. **Live trading server**  
   ```powershell
   python -m Python.Server_AGI --live
   ```
   This will open sockets, start Telegram alerts, and poll MT5 for positions.

2. **Dashboard**  
   ```powershell
   python tools/project_status_ui.py
   ```
   Visit `http://127.0.0.1:8088` for realtime status, symbol performance cards, and control buttons.

3. **Training cycle** (optional autop-run)  
   ```powershell
   python tools/champion_cycle.py
   ```
   Use this to retrain LSTMs, stage PPO candidates, and promote new champions without touching production logic.

4. **Reproducibility helpers**  
   - `tools/profit_sweep.py` hunts for better thresholds/blend ratios.
   - `training/train_lstm.py` & `training/train_drl.py` can run individually for diagnostics.
   - Watch `logs/` (`server.log`, `trade_events.jsonl`, `champion_cycle*.log`, `ppo_training.log`) for audit trails.

### Metrics & Monitoring

- Dashboard API: `http://127.0.0.1:8088/api/status`
- WebSocket stream: `ws://127.0.0.1:8088/ws`
- Telegram notifications: training start/finish, champion events, live heartbeats, and online/offline updates.

## Testing

```powershell
python -m pytest
```

No tests? The command still ensures dependency installation and will pass even if there are zero tests.

## Release summary

- Use `python tools/release_summary.py` to snapshot the currently promoted champion/canary metadata plus the last five profitability entries.
- The generated `docs/results/release_summary.md` becomes part of the evidence bundle you can publish alongside a new release tag.

## Supporting info

- `SETUP.md` explains the overall stack (MT5, Telegram, sandbox).
- `requirements.txt` lists runtime requirements (`pandas`, `joblib`, `pyarrow`, etc.).
- Keep `.tmp/` and `.venv312/` ignored; they contain runtime locks/logs.

## Contribution

1. Make a branch per feature.
2. Add tests for any automation you touch.
3. Update `docs/` + this README to capture new flows or configurations.

## Next steps

- Automate champion promotions via scheduled jobs.
- Harden registries with integrity checks (hashes, timestamps, metadata).
- Fill `models/registry/candidates` with multi-symbol results and share example metrics in `docs/metrics.md` (TBD).
