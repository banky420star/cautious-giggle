# Architecture Overview

Cautious Giggle is deliberately layered to separate data, training, runtime, and control surfaces so each part can evolve independently.

## Layers
- **Data acquisition** (`Python/data_feed.py`, `Python/mt5_executor.py`): MT5 candles are fetched per symbol, normalized, and
  cached for both training and live trading. No external APIs (like Yahoo) are used in production; the entire stack expects
  MT5 connectivity.
- **Model stack**
  - *LSTM context models* (`training/train_lstm.py`) compute regime-aware scalers and burnt-in sequences for each symbol.
  - *PPO policy* (`training/train_drl.py`) trains against `drl/trading_env.py`, which mirrors the live trading reward (PnL minus drift/fees).
  - *HybridBrain* (`Python/hybrid_brain.py`) blends LSTM signals and PPO exposures using configurable `ppo_blend`.
- **Autonomous control** (`tools/champion_cycle.py`, `Python/autonomy_loop.py`, `Python/model_evaluator.py`, `Python/model_registry.py`):
  - Cycle retrains, stages candidates, runs evaluations, and writes champion/canary metadata.
  - The registry holds paths for champion/canary and writes audit events when promotions or rollbacks occur.
- **Live trading** (`Python.Server_AGI.py`):
  - Contains the main loop that polls `mt5.copy_rates_from_pos`, asks `HybridBrain` for decisions, and pushes MT5 orders through `mt5_executor`.
  - Logs trades in `trade_events.jsonl`, audits in `audit_events.jsonl`, and publishes telemetry for the dashboard and Telegram alerts.
- **Control surface** (`tools/project_status_ui.py`, `alerts/telegram_alerts.py`):
  - Provides HTTP (`/api/status`) and WebSocket (`/ws`) endpoints.
  - Offers controls for training, restarting services, and staging canaries.
  - Streams Telegram alerts for training progress, snapshots, and online/offline indicators.

## Communication flow
1. Training modules write new candidate bundles to `models/registry/candidates/<timestamp>`.
2. `autonomy_loop` evaluates these bundles versus `models/registry/active.json`.
3. Successful challengers become canaries, then champions, while failed ones log gate reasons.
4. `Server_AGI` monitors `model_registry` to hot swap symbol-specific champion paths without restarting the process.
5. Dashboard and Telegram read from server logs and model registry entries to keep operators informed.
