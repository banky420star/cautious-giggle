# Execution Review: LSTM + PPO + AGI Brain + MT5 Executor (Windows VPS)

## Goal reviewed
The current repository goal is an autonomous Observe → Learn → Validate → Deploy → Trade loop on Windows VPS + MT5, with nightly model training and live execution guarded by risk limits.

## Why dynamic lot-size / TP / SL / trailing-stop / pending orders are not happening
1. `MT5Executor.open_position()` sends only market `TRADE_ACTION_DEAL` orders with `symbol`, `volume`, `type`, and `price`.
2. No `sl`, `tp`, `deviation`, `type_time`, `type_filling`, or pending-order action/type fields are present in the request payload.
3. Because there are no pending-order action codes (`TRADE_ACTION_PENDING`) and no pending types (`BUY_LIMIT`, `SELL_LIMIT`), buy-limit/sell-limit placement is impossible in the current implementation.
4. Lot sizing is not dynamically volatility- or ATR-based; volume comes from `abs(delta)` in `reconcile_exposure()`, where `delta = target_lots - net_lots` and `target_lots = round(target_exposure * max_lots, 2)`.
5. `max_lots` is effectively a static cap loaded from risk config, so dynamic per-trade risk sizing is not implemented.

## Why volatility is effectively the only explicit LSTM signal
1. The LSTM model predicts 3 classes mapped to `LOW_VOLATILITY`, `MED_VOLATILITY`, `HIGH_VOLATILITY`.
2. Training labels in `train_lstm.py` are generated from absolute next-return magnitude thresholds (volatility buckets), not directional class labels.
3. Therefore this LSTM branch is a volatility classifier, not an explicit directional BUY/SELL predictor.

## How LSTM training currently works
1. Fetch OHLCV per symbol using `fetch_training_data()`.
2. Scale features with `MinMaxScaler`.
3. Build rolling sequences of length 60.
4. Label each sequence using future return magnitude buckets:
   - high magnitude -> class 2
   - medium magnitude -> class 1
   - low magnitude -> class 0
5. Train `AGIModel` (3-layer LSTM + linear head) using cross-entropy.
6. Save model weights and scaler; register candidate in model registry.

## How PPO training currently works
1. Build `TradingEnv` episodes from continuous OHLCV history with observation window + portfolio state.
2. PPO action space is continuous `[-1, 1]` and interpreted as target position fraction.
3. Reward combines step return minus transaction/spread costs and drawdown penalties.
4. PPO policy uses `LSTMFeatureExtractor`, which runs sequence observations through `SmartAGI`'s LSTM feature path and concatenates portfolio state.
5. Best evaluated PPO checkpoint and matching `VecNormalize` statistics are saved and staged as a candidate.

## AGI Brain and execution-path reality check
1. `Server_AGI.py` currently initializes MT5, risk, executor, and brain but only sends Telegram heartbeat in a loop.
2. The shown server file does not execute a socket trade-decision route or call `brain.live_trade()`.
3. `HybridBrain` itself is minimal: it only forwards `live_trade(symbol, exposure, max_lots)` to executor reconciliation after risk check.
4. Result: architecture docs describe an advanced autonomous loop, but the checked-in execution path is currently thin and missing full live decision wiring.

## Windows VPS execution readiness (what exists vs gaps)
### Exists
- `scripts/setup_vps.bat` creates venv, installs requirements, and prepares directories.
- `scripts/smoke_test.ps1` validates syntax, imports, and config parse.
- `start_server.ps1` and `vps_launch_all.bat` provide launch helpers.

### Gaps / risks
- `start_server.ps1` contains a hard-coded AGI token; this is unsafe for production.
- `config.yaml.example` lacks the `risk` section required by `RiskEngine` (`max_daily_loss`, `max_daily_trades`, `max_lots`).
- `Server_AGI.py` in current form does not show end-to-end command handling from n8n to trade execution.

## Bottom line
The bot is not doing dynamic entries with lot-size/TP/SL/TT/pending limits because those fields and order types are not implemented in `MT5Executor` requests, and the live server path currently shown is heartbeat-oriented rather than full trade-routing. The LSTM branch is trained as a volatility classifier, while PPO is trained on continuous position control in `TradingEnv`; live execution only partially reflects that design today.
