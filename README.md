# cautious-giggle — Autonomous AGI Trading Hedge Fund (2026)

**Full autonomous Observe → Learn → Validate → Deploy → Trade loop**

## 🤖 The Self-Evolving Loop
This system is designed for total autonomy on a Windows VPS + MetaTrader 5, with full **macOS development support** via Yahoo Finance:
1. **Observe**: Fetches high-fidelity OHLCV data via MT5 (Windows) or Yahoo Finance with FX volume proxy (macOS/Linux).
2. **Learn**: Nightly training cycles refine **joint LSTM-PPO** models on combined multi-asset datasets.
3. **Validate**: The `AutonomyLoop` backtests candidates via walk-forward simulation. Winners are staged as **Canary**.
4. **Deploy**: The `HybridBrain` hot-swaps active models from the `ModelRegistry` in real-time.
5. **Trade**: Market execution via MT5 with spread-aware deadzones, volatility gating, and Canary risk scaling.
6. **Monitor**: Real-time PnL polling rolls back canaries if they misbehave.
7. **Self-Improve**: Optional `GrokTradingAgent` reads gradient flow and walk-forward reports to suggest hyperparameter tweaks via the xAI API.

## Architecture

```
┌──────────────┐    every 5 min    ┌──────────────┐
│     n8n      │ ───────────────►  │  agi_n8n     │
│  Orchestrator│                   │  _bridge.py  │
│  (5678)      │ ◄──── JSON ─────  │              │
└──────────────┘                   └──────┬───────┘
                                          │ socket :9090
                                   ┌──────▼───────┐
                                   │  Server_AGI  │ ◄───┐
                                   │   .py        │     │ Monitoring
                                   └──┬───┬───┬───┘     │
                                      │   │   │         │
                          ┌───────────┘   │   └───── AutonomyLoop
                          ▼               ▼               │
                   ┌────────────┐  ┌────────────┐  ┌──────▼─────┐
                   │HybridBrain │  │ risk_engine│  │ModelRegistry│
                   │ (PPO+LSTM) │  │(KillSwitch)│  │ (Promotion) │
                   └──────┬─────┘  └────────────┘  └────────────┘
                          │
                   ┌──────▼─────┐
                   │  data_feed │
                   │ (MT5 / YF) │
                   └────────────┘
```

## Key Components

| File | Purpose |
|------|---------|
| `Python/Server_AGI.py` | Main Engine — Concurrent Autonomy, Risk Polling, and Socket Server |
| `Python/hybrid_brain.py` | RL Executor — PPO-first policy with LSTM volatility gating, deadzones, and Canary risk scaling |
| `Python/autonomy_loop.py` | Orchestrator — Manages the Train → Evaluate → Canary → Promote/Rollback lifecycle |
| `Python/model_registry.py` | Ledger — Champion/Canary versioning with `save_candidate()` and `evaluate_and_stage_canary()` |
| `Python/data_feed.py` | High-fidelity data handler with FX volume proxies, Yahoo Finance fallback, and MT5 integration |
| `Python/agi_brain.py` | LSTM SmartAGI — 3-layer LSTM for volatility regime classification (LOW/MED/HIGH) |
| `Python/risk_engine.py` | Risk guardrails — Daily loss limits, trade caps, drawdown kill switch |
| `Python/mt5_executor.py` | Trade execution — Live MT5 on Windows, automatic dry-run logging on macOS |
| `Python/backtester.py` | Uses TradingEnv to replay PPO on historical data for promotion gating |
| `training/train_drl.py` | Joint PPO+LSTM training with curriculum learning and gradient diagnostics |
| `training/train_lstm.py` | Standalone LSTM volatility classifier training |
| `drl/trading_env.py` | Gymnasium env — Continuous position management with commission + spread modeling |
| `drl/lstm_feature_extractor.py` | SB3 custom feature extractor for joint LSTM-PPO gradient flow |
| `drl/ppo_agent.py` | Standalone PPO inference helper |
| `evaluation/walk_forward_test.py` | Out-of-sample walk-forward evaluation |
| `analysis/gradient_flow_analyzer.py` | Real-time LSTM gradient diagnostics via TensorBoard |
| `agents/grok_trading_agent.py` | Grok self-improver — reads telemetry and suggests hyperparameter patches via xAI API |

## Quick Start

### macOS Development (Dry-Run Mode)

```bash
# 1. Create virtual environment
python3 -m venv .venv && source .venv/bin/activate

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run smoke test
python smoke_test.py

# 4. Train LSTM (volatility classifier)
python -m training.train_lstm --epochs 50

# 5. Train Joint LSTM-PPO
python -m training.train_drl

# 6. Start server (dry-run, no MT5)
python -m Python.Server_AGI
```

### Windows VPS Deployment (Live Trading)

1. **Setup Env Vars**:
   ```powershell
   $env:AGI_TOKEN="your_secure_token"
   $env:AGI_AUTONOMY_AUTO_CANARY="true"
   $env:AGI_PNL_POLL="true"
   $env:TELEGRAM_BOT_TOKEN="your_telegram_token"
   $env:TELEGRAM_CHAT_ID="your_chat_id"
   ```

2. **Boot the Server**:
   ```powershell
   python -m Python.Server_AGI --live
   ```

3. **Start the n8n Orchestrator** (separate terminal):
   ```powershell
   npm install -g n8n
   $env:NODES_EXCLUDE="[]"
   n8n start
   ```

4. **Monitor Autonomy**:
   Check `logs/ppo_training.log` or the Console for Model Promotion signals.

### Docker

```bash
docker build -t cautious-giggle .
docker run -p 9090:9090 \
  -e AGI_TOKEN=your_token \
  -e TELEGRAM_BOT_TOKEN=your_token \
  -e TELEGRAM_CHAT_ID=your_id \
  cautious-giggle
```

## Model Lifecycle

```
train_drl.py / train_lstm.py
        │
        ▼
   Candidate (models/registry/candidates/<timestamp>/)
        │
        ▼ evaluate_and_stage_canary() or AutonomyLoop
   Canary (25% lot size via CANARY_LOT_MULT)
        │
        ▼ ≥10 trades + PnL ≥ 0 → promote | loss/DD exceeded → rollback
   Champion (full position sizing)
```

## Risk Management

- **Canary Mode**: New models trade with 25% risk (configurable via `CANARY_LOT_MULT`).
- **Kill Switch**: Realized PnL polling from MT5 triggers instant halts if daily loss limits are hit.
- **Max Drawdown**: Configurable per-model and global limits.
- **Trade Cap**: Hard daily trade limit (default: 20).
- **Error Halt**: 3 consecutive execution errors trigger automatic trading halt.

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AGI_HOST` | `127.0.0.1` | Socket server bind address |
| `AGI_PORT` | `9090` | Socket server port |
| `AGI_TOKEN` | _(empty)_ | Auth token for socket commands |
| `AGI_AUTONOMY_INTERVAL_SEC` | `3600` | Autonomy loop check interval |
| `AGI_AUTONOMY_TRAIN` | `false` | Enable nightly retraining |
| `AGI_AUTONOMY_AUTO_CANARY` | `true` | Auto-stage winning candidates |
| `CANARY_MIN_TRADES` | `10` | Min trades before canary promotion |
| `CANARY_MAX_LOSS` | `75` | Max canary loss before rollback ($) |
| `CANARY_MAX_DD` | `0.12` | Max canary drawdown before rollback |
| `CANARY_LOT_MULT` | `0.25` | Canary position size multiplier |
| `TELEGRAM_BOT_TOKEN` | — | Telegram bot token |
| `TELEGRAM_CHAT_ID` | — | Telegram chat ID |
| `XAI_API_KEY` | — | xAI API key for Grok agent |

---
**Risk warning:** For simulation/education only. Trade at your own risk.
