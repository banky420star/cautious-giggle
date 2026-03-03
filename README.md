# cautious-giggle — Autonomous AGI Trading Hedge Fund (2026)

**Full autonomous Observe → Learn → Validate → Deploy → Trade loop**

## 🤖 The Self-Evolving Loop
This system is designed for total autonomy on a Windows VPS + MetaTrader 5:
1. **Observe**: Fetches high-fidelity OHLCV data via MT5 or Yahoo Finance (with Volume Proxy).
2. **Learn**: Nightly training cycles refine PPO and LSTM models on combined multi-asset datasets.
3. **Validate**: The `AutonomyLoop` backtests candidates. Winning candidates are staged as **Canary**.
4. **Deploy**: The `HybridBrain` hot-swaps active models in real-time.
5. **Trade**: Market execution via MT5 with spread-aware deadzones and risk guardrails.
6. **Monitor**: Real-time PnL polling rollback canaries if they misbehave.

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
| `Python/hybrid_brain.py` | RL Executor — PPO-first policy with deadzones and Canary risk scaling |
| `Python/autonomy_loop.py` | Orchestrator — Manages the Train -> Evaluate -> Promote lifecycle |
| `Python/model_registry.py` | Ledger — Manages Champion/Canary versioning and hot-swaps |
| `Python/data_feed.py` | High-fidelity data handler with FX volume proxies and MT5 integration |
| `training/train_drl.py` | DRL Trainer — Joint PPO+LSTM training with curriculum learning |

## Quick Start (VPS Deployment)

1. **Setup Env Vars**:
   ```powershell
   $env:AGI_TOKEN="your_secure_token"
   $env:AGI_AUTONOMY_AUTO_CANARY="true"
   $env:AGI_PNL_POLL="true"
   ```

2. **Boot the Server**:
   ```powershell
   python -m Python.Server_AGI --live
   ```

3. **Start the n8n Orchestrator**:
   By default n8n blocks terminal commands. You MUST unlock it on Windows VPS. Open a second PowerShell:
   ```powershell
   npm install -g n8n
   $env:NODES_EXCLUDE="[]"
   n8n start
   ```

4. **Monitor Autonomy**:
   Check `logs/ppo_training.log` or the Console for Model Promotion signals.


## Live Command API (n8n/CLI)
The socket server on `AGI_HOST:AGI_PORT` accepts one-line JSON requests:
- `health`
- `risk_status`
- `predict`
- `trade`

`predict` runs the full hybrid decision cycle **without execution**, while `trade` runs decision + MT5 execution:
1. Pull latest MT5 bars
2. Classify volatility with LSTM (`LOW/MED/HIGH_VOLATILITY`)
3. Generate continuous exposure with PPO
4. Apply dynamic lot multiplier + SL/TP points
5. Reconcile exposure in MT5 using market or optional pending limit orders

Control execution via env vars:
- `AGI_USE_LIMIT_ORDERS=true|false`
- `AGI_LIMIT_OFFSET_POINTS=30`
- `AGI_PPO_DEADZONE=0.08`
- `AGI_AUTO_DYNAMIC_ENTRY=true|false`
- `AGI_SOCKET_RETRIES=3` (for `agi_n8n_bridge.py`)

## Risk Management (Vitals)

- **Canary Mode**: New models trade with 25% risk (configurable via `CANARY_LOT_MULT`).
- **Kill Switch**: Realized PnL polling from MT5 triggers instant halts if daily loss limits are hit.
- **Cooldowns**: Enforced 45s cooldowns prevent position flip-flopping due to noise.

---
**Risk warning:** For simulation/education only. Trade at your own risk.


**Production note:** run at least 2–4 weeks demo forward validation before live capital; see `PRODUCTION_READINESS.md`.
