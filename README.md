# cautious-giggle — Production AGI Trading Hedge Fund (2026)

**Full autonomous self-learning MT5 trading system**

## Architecture

```
┌──────────────┐    every 5 min    ┌──────────────┐
│     n8n      │ ───────────────►  │  agi_n8n     │
│  Orchestrator│                   │  _bridge.py  │
│  (5678)      │ ◄──── JSON ─────  │              │
└──────────────┘                   └──────┬───────┘
                                          │ socket :9090
                                   ┌──────▼───────┐
                                   │  Server_AGI  │
                                   │   .py        │
                                   └──┬───┬───┬───┘
                                      │   │   │
                          ┌───────────┘   │   └───────────┐
                          ▼               ▼               ▼
                   ┌────────────┐  ┌────────────┐  ┌────────────┐
                   │  agi_brain │  │ risk_engine│  │  telegram  │
                   │  (LSTM)    │  │ (sizing,   │  │  _alerts   │
                   │  PyTorch   │  │  DD, SL/TP)│  │            │
                   └──────┬─────┘  └────────────┘  └────────────┘
                          │
                   ┌──────▼─────┐
                   │  data_feed │
                   │  (Yahoo    │
                   │   Finance) │
                   └────────────┘
```

## Components

| File | Purpose |
|------|---------|
| `Python/Server_AGI.py` | Socket server — receives commands, runs predictions, enforces risk |
| `Python/agi_brain.py` | LSTM neural network (3-layer, MPS accelerated) |
| `Python/agi_n8n_bridge.py` | CLI bridge between n8n and the AGI server |
| `Python/data_feed.py` | Real-time data from Yahoo Finance with caching |
| `Python/risk_engine.py` | Position sizing, drawdown limits, SL/TP, daily caps |
| `Python/backtester.py` | VectorBT backtester on real historical data |
| `drl/trading_env.py` | Gymnasium environment for DRL training |
| `drl/ppo_agent.py` | PPO self-learning agent (Stable-Baselines3) |
| `training/train_lstm.py` | LSTM training on multi-symbol real data |
| `training/train_drl.py` | DRL PPO training on real market data |
| `alerts/telegram_alerts.py` | Telegram notifications via HTTP API |
| `n8n-workflow/mt5-autonomous.json` | n8n workflow (import into n8n UI) |
| `config.yaml` | Central configuration (symbols, risk, Telegram) |

## Quick Docker start (recommended)

```bash
git clone https://github.com/banky420star/cautious-giggle.git
cd cautious-giggle
docker compose up --build
```

This launches:
- **AGI Server** on port `9090`
- **n8n** on port `5678` (open http://localhost:5678)
- **Redis** on port `6379`

## Training

```bash
# Train LSTM brain (all symbols)
python training/train_lstm.py

# Train DRL PPO agent
python training/train_drl.py

# Run backtester
python Python/backtester.py
```

Full manual setup → `SETUP.md`

**Risk warning:** For simulation/education only. Trade at your own risk.
