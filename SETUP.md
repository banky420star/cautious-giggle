# SETUP GUIDE — cautious-giggle AGI Trading System

## Option 1: Docker (easiest — recommended)

```bash
git clone https://github.com/banky420star/cautious-giggle.git
cd cautious-giggle
docker compose up --build
```

This starts the AGI server (`:9090`), n8n (`:5678`), and Redis (`:6379`).

Open **http://localhost:5678** → import `n8n-workflow/mt5-autonomous.json` → activate the workflow.

---

## Option 2: Manual Setup (Mac)

### Prerequisites
- Python 3.11+ (Python 3.12 recommended)
- pip / venv

### 1. Clone & install
```bash
git clone https://github.com/banky420star/cautious-giggle.git
cd cautious-giggle
python3 -m venv venv && source venv/bin/activate
pip install -r requirements.txt
```

### 2. Configure
Edit `config.yaml`:
```yaml
trading:
  symbols: ["EURUSD", "GBPUSD", "XAUUSD"]
  timeframe: "M5"
  risk_percent: 1.0
  max_drawdown: 10.0
  confidence_threshold: 0.85

telegram:
  token: "YOUR_BOT_TOKEN"
  chat_id: "YOUR_CHAT_ID"

drl:
  total_timesteps: 500000
  model_path: "models/ppo_trading.zip"
```

### 3. Train models (recommended before live)
```bash
# LSTM brain
python training/train_lstm.py

# DRL PPO agent
python training/train_drl.py
```

### 4. Run the AGI server
```bash
python Python/Server_AGI.py
```
Server listens on `127.0.0.1:9090`.

### 5. Start n8n
```bash
npm install -g n8n
n8n start
```
Open http://localhost:5678 → import `n8n-workflow/mt5-autonomous.json` → activate.

### 6. Backtest (optional)
```bash
python Python/backtester.py
```

### 7. Test the bridge manually
```bash
python Python/agi_n8n_bridge.py ANALYZE EURUSD
python Python/agi_n8n_bridge.py TRADE GBPUSD
```

---

## File Structure
```
cautious-giggle/
├── Python/
│   ├── Server_AGI.py        # Socket server
│   ├── agi_brain.py          # LSTM model
│   ├── agi_n8n_bridge.py     # n8n CLI bridge
│   ├── data_feed.py          # Yahoo Finance data
│   ├── risk_engine.py        # Risk management
│   └── backtester.py         # VectorBT backtester
├── drl/
│   ├── trading_env.py        # Gymnasium environment
│   └── ppo_agent.py          # PPO agent
├── training/
│   ├── train_lstm.py         # LSTM training
│   └── train_drl.py          # DRL training
├── alerts/
│   └── telegram_alerts.py    # Telegram notifications
├── n8n-workflow/
│   └── mt5-autonomous.json   # n8n workflow
├── config.yaml               # Configuration
├── requirements.txt          # Python dependencies
├── Dockerfile                # Docker image
├── docker-compose.yml        # Docker services
└── README.md                 # Overview
```

---

## Telegram Setup
1. Create a bot via [@BotFather](https://t.me/BotFather)
2. Get your chat ID from [@userinfobot](https://t.me/userinfobot)
3. Put both in `config.yaml`

---

**Risk warning:** For simulation/education only. Trade at your own risk.
