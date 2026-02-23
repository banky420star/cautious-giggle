# SETUP GUIDE (Mac + Docker)

## Docker (easiest — recommended)
docker compose up --build

## Manual Mac
1. Install MT5 from metatrader5.com
2. Download Market Replay tools from mql5.com/en/articles/12828
3. `cd cautious-giggle`
4. `python3 -m venv venv && source venv/bin/activate && pip install -r requirements.txt`
5. Fill `config.yaml` with your Telegram token/chat_id
6. `python Python/Server_AGI.py` (keep running)
7. `n8n start` → import n8n-workflow/mt5-autonomous.json → activate
8. Open control.xlsx + MT5 → click "Start AGI"

DRL training: `python training/train_drl.py`
Backtest: `python Python/backtester.py`
