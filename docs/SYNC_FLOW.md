# Data & Sync Flow

Cautious Giggle is built to keep live trading, training, and observability in sync.

1. **Data acquisition**  
   - `Python/data_feed.py` pulls MT5 candles for each symbol with `copy_rates_from_pos`.  
   - Candle windows (default 60d at `M5`) power both training and live decision-making to avoid feature drift.
2. **Training synchronization**  
   - `training/train_lstm.py` and `training/train_drl.py` both read from the same data feed to ensure training uses identical sources.  
   - Config glue (`config.yaml`) defines `training.lstm_period`, `training.lstm_interval`, `drl.period`, `drl.interval`, `trading.symbols`, and `mt5` credentials so every training run sees the same environment.
3. **Registry sync**  
   - Training outputs live under `models/registry/candidates/<timestamp>`.  
   - The `model_registry` exposes `active.json` to the running server. When a new champion appears, `Server_AGI` hot-loads the updated scaler + PPO bundle path per symbol.  
   - Telegram and the dashboard read the same registry to show which model is champion/canary.
4. **Observability sync**  
   - `tools/project_status_ui.py` polls the trading server and the registry every 2 seconds and exposes `/api/status`.  
   - WebSocket clients (dashboard + controlling UI) receive updates whenever trade decisions, training toggles, or promotions occur.  
   - Telegram alerts share the same story by replaying heartbeats, training start/completion, and gate verdicts.
5. **Manual overrides**  
   - Control actions (`start_lstm`, `start_drl`, `run_cycle`) spawn isolated processes guarded against duplicates.  
   - Logs (`logs/audit_events.jsonl`, `logs/trade_events.jsonl`, `logs/ppo_training.log`) provide a timeline of every decision, ensuring manual or remote troubleshooting is aligned with what the next training cycle will see.
