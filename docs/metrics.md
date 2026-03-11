# Metrics & Trading Narrative

## Architecture & flow
- Market data arrives through the MT5 connector (MetaTrader5.copy_rates_from_pos and Python/mt5_executor) and is transformed into symbol-specific LSTM context features via 	raining/train_lstm.py. Those curated scalers and classifiers feed into the shared PPO trainer in 	raining/train_drl.py, which writes candidates into models/registry/candidates.
- The autonomy loop (Python/autonomy_loop.py) evaluates each candidate with model_evaluator.py, gates promotion with Python.Server_AGI.py's risk/risk cooldown logic, and records the champion/canary pair inside models/registry/active.json so the live server knows which artifacts to load.
- The trading server (Python.Server_AGI.py) fuses LSTM and PPO signals, enforces exposure limits, and logs every decision to logs/server.log while the control surface (	ools/project_status_ui.py) mirrors status, WebSocket telemetry, and Telegram alerts.

## Profitability snapshot
- 	ools/release_summary.py now records the current champion (models/registry/candidates/20260308_073222) and canary (.../20260308_215837), both PPO models operating on XAUUSDm 5-minute data, engineered_v2 features, vecnorm_v1 normalization, and PPO risk-adjusted reward settings (learning_rate=0.0001, 
_steps=4096, gamma=0.995, clip_range=0.2, ent_coef=0.005).
- The latest five profitability snapshots (all 2026-03-10T13:46:28.*) show a short position holding roughly half the allowed exposure (position ˜ -0.499), equity oscillating around 685 with a repeated SL exit (profit ˜ -0.0068), perfect trailing efficiency (1.0), and the same max adverse/favorable movement each tick (max_adverse=0.00512, max_favorable=0.00144). These entries live in docs/results/release_summary.md alongside the champion/canary metadata, and they can be plotted or linked into dashboards via the profitability.jsonl stream.
- The profitability story therefore ties the promoted candidate to concrete performance metrics: the champion/canary pair share the same hyperparameters and reward version, so the release summary is the single source of truth for  what is live and how profitable has it been recently.

## Observability & learning
- Tail logs/server.log/logs/trade_events.jsonl to see each new decision and event, and review logs/ppo_training.log or logs/audit_events.jsonl when you need training runs or autonomy evaluations.
- The UI logs (logs/ui_stdout.log/ui_stderr.log) and the WebSocket endpoint (	ools/project_status_ui.py at http://127.0.0.1:8088) create the operational view that bridges monitoring with trading reality.

## Next steps
1. Run python tools/profit_sweep.py to map out sensitivity between thresholds, blend ratios, and the profitability snapshots already captured.
2. Keep docs/results/release_summary.md fresh with python tools/release_summary.py (it now captures the last five equity snapshots for fast evidence bundling).
3. When sharing the trading story, refer to this doc plus the release summary so reviewers can see how MT5 candles ? LSTM features ? PPO policies produced the configured champion/canary and how that duo has behaved in the last snapshot.
