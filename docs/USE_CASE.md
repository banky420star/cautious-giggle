## Example Use Case

1. Operator starts `Server_AGI` in live mode.
2. System pulls live bars from MT5 for configured symbols/timeframe.
3. LSTM predicts regime and PPO proposes exposure.
4. Hybrid brain blends outputs and sends intent to risk engine.
5. Risk engine enforces drawdown, exposure, and margin constraints.
6. MT5 executor places/updates orders.
7. Audit/trade logs update and Telegram broadcasts read-only events.
8. Nightly cycle trains challengers and evaluates promotion gates.
