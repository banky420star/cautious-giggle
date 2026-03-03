# Production Readiness Gate (MT5 Live Trading)

This project is **not production-ready yet** by default. Use the checklist below as a go-live gate.

## Must-pass before live capital
1. **Config integrity**
   - `config.yaml` contains `trading`, `risk`, and `drl` sections.
   - `risk.max_daily_loss`, `risk.max_daily_trades`, `risk.max_lots` are set to broker-safe values.

2. **Auth/security**
   - `AGI_TOKEN` is set in environment (do not run unsigned).
   - VPS firewall restricts `AGI_PORT` access to trusted hosts.

3. **Model validity**
   - LSTM model/scaler exist and load correctly.
   - PPO model + matching `VecNormalize` stats exist and load correctly.
   - Registry can hot-swap canary/champion with successful inference continuity.

4. **Execution safety**
   - Verify market execution and pending limit execution both work on demo.
   - Verify SL/TP values are accepted by broker symbol constraints.
   - Confirm position reconciliation does not over-size under rapid signal changes.

5. **Runtime reliability**
   - Server stays up 24h+ without memory/thread growth issues.
   - Socket API handles malformed/partial input without crash.
   - Telegram heartbeat and risk status endpoints remain responsive.

6. **Risk controls**
   - Daily loss halt trigger tested end-to-end.
   - Max trades/day trigger tested end-to-end.
   - MT5 error threshold halt tested with forced order failures.

7. **Backtest/forward test evidence**
   - Walk-forward test period and out-of-sample report reviewed.
   - Minimum 2–4 weeks forward demo test on target broker/spread/session.

## Practical timeline
If all engineering work is already merged:
- **Hardening + bugfix cycle:** 2–4 days
- **Demo forward validation:** 2–4 weeks
- **Go-live with small capital:** after passing all gates above

Do not skip the forward-demo period; training metrics alone are insufficient for live deployment.
