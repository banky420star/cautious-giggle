# Incident Response — cautious-giggle

## error_halt = True

**Cause**: 3+ consecutive order execution errors.

**Detection**: Telegram alert "RISK_HALT_SET reason=consecutive_errors". Server log: `RISK_HALT_SET reason=consecutive_errors count=3`.

**Impact**: All trading stops. The halt persists across day rolls and restarts.

**Action**:
1. Check MT5 terminal — is it connected? Is the market open?
2. Check `logs/server.log` for the specific order errors
3. Fix the root cause (e.g., reconnect MT5, fix symbol config)
4. Restart Server_AGI — this creates a fresh RiskEngine. The saved state will restore `error_halt=True`, so you must either:
   - Delete `logs/risk_engine_state.json` before restarting, OR
   - Manually clear: `python -c "import json,os; d=json.load(open('logs/risk_engine_state.json')); d['error_halt']=False; d['halt']=False; d['error_count']=0; json.dump(d, open('logs/risk_engine_state.json','w'), indent=2)"`

## max_daily_loss Triggered

**Cause**: Realized P&L for the day exceeded the configured `risk.max_daily_loss`.

**Detection**: Telegram alert. Log: `RISK_HALT_SET reason=daily_loss`.

**Impact**: No new trades until UTC midnight (auto-resets).

**Action**:
1. Check open positions in MT5 — they remain open
2. To manually close positions, use MT5 terminal directly
3. Trading resumes automatically at next UTC day roll
4. If you need to resume trading immediately: restart Server_AGI and delete `logs/risk_engine_state.json` (NOT recommended)

## MT5 Disconnected Mid-Session

**Cause**: Network issue, MT5 terminal crash, broker maintenance.

**Detection**: `mt5.terminal_info()` returns None in heartbeat. Orders fail with error.

**Impact**: The main loop continues but all MT5 calls return None/fail. Open positions remain on the broker side.

**Action**:
1. Check MT5 terminal on the VPS — restart if needed
2. Server_AGI will reconnect on next loop iteration if MT5 comes back
3. If MT5 stays down, `record_error()` will eventually trigger `error_halt`

## Canary Rollback Fired

**Cause**: Canary model's realized P&L dropped below `-CANARY_MAX_LOSS` or drawdown exceeded `CANARY_MAX_DD`.

**Detection**: Telegram alert "Canary rollback [symbol]".

**Impact**: The canary model is removed and the previous champion is restored. Trading continues with the champion.

**Action**:
1. Check `models/registry/active.json` — canary should be null for that symbol
2. Review canary metrics in the audit log
3. If you want to stage a new candidate: run `python tools/champion_cycle.py`
