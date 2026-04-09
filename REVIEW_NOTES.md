# Executor Discipline Phase 1 - Review Notes

## Changes Applied (2026-04-09)

### 1. Executor Discipline (`Python/mt5_executor.py`)
- **Same-side hold logic**: Profitable/breakeven positions are held when target reduction < 25%
- **Partial close support**: Closes smallest positions first instead of close-all + reopen
- **25% rebalance threshold**: Small target changes no longer trigger churn
- **Trade recording gated on execution**: `risk.record_trade()` only on actual fills

### 2. PPO Diagnostics (`Python/hybrid_brain.py`)
- Fixed `PPO_DIAG %s | %s` logging bug (was printing literal `%s`)
- PPO_DIAG now logs: raw_action, decoded_target, min_abs, source, obs_dim
- Added `ppo_skip_reason()` method: returns "ppo_disabled", "no_model_loaded", "disabled_after_errors", "below_threshold"

### 3. Decision Hierarchy (`Python/Server_AGI.py`)
- Trade memory gate: Scales exposure 25-75% when profit_factor < 1.15 or expectancy < -1.0
- PPO-first fallback: When PPO unavailable, re-blends with boosted AGI weight (50%) + 50% safety discount
- DECISION log now includes: ppo_skip, scenario, memory metrics
- ACTION log now includes: wanted, vetoed, memory, scenario
- EXEC_TRACE log shows supervisor decision + fallback mode
- Increased live bar count from 220 to 2000 for proper HTF feature calculation
- Wired `supervisor.record_trade_result()` into close-deal loop

### 4. Risk Supervisor (`Python/risk_supervisor.py`)
- Loss-streak throttle: 5 consecutive losses -> 10min symbol cooldown
- Drawdown-based size reduction: 5%->50%, 10%->25%, 15%->halt
- Max 10 adjustments per symbol per hour
- Exposure limits multiplied by drawdown factor

### 5. Trade History API (`tools/project_status_ui.py`)
- `GET /api/trades` - trade history with symbol/side/days/limit/offset filtering
- `GET /api/decisions` - recent DECISION/EXEC_TRACE/ACTION log lines
- `GET /api/learning` - trade learning metrics

### 6. Evaluation Pipeline (`config.yaml.example`, `tools/champion_cycle.py`)
- Loosened eval gates: max_drawdown 0.20, min_sharpe 0.05, min_return 0.001
- Auto-promote first passing candidate when no champion exists
- score_margin 0.0, min_pass_rate 0.5, min_forward_win_rate 0.50

### 7. Professional Launcher (`launch_agi_pro.ps1`)
- Pre-flight checks (venv, config, session)
- Starts dashboard + trading server in background
- Opens browser, health monitoring, clean shutdown handler

### 8. Training Pipeline Fixes (Phase 2)
- **1D action space**: `shape=(6,)` → `shape=(1,)` in trading_env.py. `action_version` = "target_exposure_v1"
- **1D decode defaults**: Added TP=1.2%, SL=0.8% defaults so trades don't close immediately
- **Reward rebalanced**: cost_penalty 5.0→1.5, churn_penalty 0.5→0.1, growth 8.0→10.0, sharpe_bonus 1.0→1.5
- **Spread realistic**: Default 2bps→12bps; BTC min 15bps, XAU min 10bps (per-symbol)
- **Train/eval split**: Forward-only 80/20 split (agents implementing in train_drl.py)
- **Feature switch**: `ultimate_150` → `engineered_v2` everywhere (config, training, dreamer)
- **Timesteps**: 500k → 2M
- **Network size**: engineered_v2 uses [256,128] net_arch + 128-dim extractor (was [512,256] + 256)

### 9. Block Removal (Phase 2)
- **Fresh start on boot**: `risk.reset_halt()` + `supervisor.clear_blocks()` at startup, peak equity reset to current
- **RiskEngine**: Added `reset_halt(current_equity)` method; error threshold 3→10
- **RiskSupervisor**: Added `clear_blocks()` method; drawdown tiers loosened (5/10/15% → 8/12/18%)
- **Max drawdown halt**: 8% → 15% (calculated from current equity, not historical peak)
- **Spread guard**: 25bps → 35bps (BTC commonly exceeds 25bps)
- **Trade interval**: 45s → 30s
- **Loss streak**: threshold 5→7, cooldown 10→5 min
- **Adjustments/hour**: 10 → 20
- **Memory gate**: Min trades 20→30, returns 1.0 (not 0.0) when insufficient data
- **Memory thresholds**: PF trigger 1.15→0.9, expectancy 0.0→-1.0, loss streak 3→6
- **Memory penalties**: Softened from 25/50/75% to 40/60/80%
- **Lookback**: 30 days → 7 days (old losing trades drop off fast)
- **Decision flow fix**: Memory gate now applies AFTER PPO fallback reblend (was being overwritten)
- **Canary gates**: BTCUSDm max_drawdown 8%→15%, min_trades 60→30
- **Evaluator defaults**: Hardcoded impossible gates replaced with achievable ones

### 10. Feature Pipeline Consistency
- **Dreamer default**: Changed from `ULTIMATE_150` to `ENGINEERED_V2` in dreamer_policy.py

---

## Remaining Roadmap

### HIGH (next)
1. **Refactor training env to continuous exposure** - Currently uses discrete trade open/close; should adjust position every step
2. **Add volatility-normalized position sizing** - 0.20 lots BTC != 0.20 lots XAU in USD risk
3. **Fix Dreamer discrete/continuous mismatch** - Dreamer outputs {-1,0,1}, PPO outputs continuous [-1,1]

### MEDIUM (first month)
4. **Per-hour and per-regime performance tracking** - learning loop should identify weak time windows
5. **Walk-forward validation** - proper rolling window backtesting
6. **Add dry-run/paper mode** - validate before risking real money

### NICE-TO-HAVE
7. Remove ensemble dead code (disabled but complex)
8. Add canary live validation before champion promotion
9. Reduce profitability.jsonl per-step disk writes during training
