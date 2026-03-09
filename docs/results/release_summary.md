# Release Summary

**Timestamp:** "2026-03-09T01:59:07.824800"

## Champion Overview
- Path: None
- Metadata: {}

## Canary Overview
- Path: C:\windows\system32\cautious-giggle\models\registry\candidates\20260308_073222
- Metadata: {"path": "C:\\windows\\system32\\cautious-giggle\\models\\registry\\candidates\\20260308_073222", "type": "ppo", "symbol": "XAUUSDm", "symbols": ["XAUUSDm"], "timeframe": "5m", "period": "90d", "candles": 100000, "timesteps": 1000, "data_source": "mt5", "feature_set_version": "engineered_v2", "normalization_version": "vecnorm_v1", "reward": {}, "reward_version": "v2_risk_adjusted", "ppo_params": {"learning_rate": 0.0001, "n_steps": 4096, "batch_size": 512, "n_epochs": 10, "gamma": 0.995, "gae_lambda": 0.95, "clip_range": 0.2, "ent_coef": 0.005, "vf_coef": 0.5, "max_grad_norm": 0.5, "target_kl": 0.01, "use_sde": true, "sde_sample_freq": 4}, "windows": {"train": "90d", "validate": "120d", "forward": []}, "source": "EvalCallback best_model.zip + matching VecNormalize", "date": "2026-03-08T06:32:22.245169+00:00"}

## Profitability recent snapshots
- `2026-03-09T01:48:56.743915` equity=10000.00 position=0.0 profitability={"equity_curve": [10000.0], "trade_metrics": {"exit_type": "tp", "profit": 0.0, "exit_quality_reward": 1.0, "trailing_efficiency": 0.0, "breakeven_reward": 0.0, "max_favorable": 0.0, "max_adverse": 0.0, "trailing_moves": 0}}
