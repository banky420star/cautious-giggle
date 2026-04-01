# Runbook — cautious-giggle Trading System

## Startup Sequence

1. Ensure `config.yaml` exists (copy from `config.yaml.example` if first run)
2. Set required environment variables (see `docs/secrets_management.md`)
3. Start Server_AGI:
   - Windows: `.\start_server.ps1`
   - Linux/Docker: `./start_server.sh` or `docker-compose up -d`
4. Verify Telegram sends "Trading engine initialized" message
5. (Optional) Start the dashboard: `python tools/project_status_ui.py`
   - Dashboard available at `http://127.0.0.1:8088`

## Shutdown Sequence

1. Send SIGTERM to the Server_AGI process (or Ctrl+C)
2. Server saves risk engine state and calls `mt5.shutdown()`
3. Telegram sends "Runtime exited" message
4. Lock file is automatically removed

## Daily Operations

- **Daily limit reset**: Happens automatically at UTC midnight. The risk engine calls `reset_daily()` which clears P&L counters and trade counts. Error halts are NOT auto-cleared.
- **Heartbeat**: Sent to Telegram every `AGI_HEARTBEAT_SEC` (default 600s). Includes equity, balance, positions, model state.

## Manual Champion Promotion

```bash
python -c "from Python.model_registry import ModelRegistry; r = ModelRegistry(); r.promote_canary_to_champion(symbol='XAUUSDm', force=True)"
```

## Manual Rollback to Champion

```bash
python -c "from Python.model_registry import ModelRegistry; r = ModelRegistry(); r.rollback_to_champion(symbol='XAUUSDm')"
```

## Check Canary State

```bash
python -c "import json; print(json.dumps(json.load(open('models/registry/active.json')), indent=2))"
```

## Verify Risk Engine State

```bash
python -c "import json; print(json.dumps(json.load(open('logs/risk_engine_state.json')), indent=2))"
```
