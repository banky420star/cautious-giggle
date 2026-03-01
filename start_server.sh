#!/bin/bash

# Network Config
export AGI_HOST="0.0.0.0"
export AGI_PORT="9090"
export AGI_TOKEN="fuckyou2/"

# Autonomy & Polling
export AGI_AUTONOMY_AUTO_CANARY="true"
export AGI_PNL_POLL="true"

# Risk & Cooldowns
export AGI_COOLDOWN_SEC="45"
export AGI_MIN_HOLD_SEC="120"
export CANARY_LOT_MULT="0.25"

# Deadzones (Spread / Noise Filters)
export AGI_DZ_EURUSD="0.18"
export AGI_DZ_GBPUSD="0.20"
export AGI_DZ_XAUUSD="0.22"

echo "🚀 Starting Grok AGI Server on Port $AGI_PORT with Token '$AGI_TOKEN' in LIVE mode..."

# If you prefer to use your python venv, uncomment the next line:
# source venv/bin/activate

python -m Python.Server_AGI --live
