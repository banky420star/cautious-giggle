#!/bin/bash
set -euo pipefail

load_env_file() {
    local env_file="$1"
    [ -f "$env_file" ] || return 0
    while IFS= read -r line || [ -n "$line" ]; do
        line="${line%%#*}"
        line="${line%%$'\r'}"
        line="${line#"${line%%[![:space:]]*}"}"
        line="${line%"${line##*[![:space:]]}"}"
        [ -z "$line" ] && continue
        case "$line" in
            *=*)
                key="${line%%=*}"
                value="${line#*=}"
                key="${key%"${key##*[![:space:]]}"}"
                key="${key#"${key%%[![:space:]]*}"}"
                if [ -n "$key" ] && [ -z "${!key+x}" ]; then
                    export "$key=$value"
                fi
                ;;
        esac
    done < "$env_file"
}

# Load local environment overrides if present.
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
load_env_file "$SCRIPT_DIR/.env"

if [ -z "${AGI_TOKEN:-}" ]; then
    echo "FATAL: AGI_TOKEN must be set in the environment or in $SCRIPT_DIR/.env. See .env.example."
    exit 1
fi

# Network Config
export AGI_HOST="${AGI_HOST:-0.0.0.0}"
export AGI_PORT="${AGI_PORT:-9090}"

# Autonomy & Polling
export AGI_AUTONOMY_AUTO_CANARY="${AGI_AUTONOMY_AUTO_CANARY:-true}"
export AGI_PNL_POLL="${AGI_PNL_POLL:-true}"

# Risk & Cooldowns
export AGI_COOLDOWN_SEC="${AGI_COOLDOWN_SEC:-45}"
export AGI_MIN_HOLD_SEC="${AGI_MIN_HOLD_SEC:-120}"
export CANARY_LOT_MULT="${CANARY_LOT_MULT:-0.25}"

# Deadzones (Spread / Noise Filters)
export AGI_DZ_EURUSD="${AGI_DZ_EURUSD:-0.18}"
export AGI_DZ_GBPUSD="${AGI_DZ_GBPUSD:-0.20}"
export AGI_DZ_XAUUSD="${AGI_DZ_XAUUSD:-0.22}"

echo "Starting Grok AGI Server on Port $AGI_PORT with AGI_TOKEN sourced from env..."

# Pre-flight checks

# 1. Config check
if [ ! -f "$SCRIPT_DIR/config.yaml" ]; then
    echo "FATAL: config.yaml not found. Copy config.yaml.example and configure."
    exit 1
fi

# 2. Models directory
if [ ! -d "$SCRIPT_DIR/models" ]; then
    echo "WARNING: models/ directory not found - creating empty. No models available."
    mkdir -p "$SCRIPT_DIR/models"
fi

# 3. Stale lock check
LOCK_FILE="$SCRIPT_DIR/.tmp/server_agi.lock"
if [ -f "$LOCK_FILE" ]; then
    LOCK_PID=$(cat "$LOCK_FILE" 2>/dev/null)
    if kill -0 "$LOCK_PID" 2>/dev/null; then
        echo "FATAL: Server_AGI is already running (PID $LOCK_PID). Kill it first or remove $LOCK_FILE."
        exit 1
    else
        echo "WARNING: Removing stale lock file (PID $LOCK_PID is dead)."
        rm -f "$LOCK_FILE"
    fi
fi

# If you prefer to use your python venv, uncomment the next line:
# source venv/bin/activate

python -c "from Python.config_utils import load_project_config; load_project_config(r'$SCRIPT_DIR', live_mode=True)"

python -m Python.Server_AGI --live
