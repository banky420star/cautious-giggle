#!/bin/zsh
# Live terminal dashboard (read-only). Ctrl-C to close; the bot keeps running.
cd "$(dirname "$0")/../.." || exit 1
source scripts/mac/account_env.sh
exec .venv/bin/python tools/monitor_tui.py
