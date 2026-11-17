#!/bin/zsh
# Live terminal dashboard (read-only). Ctrl-C to close; the bot keeps running.
cd "$(dirname "$0")/../.." || exit 1
source ~/.exness/env
exec .venv/bin/python tools/monitor_tui.py
