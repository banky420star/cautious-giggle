#!/bin/zsh
# "Cautious Giggle" app icon: start the full package and open the dashboard.
#   bot (trading)  ·  graduation manager (training autopilot)  ·  web dashboard  ·  terminal view
# Each piece is started only if it isn't already running (checked by its real process, not a pid file).
ROOT="${0:A:h:h:h}"   # the project folder (this script lives in scripts/mac)
note() { osascript -e "display notification \"$1\" with title \"Cautious Giggle\"" >/dev/null 2>&1; }
if [ ! -d "$ROOT" ]; then
  osascript -e 'display dialog "The Cautious Giggle folder is missing." & return & return & "If it lives on an external drive, plug it in, then open Cautious Giggle again." with title "Cautious Giggle" buttons {"OK"} default button 1 with icon caution' >/dev/null 2>&1
  exit 1
fi
cd "$ROOT" || exit 1
source scripts/mac/account_env.sh || { osascript -e 'display dialog "Missing Exness account settings (~/.exness/env)." with title "Cautious Giggle" buttons {"OK"}' >/dev/null; exit 1; }
export PYTHONPATH="$ROOT"
mkdir -p logs .tmp
running() { pgrep -f "$1" >/dev/null 2>&1; }
detach() { nohup .venv/bin/python -c "import os,sys; os.setsid(); os.execvp(sys.argv[1], sys.argv[1:])" .venv/bin/python "$@" >/dev/null 2>&1 < /dev/null & }

# 1) trading bot: the Python process itself, via its lock-file pid (the restart loop alone is not enough)
bot_alive() {
  local pid; pid=$(cat .tmp/server_agi.lock 2>/dev/null) || return 1
  [ -n "$pid" ] && ps -o command= -p "$pid" 2>/dev/null | grep -q "Server_AGI.py"
}
if ! bot_alive; then
  scripts/mac/stop_bot.sh >/dev/null 2>&1
  [ -f .tmp/supervisor.pid ] && kill "$(cat .tmp/supervisor.pid)" 2>/dev/null
  rm -f .tmp/stop .tmp/server_agi.lock
  scripts/mac/start_bot.sh >/dev/null 2>&1
  note "Trading bot started"
fi

# 2) training autopilot (trains each market, runs its exam, retries with the next reward scenario)
if ! running "tools/graduation.py run"; then
  detach tools/graduation.py run
  note "Training autopilot started"
fi

# 3) terminal view (second screen)
if ! running "tools/simple_tui.py"; then
  osascript -e "tell application \"Terminal\" to do script \"cd '$ROOT' && source scripts/mac/account_env.sh && PYTHONPATH=\\\"\$PWD\\\" .venv/bin/python tools/simple_tui.py\"" >/dev/null 2>&1
fi

# 4) web dashboard in its own window
exec scripts/mac/web.sh
