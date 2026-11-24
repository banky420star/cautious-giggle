#!/bin/zsh
# Start the trading bot in the background (keeps running after you close Terminal).
# Auto-restarts if it crashes; caffeinate keeps the Mac awake while it runs.
cd "$(dirname "$0")/../.." || exit 1
source scripts/mac/account_env.sh || exit 1
export PYTHONPATH="$PWD"
mkdir -p logs .tmp
if [ -f .tmp/supervisor.pid ] && kill -0 "$(cat .tmp/supervisor.pid)" 2>/dev/null; then
  echo "bot already running (supervisor pid $(cat .tmp/supervisor.pid))"; exit 0
fi
ROOT="$PWD"
rm -f .tmp/stop   # before the loop starts: a leftover stop file would end the new loop at once
# Absolute paths + cd every round: if the drive remounts, a loop on relative paths loses its folder and
# never restarts the bot again (2026-11-24). The 30 s wait also covers the drive coming back.
nohup caffeinate -is zsh -c '
  ROOT="$1"
  while [ ! -f "$ROOT/.tmp/stop" ]; do
    if cd "$ROOT" 2>/dev/null && [ -x .venv/bin/python ]; then
      echo "$(date "+%F %T") starting Server_AGI" >> "$ROOT/logs/server_stdout.log"
      .venv/bin/python Python/Server_AGI.py >> "$ROOT/logs/server_stdout.log" 2>&1
      rc=$?
      [ -f "$ROOT/.tmp/stop" ] && break
      echo "$(date "+%F %T") Server_AGI exited ($rc), restarting in 30s" >> "$ROOT/logs/server_stdout.log"
    fi
    sleep 30
  done
  rm -f "$ROOT/.tmp/stop"
' _ "$ROOT" > /dev/null 2>&1 &
echo $! > .tmp/supervisor.pid
echo "bot started (supervisor pid $!). Monitor with: scripts/mac/monitor.sh"
