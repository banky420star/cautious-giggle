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
nohup caffeinate -is zsh -c '
  while [ ! -f .tmp/stop ]; do
    echo "$(date "+%F %T") starting Server_AGI" >> logs/server_stdout.log
    .venv/bin/python Python/Server_AGI.py >> logs/server_stdout.log 2>&1
    [ -f .tmp/stop ] && break
    echo "$(date "+%F %T") Server_AGI exited ($?), restarting in 30s" >> logs/server_stdout.log
    sleep 30
  done
  rm -f .tmp/stop
' > /dev/null 2>&1 &
echo $! > .tmp/supervisor.pid
rm -f .tmp/stop
echo "bot started (supervisor pid $!). Monitor with: scripts/mac/monitor.sh"
