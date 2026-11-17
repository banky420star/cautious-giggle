#!/bin/zsh
# Used by the "Cautious Giggle" app icon: make sure the bot is running, then open the web dashboard.
cd "$(dirname "$0")/../.."
PID=$(cat .tmp/supervisor.pid 2>/dev/null)
if [ -z "$PID" ] || ! kill -0 "$PID" 2>/dev/null; then
  rm -f .tmp/stop
  scripts/mac/start_bot.sh > /dev/null 2>&1
fi
exec scripts/mac/web.sh
