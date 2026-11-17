#!/bin/zsh
# Stop the bot. Open positions stay open with their broker-side SL/TP.
cd "$(dirname "$0")/../.." || exit 1
touch .tmp/stop
[ -f .tmp/server_agi.lock ] && kill "$(cat .tmp/server_agi.lock)" 2>/dev/null
[ -f .tmp/supervisor.pid ] && kill "$(cat .tmp/supervisor.pid)" 2>/dev/null
pkill -f "Python/Server_AGI.py" 2>/dev/null
rm -f .tmp/supervisor.pid
echo "bot stopped (positions keep their SL/TP at the broker)"
