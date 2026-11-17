#!/bin/zsh
# Wait for the training cycle to finish, give unapproved symbols a demo-exploration model,
# then start the bot. Log: logs/after_training.log
cd "$(dirname "$0")/../.." || exit 1
source ~/.exness/env
export PYTHONPATH="$PWD"
while pgrep -f "tools/champion_cycle.py" >/dev/null; do sleep 30; done
echo "$(date '+%F %T') training finished" >> logs/after_training.log
.venv/bin/python tools/demo_explore.py >> logs/after_training.log 2>&1
scripts/mac/stop_bot.sh >> logs/after_training.log 2>&1; sleep 3; rm -f .tmp/stop
scripts/mac/start_bot.sh >> logs/after_training.log 2>&1
