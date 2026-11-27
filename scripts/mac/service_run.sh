#!/bin/zsh
# Run one Cautious Giggle service in the foreground for launchd (which restarts it if it exits).
#   service_run.sh bot        the trading bot (keeps the Mac awake while it runs)
#   service_run.sh dashboard  the war-room dashboard on http://127.0.0.1:8788
#   service_run.sh scout      once a day: test and add the next new market (tools/market_scout.py)
#   service_run.sh dailypnl   once a day at 00:05: yesterday's P&L to Telegram (tools/daily_pnl.py)
ROOT="${0:A:h:h:h}"   # the project folder (this script lives in scripts/mac)
# the project lives on an external drive: wait for it after a reboot/login
for i in {1..120}; do [ -f "$ROOT/Python/Server_AGI.py" ] && break; sleep 5; done
cd "$ROOT" || exit 1
source scripts/mac/account_env.sh >/dev/null 2>&1 || { echo "$(date '+%F %T') account_env.sh failed" >&2; exit 1; }
export PYTHONPATH="$ROOT" PYTHONUNBUFFERED=1
mkdir -p logs .tmp
case "$1" in
  bot)
    [ -f .tmp/stop ] && { echo "$(date '+%F %T') .tmp/stop present: bot held stopped (scripts/mac/service.sh start clears it)"; sleep 300; exit 0; }
    echo "$(date '+%F %T') service: starting Server_AGI"
    exec caffeinate -is .venv/bin/python Python/Server_AGI.py ;;
  dashboard)
    echo "$(date '+%F %T') service: starting dashboard"
    exec .venv/bin/python tools/web_dashboard.py --port 8788 ;;
  scout)
    echo "$(date '+%F %T') service: daily market scout"
    exec .venv/bin/python tools/market_scout.py ;;
  dailypnl)
    echo "$(date '+%F %T') service: daily P&L report"
    exec .venv/bin/python tools/daily_pnl.py ;;
  *) echo "usage: $0 bot|dashboard|scout|dailypnl" >&2; exit 2 ;;
esac
