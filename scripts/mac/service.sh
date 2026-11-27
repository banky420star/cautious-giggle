#!/bin/zsh
# Cautious Giggle as macOS background services (launchd): start at login, restart on crash, Mac kept awake.
#   scripts/mac/service.sh install    install + start bot and dashboard (replaces start_bot.sh's supervisor)
#   scripts/mac/service.sh status     are they running? last log lines
#   scripts/mac/service.sh restart    restart the bot (e.g. after a code change)
#   scripts/mac/service.sh stop       stop the bot (positions keep their broker SL/TP); start: start it again
#   scripts/mac/service.sh uninstall  remove both services
ROOT="${0:A:h:h:h}"   # the project folder (this script lives in scripts/mac)
LA="$HOME/Library/LaunchAgents"
UID_=$(id -u)
BOT=com.cautiousgiggle.bot
DASH=com.cautiousgiggle.dashboard

plist() {  # label, arg, logfile
  cat > "$LA/$1.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>$1</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$ROOT/scripts/mac/service_run.sh</string><string>$2</string></array>
  <key>RunAtLoad</key><true/>
  <key>KeepAlive</key><true/>
  <key>ThrottleInterval</key><integer>30</integer>
  <key>ProcessType</key><string>Interactive</string>
  <key>StandardOutPath</key><string>$ROOT/logs/$3</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/$3</string>
</dict></plist>
PL
}

case "$1" in
  install)
    # retire the hand-started supervisor and any running copies
    touch "$ROOT/.tmp/stop"
    [ -f "$ROOT/.tmp/supervisor.pid" ] && kill "$(cat "$ROOT/.tmp/supervisor.pid")" 2>/dev/null
    pkill -f "caffeinate -is zsh -c" 2>/dev/null
    pkill -f "MacOS/Python Python/Server_AGI.py" 2>/dev/null
    pkill -f "tools/web_dashboard.py --port 8788" 2>/dev/null
    sleep 3; rm -f "$ROOT/.tmp/stop" "$ROOT/.tmp/supervisor.pid" "$ROOT/.tmp/server_agi.lock"
    mkdir -p "$LA"
    plist $BOT bot server_stdout.log
    plist $DASH dashboard web_dashboard.log
    for l in $BOT $DASH; do launchctl bootout gui/$UID_/$l 2>/dev/null; launchctl bootstrap gui/$UID_ "$LA/$l.plist"; done
    echo "installed: bot + dashboard start at login and restart if they stop" ;;
  status)
    for l in $BOT $DASH; do
      st=$(launchctl print gui/$UID_/$l 2>/dev/null | awk -F'= ' '/^\tstate/{print $2; exit}')
      pid=$(launchctl print gui/$UID_/$l 2>/dev/null | awk -F'= ' '/^\tpid/{print $2; exit}')
      echo "$l: ${st:-not installed} ${pid:+(pid $pid)}"
    done
    tail -2 "$ROOT/logs/server.log" | cut -c1-160 ;;
  restart) rm -f "$ROOT/.tmp/stop"; launchctl kickstart -k gui/$UID_/$BOT && echo "bot restarted" ;;
  stop)  touch "$ROOT/.tmp/stop"; launchctl kill TERM gui/$UID_/$BOT; echo "bot stopped (positions keep their SL/TP at the broker); 'start' to resume" ;;
  start) rm -f "$ROOT/.tmp/stop"; launchctl kickstart -k gui/$UID_/$BOT && echo "bot started" ;;
  uninstall)
    for l in $BOT $DASH; do launchctl bootout gui/$UID_/$l 2>/dev/null; rm -f "$LA/$l.plist"; done
    echo "services removed (use scripts/mac/start_bot.sh to run by hand)" ;;
  *) sed -n '2,9p' "$0" ;;
esac
