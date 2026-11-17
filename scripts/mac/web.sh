#!/bin/zsh
# Start the web dashboard server (if needed) and open it in its own window.
cd "$(dirname "$0")/../.."
export PYTHONPATH="$PWD"
[ -f ~/.exness/env ] && source ~/.exness/env
PORT=8787
if ! curl -s -o /dev/null --max-time 2 "http://127.0.0.1:$PORT/api/state"; then
  [ -d web/dist ] || (cd web && npm run build >/dev/null 2>&1)
  nohup .venv/bin/python tools/web_dashboard.py --port $PORT > logs/web_dashboard.log 2>&1 < /dev/null &
  disown
  for i in {1..30}; do curl -s -o /dev/null --max-time 1 "http://127.0.0.1:$PORT/" && break; sleep 0.5; done
fi
if [ -d "/Applications/Google Chrome.app" ]; then
  open -na "Google Chrome" --args --app="http://127.0.0.1:$PORT" --window-size=1440,960
else
  open "http://127.0.0.1:$PORT"
fi
