#!/bin/zsh
# Cautious Giggle guided installer (double-click "Install Cautious Giggle.command" in the project folder).
# Safe to run again: every step checks what is already there and only asks before replacing it.
#   1 tools (Homebrew, Python 3.13, Node)   2 Python packages   3 Ollama + the Gemma model
#   4 Jev key   5 Exness demo + live accounts   6 config.yaml   7 Telegram (optional)
#   8 dashboard   9 app icon + nightly P&L report   10 start the bot
ROOT="${0:A:h:h:h}"
cd "$ROOT" || exit 1
GEMMA=gemma4:e4b-it-qat
KRONOS_REV=67b630e
B=$'\e[1m'; D=$'\e[2m'; G=$'\e[32m'; Y=$'\e[33m'; R=$'\e[31m'; X=$'\e[0m'

step() { print "\n${B}━━ $1${X}"; }
ok()   { print "  ${G}✓${X} $1"; }
warn() { print "  ${Y}!${X} $1"; }
die()  { print "\n  ${R}✗ $1${X}\n"; read -k1 "?Press any key to close."; exit 1; }
yes()  { local a h="Y/n"; [[ ${2:-Y} == N ]] && h="y/N"; read "a?  $1 ${D}[$h]${X}: "; a=${a:-${2:-Y}}; [[ $a == [Yy]* ]]; }

clear
print "${B}Cautious Giggle setup${X}"
print "${D}Sets up the trading bot on this Mac. It asks for your Jev key and your Exness demo and live API keys."
print "Keys are saved only on this Mac, in files only your user can read. Press Ctrl-C to stop at any time.${X}"
[[ $(uname) == Darwin ]] || die "This installer is for macOS."

# ---------------------------------------------------------------- 1 tools
step "1/10  Tools"
if ! command -v brew >/dev/null; then
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do [[ -x $p ]] && eval "$($p shellenv)"; done
fi
if ! command -v brew >/dev/null; then
  print "  Homebrew (the Mac package manager) installs Python, Node and Ollama."
  yes "Install Homebrew now? (asks for your Mac password)" || die "Homebrew is needed. See https://brew.sh"
  /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)" || die "Homebrew install failed."
  for p in /opt/homebrew/bin/brew /usr/local/bin/brew; do [[ -x $p ]] && eval "$($p shellenv)"; done
fi
ok "Homebrew"
PY=$(command -v python3.13 || ls "$(brew --prefix 2>/dev/null)"/bin/python3.13 2>/dev/null)
if [[ -z $PY ]]; then
  print "  Installing Python 3.13…"; brew install python@3.13 || die "Python 3.13 install failed."
  PY=$(brew --prefix)/bin/python3.13
fi
ok "Python ($($PY -V))"
command -v npm >/dev/null || { print "  Installing Node (for the dashboard)…"; brew install node || warn "Node install failed: the dashboard won't build."; }
command -v npm >/dev/null && ok "Node $(node -v)"
command -v git >/dev/null || xcode-select --install 2>/dev/null

# ---------------------------------------------------------------- 2 python packages
step "2/10  Python packages ${D}(a few GB the first time; 5-15 minutes)${X}"
if [[ ! -x .venv/bin/python ]]; then
  rm -f .venv 2>/dev/null   # a broken link from another Mac
  $PY -m venv .venv || die "Could not create the Python environment."
fi
.venv/bin/python -m pip install -q --upgrade pip
.venv/bin/python -m pip install -q -r requirements-mac.txt || die "Package install failed (see the message above). Run the installer again to retry."
ok "Packages installed"
mkdir -p logs .tmp models data

# Kronos (price forecaster used by the main bot): a pinned copy outside the project
KR=~/.cautious-giggle/kronos
if [[ ! -d $KR/.git ]]; then
  git clone -q https://github.com/shiyu-coder/Kronos.git "$KR" 2>/dev/null && git -C "$KR" checkout -q $KRONOS_REV \
    && ok "Kronos forecaster" || warn "Couldn't download Kronos; the bot runs without it."
else ok "Kronos forecaster"; fi

# ---------------------------------------------------------------- 3 ollama + gemma
step "3/10  Gemma (local AI model, runs on this Mac)"
if ! command -v ollama >/dev/null && [[ ! -d /Applications/Ollama.app ]]; then
  print "  Ollama runs Gemma locally. Installing it…"
  brew install --cask ollama 2>/dev/null || brew install ollama || warn "Ollama install failed: get it from https://ollama.com/download and run this again."
fi
if ! curl -s -m 3 http://127.0.0.1:11434/api/tags >/dev/null; then
  [[ -d /Applications/Ollama.app ]] && open -ga Ollama || (command -v ollama >/dev/null && nohup ollama serve >/dev/null 2>&1 &)
  for i in {1..20}; do curl -s -m 2 http://127.0.0.1:11434/api/tags >/dev/null && break; sleep 1; done
fi
if curl -s -m 3 http://127.0.0.1:11434/api/tags | grep -q "\"$GEMMA\""; then
  ok "Gemma ($GEMMA) already downloaded"
elif command -v ollama >/dev/null || [[ -x /Applications/Ollama.app/Contents/Resources/ollama ]]; then
  OL=$(command -v ollama || print /Applications/Ollama.app/Contents/Resources/ollama)
  print "  Downloading Gemma ($GEMMA, about 5 GB)…"
  $OL pull $GEMMA && ok "Gemma ready" || warn "Gemma download failed: run 'ollama pull $GEMMA' later. The Gemma trader and news reader wait for it."
else
  warn "Ollama isn't installed, so Gemma is skipped. The other traders still work."
fi

# ---------------------------------------------------------------- 4 jev
step "4/10  Jev key"
.venv/bin/python tools/install_setup.py jev

# ---------------------------------------------------------------- 5 exness
step "5/10  Exness accounts"
print "  ${D}The bot starts on the DEMO account. Live (real money) is optional and only used when you switch to it"
print "  in the app (it asks you to type GO LIVE).${X}"
until .venv/bin/python tools/install_setup.py exness demo; do
  yes "The demo account isn't working yet. Try again?" || { warn "Skipped: the bot can't trade until the demo account is set up (run this installer again)."; break; }
done
if yes "Set up a LIVE (real money) account too?" N; then
  until .venv/bin/python tools/install_setup.py exness live; do
    yes "Try the live account again?" || break
  done
fi

# ---------------------------------------------------------------- 6 config
step "6/10  Settings"
.venv/bin/python tools/install_setup.py config

# ---------------------------------------------------------------- 7 telegram
step "7/10  Phone alerts"
.venv/bin/python tools/install_setup.py telegram

# ---------------------------------------------------------------- 8 dashboard
step "8/10  Dashboard"
if command -v npm >/dev/null; then
  (cd web && npm install --silent >/dev/null 2>&1 && npm run build >/dev/null 2>&1) && ok "Dashboard built (http://localhost:8788)" \
    || warn "Dashboard build failed; run 'cd web && npm install && npm run build' later."
else warn "No Node, so no dashboard."; fi

# ---------------------------------------------------------------- 9 app + nightly report
step "9/10  App and nightly P&L report"
APP=~/Applications/"Cautious Giggle.app"
mkdir -p ~/Applications
osacompile -o "$APP" -e "do shell script \"/bin/zsh '$ROOT/scripts/mac/open_app.sh' >/dev/null 2>&1 &\"" 2>/dev/null \
  && cp assets/app.icns "$APP/Contents/Resources/applet.icns" 2>/dev/null && touch "$APP" && ok "App: ~/Applications/Cautious Giggle.app"
[[ -e ~/Desktop/"Cautious Giggle.app" ]] || ln -s "$APP" ~/Desktop/"Cautious Giggle.app" 2>/dev/null && ok "Shortcut on the Desktop"
LA=~/Library/LaunchAgents; mkdir -p "$LA"
cat > "$LA/com.cautiousgiggle.dailypnl.plist" <<PL
<?xml version="1.0" encoding="UTF-8"?>
<!DOCTYPE plist PUBLIC "-//Apple//DTD PLIST 1.0//EN" "http://www.apple.com/DTDs/PropertyList-1.0.dtd">
<plist version="1.0"><dict>
  <key>Label</key><string>com.cautiousgiggle.dailypnl</string>
  <key>ProgramArguments</key><array><string>/bin/zsh</string><string>$ROOT/scripts/mac/service_run.sh</string><string>dailypnl</string></array>
  <key>StartCalendarInterval</key><dict><key>Hour</key><integer>0</integer><key>Minute</key><integer>5</integer></dict>
  <key>RunAtLoad</key><false/>
  <key>ProcessType</key><string>Background</string>
  <key>StandardOutPath</key><string>$ROOT/logs/daily_pnl.log</string>
  <key>StandardErrorPath</key><string>$ROOT/logs/daily_pnl.log</string>
</dict></plist>
PL
launchctl bootout gui/$(id -u)/com.cautiousgiggle.dailypnl 2>/dev/null
launchctl bootstrap gui/$(id -u) "$LA/com.cautiousgiggle.dailypnl.plist" 2>/dev/null && ok "Daily P&L report at 00:05"

# ---------------------------------------------------------------- 10 start
step "10/10  Start"
if [[ -f ~/.exness/env ]] && yes "Start the bot now (on the DEMO account; it keeps running in the background)?"; then
  [[ -f ~/.exness/active ]] || print demo > ~/.exness/active   # a new install starts on demo; a re-run keeps the choice
  zsh scripts/mac/service.sh install && ok "Bot running. Open the app on the Desktop to watch it."
fi
print "\n${G}${B}Setup finished.${X} Run this installer again any time to change keys or accounts."
read -k1 "?Press any key to close."
