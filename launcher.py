"""
Cautious Giggle Launcher.

Open → Login → Start Trading.
That's it.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from Python.config_utils import load_project_config

BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
ENV_PATH = BASE_DIR / ".env"
ICON_PATH = BASE_DIR / "cautious_giggle.ico"
TMP_DIR = BASE_DIR / ".tmp"
LOGS_DIR = BASE_DIR / "logs"
RUNTIME_DIR = BASE_DIR / "runtime"
PORT = 8088
HOST = "127.0.0.1"
TITLE = "Cautious Giggle"
WIDTH, HEIGHT = 1100, 780
NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)

RISK_PROFILES = {
    "conservative": {"risk_percent": 0.5, "max_dd": 8.0, "max_daily_loss_pct": 2.0},
    "balanced":     {"risk_percent": 1.0, "max_dd": 12.0, "max_daily_loss_pct": 4.0},
    "aggressive":   {"risk_percent": 2.0, "max_dd": 18.0, "max_daily_loss_pct": 6.0},
}

# ---------------------------------------------------------------------------
# Login HTML
# ---------------------------------------------------------------------------
LOGIN_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>Cautious Giggle — Connect</title>
<style>
:root{
  --bg:#06111d;--bg2:#0d1c2d;--panel:rgba(8,16,26,.82);
  --line:rgba(255,255,255,.08);--text:#eef6ff;--muted:#8ea3c2;
  --cyan:#4fd6ff;--lime:#8de570;--amber:#ffbf4d;--red:#ff6b7d;
}
*{box-sizing:border-box;margin:0}
html,body{height:100%;font-family:"Segoe UI Variable Text","Segoe UI",sans-serif;color:var(--text)}
body{
  overflow:hidden;
  background:
    radial-gradient(900px 500px at 80% -10%,rgba(79,214,255,.12),transparent 55%),
    radial-gradient(600px 400px at 0% 0%,rgba(141,229,112,.07),transparent 45%),
    linear-gradient(180deg,var(--bg),var(--bg2));
  display:flex;align-items:center;justify-content:center;
}
body::before{
  content:"";position:fixed;inset:0;
  background:
    linear-gradient(rgba(255,255,255,.03) 1px,transparent 1px),
    linear-gradient(90deg,rgba(255,255,255,.03) 1px,transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(circle at center,black 45%,transparent 90%);
}
.card{
  position:relative;width:480px;
  border:1px solid var(--line);border-radius:28px;
  background:var(--panel);backdrop-filter:blur(16px);
  box-shadow:0 28px 60px rgba(0,0,0,.35);
  padding:36px 32px;
}
.eyebrow{color:var(--cyan);font-size:11px;letter-spacing:.22em;text-transform:uppercase}
h1{font-size:28px;letter-spacing:-.04em;margin-top:10px}
.sub{color:var(--muted);font-size:13px;line-height:1.6;margin-top:8px}
.fields{display:grid;gap:14px;margin-top:24px}
label{display:block;color:var(--muted);font-size:11px;letter-spacing:.14em;text-transform:uppercase;margin-bottom:6px}
input{
  width:100%;padding:12px 14px;border-radius:14px;border:1px solid var(--line);
  background:rgba(255,255,255,.04);color:var(--text);font-size:14px;outline:none;
  transition:border .2s;
}
input:focus{border-color:var(--cyan)}
input::placeholder{color:rgba(142,163,194,.5)}
.btn{
  margin-top:20px;width:100%;padding:14px;border:none;border-radius:16px;
  background:linear-gradient(135deg,var(--cyan),#9f8bff);color:#06111d;
  font-size:14px;font-weight:700;letter-spacing:.03em;cursor:pointer;
  transition:opacity .2s;
}
.btn:hover{opacity:.88}
.btn:disabled{opacity:.45;cursor:not-allowed}
.status{
  margin-top:16px;padding:12px 14px;border-radius:14px;
  font-size:12px;line-height:1.6;display:none;
}
.status.info{display:block;border:1px solid rgba(79,214,255,.2);background:rgba(79,214,255,.06);color:var(--cyan)}
.status.ok{display:block;border:1px solid rgba(141,229,112,.25);background:rgba(141,229,112,.08);color:var(--lime)}
.status.err{display:block;border:1px solid rgba(255,107,125,.25);background:rgba(255,107,125,.08);color:var(--red)}
.equity-row{
  margin-top:18px;display:none;
  padding:16px;border-radius:18px;border:1px solid var(--line);
  background:rgba(255,255,255,.03);
}
.equity-row.show{display:block}
.equity-row .k{color:var(--muted);font-size:11px;letter-spacing:.14em;text-transform:uppercase}
.equity-row .v{font-size:24px;font-weight:700;margin-top:6px;color:var(--lime)}
.equity-row .d{color:var(--muted);font-size:12px;margin-top:4px}
.launch-btn{
  margin-top:16px;width:100%;padding:14px;border:none;border-radius:16px;
  background:linear-gradient(135deg,var(--lime),var(--cyan));color:#06111d;
  font-size:14px;font-weight:700;cursor:pointer;display:none;
}
.launch-btn.show{display:block}
.launch-btn:disabled{opacity:.45;cursor:not-allowed}
.mode-select{margin-top:18px;display:none}
.mode-select.show{display:block}
.mode-select label{margin-bottom:8px}
.mode-options{display:grid;grid-template-columns:1fr 1fr 1fr;gap:10px}
.mode-opt{
  padding:12px;border-radius:14px;border:2px solid var(--line);
  background:rgba(255,255,255,.03);cursor:pointer;text-align:center;
  transition:border .2s,background .2s;
}
.mode-opt:hover{border-color:rgba(79,214,255,.3)}
.mode-opt.selected{border-color:var(--cyan);background:rgba(79,214,255,.08)}
.mode-opt .mode-name{font-size:13px;font-weight:700;color:var(--text)}
.mode-opt .mode-risk{font-size:11px;color:var(--muted);margin-top:4px}
.mode-opt .mode-desc{font-size:10px;color:var(--muted);margin-top:2px}
</style>
</head>
<body>
<div class="card">
  <div class="eyebrow">MetaTrader 5 Connection</div>
  <h1>Connect your account</h1>
  <div class="sub">Enter your MT5 broker credentials. We'll validate the connection and read your account equity before starting.</div>

  <div class="fields">
    <div>
      <label>MT5 Login (Account Number)</label>
      <input type="text" id="login" placeholder="e.g. 12345678" autocomplete="off">
    </div>
    <div>
      <label>MT5 Password</label>
      <input type="password" id="password" placeholder="Your trading password">
    </div>
    <div>
      <label>MT5 Server</label>
      <input type="text" id="server" placeholder="e.g. Exness-MT5Real6">
    </div>
  </div>

  <button class="btn" id="connect-btn" onclick="doConnect()">Connect & Validate</button>
  <div class="status" id="status"></div>

  <div class="equity-row" id="equity-row">
    <div class="k">Account Equity</div>
    <div class="v" id="equity-val">—</div>
    <div class="d" id="equity-detail">This is your risk baseline.</div>
  </div>

  <div class="mode-select" id="mode-select">
    <label>Risk Mode</label>
    <div class="mode-options">
      <div class="mode-opt" onclick="selectMode('conservative',this)">
        <div class="mode-name">Conservative</div>
        <div class="mode-risk">0.5% risk</div>
        <div class="mode-desc">Small lots, tight stops</div>
      </div>
      <div class="mode-opt selected" onclick="selectMode('balanced',this)">
        <div class="mode-name">Balanced</div>
        <div class="mode-risk">1.0% risk</div>
        <div class="mode-desc">Default scaling</div>
      </div>
      <div class="mode-opt" onclick="selectMode('aggressive',this)">
        <div class="mode-name">Aggressive</div>
        <div class="mode-risk">2.0% risk</div>
        <div class="mode-desc">Larger lots, wider range</div>
      </div>
    </div>
  </div>

  <button class="launch-btn" id="launch-btn" onclick="doLaunch()">Start Trading</button>
</div>

<script>
var selectedMode = 'balanced';

function selectMode(mode, el) {
  selectedMode = mode;
  document.querySelectorAll('.mode-opt').forEach(e => e.classList.remove('selected'));
  el.classList.add('selected');
}

function setStatus(cls, msg) {
  var el = document.getElementById('status');
  el.className = 'status ' + cls;
  el.textContent = msg;
}

async function autoConnectSaved() {
  if (!window.pywebview || !window.pywebview.api || !window.pywebview.api.get_saved_mt5) {
    return;
  }
  try {
    var creds = await window.pywebview.api.get_saved_mt5();
    if (creds.login && creds.password && creds.server) {
      document.getElementById('login').value = creds.login;
      document.getElementById('password').value = creds.password;
      document.getElementById('server').value = creds.server;
      await doConnect();
    }
  } catch (e) {
    console.warn('Auto connect failed', e);
  }
}

window.addEventListener('pywebviewready', () => {
  autoConnectSaved();
});

async function doConnect() {
  var login = document.getElementById('login').value.trim();
  var password = document.getElementById('password').value.trim();
  var server = document.getElementById('server').value.trim();
  if (!login || !password || !server) { setStatus('err', 'All fields are required.'); return; }

  var btn = document.getElementById('connect-btn');
  btn.disabled = true;
  btn.textContent = 'Connecting...';
  setStatus('info', 'Connecting to MT5...');

  try {
    var r = await window.pywebview.api.validate_mt5(login, password, server);
    if (r.success) {
      setStatus('ok', 'Connected! ' + r.account_name + ' | $' + r.balance.toFixed(2));
      document.getElementById('equity-val').textContent = '$' + r.equity.toFixed(2);
      document.getElementById('equity-detail').textContent =
        'Balance: $' + r.balance.toFixed(2) + ' | Leverage: 1:' + r.leverage + ' | ' + r.account_name;
      document.getElementById('equity-row').classList.add('show');
      document.getElementById('mode-select').classList.add('show');
      document.getElementById('launch-btn').classList.add('show');
      btn.textContent = 'Connected';
    } else {
      setStatus('err', r.error);
      btn.disabled = false;
      btn.textContent = 'Retry';
    }
  } catch(e) {
    setStatus('err', String(e));
    btn.disabled = false;
    btn.textContent = 'Retry';
  }
}

async function doLaunch() {
  var btn = document.getElementById('launch-btn');
  btn.disabled = true;
  btn.textContent = 'Starting...';
  setStatus('info', 'Resetting state, starting trading engine...');

  try {
    var r = await window.pywebview.api.start_trading(selectedMode);
    if (r.success) {
      setStatus('ok', r.message);
      btn.textContent = 'Running';
    } else {
      setStatus('err', r.error);
      btn.disabled = false;
      btn.textContent = 'Retry';
    }
  } catch(e) {
    setStatus('err', String(e));
    btn.disabled = false;
    btn.textContent = 'Retry';
  }
}
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# API
# ---------------------------------------------------------------------------
class LauncherAPI:
    def __init__(self, window_ref):
        self._window = window_ref
        self._login: str = ""
        self._password: str = ""
        self._server: str = ""
        self._equity: float = 0.0
        self._balance: float = 0.0

    # -- Step 1: Validate credentials, get equity --------------------------
    def validate_mt5(self, login: str, password: str, server: str) -> dict:
        try:
            import MetaTrader5 as mt5
        except ImportError:
            return {"success": False, "error": "MetaTrader5 not installed"}

        try:
            mt5.shutdown()
        except Exception:
            pass

        if not mt5.initialize():
            return {"success": False, "error": "MT5 terminal failed to initialize"}

        if not mt5.login(int(login), password=password, server=server):
            err = mt5.last_error()
            mt5.shutdown()
            return {"success": False, "error": f"Login failed: {err}"}

        info = mt5.account_info()
        if info is None:
            mt5.shutdown()
            return {"success": False, "error": "Could not read account info"}

        self._login = str(login)
        self._password = password
        self._server = server
        self._equity = float(info.equity)
        self._balance = float(info.balance)

        result = {
            "success": True,
            "equity": self._equity,
            "balance": self._balance,
            "leverage": int(info.leverage),
            "account_name": str(info.name),
            "currency": str(info.currency),
        }
        mt5.shutdown()  # Let the bot reconnect clean
        return result

    def get_saved_mt5(self) -> dict:
        login = os.environ.get("MT5_LOGIN", "").strip()
        password = os.environ.get("MT5_PASSWORD", "").strip()
        server = os.environ.get("MT5_SERVER", "").strip()
        if not (login and password and server):
            try:
                cfg = load_project_config(str(BASE_DIR), live_mode=False) or {}
            except Exception:
                cfg = {}
            mt5_cfg = cfg.get("mt5", {}) if isinstance(cfg, dict) else {}
            def resolve(raw: Any) -> str:
                val = str(raw or "").strip()
                if val.upper().startswith("ENV:"):
                    env_name = val.split(":", 1)[1].strip()
                    return os.environ.get(env_name, "").strip()
                return val
            login = login or resolve(mt5_cfg.get("login", ""))
            password = password or resolve(mt5_cfg.get("password", ""))
            server = server or resolve(mt5_cfg.get("server", ""))
        if login and password and server:
            return {"login": login, "password": password, "server": server}
        return {}

    # -- Step 2: One clean call. That's it. --------------------------------
    def start_trading(self, risk_mode: str = "balanced") -> dict:
        if not self._login:
            return {"success": False, "error": "Connect first"}

        try:
            profile = RISK_PROFILES.get(risk_mode, RISK_PROFILES["balanced"])
            max_lots = _calculate_lot(self._equity, profile["risk_percent"])
            max_daily_loss = round(self._equity * profile["max_daily_loss_pct"] / 100, 2)

            # 1. Write session config
            session = {
                "login": self._login,
                "server": self._server,
                "start_equity": round(self._equity, 2),
                "balance": round(self._balance, 2),
                "mode": risk_mode,
                "risk_percent": profile["risk_percent"],
                "max_dd_pct": profile["max_dd"],
                "max_lots": max_lots,
                "max_daily_loss": max_daily_loss,
                "cold_start": True,
                "timestamp": time.time(),
            }
            RUNTIME_DIR.mkdir(parents=True, exist_ok=True)
            (RUNTIME_DIR / "session.json").write_text(json.dumps(session, indent=2), encoding="utf-8")

            # 2. Write .env (so subprocesses inherit MT5 creds)
            _write_env({
                "MT5_LOGIN": self._login,
                "MT5_PASSWORD": self._password,
                "MT5_SERVER": self._server,
            })

            # 3. Reset risk state — clean slate, no emotional damage
            _reset_risk(self._equity, max_daily_loss)

            # 4. Kill ghosts, clear locks
            _kill_old_processes()
            _clear_locks()

            # 5. Start everything
            _start_services()

            # 6. Navigate to dashboard (deferred so the JS callback settles
            #    before the page context is destroyed by load_url)
            dashboard_url = f"http://{HOST}:{PORT}/"
            _wait_for_url(dashboard_url, timeout=20)

            def _navigate():
                time.sleep(0.5)
                w = self._window()
                if w:
                    w.load_url(dashboard_url)

            threading.Thread(target=_navigate, daemon=True).start()

            return {
                "success": True,
                "message": f"{risk_mode.title()} mode | ${self._equity:.2f} equity | Max lots: {max_lots}",
            }
        except Exception as exc:
            return {"success": False, "error": str(exc)}


# ---------------------------------------------------------------------------
# Helpers — each one does exactly one thing
# ---------------------------------------------------------------------------
def _calculate_lot(equity: float, risk_percent: float) -> float:
    lots = round((equity * risk_percent) / 100_000, 2)
    return max(0.01, min(lots, 5.0))


def _write_env(creds: dict) -> None:
    existing = {}
    if ENV_PATH.exists():
        for line in ENV_PATH.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if line and not line.startswith("#") and "=" in line:
                k, v = line.split("=", 1)
                existing[k.strip()] = v.strip()
    existing.update(creds)
    ENV_PATH.write_text("\n".join(f"{k}={v}" for k, v in existing.items()) + "\n", encoding="utf-8")
    for k, v in creds.items():
        os.environ[k] = v


def _reset_risk(equity: float, max_daily_loss: float) -> None:
    """Reset risk engine. No leftover state. No 'why is it down 12% already'."""
    LOGS_DIR.mkdir(parents=True, exist_ok=True)

    (LOGS_DIR / "risk_engine_state.json").write_text(json.dumps({
        "peak_equity": equity,
        "current_dd": 0.0,
        "halt": False,
        "error_halt": False,
        "error_count": 0,
        "realized_pnl_today": 0.0,
        "daily_trades": 0,
        "daily_trades_by_symbol": {},
        "daily_losing_trades_by_symbol": {},
        "last_reset_day": time.strftime("%Y-%m-%d"),
    }, indent=2), encoding="utf-8")

    (LOGS_DIR / "risk_supervisor_state.json").write_text(json.dumps({
        "halt_until": None,
        "last_trade_at": {},
    }), encoding="utf-8")


def _kill_old_processes() -> None:
    """Kill any leftover Python trading processes."""
    import re
    try:
        out = subprocess.check_output(
            ["tasklist", "/FI", "IMAGENAME eq python.exe", "/FO", "CSV"],
            text=True, creationflags=NO_WINDOW,
        )
        for line in out.strip().splitlines()[1:]:
            # Don't kill ourselves
            parts = line.strip('"').split('","')
            if len(parts) >= 2:
                pid = int(parts[1].strip('"'))
                if pid != os.getpid():
                    try:
                        subprocess.run(
                            ["taskkill", "/F", "/PID", str(pid)],
                            creationflags=NO_WINDOW,
                            capture_output=True,
                        )
                    except Exception:
                        pass
    except Exception:
        pass


def _clear_locks() -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for lock in TMP_DIR.glob("*.lock"):
        try:
            lock.unlink()
        except Exception:
            pass


def _launch(args: list[str], log_name: str, env_updates: dict | None = None) -> subprocess.Popen:
    """Start a process with output going to a log file, not /dev/null."""
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    LOGS_DIR.mkdir(parents=True, exist_ok=True)
    log_file = open(LOGS_DIR / f"{log_name}.log", "a", encoding="utf-8")
    return subprocess.Popen(
        args, cwd=str(BASE_DIR), env=env,
        stdout=log_file, stderr=subprocess.STDOUT,
        creationflags=NO_WINDOW,
    )


def _start_services() -> None:
    """Start the trading stack. Dashboard, trading engine, training loop."""
    # Dashboard
    ui = BASE_DIR / "tools" / "project_status_ui.py"
    if ui.exists():
        _launch([PYTHON, "-u", str(ui)], "ui_server",
                env_updates={"AGI_UI_HOST": HOST, "AGI_UI_PORT": str(PORT)})
        time.sleep(2)

    # Trading engine
    _launch([PYTHON, "-u", "-m", "Python.Server_AGI", "--live"], "server_agi")
    time.sleep(1)

    # Training loop (champion cycle)
    cycle = BASE_DIR / "tools" / "champion_cycle_loop.py"
    if cycle.exists():
        _launch([PYTHON, "-u", str(cycle), "--interval-minutes", "30"], "champion_cycle")

    # Watchdog
    watchdog = BASE_DIR / "tools" / "watchdog.py"
    if watchdog.exists():
        _launch([PYTHON, "-u", str(watchdog)], "watchdog")


def _wait_for_url(url: str, timeout: float = 20) -> bool:
    import urllib.request, urllib.error
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with urllib.request.urlopen(url, timeout=2) as r:
                if 200 <= r.status < 500:
                    return True
        except (urllib.error.URLError, TimeoutError, OSError):
            pass
        time.sleep(0.5)
    return False


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    import webview

    window = webview.create_window(
        TITLE, html=LOGIN_HTML,
        width=WIDTH, height=HEIGHT,
        min_size=(520, 640), resizable=True,
        text_select=True, zoomable=True, easy_drag=False,
    )

    if ICON_PATH.exists():
        try:
            window.icon = str(ICON_PATH)
        except Exception:
            pass

    import weakref
    api = LauncherAPI(weakref.ref(window))
    window.expose(api.validate_mt5, api.start_trading)
    webview.start(debug="--debug" in sys.argv, http_server=False)


if __name__ == "__main__":
    main()
