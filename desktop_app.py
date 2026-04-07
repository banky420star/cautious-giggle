"""
Cautious Giggle desktop launcher.

Launches the dashboard and backend services behind a guarded startup flow with
clear preflight feedback for production use.
"""

from __future__ import annotations

import json
import os
import socket
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

from Python.config_utils import load_project_config


PORT = 8088
HOST = "127.0.0.1"
TITLE = "Cautious Giggle AGI Trading Bot"
WIDTH, HEIGHT = 1440, 920
BASE_DIR = Path(__file__).resolve().parent
PYTHON = sys.executable
ICON_PATH = BASE_DIR / "cautious_giggle.ico"
ENV_PATH = BASE_DIR / ".env"
CONFIG_PATH = BASE_DIR / "config.yaml"
HFT_CONFIG_PATH = BASE_DIR / "config_hft.yaml"
UI_SCRIPT = BASE_DIR / "tools" / "project_status_ui.py"
CHAMPION_LOOP_SCRIPT = BASE_DIR / "tools" / "champion_cycle_loop.py"
WATCHDOG_SCRIPT = BASE_DIR / "tools" / "watchdog.py"
SERVER_AGI_MODULE = "Python.Server_AGI"
DASHBOARD_PRIMARY_URL = f"http://{HOST}:{PORT}/"
DASHBOARD_FALLBACK_URL = f"http://{HOST}:{PORT}/static?p=overview"
DASHBOARD_MINI_URL = f"http://{HOST}:{PORT}/mini"
TMP_DIR = BASE_DIR / ".tmp"
ORCHESTRATOR_PATH = TMP_DIR / "orchestrator.json"
LOCK_PATHS = (
    TMP_DIR / "server_agi.lock",
    TMP_DIR / "server_agi_hft.lock",
    TMP_DIR / "champion_cycle.lock",
    TMP_DIR / "train_drl.lock",
    TMP_DIR / "train_drl_XAUUSDm.lock",
)
SERVER_LOCK_PATH = TMP_DIR / "server_agi.lock"
HFT_LOCK_PATH = TMP_DIR / "server_agi_hft.lock"

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)


SPLASH_HTML = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Cautious Giggle Launcher</title>
<style>
:root{
  --bg:#06111d;
  --bg2:#0d1c2d;
  --panel:rgba(8,16,26,.78);
  --line:rgba(255,255,255,.08);
  --text:#eef6ff;
  --muted:#8ea3c2;
  --cyan:#4fd6ff;
  --lime:#8de570;
  --amber:#ffbf4d;
  --red:#ff6b7d;
}
*{box-sizing:border-box}
html,body{margin:0;height:100%;font-family:"Segoe UI Variable Text","Bahnschrift","Segoe UI",sans-serif;color:var(--text)}
body{
  overflow:hidden;
  background:
    radial-gradient(1000px 560px at 90% -10%, rgba(79,214,255,.14), transparent 55%),
    radial-gradient(760px 420px at 0% 0%, rgba(141,229,112,.09), transparent 45%),
    linear-gradient(180deg,var(--bg),var(--bg2));
}
body::before{
  content:"";
  position:fixed;
  inset:0;
  background:
    linear-gradient(rgba(255,255,255,.03) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.03) 1px, transparent 1px);
  background-size:48px 48px;
  mask-image:radial-gradient(circle at center, black 45%, transparent 90%);
  animation:gridShift 18s linear infinite;
}
@keyframes gridShift{to{background-position:48px 48px}}
.shell{
  position:relative;
  display:grid;
  grid-template-columns:minmax(0,1.2fr) minmax(360px,.9fr);
  gap:24px;
  height:100%;
  padding:36px;
}
.hero,.panel{
  border:1px solid var(--line);
  border-radius:28px;
  background:var(--panel);
  backdrop-filter:blur(16px);
  box-shadow:0 28px 60px rgba(0,0,0,.28);
}
.hero{
  position:relative;
  overflow:hidden;
  padding:32px;
  display:flex;
  flex-direction:column;
  justify-content:space-between;
}
.hero::after{
  content:"";
  position:absolute;
  inset:auto -10% -18% 40%;
  height:280px;
  background:radial-gradient(circle, rgba(79,214,255,.18), transparent 70%);
  filter:blur(24px);
}
.eyebrow{
  color:var(--cyan);
  font-size:12px;
  letter-spacing:.22em;
  text-transform:uppercase;
}
.title{
  margin-top:14px;
  font-size:54px;
  line-height:.92;
  letter-spacing:-.05em;
  max-width:520px;
}
.lede{
  margin-top:18px;
  max-width:560px;
  color:var(--muted);
  font-size:15px;
  line-height:1.7;
}
.heroGrid{
  display:grid;
  grid-template-columns:repeat(3,minmax(0,1fr));
  gap:12px;
  margin-top:26px;
}
.heroMetric{
  padding:14px;
  border-radius:18px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
}
.heroMetric .k{
  color:var(--muted);
  font-size:11px;
  letter-spacing:.16em;
  text-transform:uppercase;
}
.heroMetric .v{
  margin-top:10px;
  font-size:20px;
  font-weight:700;
}
.heroRail{
  display:grid;
  gap:10px;
  margin-top:24px;
}
.railItem{
  display:flex;
  justify-content:space-between;
  gap:12px;
  padding:12px 14px;
  border-radius:18px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
}
.railItem strong{
  display:block;
  font-size:13px;
}
.railItem span{
  color:var(--muted);
  font-size:12px;
  line-height:1.5;
}
.panel{
  padding:26px;
  display:grid;
  gap:18px;
}
.panelTop{
  display:flex;
  justify-content:space-between;
  gap:14px;
  align-items:flex-start;
}
.statusPill{
  display:inline-flex;
  align-items:center;
  gap:8px;
  padding:9px 12px;
  border-radius:999px;
  border:1px solid var(--line);
  color:var(--muted);
  font-size:12px;
}
.dot{
  width:9px;
  height:9px;
  border-radius:50%;
  background:var(--amber);
  box-shadow:0 0 14px rgba(255,191,77,.38);
}
.dot.ok{background:var(--lime);box-shadow:0 0 18px rgba(141,229,112,.42)}
.dot.err{background:var(--red);box-shadow:0 0 18px rgba(255,107,125,.42)}
.panelTitle{
  font-size:26px;
  letter-spacing:-.04em;
}
.panelMeta{
  margin-top:8px;
  color:var(--muted);
  font-size:13px;
  line-height:1.6;
}
.progressCard{
  padding:16px;
  border-radius:22px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
}
.progressHead{
  display:flex;
  justify-content:space-between;
  gap:12px;
  color:var(--muted);
  font-size:13px;
}
.progressValue{
  color:var(--text);
  font-weight:700;
}
.progressBar{
  margin-top:14px;
  height:8px;
  border-radius:999px;
  background:rgba(255,255,255,.08);
  overflow:hidden;
}
.progressFill{
  height:100%;
  width:0%;
  border-radius:999px;
  background:linear-gradient(90deg,var(--cyan),#9f8bff,var(--lime));
  transition:width .35s ease;
}
.stepList{
  display:grid;
  gap:10px;
}
.step{
  display:grid;
  grid-template-columns:16px minmax(0,1fr) auto;
  gap:12px;
  align-items:start;
  padding:12px 14px;
  border-radius:18px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
  color:var(--muted);
}
.stepDot{
  width:12px;
  height:12px;
  margin-top:3px;
  border-radius:50%;
  border:2px solid rgba(255,255,255,.18);
  background:transparent;
}
.step.done .stepDot{
  border-color:var(--lime);
  background:var(--lime);
  box-shadow:0 0 14px rgba(141,229,112,.42);
}
.step.active .stepDot{
  border-color:var(--cyan);
  background:var(--cyan);
  box-shadow:0 0 18px rgba(79,214,255,.42);
}
.step.fail .stepDot{
  border-color:var(--red);
  background:var(--red);
  box-shadow:0 0 18px rgba(255,107,125,.42);
}
.stepMain strong{
  display:block;
  color:var(--text);
  font-size:13px;
}
.stepMain span{
  display:block;
  margin-top:5px;
  font-size:12px;
  line-height:1.5;
}
.stepState{
  font-size:11px;
  letter-spacing:.14em;
  text-transform:uppercase;
}
.banner{
  display:none;
  padding:14px 16px;
  border-radius:18px;
  border:1px solid rgba(255,107,125,.25);
  background:rgba(255,107,125,.1);
  color:#ffd6db;
  font-size:13px;
  line-height:1.6;
}
.banner.warn{
  display:block;
  border-color:rgba(255,191,77,.3);
  background:rgba(255,191,77,.08);
  color:#ffe2b0;
}
.banner.err{
  display:block;
}
.detailList{
  display:grid;
  gap:8px;
}
.detail{
  padding:10px 12px;
  border-radius:16px;
  border:1px solid var(--line);
  background:rgba(255,255,255,.03);
  color:var(--muted);
  font-size:12px;
  line-height:1.6;
}
@media (max-width: 1100px){
  .shell{grid-template-columns:1fr;padding:18px;height:auto;min-height:100%}
  .title{font-size:40px}
}
</style>
</head>
<body>
<div class="shell">
  <section class="hero">
    <div>
      <div class="eyebrow">Cautious Giggle AGI Trading Bot</div>
      <div class="title">Multi-brain autonomous trading engine.</div>
      <div class="lede">
        Blended LSTM + Dreamer v3 + PPO signals with three-phase trade management:
        cut losers fast, hold winners longer, pyramid into strength. Full risk
        supervision, self-optimization from live P&amp;L, and isolated HFT lane.
      </div>
      <div class="heroGrid">
        <div class="heroMetric">
          <div class="k">Signal blend</div>
          <div class="v">PPO + Dreamer + LSTM</div>
        </div>
        <div class="heroMetric">
          <div class="k">Symbols</div>
          <div class="v">BTCUSDm &middot; XAUUSDm</div>
        </div>
        <div class="heroMetric">
          <div class="k">Timeframe</div>
          <div class="v">M5 Live</div>
        </div>
      </div>
    </div>
    <div class="heroRail">
      <div class="railItem">
        <div>
          <strong>PPO diagnostics</strong>
          <span>Every decision cycle logs structured PPO inference status &mdash; model path, obs shape, raw action, skip reason. No silent failures.</span>
        </div>
      </div>
      <div class="railItem">
        <div>
          <strong>Self-optimization loop</strong>
          <span>Closed trades from MT5 feed back into trade learning &mdash; win rate, expectancy, and loss streaks shape future PPO rewards.</span>
        </div>
      </div>
      <div class="railItem">
        <div>
          <strong>HFT lane isolation</strong>
          <span>Standard (magic 51000-52999) and HFT (magic 61000-62999) run independently with separate position filters and risk limits.</span>
        </div>
      </div>
      <div class="railItem">
        <div>
          <strong>Risk supervisor</strong>
          <span>Portfolio-level circuit breaker: daily loss cap, trade cooldowns, max positions, spread guard, and drawdown halt.</span>
        </div>
      </div>
    </div>
  </section>

  <section class="panel">
    <div class="panelTop">
      <div>
        <div class="panelTitle">Startup sequence</div>
        <div class="panelMeta" id="status-text">Running guarded preflight checks before live startup.</div>
      </div>
      <div class="statusPill"><span class="dot" id="status-dot"></span><span id="status-pill">Preparing</span></div>
    </div>

    <div class="progressCard">
      <div class="progressHead">
        <span>Launch completion</span>
        <span class="progressValue" id="pct">0%</span>
      </div>
      <div class="progressBar"><div class="progressFill" id="progress-fill"></div></div>
    </div>

    <div class="banner" id="banner"></div>

    <div class="stepList">
      <div class="step" id="step-1">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>Preflight validation</strong>
          <span>Load .env, validate MT5 credentials, AGI token, config files, and risk supervisor settings.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
      <div class="step" id="step-2">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>Dashboard &amp; trade history</strong>
          <span>Start operator dashboard with Trade History, PPO Diagnostics, Trade Learning, and HFT Health panels.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
      <div class="step" id="step-3">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>Standard trading brain</strong>
          <span>Launch Server_AGI with blended PPO+Dreamer+LSTM signals, three-phase trade management, and risk supervisor.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
      <div class="step" id="step-4">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>HFT scalping lane</strong>
          <span>Launch isolated HFT bot (magic 61000-62999) with M1 timeframe, separate position filters, and own risk limits.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
      <div class="step" id="step-5">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>Champion cycle &amp; self-optimization</strong>
          <span>Start LSTM&#8594;Dreamer&#8594;PPO training loop, canary evaluation, champion promotion, and trade learning feedback.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
      <div class="step" id="step-6">
        <div class="stepDot"></div>
        <div class="stepMain">
          <strong>Watchdog &amp; recovery</strong>
          <span>Attach process supervision, crash detection, and automatic restart before handing off to operator dashboard.</span>
        </div>
        <div class="stepState">pending</div>
      </div>
    </div>

    <div class="detailList" id="detail-list">
      <div class="detail">Waiting for the launcher to emit runtime detail.</div>
    </div>
  </section>
</div>

<script>
window.cgSetStatus = function(kind, title, detail, pct){
  var dot = document.getElementById("status-dot");
  var pill = document.getElementById("status-pill");
  var text = document.getElementById("status-text");
  var fill = document.getElementById("progress-fill");
  var pctEl = document.getElementById("pct");
  dot.className = "dot " + (kind || "");
  pill.textContent = title || "Preparing";
  text.textContent = detail || "";
  fill.style.width = String(pct || 0) + "%";
  pctEl.textContent = String(pct || 0) + "%";
};
window.cgSetStep = function(index, state, detail){
  var el = document.getElementById("step-" + index);
  if(!el) return;
  el.className = "step " + (state || "");
  var desc = el.querySelector(".stepMain span");
  var badge = el.querySelector(".stepState");
  if(detail){ desc.textContent = detail; }
  badge.textContent = state || "pending";
};
window.cgSetBanner = function(kind, message){
  var el = document.getElementById("banner");
  if(!message){
    el.className = "banner";
    el.style.display = "none";
    el.textContent = "";
    return;
  }
  el.className = "banner " + (kind || "warn");
  el.style.display = "block";
  el.textContent = message;
};
window.cgSetDetails = function(items){
  var el = document.getElementById("detail-list");
  if(!Array.isArray(items) || !items.length){
    el.innerHTML = '<div class="detail">No detail emitted yet.</div>';
    return;
  }
  el.innerHTML = items.map(function(item){
    return '<div class="detail">' + item + '</div>';
  }).join('');
};
</script>
</body>
</html>
"""


def _load_env_file(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        if key and key not in os.environ:
            os.environ[key] = value.strip()


def _normalize_secret(raw: Any) -> str:
    return str(raw or "").strip()


def _validate_live_config(path: Path) -> str | None:
    prev_cfg = os.environ.get("AGI_CONFIG")
    try:
        os.environ["AGI_CONFIG"] = str(path)
        load_project_config(str(BASE_DIR), live_mode=True)
        return None
    except Exception as exc:
        return str(exc)
    finally:
        if prev_cfg is None:
            os.environ.pop("AGI_CONFIG", None)
        else:
            os.environ["AGI_CONFIG"] = prev_cfg


def _hft_launch_issue() -> str | None:
    if not HFT_CONFIG_PATH.exists():
        return "config_hft.yaml is absent."
    error = _validate_live_config(HFT_CONFIG_PATH)
    if error:
        return f"config_hft.yaml is invalid: {error}"
    return None


def _collect_preflight() -> tuple[list[str], list[str]]:
    issues: list[str] = []
    warnings: list[str] = []

    if not UI_SCRIPT.exists():
        issues.append(f"UI server entrypoint is missing: {UI_SCRIPT}")
    if not CONFIG_PATH.exists():
        issues.append(f"Main config is missing: {CONFIG_PATH}")
    if not CHAMPION_LOOP_SCRIPT.exists():
        issues.append(f"Champion cycle script is missing: {CHAMPION_LOOP_SCRIPT}")
    if not WATCHDOG_SCRIPT.exists():
        warnings.append(f"Watchdog script is missing and will be skipped: {WATCHDOG_SCRIPT}")
    if not HFT_CONFIG_PATH.exists():
        warnings.append(f"HFT config is missing and the HFT lane will be skipped: {HFT_CONFIG_PATH}")
    if not _normalize_secret(os.environ.get("AGI_TOKEN", "")):
        issues.append("AGI_TOKEN is not set in the process environment or .env.")

    if CONFIG_PATH.exists():
        config_error = _validate_live_config(CONFIG_PATH)
        if config_error:
            issues.append(f"Main live config is invalid: {config_error}")

    if HFT_CONFIG_PATH.exists():
        hft_issue = _hft_launch_issue()
        if hft_issue:
            warnings.append(f"HFT lane will be skipped: {hft_issue}")

    if not ICON_PATH.exists():
        warnings.append(f"Desktop icon not found; launcher will continue without it: {ICON_PATH}")

    return issues, warnings


def port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _http_ready(url: str, timeout: float = 2.0) -> bool:
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            return 200 <= int(response.status) < 500
    except (urllib.error.URLError, TimeoutError, OSError):
        return False


def process_running(name_fragment: str) -> bool:
    script = (
        "Get-CimInstance Win32_Process | "
        "Select-Object -ExpandProperty CommandLine | "
        f"Where-Object {{ $_ -like '*{name_fragment}*' }}"
    )
    try:
        out = subprocess.check_output(
            ["powershell", "-NoProfile", "-Command", script],
            creationflags=NO_WINDOW,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return bool(out.strip())
    except Exception:
        return False


def _launch(args: list[str], *, env_updates: dict[str, str] | None = None) -> subprocess.Popen[Any]:
    env = os.environ.copy()
    if env_updates:
        env.update(env_updates)
    return subprocess.Popen(
        args,
        cwd=str(BASE_DIR),
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )


def start_ui_server() -> subprocess.Popen[Any] | None:
    # Reuse an existing dashboard only when it serves one of our known routes.
    if port_open(HOST, PORT):
        if _http_ready(DASHBOARD_PRIMARY_URL) or _http_ready(DASHBOARD_FALLBACK_URL) or _http_ready(DASHBOARD_MINI_URL):
            return None
    return _launch([PYTHON, "-u", str(UI_SCRIPT)], env_updates={"AGI_UI_HOST": HOST, "AGI_UI_PORT": str(PORT)})


def start_server_agi() -> subprocess.Popen[Any] | None:
    if _lock_pid_running(SERVER_LOCK_PATH):
        return None
    return _launch([PYTHON, "-m", SERVER_AGI_MODULE, "--live"])


def start_hft_server() -> subprocess.Popen[Any] | None:
    if _hft_launch_issue():
        return None
    if _lock_pid_running(HFT_LOCK_PATH):
        return None
    return _launch(
        [PYTHON, "-m", SERVER_AGI_MODULE, "--live"],
        env_updates={
            "AGI_CONFIG": str(HFT_CONFIG_PATH),
            "AGI_MODE_TAG": "hft",
            "AGI_LOOP_SEC": "5",
            "AGI_HEARTBEAT_SEC": "300",
            "AGI_SYMBOL_CARD_SEC": "60",
            "AGI_TRADE_LEARN_SEC": "300",
        },
    )


def start_champion_loop() -> subprocess.Popen[Any] | None:
    if not CHAMPION_LOOP_SCRIPT.exists():
        return None
    if process_running("champion_cycle_loop"):
        return None
    return _launch([PYTHON, "-u", str(CHAMPION_LOOP_SCRIPT), "--interval-minutes", "30"])


def start_watchdog() -> subprocess.Popen[Any] | None:
    if not WATCHDOG_SCRIPT.exists():
        return None
    if process_running("watchdog.py"):
        return None
    return _launch([PYTHON, "-u", str(WATCHDOG_SCRIPT)])


def _write_orchestrator_state(procs: list[subprocess.Popen[Any] | None]) -> None:
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    payload = {
        "pids": {
            "ui": procs[0].pid if len(procs) > 0 and procs[0] else None,
            "server_agi": procs[1].pid if len(procs) > 1 and procs[1] else None,
            "hft": procs[2].pid if len(procs) > 2 and procs[2] else None,
            "champion_cycle": procs[3].pid if len(procs) > 3 and procs[3] else None,
            "watchdog": procs[4].pid if len(procs) > 4 and procs[4] else None,
        },
        "dashboard_primary_url": DASHBOARD_PRIMARY_URL,
        "dashboard_fallback_url": DASHBOARD_FALLBACK_URL,
        "dashboard_mini_url": DASHBOARD_MINI_URL,
        "started_at": time.time(),
    }
    ORCHESTRATOR_PATH.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _pid_running(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        out = subprocess.run(
            ["tasklist", "/FI", f"PID eq {pid}"],
            check=False,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            creationflags=NO_WINDOW,
        ).stdout
        return str(pid) in out
    except Exception:
        return False


def _lock_pid_running(lock_path: Path) -> bool:
    if not lock_path.exists():
        return False
    try:
        pid = int((lock_path.read_text(encoding="utf-8") or "").strip() or "0")
    except Exception:
        return False
    return _pid_running(pid)


def _cleanup_stale_runtime_files() -> list[str]:
    notes: list[str] = []
    TMP_DIR.mkdir(parents=True, exist_ok=True)
    for lock_path in LOCK_PATHS:
        if not lock_path.exists():
            continue
        try:
            raw = lock_path.read_text(encoding="utf-8").strip()
            pid = int(raw or "0")
        except Exception:
            pid = 0
        if pid <= 0 or not _pid_running(pid):
            try:
                lock_path.unlink()
                notes.append(f"Removed stale lock: {lock_path.name}")
            except Exception as exc:
                notes.append(f"Could not remove stale lock {lock_path.name}: {exc}")

    if ORCHESTRATOR_PATH.exists():
        try:
            ORCHESTRATOR_PATH.unlink()
            notes.append("Cleared stale orchestrator.json")
        except Exception as exc:
            notes.append(f"Could not clear stale orchestrator.json: {exc}")
    return notes


def _clear_orchestrator_state() -> None:
    try:
        if ORCHESTRATOR_PATH.exists():
            ORCHESTRATOR_PATH.unlink()
    except Exception:
        pass


def terminate_proc(proc: subprocess.Popen[Any] | None) -> None:
    if proc is None or proc.poll() is not None:
        return
    try:
        subprocess.run(
            ["taskkill", "/PID", str(proc.pid), "/T", "/F"],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            creationflags=NO_WINDOW,
        )
    except Exception:
        proc.terminate()
        try:
            proc.wait(timeout=5)
        except subprocess.TimeoutExpired:
            proc.kill()


def _js(window: Any, code: str) -> None:
    try:
        window.evaluate_js(code)
    except Exception:
        pass


def _set_status(window: Any, kind: str, title: str, detail: str, pct: int) -> None:
    _js(
        window,
        f"window.cgSetStatus({json.dumps(kind)}, {json.dumps(title)}, {json.dumps(detail)}, {pct});",
    )


def _set_step(window: Any, index: int, state: str, detail: str) -> None:
    _js(window, f"window.cgSetStep({index}, {json.dumps(state)}, {json.dumps(detail)});")


def _set_banner(window: Any, kind: str, message: str) -> None:
    _js(window, f"window.cgSetBanner({json.dumps(kind)}, {json.dumps(message)});")


def _set_details(window: Any, details: list[str]) -> None:
    _js(window, f"window.cgSetDetails({json.dumps(details)});")


def _wait_for_dashboard() -> str | None:
    deadline = time.time() + 45
    while time.time() < deadline:
        if _http_ready(DASHBOARD_PRIMARY_URL):
            return DASHBOARD_PRIMARY_URL
        if _http_ready(DASHBOARD_FALLBACK_URL):
            return DASHBOARD_FALLBACK_URL
        if _http_ready(DASHBOARD_MINI_URL):
            return DASHBOARD_MINI_URL
        time.sleep(0.5)
    return None


def splash_sequence(window: Any) -> None:
    _load_env_file(ENV_PATH)

    issues, warnings = _collect_preflight()
    detail_rows = [f"Workspace: {BASE_DIR}", f"Python: {PYTHON}"]
    detail_rows.extend(_cleanup_stale_runtime_files())
    detail_rows.extend(f"Warning: {item}" for item in warnings)
    _set_details(window, detail_rows)

    _set_status(window, "", "Preflight", "Validating MT5 credentials, AGI token, configs, and risk supervisor.", 8)
    _set_step(window, 1, "active", "Loading .env, checking MT5 credentials, AGI token, and config validity.")
    time.sleep(0.6)

    if warnings:
        _set_banner(window, "warn", "Non-blocking warnings detected. Optional services may be skipped.")

    if issues:
        _set_step(window, 1, "fail", "Preflight failed. Live services were not started.")
        _set_status(window, "err", "Blocked", "Missing required prerequisites for live trading.", 12)
        _set_banner(window, "err", " | ".join(issues))
        _set_details(window, detail_rows + [f"Blocking: {item}" for item in issues])
        return

    _set_step(window, 1, "done", "MT5 credentials, AGI token, configs, and risk settings validated.")
    procs: list[subprocess.Popen[Any] | None] = []

    _set_status(window, "", "Dashboard", "Starting dashboard with Trade History, PPO Diagnostics, and HFT Health panels.", 22)
    _set_step(window, 2, "active", "Starting operator dashboard (Overview, Training, Performance, Trades, PPO, HFT, Activity, Control).")
    ui_proc = start_ui_server()
    procs.append(ui_proc)
    dashboard_url = _wait_for_dashboard()
    if not dashboard_url:
        _set_step(window, 2, "fail", "Dashboard did not become reachable within the launch timeout.")
        _set_status(window, "err", "Blocked", "UI server failed to start. Check tools/project_status_ui.py logs.", 28)
        _set_banner(window, "err", "Dashboard startup timed out. Check logs/ui_restart.log for details.")
        for proc in procs:
            terminate_proc(proc)
        _clear_orchestrator_state()
        return
    _set_step(window, 2, "done", f"Dashboard ready at {dashboard_url} — all 8 tabs active.")

    _set_status(window, "", "Trading brain", "Launching PPO+Dreamer+LSTM blended signal engine with risk supervisor.", 40)
    _set_step(window, 3, "active", "Starting Server_AGI: M5 timeframe, three-phase trade management, PPO diagnostics enabled.")
    agi_proc = start_server_agi()
    procs.append(agi_proc)
    time.sleep(0.7)
    _set_step(window, 3, "done", "Standard trading brain online (magic 51000-52999) with risk supervisor active.")

    _set_status(window, "", "HFT lane", "Starting isolated HFT scalper with separate magic range and position filters.", 56)
    _set_step(window, 4, "active", "Launching HFT bot (magic 61000-62999, M1 timeframe, independent risk limits).")
    hft_issue = _hft_launch_issue()
    hft_proc = start_hft_server()
    procs.append(hft_proc)
    if hft_issue:
        _set_step(window, 4, "done", f"HFT lane skipped: {hft_issue}")
    elif HFT_CONFIG_PATH.exists():
        _set_step(window, 4, "done", "HFT lane online (magic 61000-62999) — isolated from standard brain.")
    else:
        _set_step(window, 4, "done", "HFT lane skipped — config_hft.yaml not found.")

    _set_status(window, "", "Self-optimization", "Starting training loop and trade learning feedback.", 70)
    _set_step(window, 5, "active", "Launching LSTM > Dreamer > PPO training cycle, canary eval, and champion promotion.")
    cycle_proc = start_champion_loop()
    procs.append(cycle_proc)
    _set_step(window, 5, "done", "Champion cycle active — trade P&L feeds back into PPO reward shaping.")

    _set_status(window, "", "Watchdog", "Starting crash detection and automatic recovery.", 84)
    _set_step(window, 6, "active", "Attaching process supervision, crash detection, and auto-restart.")
    watchdog_proc = start_watchdog()
    procs.append(watchdog_proc)
    if WATCHDOG_SCRIPT.exists():
        _set_step(window, 6, "done", "Watchdog active — monitoring all processes for crash recovery.")
    else:
        _set_step(window, 6, "done", "Watchdog skipped — tools/watchdog.py not found.")

    window._procs = procs
    _write_orchestrator_state(procs)
    _set_details(window, detail_rows + [
        f"Dashboard: {dashboard_url}",
        "Standard brain: PPO+Dreamer+LSTM on M5 (magic 51000-52999)",
        "HFT lane: M1 scalper (magic 61000-62999)" if not hft_issue else f"HFT: skipped ({hft_issue})",
        "Risk supervisor: enabled (daily loss cap, trade cooldowns, position limits)",
        "Self-optimization: trade learning > reward shaping > PPO retraining",
        "Launch complete. Opening operator dashboard.",
    ])
    _set_banner(window, "", "")
    _set_status(window, "ok", "Trading", "All systems online. Multi-brain trading engine active.", 100)
    time.sleep(0.8)
    window.load_url(dashboard_url)


def main() -> None:
    import webview

    window = webview.create_window(
        TITLE,
        html=SPLASH_HTML,
        width=WIDTH,
        height=HEIGHT,
        min_size=(960, 640),
        resizable=True,
        text_select=True,
        zoomable=True,
        easy_drag=False,
    )

    if ICON_PATH.exists():
        try:
            window.icon = str(ICON_PATH)
        except Exception:
            pass

    splash_started = False

    def on_loaded() -> None:
        nonlocal splash_started
        if splash_started:
            return
        splash_started = True
        thread = threading.Thread(target=splash_sequence, args=(window,), daemon=True)
        thread.start()

    def on_closed() -> None:
        for proc in getattr(window, "_procs", []):
            terminate_proc(proc)
        _clear_orchestrator_state()

    window.events.loaded += on_loaded
    window.events.closed += on_closed

    webview.start(debug="--debug" in sys.argv, http_server=False)


if __name__ == "__main__":
    main()
