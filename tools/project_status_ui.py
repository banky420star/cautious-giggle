import asyncio
import json
import os
import re
import subprocess
import sys
from collections import defaultdict
from datetime import datetime, timedelta, timezone

from aiohttp import web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

try:
    import yaml
except Exception:
    yaml = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from alerts.telegram_alerts import TelegramAlerter
from Python.model_registry import ModelRegistry

LOG_DIR = os.path.join(ROOT, "logs")
ACTIVE_PATH = os.path.join(ROOT, "models", "registry", "active.json")


def _venv_python():
    return os.path.join(ROOT, ".venv312", "Scripts", "python.exe")


def _tail(path, lines=60):
    if not os.path.exists(path):
        return []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            data = f.readlines()
        return [x.rstrip("\n") for x in data[-lines:]]
    except Exception:
        return []


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd=ROOT)
    except Exception as exc:
        return f"ERROR: {exc}"


def _run_ps(command):
    return _run(["powershell", "-NoProfile", "-Command", command])


def _powershell_json(command):
    raw = _run_ps(command)
    try:
        data = json.loads(raw)
        if isinstance(data, dict):
            return [data]
        if isinstance(data, list):
            return data
    except Exception:
        pass
    return []


def _load_cfg():
    cfg_path = os.path.join(ROOT, "config.yaml")
    if not os.path.exists(cfg_path) or yaml is None:
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _build_alerter():
    cfg = _load_cfg()
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    if not token or not chat_id:
        return TelegramAlerter(None, None)
    return TelegramAlerter(token, str(chat_id))



def _init_mt5_from_cfg():
    if mt5 is None:
        return False
    cfg = _load_cfg()
    mt5_cfg = cfg.get("mt5", {}) if isinstance(cfg, dict) else {}
    login_raw = os.environ.get("MT5_LOGIN") or _resolve_cfg_value(mt5_cfg.get("login", 0))
    password = os.environ.get("MT5_PASSWORD") or _resolve_cfg_value(mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER") or _resolve_cfg_value(mt5_cfg.get("server", ""))
    try:
        login = int(login_raw or 0)
    except Exception:
        login = 0
    if login and password and server:
        return bool(mt5.initialize(login=login, password=password, server=server))
    return bool(mt5.initialize())
def _active_models():
    if not os.path.exists(ACTIVE_PATH):
        return {"champion": None, "canary": None}
    try:
        with open(ACTIVE_PATH, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"champion": None, "canary": None}


def _processes():
    rows = _powershell_json(
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Select-Object ProcessId,ParentProcessId,Name,CommandLine,CreationDate | ConvertTo-Json -Depth 4"
    )
    out = []
    for p in rows:
        cmd = str(p.get("CommandLine") or "")
        name = str(p.get("Name") or "")
        if "cautious-giggle" in cmd.lower() or name.lower().startswith("python"):
            out.append(
                {
                    "pid": p.get("ProcessId"),
                    "ppid": p.get("ParentProcessId"),
                    "name": name,
                    "cmd": cmd,
                    "created": p.get("CreationDate"),
                }
            )
    return out


def _filter_cmd(procs, token):
    t = token.lower().replace("\\", "/")
    out = []
    for p in procs:
        cmd = (p.get("cmd") or "").lower().replace("\\", "/")
        if t in cmd:
            out.append(p)
    return out

def _is_running(token: str) -> bool:
    return len(_filter_cmd(_processes(), token)) > 0


def _latest_training_progress() -> dict:
    out = {
        "drl_symbol": None,
        "drl_timesteps": None,
        "drl_candles": None,
        "lstm_symbol": None,
        "lstm_epoch": None,
        "lstm_epochs_total": None,
        "train_error": None,
    }
    ppo_lines = _tail(os.path.join(LOG_DIR, "ppo_training.log"), 200)
    lstm_lines = _tail(os.path.join(LOG_DIR, "lstm_training.log"), 200)

    drl_re = re.compile(r"DRL Training \| symbols=\['([^']+)'\].*timesteps=([0-9,]+).*(?:candles=([0-9,]+))?")
    lstm_re = re.compile(r"([A-Za-z0-9_]+)\s*\|\s*epoch\s+(\d+)\s*/\s*(\d+)")
    err_re = re.compile(r"(Authorization failed|insufficient MT5 data|MT5 initialize failed)", re.IGNORECASE)

    for line in reversed(ppo_lines):
        m = drl_re.search(line)
        if m:
            out["drl_symbol"] = m.group(1)
            out["drl_timesteps"] = m.group(2)
            out["drl_candles"] = m.group(3) if m.lastindex and m.lastindex >= 3 else None
            break

    for line in reversed(lstm_lines):
        m = lstm_re.search(line)
        if m:
            out["lstm_symbol"] = m.group(1)
            out["lstm_epoch"] = m.group(2)
            out["lstm_epochs_total"] = m.group(3)
            break

    for line in reversed(ppo_lines + lstm_lines):
        if err_re.search(line):
            out["train_error"] = line
            break

    return out


def _training_state(procs):
    drl = _filter_cmd(procs, "training/train_drl.py")
    lstm = _filter_cmd(procs, "training/train_lstm.py")
    cycle = _filter_cmd(procs, "tools/champion_cycle_loop.py")
    progress = _latest_training_progress()
    return {
        "drl_running": len(drl) > 0,
        "lstm_running": len(lstm) > 0,
        "cycle_running": len(cycle) > 0,
        "drl_pids": [p["pid"] for p in drl],
        "lstm_pids": [p["pid"] for p in lstm],
        "cycle_pids": [p["pid"] for p in cycle],
        "drl_symbol": progress.get("drl_symbol"),
        "drl_timesteps": progress.get("drl_timesteps"),
        "drl_candles": progress.get("drl_candles"),
        "lstm_symbol": progress.get("lstm_symbol"),
        "lstm_epoch": progress.get("lstm_epoch"),
        "lstm_epochs_total": progress.get("lstm_epochs_total"),
        "train_error": progress.get("train_error"),
    }


def _server_state(procs):
    servers = _filter_cmd(procs, "python.server_agi")
    return {"running": len(servers) > 0, "pids": [p["pid"] for p in servers]}


def _mt5_snapshot():
    base = {
        "connected": False,
        "balance": None,
        "equity": None,
        "profit": None,
        "free_margin": None,
        "open_positions": 0,
        "positions": [],
    }
    if mt5 is None:
        return base

    try:
        if not _init_mt5_from_cfg():
            return base

        info = mt5.account_info()
        if info:
            base["connected"] = True
            base["balance"] = float(info.balance)
            base["equity"] = float(info.equity)
            base["profit"] = float(info.profit)
            base["free_margin"] = float(info.margin_free)

        positions = mt5.positions_get() or []
        base["open_positions"] = len(positions)
        rows = []
        for p in positions:
            rows.append(
                {
                    "ticket": int(p.ticket),
                    "symbol": str(p.symbol),
                    "type": "BUY" if int(p.type) == 0 else "SELL",
                    "volume": float(p.volume),
                    "profit": float(p.profit),
                    "open_price": float(p.price_open),
                    "current_price": float(p.price_current),
                    "sl": float(p.sl) if p.sl else 0.0,
                    "tp": float(p.tp) if p.tp else 0.0,
                }
            )
        base["positions"] = rows
    except Exception:
        return base

    return base


def _mt5_symbol_perf(days=7, max_points=24):
    if mt5 is None:
        return []
    try:
        if not _init_mt5_from_cfg():
            return []
        end = datetime.now(timezone.utc)
        start = end - timedelta(days=int(days))
        deals = mt5.history_deals_get(start, end)
        if not deals:
            return []

        by_symbol = defaultdict(list)
        for d in deals:
            if int(getattr(d, "entry", -1)) != int(mt5.DEAL_ENTRY_OUT):
                continue
            by_symbol[str(getattr(d, "symbol", "?"))].append(d)

        out = []
        for sym, rows in by_symbol.items():
            rows = sorted(rows, key=lambda x: int(getattr(x, "time", 0)))
            pnl = 0.0
            wins = 0
            trades = 0
            curve = []
            for d in rows:
                val = float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))
                pnl += val
                trades += 1
                if val > 0:
                    wins += 1
                curve.append(round(pnl, 2))

            if len(curve) > max_points:
                step = max(1, len(curve) // max_points)
                curve = curve[::step][-max_points:]

            out.append(
                {
                    "symbol": sym,
                    "trades": trades,
                    "wins": wins,
                    "win_rate": round((wins / trades) * 100.0, 2) if trades else 0.0,
                    "pnl": round(pnl, 2),
                    "curve": curve,
                }
            )
        out.sort(key=lambda x: x["pnl"], reverse=True)
        return out[:16]
    except Exception:
        return []


def read_status():
    procs = _processes()
    reg = ModelRegistry()
    canary_ok, canary_reason = reg.can_promote_canary()
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ROOT,
        "active_models": _active_models(),
        "canary_gate": {"ready": bool(canary_ok), "reason": canary_reason},
        "server": _server_state(procs),
        "training": _training_state(procs),
        "account": _mt5_snapshot(),
        "symbol_perf": _mt5_symbol_perf(7),
        "logs": {
            "server": _tail(os.path.join(LOG_DIR, "server.log"), 50),
            "lstm": _tail(os.path.join(LOG_DIR, "lstm_training.log"), 50),
            "ppo": _tail(os.path.join(LOG_DIR, "ppo_training.log"), 50),
            "audit": _tail(os.path.join(LOG_DIR, "audit_events.jsonl"), 30),
        },
    }


def _spawn(args, stdout_name, stderr_name, env=None):
    os.makedirs(LOG_DIR, exist_ok=True)
    out = open(os.path.join(LOG_DIR, stdout_name), "a", encoding="utf-8")
    err = open(os.path.join(LOG_DIR, stderr_name), "a", encoding="utf-8")
    cmd_env = os.environ.copy()
    if env:
        cmd_env.update(env)
    try:
        proc = subprocess.Popen(args, cwd=ROOT, stdout=out, stderr=err, env=cmd_env)
        return proc.pid
    finally:
        out.close()
        err.close()


def _kill_by_token(token):
    token = token.lower()
    killed = []
    for p in _processes():
        cmd = (p.get("cmd") or "").lower()
        if token in cmd:
            pid = int(p["pid"])
            subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"], check=False)
            killed.append(pid)
    return killed


def control_action(action, payload):
    reg = ModelRegistry()
    try:
        if action == "start_drl":
            if _is_running("training/train_drl.py"):
                return {"ok": True, "message": "PPO training already running"}
            timesteps = str(int(payload.get("timesteps", 500000)))
            pid = _spawn([_venv_python(), "training/train_drl.py"], "train_drl_ui_stdout.log", "train_drl_ui_stderr.log", env={"AGI_DRL_TIMESTEPS": timesteps})
            return {"ok": True, "message": f"PPO training started pid={pid}, timesteps={timesteps}"}

        if action == "stop_drl":
            ids = _kill_by_token("training/train_drl.py")
            return {"ok": True, "message": f"Stopped DRL pids={ids}"}

        if action == "start_lstm":
            if _is_running("training/train_lstm.py"):
                return {"ok": True, "message": "LSTM training already running"}
            pid = _spawn([_venv_python(), "training/train_lstm.py"], "train_lstm_ui_stdout.log", "train_lstm_ui_stderr.log")
            return {"ok": True, "message": f"LSTM training started pid={pid}"}

        if action == "stop_lstm":
            ids = _kill_by_token("training/train_lstm.py")
            return {"ok": True, "message": f"Stopped LSTM pids={ids}"}

        if action == "run_cycle":
            if _is_running("tools/champion_cycle_loop.py") or _is_running("tools/champion_cycle.py"):
                return {"ok": True, "message": "Champion cycle already running"}
            pid = _spawn([_venv_python(), "tools/champion_cycle.py"], "champion_cycle_stdout.log", "champion_cycle_stderr.log")
            return {"ok": True, "message": f"Champion cycle started pid={pid}"}

        if action == "restart_server":
            _kill_by_token("python.server_agi")
            pid = _spawn([_venv_python(), "-m", "Python.Server_AGI", "--live"], "server_stdout.log", "server_stderr.log")
            return {"ok": True, "message": f"Server restarted pid={pid}"}

        if action == "set_canary_latest":
            cands = sorted(
                [os.path.join(reg.candidates_dir, d) for d in os.listdir(reg.candidates_dir) if os.path.isdir(os.path.join(reg.candidates_dir, d))],
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
            if not cands:
                return {"ok": False, "message": "No candidates found"}
            reg.set_canary(cands[0])
            return {"ok": True, "message": f"Canary set to {cands[0]}"}

        if action == "promote_canary":
            reg.promote_canary_to_champion()
            return {"ok": True, "message": "Canary promoted to champion"}

        if action == "promote_canary_force":
            reg.promote_canary_to_champion(force=True)
            return {"ok": True, "message": "Canary force-promoted to champion"}

        if action == "rollback_canary":
            reg.rollback_to_champion()
            return {"ok": True, "message": "Canary rolled back to champion"}
    except Exception as exc:
        return {"ok": False, "message": str(exc)}

    return {"ok": False, "message": f"Unknown action: {action}"}


HTML = """<!doctype html><html><head><meta charset='utf-8'/><meta name='viewport' content='width=device-width, initial-scale=1'/><title>AGI TradeOS Live</title>
<style>
:root{--bg0:#07090f;--bg1:#0c111d;--glass:rgba(17,23,38,.58);--line:rgba(255,255,255,.12);--ink:#edf2ff;--muted:#9ca8c6;--good:#38f4a3;--bad:#ff6a7e}
*{box-sizing:border-box}body{margin:0;font-family:"SF Pro Display","Segoe UI",sans-serif;color:var(--ink);background:radial-gradient(1200px 700px at 90% -10%, rgba(143,123,255,.22), transparent 60%),radial-gradient(900px 500px at -10% 10%, rgba(108,209,255,.20), transparent 60%),linear-gradient(160deg,var(--bg0),var(--bg1));min-height:100vh}
.shell{max-width:1320px;margin:0 auto;padding:18px}.top{display:flex;justify-content:space-between;align-items:center;gap:16px;margin-bottom:12px;padding:16px 18px;border:1px solid var(--line);border-radius:20px;background:var(--glass)}
.title{font-size:28px;font-weight:700}.sub{color:var(--muted);font-size:13px}.grid{display:grid;grid-template-columns:repeat(12,1fr);gap:12px}.card{background:var(--glass);border:1px solid var(--line);border-radius:18px;padding:14px}
.kpi{grid-column:span 3}.kpi .label{font-size:12px;color:var(--muted)}.kpi .val{font-size:26px;font-weight:700;margin-top:4px}.good{color:var(--good)}.bad{color:var(--bad)}.wide{grid-column:span 6}.full{grid-column:1/-1}
.head{font-size:12px;color:var(--muted);margin-bottom:8px;text-transform:uppercase;letter-spacing:.7px}.mono{font-family:Consolas,monospace;font-size:12px;line-height:1.45;white-space:pre-wrap;max-height:260px;overflow:auto}
.chip{display:inline-flex;align-items:center;padding:4px 8px;border:1px solid var(--line);border-radius:999px;font-size:11px;color:var(--muted);margin-right:6px;margin-bottom:6px}.btn{background:#16233d;border:1px solid #334b7a;color:#dce9ff;padding:8px 10px;border-radius:10px;cursor:pointer;font-size:12px}
.controls{display:flex;flex-wrap:wrap;gap:8px}.symGrid{display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:10px}.sym{padding:10px;border:1px solid var(--line);border-radius:12px;background:rgba(8,13,22,.45)} .spark{width:100%;height:48px}
table{width:100%;border-collapse:collapse;font-size:12px}th,td{padding:8px 6px;border-bottom:1px solid rgba(255,255,255,.08);text-align:left}th{color:var(--muted)}
@media (max-width:980px){.kpi{grid-column:span 6}.wide{grid-column:1/-1}}
</style></head><body>
<div class='shell'><div class='top'><div><div class='title'>AGI TradeOS Live</div><div class='sub' id='meta'>connecting...</div></div><div class='sub' id='live'>WebSocket</div></div>
<div class='grid'>
<div class='card kpi'><div class='label'>Balance</div><div class='val' id='balance'>-</div></div><div class='card kpi'><div class='label'>Equity</div><div class='val' id='equity'>-</div></div><div class='card kpi'><div class='label'>Open Trades</div><div class='val' id='trades'>-</div></div><div class='card kpi'><div class='label'>Unrealized PnL</div><div class='val' id='pnl'>-</div></div>
<div class='card wide'><div class='head'>Models / Runtime</div><div id='models'></div><div id='runtime'></div></div><div class='card wide'><div class='head'>Training</div><div id='training'></div></div>
<div class='card full'><div class='head'>Controls</div><div class='controls'><button class='btn' onclick="act('start_lstm')">Start LSTM</button><button class='btn' onclick="act('stop_lstm')">Stop LSTM</button><button class='btn' onclick="act('start_drl',{timesteps:500000})">Start PPO</button><button class='btn' onclick="act('stop_drl')">Stop PPO</button><button class='btn' onclick="act('run_cycle')">Run Full Cycle</button><button class='btn' onclick="act('set_canary_latest')">Set Latest Canary</button><button class='btn' onclick="act('promote_canary')">Promote Canary</button><button class='btn' onclick="act('rollback_canary')">Rollback Canary</button><button class='btn' onclick="act('restart_server')">Restart Server</button></div><div class='sub' id='ctrlMsg'></div></div>
<div class='card full'><div class='head'>Per-Symbol Performance (7d)</div><div class='symGrid' id='symGrid'></div></div>
<div class='card full'><div class='head'>Open Positions</div><div style='overflow:auto'><table><thead><tr><th>Ticket</th><th>Symbol</th><th>Side</th><th>Volume</th><th>PnL</th><th>Open</th><th>Current</th><th>SL</th><th>TP</th></tr></thead><tbody id='pos'></tbody></table></div></div>
<div class='card wide'><div class='head'>Server Log</div><div class='mono' id='server'></div></div><div class='card wide'><div class='head'>PPO Log</div><div class='mono' id='ppo'></div></div><div class='card wide'><div class='head'>LSTM Log</div><div class='mono' id='lstm'></div></div><div class='card wide'><div class='head'>Audit</div><div class='mono' id='audit'></div></div>
</div></div>
<script>
const byId=(i)=>document.getElementById(i), fmt=(v)=>v===null||v===undefined?'-':Number(v).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
function spark(points){ if(!points||points.length<2) return ''; const w=220,h=48,p=4; const min=Math.min(...points),max=Math.max(...points),span=(max-min)||1; const coords=points.map((v,i)=>{const x=p+i*(w-2*p)/(points.length-1); const y=h-p-((v-min)/span)*(h-2*p); return `${x},${y}`}).join(' '); return `<svg class='spark' viewBox='0 0 ${w} ${h}'><polyline fill='none' stroke='#6cd1ff' stroke-width='2' points='${coords}'/></svg>`; }
function render(d){ const a=d.account||{},t=d.training||{},s=d.server||{},m=d.active_models||{}; byId('meta').textContent=`UTC ${d.timestamp_utc}`; byId('balance').textContent=fmt(a.balance); byId('equity').textContent=fmt(a.equity); byId('trades').textContent=String(a.open_positions??0); const p=a.profit??0; const pe=byId('pnl'); pe.textContent=fmt(p); pe.className='val '+(p>=0?'good':'bad'); byId('models').innerHTML=`<span class='chip'>Champion: ${m.champion||'none'}</span><span class='chip'>Canary: ${m.canary||'none'}</span>`; byId('runtime').innerHTML=`<span class='chip'>Server: ${s.running?'RUNNING':'STOPPED'}</span><span class='chip'>PIDs: ${(s.pids||[]).join(', ')||'-'}</span><span class='chip'>MT5: ${a.connected?'CONNECTED':'DISCONNECTED'}</span>`; byId('training').innerHTML=`<span class='chip'>PPO: ${t.drl_running?'TRAINING':'IDLE'}</span><span class='chip'>PPO Symbol: ${t.drl_symbol||'-'}</span><span class='chip'>PPO Steps: ${t.drl_timesteps||'-'}</span><span class='chip'>PPO Candles: ${t.drl_candles||'-'}</span><span class='chip'>LSTM: ${t.lstm_running?'TRAINING':'IDLE'}</span><span class='chip'>LSTM Symbol: ${t.lstm_symbol||'-'}</span><span class='chip'>LSTM Epoch: ${(t.lstm_epoch&&t.lstm_epochs_total)?`${t.lstm_epoch}/${t.lstm_epochs_total}`:'-'}</span><span class='chip'>Cycle: ${t.cycle_running?'RUNNING':'IDLE'}</span><span class='chip'>PPO PID(s): ${(t.drl_pids||[]).join(', ')||'-'}</span><span class='chip'>LSTM PID(s): ${(t.lstm_pids||[]).join(', ')||'-'}</span><span class='chip'>Cycle PID(s): ${(t.cycle_pids||[]).join(', ')||'-'}</span><span class='chip'>Train Error: ${t.train_error||'none'}</span>`;
const rows=(a.positions||[]).map(p=>`<tr><td>${p.ticket}</td><td>${p.symbol}</td><td>${p.type}</td><td>${p.volume}</td><td class='${p.profit>=0?'good':'bad'}'>${fmt(p.profit)}</td><td>${p.open_price}</td><td>${p.current_price}</td><td>${p.sl}</td><td>${p.tp}</td></tr>`).join(''); byId('pos').innerHTML=rows||'<tr><td colspan="9">No open trades</td></tr>';
const cards=(d.symbol_perf||[]).map(s=>`<div class='sym'><div style='display:flex;justify-content:space-between'><strong>${s.symbol}</strong><span class='${s.pnl>=0?'good':'bad'}'>${fmt(s.pnl)}</span></div><div class='sub'>Trades ${s.trades} | Win ${s.win_rate}%</div>${spark(s.curve)}</div>`).join(''); byId('symGrid').innerHTML=cards||'<div class="sub">No closed deals in selected window.</div>';
byId('server').textContent=(d.logs?.server||[]).join(String.fromCharCode(10)); byId('ppo').textContent=(d.logs?.ppo||[]).join(String.fromCharCode(10)); byId('lstm').textContent=(d.logs?.lstm||[]).join(String.fromCharCode(10)); byId('audit').textContent=(d.logs?.audit||[]).join(String.fromCharCode(10)); }
let ws;
async function pollOnce(){ try{ const r=await fetch('/api/status',{cache:'no-store'}); if(!r.ok) return; const d=await r.json(); render(d); if(ws==null||ws.readyState!==1){ byId('live').textContent='HTTP Polling'; } }catch(_){} }
function connect(){ const proto=location.protocol==='https:'?'wss':'ws'; ws=new WebSocket(`${proto}://${location.host}/ws`); ws.onopen=()=>byId('live').textContent='WebSocket Connected'; ws.onclose=()=>{byId('live').textContent='Reconnecting...'; setTimeout(connect,1200)}; ws.onerror=()=>ws.close(); ws.onmessage=(ev)=>{try{render(JSON.parse(ev.data))}catch(_){}}; }
async function act(action,payload={}){ try{ const r=await fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({action,...payload})}); const d=await r.json(); byId('ctrlMsg').textContent=`${new Date().toLocaleTimeString()} :: ${d.message||'ok'}`; }catch(e){ byId('ctrlMsg').textContent='Control failed: '+e; } }
connect();
pollOnce();
setInterval(pollOnce, 2000);
</script></body></html>"""


async def index(_request):
    return web.Response(text=HTML, content_type="text/html")


async def api_status(_request):
    return web.json_response(read_status())


async def api_control(request):
    data = await request.json()
    action = str(data.get("action", "")).strip()
    result = control_action(action, data)
    alerter = request.app.get("alerter")
    if alerter and result.get("ok"):
        alerter.alert(f"UI control executed: {action} | {result.get('message')}")
    return web.json_response(result)


async def ws_status(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str(json.dumps(read_status(), ensure_ascii=False))
            await asyncio.sleep(2)
    except Exception:
        pass
    finally:
        await ws.close()
    return ws


async def notify_loop(app):
    alerter = app.get("alerter")
    prev = None
    while True:
        try:
            d = read_status()
            cur = {
                "server": d["server"]["running"],
                "drl": d["training"]["drl_running"],
                "lstm": d["training"]["lstm_running"],
                "champ": d["active_models"].get("champion"),
                "canary": d["active_models"].get("canary"),
            }
            if prev is not None and alerter is not None:
                for k, v in cur.items():
                    if v != prev.get(k):
                        alerter.alert(f"Dashboard event: {k} changed -> {v}")
            prev = cur
        except Exception:
            pass
        await asyncio.sleep(8)


async def on_startup(app):
    app["alerter"] = _build_alerter()
    app["notify_task"] = asyncio.create_task(notify_loop(app))


def run(host="127.0.0.1", port=8088):
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/control", api_control)
    app.router.add_get("/ws", ws_status)
    app.on_startup.append(on_startup)
    web.run_app(app, host=host, port=int(port))


if __name__ == "__main__":
    run(host=os.environ.get("AGI_UI_HOST", "127.0.0.1"), port=int(os.environ.get("AGI_UI_PORT", "8088")))







