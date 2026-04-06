import asyncio
import ast
import json
import os
import re
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo

from aiohttp import web

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

# Mirror Server_AGI startup so the dashboard sees the same MT5/Telegram config.
ENV_PATH = os.path.join(ROOT, ".env")
if os.path.exists(ENV_PATH):
    try:
        with open(ENV_PATH, "r", encoding="utf-8") as _env_file:
            for _line in _env_file:
                _line = _line.strip()
                if _line and not _line.startswith("#") and "=" in _line:
                    _k, _v = _line.split("=", 1)
                    os.environ.setdefault(_k.strip(), _v.strip())
    except Exception:
        pass

try:
    import yaml
except Exception:
    yaml = None

try:
    import MetaTrader5 as mt5
except Exception:
    mt5 = None

from alerts.telegram_alerts import TelegramAlerter
from Python.config_utils import DEFAULT_TRADING_SYMBOLS
from Python.pattern_recognition import get_pattern_library
from Python.perpetual_improvement import export_perpetual_improvement_state
from Python.model_registry import ModelRegistry

LOG_DIR = os.path.join(ROOT, "logs")
UI_ASSET_DIR = os.path.join(ROOT, "tools", "ui_assets")
UI_HTML_PATH = os.path.join(UI_ASSET_DIR, "project_status_ui.html")
MINI_UI_HTML_PATH = os.path.join(UI_ASSET_DIR, "telegram_mini_app.html")
ACTIVE_PATH = os.path.join(ROOT, "models", "registry", "active.json")
EVENT_INTEL_PATH = os.path.join(LOG_DIR, "event_intel_state.json")
ACCOUNT_HISTORY_PATH = os.path.join(LOG_DIR, "account_history.jsonl")
LOG_TS_FMT = "%Y-%m-%d %H:%M:%S"
ACCOUNT_HISTORY_INTERVAL_SECONDS = 5
_JSONL_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per JSONL file before rotation
FRONTEND_DIST_DIR = os.path.join(ROOT, "frontend", "dist")


def _ui_pattern_library(limit_per_symbol: int = 1) -> dict:
    try:
        raw = get_pattern_library() or {}
    except Exception:
        raw = {}
    if not isinstance(raw, dict):
        return {}
    configured = {str(sym) for sym in _configured_symbols()}
    by_symbol = {}
    for pattern_name, payload in raw.items():
        if not isinstance(payload, dict):
            continue
        symbol = str(payload.get("symbol") or "").strip()
        if not symbol or (configured and symbol not in configured):
            continue
        discovered_at = str(payload.get("discovered_at") or "")
        entry = {"pattern_name": pattern_name, **payload}
        by_symbol.setdefault(symbol, []).append((discovered_at, pattern_name, entry))
    compact = {}
    for symbol, rows in by_symbol.items():
        rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
        for _, pattern_name, entry in rows[: max(1, int(limit_per_symbol))]:
            compact[pattern_name] = entry
    return compact


def _rotate_jsonl_if_needed(path: str) -> None:
    """Rename path -> path.1 (keeping one backup) when the file exceeds _JSONL_MAX_BYTES."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= _JSONL_MAX_BYTES:
            backup = path + ".1"
            if os.path.exists(backup):
                os.remove(backup)
            os.rename(path, backup)
    except Exception:
        pass
TELEGRAM_CARD_SYNC_SECONDS = 45
_PATT_SERVICE_ADDED = False
from aiohttp import web as _aioweb
def _noop(*a, **k):
    pass

async def api_patterns(_request):
    try:
        lib = _ui_pattern_library()
        items = []
        for k, v in lib.items():
            if isinstance(v, dict):
                items.append({**v, 'pattern_name': k})
            else:
                items.append({'pattern_name': k, 'details': v})
        return _aioweb.web.json_response(items)
    except Exception:
        return _aioweb.web.json_response([])

async def api_performance(_request):
    try:
        return _aioweb.web.json_response(export_perpetual_improvement_state())
    except Exception:
        return _aioweb.web.json_response({})

async def api_frontend(_request):
    path = os.path.join(ROOT, 'frontend', 'index.html')
    if os.path.exists(path):
        return _aioweb.web.FileResponse(path)
    return _aioweb.web.Response(text='<html><body>Frontend not found</body></html>', content_type='text/html')
_ACCOUNT_HISTORY_LAST_TS = None
_ACCOUNT_HISTORY_LAST_SIG = None
STATUS_CACHE = {
    "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    "repo_root": ROOT,
    "state": "booting",
}
_STATUS_REFRESH_TASK = None
_STATUS_REFRESH_STARTED_AT = None
_STATUS_REFRESH_DEGRADED = False


def _venv_python():
    return os.path.join(ROOT, ".venv312", "Scripts", "python.exe")


def _tail(path, lines=60):
    if not os.path.exists(path):
        return []
    try:
        # Seek-based tail: read from the end so large files aren't loaded in full.
        chunk = 8192
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            buf = b""
            pos = size
            found = 0
            while pos > 0 and found <= lines:
                read_size = min(chunk, pos)
                pos -= read_size
                f.seek(pos)
                block = f.read(read_size)
                buf = block + buf
                found = buf.count(b"\n")
            raw_lines = buf.decode("utf-8", errors="replace").splitlines()
            if raw_lines and not buf.endswith(b"\n"):
                # last line may be partial – keep it
                pass
            return raw_lines[-lines:]
    except Exception:
        return []


def _line_ts_utc(line: str):
    try:
        raw = str(line)[:19]
        dt = datetime.strptime(raw, LOG_TS_FMT)
        return dt.replace(tzinfo=_log_timezone()).astimezone(timezone.utc)
    except Exception:
        return None


_LOG_TZ_CACHE = None
_LOG_TZ_CFG_MTIME: float = -1.0


def _log_timezone():
    global _LOG_TZ_CACHE, _LOG_TZ_CFG_MTIME
    cfg_path = os.path.join(ROOT, "config.yaml")
    try:
        mtime = os.path.getmtime(cfg_path) if os.path.exists(cfg_path) else 0.0
    except Exception:
        mtime = 0.0
    if _LOG_TZ_CACHE is not None and mtime == _LOG_TZ_CFG_MTIME:
        return _LOG_TZ_CACHE
    cfg = _load_cfg()
    runtime_cfg = cfg.get("runtime", {}) if isinstance(cfg, dict) else {}
    candidates = [
        os.environ.get("AGI_LOG_TIMEZONE"),
        os.environ.get("TZ"),
        runtime_cfg.get("timezone"),
        "Europe/Berlin",
    ]
    for name in candidates:
        if not name:
            continue
        try:
            tz = ZoneInfo(str(name))
            _LOG_TZ_CACHE = tz
            _LOG_TZ_CFG_MTIME = mtime
            return tz
        except Exception:
            continue
    _LOG_TZ_CACHE = timezone.utc
    _LOG_TZ_CFG_MTIME = mtime
    return timezone.utc


def _is_recent_log_line(line: str, minutes: int = 20) -> bool:
    ts = _line_ts_utc(line)
    if ts is None:
        return False
    delta = datetime.now(timezone.utc) - ts
    return timedelta(0) <= delta <= timedelta(minutes=max(1, int(minutes)))


def _run(cmd):
    try:
        return subprocess.check_output(cmd, stderr=subprocess.STDOUT, text=True, cwd=ROOT, timeout=8)
    except subprocess.TimeoutExpired:
        return "ERROR: timeout"
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


_CFG_CACHE: dict | None = None
_CFG_MTIME: float = 0.0


def _load_cfg():
    global _CFG_CACHE, _CFG_MTIME
    cfg_path = os.path.join(ROOT, "config.yaml")
    if not os.path.exists(cfg_path) or yaml is None:
        return {}
    try:
        mtime = os.path.getmtime(cfg_path)
        if _CFG_CACHE is not None and mtime == _CFG_MTIME:
            return _CFG_CACHE
        with open(cfg_path, "r", encoding="utf-8") as f:
            _CFG_CACHE = yaml.safe_load(f) or {}
        _CFG_MTIME = mtime
        return _CFG_CACHE
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


def _trade_learning_status():
    path = os.path.join(LOG_DIR, "learning", "trade_learning_latest.json")
    empty = {
        "available": False,
        "trades": 0,
        "win_rate": 0.0,
        "expectancy": 0.0,
        "profit_factor": 0.0,
        "total_pnl": 0.0,
        "generated_at_utc": None,
        "best_symbols": [],
        "worst_symbols": [],
        "by_symbol": [],
    }
    if not os.path.exists(path):
        return empty
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
        by_symbol_raw = d.get("by_symbol", []) if isinstance(d.get("by_symbol"), list) else []
        by_symbol = []
        for row in by_symbol_raw:
            if not isinstance(row, dict):
                continue
            by_symbol.append({
                "symbol": str(row.get("symbol", "")),
                "trades": int(row.get("trades", 0)),
                "wins": int(row.get("wins", 0)),
                "losses": int(row.get("losses", 0)),
                "win_rate": round(float(row.get("win_rate", 0.0)), 2),
                "expectancy": round(float(row.get("expectancy", 0.0)), 4),
                "profit_factor": round(float(row.get("profit_factor", 0.0)), 4),
                "total_pnl": round(float(row.get("total_pnl", 0.0)), 2),
                "recent_loss_streak": int(row.get("recent_loss_streak", 0)),
                "max_loss_streak": int(row.get("max_loss_streak", 0)),
                "avg_hold_minutes": round(float(row.get("avg_hold_minutes", 0.0)), 2),
            })
        return {
            "available": True,
            "trades": int(d.get("trades", 0)),
            "win_rate": float(d.get("win_rate", 0.0)),
            "expectancy": float(d.get("expectancy", 0.0)),
            "profit_factor": float(d.get("profit_factor", 0.0)),
            "total_pnl": float(d.get("total_pnl", 0.0)),
            "generated_at_utc": d.get("generated_at_utc"),
            "best_symbols": d.get("best_symbols", [])[:3],
            "worst_symbols": d.get("worst_symbols", [])[:3],
            "by_symbol": by_symbol,
        }
    except Exception:
        return empty


def _event_intel_status():
    if not os.path.exists(EVENT_INTEL_PATH):
        return {
            "enabled": False,
            "updated_utc": None,
            "summary": {"upcoming_24h": 0, "active_window": 0, "high_upcoming_24h": 0, "high_active": 0},
            "upcoming": [],
            "active": [],
            "by_symbol": {},
            "sources": {"calendar_url": False, "news_url": False, "websocket_url": False},
        }
    try:
        with open(EVENT_INTEL_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, dict) else {}
    except Exception:
        return {
            "enabled": False,
            "updated_utc": None,
            "summary": {"upcoming_24h": 0, "active_window": 0, "high_upcoming_24h": 0, "high_active": 0},
            "upcoming": [],
            "active": [],
            "by_symbol": {},
            "sources": {"calendar_url": False, "news_url": False, "websocket_url": False},
        }


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


def _root_processes(rows):
    pid_set = {int(r.get("pid") or 0) for r in rows}
    roots = [r for r in rows if int(r.get("ppid") or 0) not in pid_set]
    return roots or rows


def _root_pids(rows):
    return [int(r.get("pid") or 0) for r in _root_processes(rows) if int(r.get("pid") or 0) > 0]


def _runtime_owner_health(procs):
    roles = [
        ("server", "python.server_agi"),
        ("ui", "tools/project_status_ui.py"),
        ("cycle", "tools/champion_cycle.py"),
        ("train_lstm", "training/train_lstm.py"),
        ("train_drl", "training/train_drl.py"),
    ]
    issues = []
    max_parallel_roots = max(1, len(_configured_symbols()))
    parallel_roles = {"train_lstm", "train_drl"}
    for role, token in roles:
        rows = _filter_cmd(procs, token)
        if not rows:
            continue
        pid_set = {int(r.get("pid") or 0) for r in rows}
        roots = [r for r in rows if int(r.get("ppid") or 0) not in pid_set]
        exes = sorted({str(r.get("name") or "").lower() + "|" + str((r.get("cmd") or "")).lower() for r in rows})
        exe_paths = sorted(
            {
                str((r.get("cmd") or "")).split(" ")[0].strip('"').lower().replace("\\", "/")
                for r in rows
                if str((r.get("cmd") or "")).strip()
            }
        )

        # Windows venv redirector chain: venv launcher roots the tree and the base
        # interpreter appears only as a child process for the same role token.
        allowed_paths = {
            "users/administrator/desktop/python.exe",
            ".venv312/scripts/python.exe",
            ".venv/scripts/python.exe",
        }
        if len(roots) == 1 and exe_paths and all(any(token in p for token in allowed_paths) for p in exe_paths):
            non_root_children_ok = True
            for r in rows:
                pid = int(r.get("pid") or 0)
                ppid = int(r.get("ppid") or 0)
                if pid != int(roots[0].get("pid") or 0) and ppid not in pid_set:
                    non_root_children_ok = False
                    break
            if non_root_children_ok:
                continue

        if len(roots) > 1 and role in parallel_roles and len(roots) <= max_parallel_roots:
            continue
        if len(roots) > 1:
            issues.append({"role": role, "type": "multiple_root_owners", "root_pids": [int(r.get("pid") or 0) for r in roots], "exe_paths": exe_paths})
        elif len(exes) > 1 and len(exe_paths) > 1:
            issues.append({"role": role, "type": "mixed_executables", "root_pids": [int(roots[0].get("pid") or 0)] if roots else [int(rows[0].get("pid") or 0)], "exe_paths": exe_paths})
    return {"ok": len(issues) == 0, "issues": issues}


def _normalize_single_owner():
    procs = _processes()
    roles = [
        "python.server_agi",
        "tools/project_status_ui.py",
        "tools/champion_cycle.py",
        "training/train_lstm.py",
        "training/train_drl.py",
    ]
    venv_hint = os.path.join(ROOT, ".venv312", "scripts", "python.exe").lower().replace("\\", "/")
    max_parallel_roots = max(1, len(_configured_symbols()))
    parallel_tokens = {"training/train_lstm.py", "training/train_drl.py"}
    killed = []
    for token in roles:
        rows = _filter_cmd(procs, token)
        if not rows:
            continue
        pid_set = {int(r.get("pid") or 0) for r in rows}
        roots = [r for r in rows if int(r.get("ppid") or 0) not in pid_set]
        if token in parallel_tokens and len(roots) <= max_parallel_roots:
            continue
        if len(roots) <= 1:
            continue
        keep = None
        for r in roots:
            cmd = str(r.get("cmd") or "").lower().replace("\\", "/")
            if venv_hint in cmd:
                keep = int(r.get("pid") or 0)
                break
        if keep is None:
            keep = int(roots[-1].get("pid") or 0)
        for r in roots:
            pid = int(r.get("pid") or 0)
            if pid and pid != keep:
                subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"], check=False)
                killed.append(pid)
        # Also remove any non-venv executable workers chained under the kept root.
        for r in rows:
            pid = int(r.get("pid") or 0)
            if not pid:
                continue
            cmd = str(r.get("cmd") or "").lower().replace("\\", "/")
            if venv_hint not in cmd and pid != keep:
                subprocess.run(["powershell", "-NoProfile", "-Command", f"Stop-Process -Id {pid} -Force"], check=False)
                killed.append(pid)
    return killed


def _is_running(token: str) -> bool:
    return len(_filter_cmd(_processes(), token)) > 0


def _parse_symbol_list(raw):
    if isinstance(raw, (list, tuple)):
        return [str(x).strip() for x in raw if str(x).strip()]

    txt = str(raw or "").strip()
    if not txt:
        return []

    try:
        parsed = ast.literal_eval(txt)
        if isinstance(parsed, (list, tuple)):
            return [str(x).strip() for x in parsed if str(x).strip()]
    except Exception:
        pass

    parts = txt.strip("[]")
    return [x.strip().strip("'\"") for x in parts.split(",") if x.strip()]


def _as_int(raw, default=0):
    try:
        return int(str(raw).replace(",", "").strip())
    except Exception:
        return default


def _as_float(raw, default=None):
    try:
        return float(str(raw).replace(",", "").strip())
    except Exception:
        return default


def _configured_symbols():
    cfg = _load_cfg()
    trading = cfg.get("trading", {}) if isinstance(cfg, dict) else {}
    configured = _parse_symbol_list(trading.get("symbols", []))
    return configured or list(DEFAULT_TRADING_SYMBOLS)


def _has_lstm_artifact(symbol: str) -> bool:
    safe = str(symbol or "").replace("/", "_")
    return os.path.exists(os.path.join(ROOT, "models", "per_symbol", f"lstm_{safe}.pt"))


def _has_dreamer_artifact(symbol: str) -> bool:
    safe = str(symbol or "").replace("/", "_")
    return os.path.exists(os.path.join(ROOT, "models", "dreamer", f"dreamer_{safe}.pt"))


def _candidate_label(path: str | None) -> str | None:
    if not path:
        return None
    normalized = str(path).replace("\\", "/").rstrip("/")
    return os.path.basename(normalized) or None


def _latest_candidates_by_symbol(symbols: list[str]) -> dict:
    root = os.path.join(ROOT, "models", "registry", "candidates")
    out = {}
    wanted = {str(symbol) for symbol in symbols if str(symbol)}
    if not wanted or not os.path.isdir(root):
        return out

    dirs = [os.path.join(root, name) for name in os.listdir(root) if os.path.isdir(os.path.join(root, name))]
    dirs.sort(key=lambda path: os.path.getmtime(path), reverse=True)

    for candidate_dir in dirs:
        meta = None
        for name in ("metadata.json", "scorecard.json"):
            path = os.path.join(candidate_dir, name)
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as handle:
                    meta = json.load(handle) or {}
                break
            except Exception:
                meta = None
        if not isinstance(meta, dict):
            continue

        symbol = str(meta.get("symbol") or "").strip()
        if not symbol or symbol not in wanted or symbol in out:
            continue

        evaluation = meta.get("evaluation", {}) if isinstance(meta.get("evaluation"), dict) else {}
        updated_utc = meta.get("registered_at") or meta.get("date")
        if not updated_utc:
            updated_utc = datetime.fromtimestamp(os.path.getmtime(candidate_dir), tz=timezone.utc).isoformat()

        out[symbol] = {
            "path": candidate_dir,
            "label": os.path.basename(candidate_dir),
            "updated_utc": updated_utc,
            "gates_passed": bool(evaluation.get("gates_passed", False)),
            "winner": bool(evaluation.get("winner", False)),
            "candidate_score": evaluation.get("candidate_score"),
        }
    return out


def _build_lstm_visual(lines, running: bool) -> dict:
    out = {
        "symbols": _configured_symbols(),
        "current_symbol": None,
        "epochs_total": None,
        "candles": None,
        "updated_utc": None,
        "queue": [],
        "summary": {
            "total_symbols": 0,
            "completed_symbols": 0,
            "active_symbols": 0,
            "failed_symbols": 0,
            "queued_symbols": 0,
            "completion_pct": 0.0,
        },
    }
    if not lines:
        return out

    start_re = re.compile(
        r"LSTM per-symbol training on .*?\|\s*symbols=(\[[^\]]*\])\s*\|\s*epochs=(\d+)(?:.*?\|\s*candles=([0-9,]+))?",
        re.IGNORECASE,
    )
    progress_re = re.compile(
        r"([A-Za-z0-9_]+)\s*\|\s*epoch\s+(\d+)\s*/\s*(\d+)\s*\|\s*loss\s+([0-9.]+)\s*\|\s*acc\s+([0-9.]+)%",
        re.IGNORECASE,
    )
    skip_re = re.compile(
        r"(?:insufficient data for|insufficient engineered rows for|no sequences for)\s+([A-Za-z0-9_]+)",
        re.IGNORECASE,
    )

    configured_symbols = list(out["symbols"])
    symbols = list(configured_symbols)
    progress_by = {
        sym: {
            "epoch": 0,
            "epochs_total": 0,
            "loss": None,
            "acc": None,
            "status": "queued",
            "updated_utc": None,
        }
        for sym in symbols
    }
    latest_symbol = None
    latest_ts = None
    max_epochs_total = 0
    candles = None

    for line in lines:
        sm = start_re.search(line)
        if sm:
            line_symbols = _parse_symbol_list(sm.group(1)) or configured_symbols
            for sym in line_symbols:
                if sym not in symbols:
                    symbols.append(sym)
                progress_by.setdefault(
                    sym,
                    {
                        "epoch": 0,
                        "epochs_total": 0,
                        "loss": None,
                        "acc": None,
                        "status": "queued",
                        "updated_utc": None,
                    },
                )
            max_epochs_total = max(max_epochs_total, _as_int(sm.group(2), 0))
            parsed_candles = _as_int(sm.group(3), 0) or None
            candles = parsed_candles or candles
            continue

        pm = progress_re.search(line)
        if pm:
            sym = str(pm.group(1))
            if sym not in symbols:
                symbols.append(sym)
            item = progress_by.setdefault(
                sym,
                {
                    "epoch": 0,
                    "epochs_total": 0,
                    "loss": None,
                    "acc": None,
                    "status": "queued",
                    "updated_utc": None,
                },
            )
            item["epoch"] = max(item["epoch"], _as_int(pm.group(2), 0))
            item["epochs_total"] = max(item["epochs_total"], _as_int(pm.group(3), 0))
            item["loss"] = _as_float(pm.group(4))
            item["acc"] = _as_float(pm.group(5))
            ts = _line_ts_utc(line)
            item["updated_utc"] = ts.isoformat() if ts else None
            latest_symbol = sym
            latest_ts = ts or latest_ts
            max_epochs_total = max(max_epochs_total, item["epochs_total"])
            continue

        fm = skip_re.search(line)
        if fm:
            reason = fm.group(0).strip()
            sym = str(fm.group(1))
            if sym not in symbols:
                symbols.append(sym)
            item = progress_by.setdefault(
                sym,
                {
                    "epoch": 0,
                    "epochs_total": max_epochs_total,
                    "loss": None,
                    "acc": None,
                    "status": "failed",
                    "fail_reason": reason,
                    "updated_utc": None,
                },
            )
            item["status"] = "failed"
            item["fail_reason"] = reason
            ts = _line_ts_utc(line)
            item["updated_utc"] = ts.isoformat() if ts else None
            latest_ts = ts or latest_ts

    if not symbols:
        return out

    out["symbols"] = symbols
    out["epochs_total"] = max_epochs_total or None
    out["candles"] = candles

    if latest_symbol is None and running and symbols:
        latest_symbol = symbols[0]

    recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=20)
    queue = []
    counts = {"done": 0, "active": 0, "failed": 0, "queued": 0}
    for sym in symbols:
        item = progress_by.get(sym, {})
        status = item.get("status", "queued")
        epoch = _as_int(item.get("epoch"), 0)
        total = _as_int(item.get("epochs_total"), max_epochs_total)
        updated_utc = None
        try:
            updated_raw = item.get("updated_utc")
            if updated_raw:
                updated_utc = datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00"))
        except Exception:
            updated_utc = None
        is_recent = updated_utc is not None and updated_utc >= recent_cutoff
        if status != "failed":
            if total > 0 and epoch >= total:
                status = "done"
            elif running and epoch > 0 and is_recent:
                status = "active"
            elif epoch > 0:
                status = "partial"
            else:
                status = "queued"

        if status == "done":
            pct = 100.0
            counts["done"] += 1
        elif status == "active":
            pct = round((epoch / total) * 100.0, 2) if total > 0 else 4.0
            pct = max(pct, 4.0)
            counts["active"] += 1
        elif status == "failed":
            pct = 0.0
            counts["failed"] += 1
        elif status == "partial":
            pct = round((epoch / total) * 100.0, 2) if total > 0 else 0.0
            counts["active"] += 1
        else:
            pct = 0.0
            counts["queued"] += 1

        entry = {
                "symbol": sym,
                "status": status,
                "epoch": epoch,
                "epochs_total": total,
                "progress_pct": pct,
                "loss": item.get("loss"),
                "acc": item.get("acc"),
                "updated_utc": item.get("updated_utc"),
            }
        if item.get("fail_reason"):
            entry["fail_reason"] = item["fail_reason"]
        queue.append(entry)

    total_symbols = len(symbols)
    completed = counts["done"]
    completion_pct = round((completed / total_symbols) * 100.0, 2) if total_symbols else 0.0

    out["current_symbol"] = latest_symbol
    out["updated_utc"] = latest_ts.isoformat() if latest_ts else None
    out["queue"] = queue
    out["summary"] = {
        "total_symbols": total_symbols,
        "completed_symbols": completed,
        "active_symbols": counts["active"],
        "failed_symbols": counts["failed"],
        "queued_symbols": counts["queued"],
        "completion_pct": completion_pct,
    }
    return out


def _build_ppo_visual(lines, running: bool) -> dict:
    out = {
        "symbols": _configured_symbols(),
        "current_symbol": None,
        "target_timesteps": None,
        "candles": None,
        "phase": "idle",
        "current_timesteps": None,
        "progress_pct": None,
        "elapsed_seconds": None,
        "eta_seconds": None,
        "candidate_ready": False,
        "candidate_path": None,
        "updated_utc": None,
        "queue": [],
        "summary": {
            "total_symbols": 0,
            "completed_symbols": 0,
            "active_symbols": 0,
            "queued_symbols": 0,
            "completion_pct": 0.0,
        },
    }
    if not lines:
        return out

    start_re = re.compile(
        r"DRL Training\s*\|\s*symbols=(\[[^\]]*\])\s*\|\s*timesteps=([0-9,]+)(?:.*?\|\s*candles=([0-9,]+))?",
        re.IGNORECASE,
    )
    progress_re = re.compile(
        r"PPO progress\s*\|\s*symbols=(\[[^\]]*\])\s*\|\s*step=([0-9,]+)\/([0-9,]+)\s*\|\s*pct=([0-9.]+)\s*\|\s*elapsed_s=(\d+)\s*\|\s*eta_s=([0-9]+|unknown)",
        re.IGNORECASE,
    )
    staged_re = re.compile(r"Candidate staged to:\s*(.+)$", re.IGNORECASE)

    configured_symbols = list(out["symbols"])
    symbols = list(configured_symbols)
    progress_by = {
        sym: {
            "current_timesteps": 0,
            "target_timesteps": 0,
            "progress_pct": 0.0,
            "elapsed_seconds": None,
            "eta_seconds": None,
            "status": "queued",
            "updated_utc": None,
        }
        for sym in symbols
    }
    started = False
    staged = None
    latest_symbol = None
    latest_ts = None
    max_target = 0
    candles = None
    recent_cutoff = datetime.now(timezone.utc) - timedelta(minutes=30)

    for line in lines:
        sm = start_re.search(line)
        if sm:
            line_symbols = _parse_symbol_list(sm.group(1)) or configured_symbols
            for sym in line_symbols:
                if sym not in symbols:
                    symbols.append(sym)
                progress_by.setdefault(
                    sym,
                    {
                        "current_timesteps": 0,
                        "target_timesteps": 0,
                        "progress_pct": 0.0,
                        "elapsed_seconds": None,
                        "eta_seconds": None,
                        "status": "queued",
                        "updated_utc": None,
                    },
                )
            max_target = max(max_target, _as_int(sm.group(2), 0))
            parsed_candles = _as_int(sm.group(3), 0) or None
            candles = parsed_candles or candles
            latest_ts = _line_ts_utc(line) or latest_ts
            continue
        if "Starting PPO training" in line:
            started = True
            latest_ts = _line_ts_utc(line) or latest_ts
            continue
        pm = progress_re.search(line)
        if pm:
            progress_symbols = _parse_symbol_list(pm.group(1)) or configured_symbols
            current_steps = _as_int(pm.group(2), 0)
            target_steps = _as_int(pm.group(3), 0)
            progress_pct = _as_float(pm.group(4)) or 0.0
            elapsed_seconds = _as_int(pm.group(5), 0)
            eta_seconds = None if str(pm.group(6)).lower() == "unknown" else (_as_int(pm.group(6), 0) or None)
            ts = _line_ts_utc(line)
            for sym in progress_symbols:
                if sym not in symbols:
                    symbols.append(sym)
                item = progress_by.setdefault(
                    sym,
                    {
                        "current_timesteps": 0,
                        "target_timesteps": 0,
                        "progress_pct": 0.0,
                        "elapsed_seconds": None,
                        "eta_seconds": None,
                        "status": "queued",
                        "updated_utc": None,
                    },
                )
                item["current_timesteps"] = max(item["current_timesteps"], current_steps)
                item["target_timesteps"] = max(item["target_timesteps"], target_steps)
                item["progress_pct"] = max(float(item["progress_pct"] or 0.0), float(progress_pct))
                item["elapsed_seconds"] = elapsed_seconds
                item["eta_seconds"] = eta_seconds
                item["updated_utc"] = ts.isoformat() if ts else item.get("updated_utc")
                latest_symbol = sym
            latest_ts = ts or latest_ts
            max_target = max(max_target, target_steps)
            continue
        staged_match = staged_re.search(line)
        if staged_match:
            staged = staged_match.group(1).strip()
            latest_ts = _line_ts_utc(line) or latest_ts

    if not symbols:
        return out

    queue = []
    counts = {"done": 0, "active": 0, "queued": 0}
    for sym in symbols:
        item = progress_by.get(sym, {})
        current_steps = _as_int(item.get("current_timesteps"), 0)
        target_steps = _as_int(item.get("target_timesteps"), max_target)
        updated_utc = None
        try:
            updated_raw = item.get("updated_utc")
            if updated_raw:
                updated_utc = datetime.fromisoformat(str(updated_raw).replace("Z", "+00:00"))
        except Exception:
            updated_utc = None
        is_recent = updated_utc is not None and updated_utc >= recent_cutoff
        if target_steps > 0 and current_steps >= target_steps:
            status = "done"
            progress_pct = 100.0
            counts["done"] += 1
        elif running and current_steps > 0 and is_recent:
            status = "active"
            progress_pct = float(item.get("progress_pct") or 0.0)
            counts["active"] += 1
        elif current_steps > 0:
            status = "partial"
            progress_pct = float(item.get("progress_pct") or 0.0)
            counts["active"] += 1
        else:
            status = "queued"
            progress_pct = 0.0
            counts["queued"] += 1
        queue.append(
            {
                "symbol": sym,
                "status": status,
                "current_timesteps": current_steps,
                "target_timesteps": target_steps,
                "progress_pct": round(progress_pct, 2),
                "elapsed_seconds": item.get("elapsed_seconds"),
                "eta_seconds": item.get("eta_seconds"),
                "updated_utc": item.get("updated_utc"),
            }
        )

    total_symbols = len(queue)
    completed = counts["done"]
    completion_pct = round((completed / total_symbols) * 100.0, 2) if total_symbols else 0.0
    out["symbols"] = symbols
    out["current_symbol"] = latest_symbol or (symbols[0] if symbols else None)
    if out["current_symbol"]:
        current_item = next((item for item in queue if item["symbol"] == out["current_symbol"]), None)
        if current_item:
            out["current_timesteps"] = current_item.get("current_timesteps")
            out["target_timesteps"] = current_item.get("target_timesteps")
            out["progress_pct"] = current_item.get("progress_pct")
            out["elapsed_seconds"] = current_item.get("elapsed_seconds")
            out["eta_seconds"] = current_item.get("eta_seconds")
    out["target_timesteps"] = out["target_timesteps"] or (max_target or None)
    out["candles"] = candles
    out["candidate_ready"] = staged is not None
    out["candidate_path"] = staged
    out["updated_utc"] = latest_ts.isoformat() if latest_ts else None
    out["queue"] = queue
    out["summary"] = {
        "total_symbols": total_symbols,
        "completed_symbols": completed,
        "active_symbols": counts["active"],
        "queued_symbols": counts["queued"],
        "completion_pct": completion_pct,
    }
    if running and counts["active"] > 1:
        out["phase"] = "parallel_optimizing"
    elif running:
        out["phase"] = "optimizing" if started else "loading"
    elif staged is not None:
        out["phase"] = "candidate_ready"
    elif started:
        out["phase"] = "stalled"
    else:
        out["phase"] = "queued"
    return out


def _build_dreamer_visual(lines, running: bool) -> dict:
    out = {
        "symbols": _configured_symbols(),
        "current_symbol": None,
        "steps": None,
        "window": None,
        "obs_dim": None,
        "phase": "queued",
        "last_saved_symbol": None,
        "updated_utc": None,
        "estimated_run_seconds": None,
        "queue": [],
        "summary": {
            "total_symbols": 0,
            "completed_symbols": 0,
            "active_symbols": 0,
            "queued_symbols": 0,
            "completion_pct": 0.0,
        },
    }
    if not lines:
        return out

    start_re = re.compile(
        r"Dreamer training start\s*\|\s*symbol=([A-Za-z0-9_]+)\s*\|\s*steps=(\d+)\s*\|\s*window=(\d+)\s*\|\s*obs_dim=(\d+)",
        re.IGNORECASE,
    )
    progress_re = re.compile(
        r"Dreamer progress\s*\|\s*symbol=([A-Za-z0-9_]+)\s*\|\s*step=(\d+)\/(\d+)\s*\|\s*pct=([0-9.]+)\s*\|\s*elapsed_s=(\d+)",
        re.IGNORECASE,
    )
    saved_re = re.compile(r"dreamer_([A-Za-z0-9_]+)\.pt", re.IGNORECASE)
    latest_ts = None
    now_utc = datetime.now(timezone.utc)
    symbols = list(out["symbols"])
    progress_by = {
        sym: {
            "symbol": sym,
            "status": "queued",
            "steps": None,
            "window": None,
            "obs_dim": None,
            "started_utc": None,
            "saved_utc": None,
            "updated_utc": None,
            "progress_pct": 0.0,
            "detail": "waiting",
        }
        for sym in symbols
    }
    run_profiles = []

    for line in lines:
        sm = start_re.search(line)
        if sm:
            sym = sm.group(1)
            ts = _line_ts_utc(line)
            item = progress_by.setdefault(
                sym,
                {
                    "symbol": sym,
                    "status": "queued",
                    "steps": None,
                    "window": None,
                    "obs_dim": None,
                    "started_utc": None,
                    "saved_utc": None,
                    "updated_utc": None,
                    "progress_pct": 0.0,
                    "detail": "waiting",
                },
            )
            item["steps"] = _as_int(sm.group(2), 0) or None
            item["window"] = _as_int(sm.group(3), 0) or None
            item["obs_dim"] = _as_int(sm.group(4), 0) or None
            item["started_utc"] = ts.isoformat() if ts else None
            item["updated_utc"] = ts.isoformat() if ts else item.get("updated_utc")
            item["saved_utc"] = None
            item["status"] = "active" if running else "partial"
            item["detail"] = "training started"
            out["current_symbol"] = sym
            out["steps"] = item["steps"]
            out["window"] = item["window"]
            out["obs_dim"] = item["obs_dim"]
            latest_ts = ts or latest_ts
            continue
        pm = progress_re.search(line)
        if pm:
            sym = pm.group(1)
            ts = _line_ts_utc(line)
            item = progress_by.setdefault(
                sym,
                {
                    "symbol": sym,
                    "status": "queued",
                    "steps": _as_int(pm.group(3), 0) or None,
                    "window": None,
                    "obs_dim": None,
                    "started_utc": None,
                    "saved_utc": None,
                    "updated_utc": None,
                    "progress_pct": 0.0,
                    "detail": "waiting",
                },
            )
            item["steps"] = _as_int(pm.group(3), 0) or item.get("steps")
            item["progress_pct"] = max(float(item.get("progress_pct") or 0.0), _as_float(pm.group(4), 0.0) or 0.0)
            item["updated_utc"] = ts.isoformat() if ts else item.get("updated_utc")
            item["status"] = "active" if running else "partial"
            item["detail"] = f"{_as_int(pm.group(2), 0):,}/{_as_int(pm.group(3), 0):,} steps"
            out["current_symbol"] = sym
            out["steps"] = item.get("steps")
            latest_ts = ts or latest_ts
            continue
        mm = saved_re.search(line)
        if mm:
            sym = mm.group(1)
            ts = _line_ts_utc(line)
            item = progress_by.setdefault(
                sym,
                {
                    "symbol": sym,
                    "status": "queued",
                    "steps": None,
                    "window": None,
                    "obs_dim": None,
                    "started_utc": None,
                    "saved_utc": None,
                    "updated_utc": None,
                    "progress_pct": 0.0,
                    "detail": "waiting",
                },
            )
            started_utc = item.get("started_utc")
            started_dt = None
            if started_utc:
                try:
                    started_dt = datetime.fromisoformat(str(started_utc).replace("Z", "+00:00"))
                except Exception:
                    started_dt = None
            item["saved_utc"] = ts.isoformat() if ts else None
            item["updated_utc"] = ts.isoformat() if ts else item.get("updated_utc")
            item["status"] = "done"
            item["progress_pct"] = 100.0
            item["detail"] = "artifact saved"
            out["last_saved_symbol"] = sym
            latest_ts = ts or latest_ts
            duration_seconds = (ts - started_dt).total_seconds() if started_dt is not None and ts is not None and ts >= started_dt else None
            steps = _as_int(item.get("steps"), 0)
            window = _as_int(item.get("window"), 0)
            if duration_seconds and steps > 0 and window > 0:
                run_profiles.append(
                    {
                        "symbol": sym,
                        "duration_seconds": duration_seconds,
                        "steps": steps,
                        "window": window,
                        "seconds_per_unit": duration_seconds / float(steps * window),
                    }
                )

    def _estimate_run_seconds(symbol: str, steps: int | None, window: int | None) -> float | None:
        if not steps or not window:
            return None
        matching = [row["seconds_per_unit"] for row in run_profiles if row.get("symbol") == symbol]
        pool = matching or [row["seconds_per_unit"] for row in run_profiles]
        if not pool:
            return None
        return round((sum(pool) / len(pool)) * float(steps * window), 2)

    estimated_run_seconds = None
    estimated_by_symbol = {}

    active_candidates = []
    for sym, item in progress_by.items():
        started_utc = item.get("started_utc")
        saved_utc = item.get("saved_utc")
        started_dt = None
        saved_dt = None
        if started_utc:
            try:
                started_dt = datetime.fromisoformat(str(started_utc).replace("Z", "+00:00"))
            except Exception:
                started_dt = None
        if saved_utc:
            try:
                saved_dt = datetime.fromisoformat(str(saved_utc).replace("Z", "+00:00"))
            except Exception:
                saved_dt = None

        is_done = saved_dt is not None and (started_dt is None or saved_dt >= started_dt)
        is_active = started_dt is not None and (saved_dt is None or started_dt > saved_dt)
        if is_done:
            item["status"] = "done"
            item["progress_pct"] = 100.0
            item["detail"] = "artifact saved"
        elif is_active and running:
            elapsed_seconds = max(0, int((now_utc - started_dt).total_seconds())) if started_dt else 0
            estimate_for_item = _estimate_run_seconds(sym, _as_int(item.get("steps"), 0), _as_int(item.get("window"), 0))
            estimated_by_symbol[sym] = estimate_for_item
            existing_progress = float(item.get("progress_pct") or 0.0)
            if estimate_for_item and estimate_for_item > 0:
                estimated_progress = round(min(96.0, max(6.0, (elapsed_seconds / estimate_for_item) * 100.0)), 2)
                item["progress_pct"] = max(existing_progress, estimated_progress)
            else:
                item["progress_pct"] = max(existing_progress, 12.0)
            item["status"] = "active"
            item["detail"] = (
                f"est. {item['progress_pct']:.0f}% of run"
                if estimate_for_item and elapsed_seconds > 0
                else "training active"
            )
            active_candidates.append((started_dt, sym))
        elif is_active:
            item["status"] = "partial"
            item["progress_pct"] = round(float(item.get("progress_pct") or 0.0), 2)
            item["detail"] = "awaiting resume"
            active_candidates.append((started_dt, sym))
        else:
            item["status"] = "queued"
            item["progress_pct"] = 0.0
            item["detail"] = "waiting"

    if active_candidates:
        active_candidates.sort(key=lambda pair: pair[0] or datetime.min.replace(tzinfo=timezone.utc))
        out["current_symbol"] = active_candidates[-1][1]
        estimated_run_seconds = estimated_by_symbol.get(out["current_symbol"])

    queue = []
    counts = {"done": 0, "active": 0, "queued": 0}
    for sym in symbols:
        item = progress_by.get(sym) or {
            "symbol": sym,
            "status": "queued",
            "steps": None,
            "window": None,
            "obs_dim": None,
            "started_utc": None,
            "saved_utc": None,
            "updated_utc": None,
            "progress_pct": 0.0,
            "detail": "waiting",
        }
        status = str(item.get("status") or "queued")
        if status == "done":
            counts["done"] += 1
        elif status == "active":
            counts["active"] += 1
        else:
            counts["queued"] += 1
        queue.append(
            {
                "symbol": sym,
                "status": status,
                "steps": item.get("steps"),
                "window": item.get("window"),
                "obs_dim": item.get("obs_dim"),
                "started_utc": item.get("started_utc"),
                "saved_utc": item.get("saved_utc"),
                "updated_utc": item.get("updated_utc"),
                "progress_pct": round(float(item.get("progress_pct") or 0.0), 2),
                "detail": item.get("detail"),
            }
        )

    total_symbols = len(queue)
    completed_symbols = counts["done"]
    completion_pct = round((completed_symbols / total_symbols) * 100.0, 2) if total_symbols else 0.0
    out["updated_utc"] = latest_ts.isoformat() if latest_ts else None
    if running:
        out["phase"] = "optimizing"
    elif active_candidates:
        out["phase"] = "stalled"
    elif out["last_saved_symbol"]:
        out["phase"] = "completed"
    out["estimated_run_seconds"] = estimated_run_seconds
    out["queue"] = queue
    out["summary"] = {
        "total_symbols": total_symbols,
        "completed_symbols": completed_symbols,
        "active_symbols": counts["active"],
        "queued_symbols": counts["queued"],
        "completion_pct": completion_pct,
    }
    return out


def _stage_completion_pct(stage: dict | None) -> float:
    if not isinstance(stage, dict):
        return 0.0
    summary = stage.get("summary") if isinstance(stage.get("summary"), dict) else {}
    return float(summary.get("completion_pct") or 0.0)


def _build_training_visuals(lstm_lines, ppo_lines, dreamer_lines, lstm_running: bool, drl_running: bool, dreamer_running: bool) -> dict:
    lstm = _build_lstm_visual(lstm_lines, lstm_running)
    ppo = _build_ppo_visual(ppo_lines, drl_running)
    dreamer = _build_dreamer_visual(dreamer_lines, dreamer_running)
    lstm["progress_pct"] = _stage_completion_pct(lstm)
    ppo["progress_pct"] = _stage_completion_pct(ppo)
    dreamer["progress_pct"] = _stage_completion_pct(dreamer)
    lstm_active = _as_int((lstm.get("summary") or {}).get("active_symbols"), 0)
    ppo_active = _as_int((ppo.get("summary") or {}).get("active_symbols"), 0)
    dreamer_active = _as_int((dreamer.get("summary") or {}).get("active_symbols"), 0)
    running_labels = []
    if lstm_running:
        running_labels.append("LSTM")
    if drl_running:
        running_labels.append("PPO")
    if dreamer_running:
        running_labels.append("Dreamer")
    if len(running_labels) >= 2:
        active_stage = "parallel"
        if len(running_labels) == 2:
            active_label = f"{running_labels[0]} and {running_labels[1]} running"
        else:
            active_label = f"{', '.join(running_labels[:-1])}, and {running_labels[-1]} running"
        if not dreamer_running and str(dreamer.get("phase") or "") == "stalled":
            active_label += " | Dreamer stalled"
    elif lstm_active > 1 or ppo_active > 1 or dreamer_active > 1:
        active_stage = "parallel"
        active_label = "Parallel pair-lane training running"
    elif lstm_running:
        active_stage = "lstm"
        active_label = "LSTM feature training in progress"
    elif dreamer_running:
        active_stage = "dreamer"
        active_label = "Dreamer world-model training in progress"
    elif drl_running:
        active_stage = "ppo"
        active_label = "PPO policy optimization in progress"
    elif ppo.get("candidate_ready"):
        active_stage = "canary"
        active_label = "Candidate staged for canary review"
    elif lstm.get("summary", {}).get("completed_symbols", 0) > 0:
        active_stage = "review"
        active_label = "Waiting for the next training stage"
    else:
        active_stage = "idle"
        active_label = "Training idle"

    return {
        "active_stage": active_stage,
        "active_label": active_label,
        "lstm": lstm,
        "ppo": ppo,
        "dreamer": dreamer,
    }


def _symbol_stage_rows(training: dict, active: dict, account: dict | None = None, server: dict | None = None) -> list[dict]:
    symbols = list(training.get("configured_symbols") or _configured_symbols())
    visual = training.get("visual", {}) if isinstance(training.get("visual"), dict) else {}
    lstm_visual = visual.get("lstm", {}) if isinstance(visual.get("lstm"), dict) else {}
    ppo_visual = visual.get("ppo", {}) if isinstance(visual.get("ppo"), dict) else {}
    dreamer_visual = visual.get("dreamer", {}) if isinstance(visual.get("dreamer"), dict) else {}
    queue = {str(item.get("symbol")): item for item in lstm_visual.get("queue", []) if str(item.get("symbol") or "")}
    ppo_queue = {str(item.get("symbol")): item for item in ppo_visual.get("queue", []) if str(item.get("symbol") or "")}
    dreamer_queue = {str(item.get("symbol")): item for item in dreamer_visual.get("queue", []) if str(item.get("symbol") or "")}
    registry_symbols = active.get("symbols", {}) if isinstance(active.get("symbols"), dict) else {}
    latest_candidates = _latest_candidates_by_symbol(symbols)
    account = account if isinstance(account, dict) else {}
    server = server if isinstance(server, dict) else {}
    positions_by_symbol = defaultdict(list)
    for position in account.get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        symbol = str(position.get("symbol") or "").strip()
        if symbol:
            positions_by_symbol[symbol].append(position)
    server_running = bool(server.get("running", False))

    current_ppo_symbol = str(training.get("drl_symbol") or ppo_visual.get("current_symbol") or "").strip()
    current_dreamer_symbol = str(dreamer_visual.get("current_symbol") or "").strip()
    last_saved_dreamer = str(dreamer_visual.get("last_saved_symbol") or "").strip()

    rows = []
    for symbol in symbols:
        lstm_item = queue.get(symbol, {})
        lstm_state = str(lstm_item.get("status") or "").strip() or ("done" if _has_lstm_artifact(symbol) else "queued")
        lstm_progress = float(lstm_item.get("progress_pct") or (100.0 if lstm_state == "done" else 0.0))
        lstm_detail = "waiting"
        epoch = _as_int(lstm_item.get("epoch"), 0)
        total = _as_int(lstm_item.get("epochs_total"), 0)
        # Override stale "failed" if artifact exists on disk (training succeeded)
        if lstm_state == "failed" and _has_lstm_artifact(symbol):
            lstm_state = "done"
            lstm_progress = 100.0
            lstm_detail = f"epoch {epoch}/{total}" if total > 0 else "complete"
        # Override stale "failed" if LSTM is currently running for this symbol
        elif lstm_state == "failed" and training.get("lstm_running"):
            lstm_state = "active"
            lstm_progress = 50.0
            lstm_detail = "retraining"
        elif lstm_state == "failed":
            lstm_detail = lstm_item.get("fail_reason") or "training failed"
        if lstm_state in {"active", "partial", "done"} and lstm_detail == "waiting":
            if total > 0:
                lstm_detail = f"epoch {epoch}/{total}"

        dreamer_item = dreamer_queue.get(symbol, {})
        dreamer_state = str(dreamer_item.get("status") or "").strip()
        if not dreamer_state:
            if training.get("dreamer_running") and current_dreamer_symbol == symbol:
                dreamer_state = "active"
            elif _has_dreamer_artifact(symbol) or last_saved_dreamer == symbol:
                dreamer_state = "done"
            else:
                dreamer_state = "queued"
        dreamer_progress = float(
            dreamer_item.get("progress_pct")
            or (100.0 if dreamer_state == "done" else 0.0)
        )
        dreamer_detail = str(dreamer_item.get("detail") or "").strip()
        if not dreamer_detail:
            dreamer_steps = _as_int(dreamer_item.get("steps") or dreamer_visual.get("steps"), 0)
            if dreamer_state == "active":
                dreamer_detail = f"steps {dreamer_steps:,}" if dreamer_steps > 0 else "optimizing"
            elif dreamer_state == "partial":
                dreamer_detail = "awaiting resume"
            elif dreamer_state == "done":
                dreamer_detail = "artifact saved"
            else:
                dreamer_detail = "waiting"

        candidate = latest_candidates.get(symbol)
        ppo_item = ppo_queue.get(symbol, {})
        ppo_state = str(ppo_item.get("status") or "").strip()
        if ppo_state:
            ppo_progress = float(
                ppo_item.get("progress_pct")
                or (100.0 if ppo_state == "done" else 0.0)
            )
            current_steps = _as_int(ppo_item.get("current_timesteps"), 0)
            target_steps = _as_int(ppo_item.get("target_timesteps"), 0)
            if ppo_state in {"active", "partial", "done"} and target_steps > 0:
                ppo_detail = f"{current_steps:,}/{target_steps:,}"
            elif ppo_state == "done":
                ppo_detail = "candidate staged"
            else:
                ppo_detail = "queued in cycle"
        elif training.get("drl_running") and current_ppo_symbol == symbol:
            ppo_state = "active"
            ppo_progress = float(ppo_visual.get("progress_pct") or 0.0)
            current_steps = _as_int(ppo_visual.get("current_timesteps"), 0)
            target_steps = _as_int(ppo_visual.get("target_timesteps"), 0)
            ppo_detail = f"{current_steps:,}/{target_steps:,}" if target_steps > 0 else "optimizing"
        elif training.get("cycle_running") and current_ppo_symbol == symbol:
            ppo_state = "queued"
            ppo_progress = 0.0
            ppo_detail = "queued in cycle"
        elif candidate:
            ppo_state = "done"
            ppo_progress = 100.0
            ppo_detail = candidate.get("label") or "candidate staged"
        else:
            ppo_state = "queued"
            ppo_progress = 0.0
            ppo_detail = "waiting"

        registry_row = registry_symbols.get(symbol, {}) if isinstance(registry_symbols.get(symbol), dict) else {}
        canary_path = registry_row.get("canary")
        canary_state = registry_row.get("canary_state", {}) if isinstance(registry_row.get("canary_state"), dict) else {}
        if canary_path:
            canary_stage = "ready" if bool(canary_state.get("passed", False)) else "testing"
            canary_detail = _candidate_label(canary_path) or "candidate attached"
        else:
            canary_stage = "waiting"
            canary_detail = canary_state.get("reason") or "none staged"

        champion_path = registry_row.get("champion")
        champion_stage = "live" if champion_path else "waiting"
        champion_detail = _candidate_label(champion_path) or "not set"
        positions = positions_by_symbol.get(symbol, [])
        if positions:
            total_profit = round(sum(float(pos.get("profit") or 0.0) for pos in positions), 2)
            trading_stage = "active"
            trading_progress = 100.0
            trading_detail = f"{len(positions)} open | pnl {total_profit:+.2f}"
        elif champion_path and server_running:
            trading_stage = "armed"
            trading_progress = 100.0
            trading_detail = "runtime live"
        elif champion_path:
            trading_stage = "paused"
            trading_progress = 0.0
            trading_detail = "runtime stopped"
        else:
            trading_stage = "waiting"
            trading_progress = 0.0
            trading_detail = "no champion"

        rows.append(
            {
                "symbol": symbol,
                "lstm": {"state": lstm_state, "progress_pct": round(lstm_progress, 2), "detail": lstm_detail},
                "dreamer": {"state": dreamer_state, "progress_pct": round(dreamer_progress, 2), "detail": dreamer_detail},
                "ppo": {"state": ppo_state, "progress_pct": round(ppo_progress, 2), "detail": ppo_detail},
                "canary": {"state": canary_stage, "detail": canary_detail},
                "champion": {"state": champion_stage, "detail": champion_detail},
                "trading": {"state": trading_stage, "progress_pct": round(trading_progress, 2), "detail": trading_detail},
            }
        )
    return rows


def _symbol_pipeline_summary(rows: list[dict]) -> dict:
    summary = {
        "symbols_total": len(rows),
        "training_active_symbols": 0,
        "canary_review_symbols": 0,
        "champion_live_symbols": 0,
        "trading_ready_symbols": 0,
        "trading_active_symbols": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        if any(str((row.get(stage) or {}).get("state") or "") == "active" for stage in ("lstm", "dreamer", "ppo")):
            summary["training_active_symbols"] += 1
        canary_state = str((row.get("canary") or {}).get("state") or "")
        if canary_state in {"testing", "ready"}:
            summary["canary_review_symbols"] += 1
        champion_state = str((row.get("champion") or {}).get("state") or "")
        if champion_state == "live":
            summary["champion_live_symbols"] += 1
        trading_state = str((row.get("trading") or {}).get("state") or "")
        if trading_state in {"armed", "active"}:
            summary["trading_ready_symbols"] += 1
        if trading_state == "active":
            summary["trading_active_symbols"] += 1
    return summary


def _latest_incidents_by_symbol(incidents: list[dict], event_name: str) -> dict[str, dict]:
    rows = {}
    for item in incidents or []:
        if str(item.get("event") or "") != event_name:
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        symbol = str(payload.get("symbol") or item.get("symbol") or "").strip()
        if symbol and symbol not in rows:
            rows[symbol] = item
    return rows


def _symbol_lane_rows(
    training: dict,
    active: dict,
    incidents: list[dict],
    account: dict | None = None,
    server: dict | None = None,
) -> list[dict]:
    stage_rows = training.get("symbol_stage_rows") if isinstance(training.get("symbol_stage_rows"), list) else None
    if stage_rows is None:
        stage_rows = _symbol_stage_rows(training, active, account=account, server=server)

    signals = _latest_incidents_by_symbol(incidents, "signal")
    actions = _latest_incidents_by_symbol(incidents, "trade_action")
    blocks = _latest_incidents_by_symbol(incidents, "risk_supervisor_block")
    registry_symbols = active.get("symbols", {}) if isinstance(active.get("symbols"), dict) else {}

    positions_by_symbol = defaultdict(list)
    for position in (account or {}).get("positions", []) or []:
        if not isinstance(position, dict):
            continue
        symbol = str(position.get("symbol") or "").strip()
        if symbol:
            positions_by_symbol[symbol].append(position)

    rows = []
    for stage_row in stage_rows:
        if not isinstance(stage_row, dict):
            continue
        symbol = str(stage_row.get("symbol") or "").strip()
        if not symbol:
            continue

        signal_item = signals.get(symbol, {})
        signal_payload = signal_item.get("payload") if isinstance(signal_item.get("payload"), dict) else {}
        action_item = actions.get(symbol, {})
        action_payload = action_item.get("payload") if isinstance(action_item.get("payload"), dict) else {}
        block_item = blocks.get(symbol, {})
        block_payload = block_item.get("payload") if isinstance(block_item.get("payload"), dict) else {}
        registry_row = registry_symbols.get(symbol, {}) if isinstance(registry_symbols.get(symbol), dict) else {}
        positions = positions_by_symbol.get(symbol, [])

        final_target = float(signal_payload.get("exposure") or 0.0)
        ppo_target = float(signal_payload.get("ppo_exposure") or 0.0)
        dreamer_target = float(signal_payload.get("dreamer_exposure") or 0.0)
        agi_bias = float(signal_payload.get("agi_bias") or 0.0)
        risk_scalar = float(signal_payload.get("risk_scalar") or 1.0)
        profile = signal_payload.get("decision_profile") if isinstance(signal_payload.get("decision_profile"), dict) else {}

        execution_state = "watching"
        if action_payload:
            request_action = str(action_payload.get("request_action") or action_payload.get("action") or "watching")
            execution_state = "executed" if bool(action_payload.get("executed")) else request_action
        elif block_payload:
            execution_state = "blocked"
        elif signal_payload:
            execution_state = "armed" if abs(final_target) > 1e-9 else "neutral"

        trading_state = str((stage_row.get("trading") or {}).get("state") or "waiting")
        trading_pnl = round(sum(float(position.get("profit") or 0.0) for position in positions), 2) if positions else 0.0
        last_update = action_item.get("ts") or signal_item.get("ts") or block_item.get("ts")

        rows.append(
            {
                "symbol": symbol,
                "pipeline": {
                    "lstm": dict(stage_row.get("lstm") or {}),
                    "dreamer": dict(stage_row.get("dreamer") or {}),
                    "ppo": dict(stage_row.get("ppo") or {}),
                    "canary": dict(stage_row.get("canary") or {}),
                    "champion": dict(stage_row.get("champion") or {}),
                    "trading": dict(stage_row.get("trading") or {}),
                },
                "registry": {
                    "champion": registry_row.get("champion"),
                    "champion_label": _candidate_label(registry_row.get("champion")),
                    "canary": registry_row.get("canary"),
                    "canary_label": _candidate_label(registry_row.get("canary")),
                    "canary_ready": bool(((registry_row.get("canary_state") or {}).get("passed"))),
                },
                "decision": {
                    "state": execution_state,
                    "regime": signal_payload.get("regime") or signal_payload.get("signal") or "UNKNOWN",
                    "direction": signal_payload.get("direction") or signal_payload.get("agi_direction") or signal_payload.get("regime") or "UNKNOWN",
                    "confidence": signal_payload.get("confidence"),
                    "risk_scalar": risk_scalar,
                    "agi_bias": agi_bias,
                    "agi_direction": signal_payload.get("agi_direction") or signal_payload.get("direction") or signal_payload.get("regime") or "UNKNOWN",
                    "agi_feature_version": signal_payload.get("agi_feature_version") or signal_payload.get("feature_version") or "unknown",
                    "ppo_target": ppo_target,
                    "dreamer_target": dreamer_target,
                    "ppo_weight_used": signal_payload.get("ppo_weight_used"),
                    "dreamer_weight_used": signal_payload.get("dreamer_weight_used"),
                    "agi_weight_used": signal_payload.get("agi_weight_used"),
                    "raw_target": signal_payload.get("raw_target"),
                    "final_target": final_target,
                    "updated_utc": signal_item.get("ts"),
                    "trade_memory": signal_payload.get("trade_memory") if isinstance(signal_payload.get("trade_memory"), dict) else {},
                },
                "execution": {
                    "state": execution_state,
                    "updated_utc": last_update,
                    "request_action": action_payload.get("request_action") or action_payload.get("action") or "watching",
                    "executed": bool(action_payload.get("executed")),
                    "side": action_payload.get("side"),
                    "entry_mode": action_payload.get("entry_mode"),
                    "lots": action_payload.get("lots"),
                    "target_lots": action_payload.get("target_lots"),
                    "lane": action_payload.get("lane"),
                    "model_source": action_payload.get("model_source"),
                    "model_version": action_payload.get("model_version"),
                    "magic": action_payload.get("magic"),
                    "comment": action_payload.get("comment"),
                    "retcode": action_payload.get("retcode"),
                    "ticket": action_payload.get("ticket"),
                    "block_reason": block_payload.get("reason"),
                },
                "profile": {
                    "ppo_weight": profile.get("ppo_weight"),
                    "dreamer_weight": profile.get("dreamer_weight"),
                    "agi_weight": profile.get("agi_weight"),
                    "min_trade_threshold": profile.get("min_trade_threshold"),
                    "max_abs_target": profile.get("max_abs_target"),
                    "cooldown_sec": profile.get("cooldown_sec"),
                },
                "position": {
                    "state": trading_state,
                    "open_positions": len(positions),
                    "floating_pnl": trading_pnl,
                },
            }
        )
    return rows


def _symbol_lane_summary(rows: list[dict]) -> dict:
    summary = {
        "symbols_total": len(rows),
        "actionable_symbols": 0,
        "executed_symbols": 0,
        "blocked_symbols": 0,
        "neutral_symbols": 0,
        "open_positions": 0,
    }
    for row in rows:
        if not isinstance(row, dict):
            continue
        decision = row.get("decision") if isinstance(row.get("decision"), dict) else {}
        execution = row.get("execution") if isinstance(row.get("execution"), dict) else {}
        position = row.get("position") if isinstance(row.get("position"), dict) else {}
        state = str(execution.get("state") or decision.get("state") or "")
        if state in {"armed", "executed"}:
            summary["actionable_symbols"] += 1
        if state == "executed":
            summary["executed_symbols"] += 1
        if state == "blocked":
            summary["blocked_symbols"] += 1
        if state in {"neutral", "watching"}:
            summary["neutral_symbols"] += 1
        summary["open_positions"] += _as_int(position.get("open_positions"), 0)
    return summary


def _latest_training_progress() -> dict:
    out = {
        "drl_symbol": None,
        "drl_timesteps": None,
        "drl_candles": None,
        "lstm_symbol": None,
        "lstm_epoch": None,
        "lstm_epochs_total": None,
        "train_error": None,
        "cycle_ppo_symbol": None,
    }
    ppo_lines = _tail(os.path.join(LOG_DIR, "ppo_training.log"), 200)
    lstm_lines = _tail(os.path.join(LOG_DIR, "lstm_training.log"), 200)
    cycle_lines = _tail(os.path.join(LOG_DIR, "champion_cycle_stderr.log"), 200)

    drl_re = re.compile(
        r"symbols=\['([^']+)'\]\s*\|\s*timesteps=([0-9,]+)(?:.*?\|\s*candles=([0-9,]+))?",
        re.IGNORECASE,
    )
    lstm_re = re.compile(r"([A-Za-z0-9_]+)\s*\|\s*epoch\s+(\d+)\s*/\s*(\d+)")
    cycle_ppo_re = re.compile(r"(?:Cycle step:\s*train PPO candidate for\s+|PPO start\s*\|\s*symbol=)([A-Za-z0-9_]+)", re.IGNORECASE)
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

    for line in reversed(cycle_lines):
        m = cycle_ppo_re.search(line)
        if m:
            out["cycle_ppo_symbol"] = m.group(1)
            break

    for line in reversed(ppo_lines + lstm_lines + cycle_lines):
        if err_re.search(line) and _is_recent_log_line(line, minutes=25):
            out["train_error"] = line
            break

    return out


def _training_state(procs):
    configured_symbols = _configured_symbols()
    drl = _filter_cmd(procs, "training/train_drl.py")
    lstm = _filter_cmd(procs, "training/train_lstm.py")
    dreamer = _filter_cmd(procs, "training/train_dreamer.py")
    cycle = _filter_cmd(procs, "tools/champion_cycle_loop.py")
    if not cycle:
        cycle = _filter_cmd(procs, "tools/champion_cycle.py")
    progress = _latest_training_progress()
    drl_running = len(drl) > 0
    lstm_running = len(lstm) > 0
    dreamer_running = len(dreamer) > 0
    lstm_lines = _tail(os.path.join(LOG_DIR, "lstm_training.log"), 800)
    ppo_lines = _tail(os.path.join(LOG_DIR, "ppo_training.log"), 400)
    dreamer_lines = _tail(os.path.join(LOG_DIR, "dreamer_training.log"), 400)
    visual = _build_training_visuals(
        lstm_lines,
        ppo_lines,
        dreamer_lines,
        lstm_running=lstm_running,
        drl_running=drl_running,
        dreamer_running=dreamer_running,
    )
    cycle_running = len(cycle) > 0
    drl_symbol = progress.get("drl_symbol") or progress.get("cycle_ppo_symbol")
    if cycle_running and not drl_running and not lstm_running and not dreamer_running and drl_symbol:
        visual["active_stage"] = "ppo"
        visual["active_label"] = f"PPO queued for {drl_symbol}"
    return {
        "drl_running": drl_running,
        "lstm_running": lstm_running,
        "dreamer_running": dreamer_running,
        "cycle_running": cycle_running,
        "configured_symbols": configured_symbols,
        "drl_pids": _root_pids(drl),
        "lstm_pids": _root_pids(lstm),
        "dreamer_pids": _root_pids(dreamer),
        "cycle_pids": _root_pids(cycle),
        "drl_symbol": drl_symbol if (drl_running or cycle_running) else None,
        "drl_timesteps": progress.get("drl_timesteps") if drl_running else None,
        "drl_candles": progress.get("drl_candles") if drl_running else None,
        "lstm_symbol": progress.get("lstm_symbol") if lstm_running else None,
        "lstm_epoch": progress.get("lstm_epoch") if lstm_running else None,
        "lstm_epochs_total": progress.get("lstm_epochs_total") if lstm_running else None,
        "train_error": progress.get("train_error"),
        "visual": visual,
        "cycle_heartbeat": _read_cycle_heartbeat(),
    }


def _read_cycle_heartbeat():
    """Read champion loop heartbeat file for cycle health visibility."""
    hb_path = os.path.join(ROOT, ".tmp", "champion_loop.heartbeat")
    if not os.path.exists(hb_path):
        return None
    try:
        with open(hb_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return None


def _server_state(procs):
    servers = _filter_cmd(procs, "python.server_agi")
    if len(servers) > 0:
        return {"running": True, "pids": [p["pid"] for p in servers]}

    # Fallback: query process table directly to avoid false negatives when the
    # shared process snapshot is stale or unavailable.
    rows = _powershell_json(
        "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
        "Where-Object { $_.CommandLine -like '*Python.Server_AGI*' } | "
        "Select-Object ProcessId | ConvertTo-Json -Depth 3"
    )
    pids = [int(r.get("ProcessId") or 0) for r in rows if int(r.get("ProcessId") or 0) > 0]
    return {"running": len(pids) > 0, "pids": pids}


def _n8n_state():
    out = {"running": False, "pid": None, "ports": [], "python_task_runner": "unknown"}
    node_rows = _powershell_json(
        "Get-CimInstance Win32_Process -Filter \"Name='node.exe'\" | "
        "Select-Object ProcessId,CommandLine | ConvertTo-Json -Depth 3"
    )
    for row in node_rows:
        cmd = str(row.get("CommandLine") or "").lower()
        if "n8n" in cmd:
            out["running"] = True
            out["pid"] = row.get("ProcessId")
            break

    netstat = _run(["cmd", "/c", "netstat -ano -p tcp"])
    ports = set()
    owner_by_port = {}
    for line in str(netstat).splitlines():
        parts = line.split()
        if len(parts) < 5:
            continue
        if parts[0].upper() != "TCP" or parts[3].upper() != "LISTENING":
            continue
        local = parts[1]
        pid = parts[4]
        try:
            port = int(local.rsplit(":", 1)[-1])
        except Exception:
            continue
        if port in (5678, 5679):
            ports.add(port)
            owner_by_port[port] = pid
    out["ports"] = sorted(ports)
    if 5678 in ports:
        out["running"] = True
        if out["pid"] is None:
            try:
                out["pid"] = int(owner_by_port.get(5678))
            except Exception:
                pass

    # n8n warns when internal Python runner is unavailable; infer from runtime capability.
    py3 = shutil.which("python3")
    out["python_task_runner"] = "missing" if py3 is None else "present"
    return out


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
            symbol = str(p.symbol)
            side = "BUY" if int(p.type) == 0 else "SELL"
            entry = float(p.price_open)
            tp = float(p.tp) if p.tp else 0.0
            sl = float(p.sl) if p.sl else 0.0
            volume = float(p.volume)
            tp_value = None
            sl_value = None
            try:
                sinfo = mt5.symbol_info(symbol)
                tick_size = float(getattr(sinfo, "trade_tick_size", 0.0) or 0.0)
                tick_value = float(getattr(sinfo, "trade_tick_value", 0.0) or 0.0)
                if tick_size > 0 and tick_value > 0:
                    usd_per_price = tick_value / tick_size
                    if side == "BUY":
                        tp_value = (tp - entry) * usd_per_price * volume
                        sl_value = (sl - entry) * usd_per_price * volume
                    else:
                        tp_value = (entry - tp) * usd_per_price * volume
                        sl_value = (entry - sl) * usd_per_price * volume
            except Exception:
                pass
            rows.append(
                {
                    "ticket": int(p.ticket),
                    "symbol": symbol,
                    "type": side,
                    "volume": volume,
                    "profit": float(p.profit),
                    "open_price": entry,
                    "current_price": float(p.price_current),
                    "sl": sl,
                    "tp": tp,
                    "tp_value_usd": None if tp_value is None else float(tp_value),
                    "sl_value_usd": None if sl_value is None else float(sl_value),
                    "expected_profit_usd": None if tp_value is None else float(tp_value),
                    "expected_loss_usd": None if sl_value is None else float(sl_value),
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


def _record_account_history(account: dict):
    global _ACCOUNT_HISTORY_LAST_TS, _ACCOUNT_HISTORY_LAST_SIG

    if not isinstance(account, dict) or not account.get("connected"):
        return

    try:
        balance = float(account.get("balance")) if account.get("balance") is not None else None
        equity = float(account.get("equity")) if account.get("equity") is not None else None
        profit = float(account.get("profit")) if account.get("profit") is not None else None
        free_margin = float(account.get("free_margin")) if account.get("free_margin") is not None else None
        open_positions = int(account.get("open_positions") or 0)
    except Exception:
        return

    if balance is None or equity is None:
        return

    now = datetime.now(timezone.utc)
    sig = (
        round(balance, 2),
        round(equity, 2),
        round(0.0 if profit is None else profit, 2),
        round(0.0 if free_margin is None else free_margin, 2),
        open_positions,
    )
    if (
        _ACCOUNT_HISTORY_LAST_TS is not None
        and sig == _ACCOUNT_HISTORY_LAST_SIG
        and (now - _ACCOUNT_HISTORY_LAST_TS).total_seconds() < ACCOUNT_HISTORY_INTERVAL_SECONDS
    ):
        return

    row = {
        "ts": now.isoformat(),
        "balance": balance,
        "equity": equity,
        "profit": 0.0 if profit is None else profit,
        "free_margin": 0.0 if free_margin is None else free_margin,
        "open_positions": open_positions,
    }
    try:
        os.makedirs(os.path.dirname(ACCOUNT_HISTORY_PATH), exist_ok=True)
        _rotate_jsonl_if_needed(ACCOUNT_HISTORY_PATH)
        with open(ACCOUNT_HISTORY_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, ensure_ascii=True) + "\n")
        _ACCOUNT_HISTORY_LAST_TS = now
        _ACCOUNT_HISTORY_LAST_SIG = sig
    except Exception:
        return


def _account_history_series(limit: int = 64):
    base = {"source": "unavailable", "labels": [], "equity": [], "drawdown_pct": []}
    if not os.path.exists(ACCOUNT_HISTORY_PATH):
        return base

    entries = []
    for raw in _tail(ACCOUNT_HISTORY_PATH, max(limit * 4, 240)):
        try:
            item = json.loads(raw)
        except Exception:
            continue
        ts_raw = item.get("ts")
        equity = item.get("equity")
        if ts_raw is None or equity is None:
            continue
        try:
            entries.append({"ts": str(ts_raw), "equity": float(equity)})
        except Exception:
            continue

    if not entries:
        return base

    deduped = []
    for item in entries:
        if deduped and deduped[-1]["ts"] == item["ts"]:
            deduped[-1] = item
        else:
            deduped.append(item)
    entries = deduped[-limit:]

    peak = None
    drawdown = []
    for item in entries:
        eq = float(item["equity"])
        peak = eq if peak is None else max(peak, eq)
        dd = 0.0 if not peak else max(0.0, (peak - eq) / peak * 100.0)
        drawdown.append(round(dd, 2))

    return {
        "source": "account_history",
        "labels": [_compact_time_label(item["ts"]) for item in entries],
        "equity": [round(float(item["equity"]), 2) for item in entries],
        "drawdown_pct": drawdown,
    }


def _compact_time_label(raw_ts):
    if not raw_ts:
        return "-"
    try:
        dt = datetime.fromisoformat(str(raw_ts).replace("Z", "+00:00"))
        return dt.astimezone().strftime("%m-%d %H:%M")
    except Exception:
        return str(raw_ts)[11:16] if len(str(raw_ts)) >= 16 else str(raw_ts)


def _profitability_chart_series(limit: int = 32):
    path = os.path.join(LOG_DIR, "profitability.jsonl")
    base = {"source": "unavailable", "labels": [], "equity": [], "drawdown_pct": []}
    if not os.path.exists(path):
        return base

    entries = []
    for raw in _tail(path, max(limit * 6, 240)):
        try:
            item = json.loads(raw)
        except Exception:
            continue
        ts_raw = item.get("ts")
        equity = item.get("equity")
        if ts_raw is None or equity is None:
            continue
        try:
            entries.append(
                {
                    "ts": str(ts_raw),
                    "equity": float(equity),
                }
            )
        except Exception:
            continue

    if not entries:
        return base

    deduped = []
    for item in entries:
        if deduped and deduped[-1]["ts"] == item["ts"]:
            deduped[-1] = item
        else:
            deduped.append(item)
    entries = deduped[-limit:]

    peak = None
    drawdown = []
    for item in entries:
        eq = float(item["equity"])
        peak = eq if peak is None else max(peak, eq)
        dd = 0.0 if not peak else max(0.0, (peak - eq) / peak * 100.0)
        drawdown.append(round(dd, 2))

    return {
        "source": "profitability_log",
        "labels": [_compact_time_label(item["ts"]) for item in entries],
        "equity": [round(float(item["equity"]), 2) for item in entries],
        "drawdown_pct": drawdown,
    }


def _symbol_pnl_chart(symbol_perf):
    rows = list(symbol_perf or [])[:8]
    return {
        "source": "mt5_deals" if rows else "unavailable",
        "labels": [str(row.get("symbol", "?")) for row in rows],
        "values": [round(float(row.get("pnl", 0.0)), 2) for row in rows],
    }


def _dashboard_charts(account: dict, symbol_perf):
    history = _account_history_series()
    profitability = _profitability_chart_series()

    preferred = history if len(history.get("equity") or []) >= 2 else profitability
    if not (preferred.get("equity") or []):
        preferred = history if history.get("equity") else profitability

    equity_values = preferred.get("equity") or []
    drawdown_values = preferred.get("drawdown_pct") or []

    if not equity_values:
        fallback_equity = None
        try:
            fallback_equity = float(account.get("equity")) if account.get("equity") is not None else None
        except Exception:
            fallback_equity = None
        if fallback_equity is not None:
            profitability = {
                "source": "mt5_snapshot",
                "labels": ["now"],
                "equity": [round(fallback_equity, 2)],
                "drawdown_pct": [0.0],
            }
            equity_values = profitability["equity"]
            drawdown_values = profitability["drawdown_pct"]

    return {
        "equity_curve": {
            "source": preferred.get("source", "unavailable"),
            "labels": preferred.get("labels", []),
            "values": equity_values,
        },
        "drawdown_curve": {
            "source": preferred.get("source", "unavailable"),
            "labels": preferred.get("labels", []),
            "values": drawdown_values,
        },
        "symbol_pnl": _symbol_pnl_chart(symbol_perf),
    }


def _file_status(path: str, stale_minutes: int = 15):
    if not os.path.exists(path):
        return {"path": path, "exists": False, "fresh": False, "updated_utc": None, "age_seconds": None}
    try:
        ts = datetime.fromtimestamp(os.path.getmtime(path), tz=timezone.utc)
        age = (datetime.now(timezone.utc) - ts).total_seconds()
        return {
            "path": path,
            "exists": True,
            "fresh": age <= max(60, int(stale_minutes) * 60),
            "updated_utc": ts.isoformat(),
            "age_seconds": round(age, 2),
        }
    except Exception:
        return {"path": path, "exists": True, "fresh": False, "updated_utc": None, "age_seconds": None}


def _source_health():
    return {
        "server_log": _file_status(os.path.join(LOG_DIR, "server.log"), stale_minutes=5),
        "ppo_log": _file_status(os.path.join(LOG_DIR, "ppo_training.log"), stale_minutes=60),
        "lstm_log": _file_status(os.path.join(LOG_DIR, "lstm_training.log"), stale_minutes=60),
        "dreamer_log": _file_status(os.path.join(LOG_DIR, "dreamer_training.log"), stale_minutes=60),
        "audit_log": _file_status(os.path.join(LOG_DIR, "audit_events.jsonl"), stale_minutes=10),
        "account_history": _file_status(ACCOUNT_HISTORY_PATH, stale_minutes=15),
        "trade_learning": _file_status(os.path.join(LOG_DIR, "learning", "trade_learning_latest.json"), stale_minutes=60),
        "event_intel": _file_status(EVENT_INTEL_PATH, stale_minutes=30),
        "active_registry": _file_status(ACTIVE_PATH, stale_minutes=1440),
    }


def _telegram_status():
    cfg = _load_cfg()
    tel = cfg.get("telegram", {}) if isinstance(cfg, dict) else {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_cfg_value(tel.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_cfg_value(tel.get("chat_id"))
    configured = bool(token and chat_id)
    cards = TelegramAlerter(None, None).state_summary(limit=14)
    card_rows = cards.get("cards") if isinstance(cards.get("cards"), list) else []
    recent_live_card = any(str((row or {}).get("delivery_status") or "") == "both" for row in card_rows)
    cards["configured"] = configured
    cards["connected"] = bool(configured or recent_live_card)
    cards["delivery_target"] = "both" if configured else "dashboard_only"
    return cards


def _risk_supervisor_halt_until() -> datetime | None:
    path = os.path.join(LOG_DIR, "risk_supervisor_state.json")
    if not os.path.exists(path):
        return None
    try:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    halt_until = data.get("halt_until")
    if not halt_until:
        return None
    try:
        return datetime.fromisoformat(str(halt_until))
    except Exception:
        return None


def _incident_feed(limit: int = 40):
    path = os.path.join(LOG_DIR, "audit_events.jsonl")
    if not os.path.exists(path):
        return []

    rows = []
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()[-max(limit * 4, 120) :]
    except Exception:
        return []

    for raw in reversed(lines):
        try:
            item = json.loads(raw)
        except Exception:
            continue
        event = str(item.get("event") or "")
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        payload_text = json.dumps(payload, ensure_ascii=True)
        merged = f"{event} {payload_text}".lower()

        severity = "info"
        if any(token in merged for token in ("error", "fail", "exception", "traceback")):
            severity = "critical"
        elif any(token in merged for token in ("warning", "halt", "risk_supervisor_block", "multiple_root_owners", "mixed_executables")):
            severity = "warning"
        elif event in {"trade_open", "trade_closed", "trade_action", "signal"}:
            severity = "activity"

        symbol = payload.get("symbol")
        summary = event.replace("_", " ").strip() or "event"
        if symbol:
            summary = f"{symbol} · {summary}"
        if event == "risk_supervisor_block":
            halt_until = _risk_supervisor_halt_until()
            now = datetime.now(timezone.utc)
            if not halt_until or now >= halt_until:
                continue
            summary = f"{symbol or 'runtime'} · blocked by risk supervisor"
        elif event == "runtime_owner_health" and payload.get("issues"):
            summary = "runtime ownership warning"
        elif event == "signal":
            summary = f"{symbol or 'symbol'} · {payload.get('regime') or payload.get('signal', 'signal')} @ {payload.get('confidence', '-')}"
        elif event == "trade_action":
            summary = (
                f"{symbol or 'symbol'} · "
                f"{payload.get('request_action') or payload.get('action') or 'action'} "
                f"| magic {payload.get('magic') or '-'}"
            )

        rows.append(
            {
                "ts": item.get("ts"),
                "event": event,
                "severity": severity,
                "symbol": symbol,
                "subsystem": event.split("_", 1)[0] if "_" in event else event,
                "summary": summary,
                "payload": payload,
            }
        )
        if len(rows) >= limit:
            break
    return rows


def _registry_summary(active: dict):
    symbols = active.get("symbols", {}) if isinstance(active, dict) else {}
    symbol_rows = []
    for symbol, cfg in sorted(symbols.items()):
        if not isinstance(cfg, dict):
            continue
        canary_state = cfg.get("canary_state", {}) if isinstance(cfg.get("canary_state"), dict) else {}
        symbol_rows.append(
            {
                "symbol": symbol,
                "champion": cfg.get("champion"),
                "canary": cfg.get("canary"),
                "canary_ready": bool(canary_state.get("passed", False)),
                "canary_reason": canary_state.get("reason"),
                "min_trades": (cfg.get("canary_policy", {}) or {}).get("min_trades"),
                "max_drawdown": (cfg.get("canary_policy", {}) or {}).get("max_drawdown"),
            }
        )
    return {
        "champion": active.get("champion"),
        "canary": active.get("canary"),
        "champion_history": list(active.get("champion_history", []) or [])[:6],
        "symbol_rows": symbol_rows,
        "symbol_count": len(symbol_rows),
    }


def _fallback_account_snapshot():
    cached = STATUS_CACHE.get("account") if isinstance(STATUS_CACHE, dict) else None
    if isinstance(cached, dict) and cached:
        return cached
    return {
        "connected": False,
        "balance": None,
        "equity": None,
        "profit": None,
        "free_margin": None,
        "open_positions": 0,
        "positions": [],
    }


def _fallback_charts():
    cached = STATUS_CACHE.get("charts") if isinstance(STATUS_CACHE, dict) else None
    if isinstance(cached, dict) and cached:
        return cached
    return {
        "equity_curve": {"source": "unavailable", "labels": [], "values": []},
        "drawdown_curve": {"source": "unavailable", "labels": [], "values": []},
        "symbol_pnl": {"source": "unavailable", "labels": [], "values": []},
    }


def _collect_status_fast(state: str = "degraded", error: str | None = None):
    procs = _processes()
    active = _active_models()
    server = _server_state(procs)
    account = _fallback_account_snapshot()
    symbol_perf = STATUS_CACHE.get("symbol_perf", []) if isinstance(STATUS_CACHE, dict) else []
    training = _training_state(procs)
    training["symbol_stage_rows"] = _symbol_stage_rows(training, active, account=account, server=server)
    training["pipeline_summary"] = _symbol_pipeline_summary(training["symbol_stage_rows"])
    incidents = _incident_feed(40)
    training["symbol_lane_rows"] = _symbol_lane_rows(training, active, incidents, account=account, server=server)
    training["lane_summary"] = _symbol_lane_summary(training["symbol_lane_rows"])
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ROOT,
        "state": state,
        "error": error,
        "active_models": active,
        "registry": _registry_summary(active),
        "canary_gate": {"ready": False, "reason": "status refresh degraded"},
        "server": server,
        "runtime_owner": _runtime_owner_health(procs),
        "n8n": _n8n_state(),
        "training": training,
        "account": account,
        "symbol_perf": symbol_perf,
        "charts": _fallback_charts(),
        "trade_learning": _trade_learning_status(),
        "event_intel": _event_intel_status(),
        "incidents": incidents,
        "source_health": _source_health(),
        "telegram": _telegram_status(),
        "logs": {
            "server": _tail(os.path.join(LOG_DIR, "server.log"), 50),
            "lstm": _tail(os.path.join(LOG_DIR, "lstm_training.log"), 50),
            "ppo": _tail(os.path.join(LOG_DIR, "ppo_training.log"), 50),
            "dreamer": _tail(os.path.join(LOG_DIR, "dreamer_training.log"), 50),
            "audit": _tail(os.path.join(LOG_DIR, "audit_events.jsonl"), 30),
        },
    }


def _collect_status():
    procs = _processes()
    reg = ModelRegistry()
    canary_ok, canary_reason = reg.can_promote_canary()
    active = _active_models()
    server = _server_state(procs)
    account = _mt5_snapshot()
    _record_account_history(account)
    symbol_perf = _mt5_symbol_perf(7)
    training = _training_state(procs)
    training["symbol_stage_rows"] = _symbol_stage_rows(training, active, account=account, server=server)
    training["pipeline_summary"] = _symbol_pipeline_summary(training["symbol_stage_rows"])
    incidents = _incident_feed(40)
    training["symbol_lane_rows"] = _symbol_lane_rows(training, active, incidents, account=account, server=server)
    training["lane_summary"] = _symbol_lane_summary(training["symbol_lane_rows"])
    # Pattern recognition & perpetual improvement hooks (best-effort)
    try:
        from Python.perpetual_improvement import get_perpetual_improvement_system
        pis = get_perpetual_improvement_system()
        last_action = pis.get_last_improvement_action() or {}
        training["perpetual_improvement"] = last_action
        # Snapshot of learning rates if available
        lr_snapshot = getattr(pis, "learning_rates", {}) if hasattr(pis, "learning_rates") else {}
        training["perpetual_improvement"]["rates_snapshot"] = dict(lr_snapshot)
        # Pattern library exposure from disk (best-effort)
        training["pattern_library"] = _ui_pattern_library()
    except Exception:
        pass
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ROOT,
        "active_models": active,
        "registry": _registry_summary(active),
        "state": "live",
        "canary_gate": {"ready": bool(canary_ok), "reason": canary_reason},
        "server": server,
        "runtime_owner": _runtime_owner_health(procs),
        "n8n": _n8n_state(),
        "training": training,
        "account": account,
        "symbol_perf": symbol_perf,
        "charts": _dashboard_charts(account, symbol_perf),
        "trade_learning": _trade_learning_status(),
        "event_intel": _event_intel_status(),
        "incidents": incidents,
        "source_health": _source_health(),
        "telegram": _telegram_status(),
        "logs": {
            "server": _tail(os.path.join(LOG_DIR, "server.log"), 50),
            "lstm": _tail(os.path.join(LOG_DIR, "lstm_training.log"), 50),
            "ppo": _tail(os.path.join(LOG_DIR, "ppo_training.log"), 50),
            "dreamer": _tail(os.path.join(LOG_DIR, "dreamer_training.log"), 50),
            "audit": _tail(os.path.join(LOG_DIR, "audit_events.jsonl"), 30),
        },
    }


def read_status(refresh_if_booting: bool = True):
    global STATUS_CACHE
    if refresh_if_booting and STATUS_CACHE.get("state") == "booting":
        try:
            STATUS_CACHE = _collect_status()
        except Exception:
            pass
    return STATUS_CACHE


def _symbol_cards_from_status(status: dict):
    cards = {}
    positions = (status.get("account", {}) or {}).get("positions", []) or []
    for row in positions:
        symbol = str(row.get("symbol") or "")
        if not symbol:
            continue
        side = str(row.get("type") or "n/a")
        cards[symbol] = {
            "signal": side,
            "confidence": None,
            "agi_exposure": None,
            "ppo_exposure": None,
            "dreamer_exposure": None,
            "blend_exposure": None,
            "open_positions": 1,
            "floating_pnl": float(row.get("profit", 0.0) or 0.0),
            "position_side": side,
            "position_volume": float(row.get("volume", 0.0) or 0.0),
            "position_entry": row.get("open_price"),
            "position_tp": row.get("tp"),
            "position_sl": row.get("sl"),
            "position_tp_value_usd": row.get("tp_value_usd"),
            "position_sl_value_usd": row.get("sl_value_usd"),
            "last_closed": {},
        }

    for item in status.get("incidents", []) or []:
        if str(item.get("event") or "") != "signal":
            continue
        payload = item.get("payload") if isinstance(item.get("payload"), dict) else {}
        symbol = str(payload.get("symbol") or item.get("symbol") or "")
        if not symbol:
            continue
        card = cards.setdefault(
            symbol,
            {
                "signal": payload.get("signal", "n/a"),
                "confidence": None,
                "agi_exposure": None,
                "ppo_exposure": None,
                "dreamer_exposure": None,
                "blend_exposure": None,
                "open_positions": 0,
                "floating_pnl": 0.0,
                "position_side": "n/a",
                "position_volume": 0.0,
                "position_entry": None,
                "position_tp": None,
                "position_sl": None,
                "position_tp_value_usd": None,
                "position_sl_value_usd": None,
                "last_closed": {},
            },
        )
        card["signal"] = payload.get("signal", card.get("signal", "n/a"))
        card["confidence"] = payload.get("confidence")
        card["agi_exposure"] = payload.get("agi_exposure")
        card["ppo_exposure"] = payload.get("ppo_exposure")
        card["dreamer_exposure"] = payload.get("dreamer_exposure")
        card["blend_exposure"] = payload.get("exposure")

    return cards


def _sync_dashboard_cards(alerter, status: dict):
    if alerter is None:
        return

    account = status.get("account", {}) or {}
    training = status.get("training", {}) or {}
    active_models = status.get("active_models", {}) or {}
    trade_learning = status.get("trade_learning", {}) or {}
    event_intel = status.get("event_intel", {}) or {}

    snapshot = {
        "balance": account.get("balance"),
        "equity": account.get("equity"),
        "free_margin": account.get("free_margin"),
        "pnl_today": trade_learning.get("total_pnl"),
        "floating": account.get("profit"),
        "open_positions": account.get("open_positions"),
    }
    alerter.heartbeat_full(
        uptime="dashboard-live",
        mt5_connected=bool(account.get("connected")),
        trading_enabled=bool(status.get("server", {}).get("running")),
        snapshot=snapshot,
        training=training,
        models=active_models,
        event_intel=event_intel,
    )
    alerter.snapshot(
        balance=account.get("balance"),
        equity=account.get("equity"),
        pnl_today=trade_learning.get("total_pnl"),
        floating=account.get("profit"),
        open_positions=account.get("open_positions"),
    )
    alerter.profitability_daily(trade_learning)
    alerter.model(
        f"champion={active_models.get('champion') or 'none'} | "
        f"canary={active_models.get('canary') or 'none'} | "
        f"gate={status.get('canary_gate', {}).get('reason') or 'n/a'}"
    )

    for symbol, payload in sorted(_symbol_cards_from_status(status).items()):
        alerter.symbol_status(symbol, payload)


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


def _clear_stale_lock(lock_name):
    path = os.path.join(ROOT, ".tmp", lock_name)
    if not os.path.exists(path):
        return False
    try:
        with open(path, "r", encoding="utf-8") as handle:
            pid = int((handle.read() or "0").strip())
    except Exception:
        pid = 0
    if pid > 0:
        try:
            os.kill(pid, 0)
            return False
        except OSError:
            pass
    try:
        os.remove(path)
        return True
    except Exception:
        return False


def _tail_text(path, lines=6):
    if not os.path.exists(path):
        return ""
    try:
        with open(path, "r", encoding="utf-8", errors="replace") as handle:
            return "".join(handle.readlines()[-lines:]).strip()
    except Exception:
        return ""


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
            _clear_stale_lock("train_drl_global.lock")
            timesteps = str(int(payload.get("timesteps", 100000)))
            pid = _spawn([_venv_python(), "training/train_drl.py"], "train_drl_ui_stdout.log", "train_drl_ui_stderr.log", env={"AGI_DRL_TIMESTEPS": timesteps})
            time.sleep(1.2)
            if not _is_running("training/train_drl.py"):
                tail = _tail_text(os.path.join(LOG_DIR, "train_drl_ui_stderr.log"))
                return {"ok": False, "message": f"PPO training failed to start. {tail}".strip()}
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
            if _is_running("training/train_lstm.py") or _is_running("training/train_drl.py"):
                return {
                    "ok": False,
                    "message": "Cannot start champion cycle while standalone LSTM/PPO trainers are running. Stop them first.",
                }
            _clear_stale_lock("champion_cycle.lock")
            pid = _spawn([_venv_python(), "tools/champion_cycle.py"], "champion_cycle_stdout.log", "champion_cycle_stderr.log")
            time.sleep(1.2)
            if not _is_running("tools/champion_cycle.py"):
                tail = _tail_text(os.path.join(LOG_DIR, "champion_cycle_stderr.log"))
                return {"ok": False, "message": f"Champion cycle failed to start. {tail}".strip()}
            return {"ok": True, "message": f"Champion cycle started pid={pid}"}

        if action == "rebuild_trade_memory":
            pid = _spawn([_venv_python(), "training/build_trade_memory.py"], "trade_memory_stdout.log", "trade_memory_stderr.log")
            return {"ok": True, "message": f"Trade memory rebuild started pid={pid}"}

        if action == "restart_server":
            _kill_by_token("python.server_agi")
            lock_path = os.path.join(ROOT, ".tmp", "server_agi.lock")
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
            pid = _spawn([_venv_python(), "-m", "Python.Server_AGI", "--live"], "server_stdout.log", "server_stderr.log")
            time.sleep(1.2)
            if not _is_running("python.server_agi"):
                try:
                    err_path = os.path.join(LOG_DIR, "server_stderr.log")
                    tail = ""
                    if os.path.exists(err_path):
                        with open(err_path, "r", encoding="utf-8", errors="replace") as f:
                            tail = "".join(f.readlines()[-3:]).strip()
                    return {"ok": False, "message": f"Server failed to start. {tail}"}
                except Exception:
                    return {"ok": False, "message": "Server failed to start."}
            return {"ok": True, "message": f"Server restarted pid={pid}"}

        if action == "normalize_owners":
            ids = _normalize_single_owner()
            return {"ok": True, "message": f"Normalized runtime owners; stopped pids={ids}"}

        if action == "set_canary_latest":
            symbol = str(payload.get("symbol") or "").strip()
            cands = sorted(
                [os.path.join(reg.candidates_dir, d) for d in os.listdir(reg.candidates_dir) if os.path.isdir(os.path.join(reg.candidates_dir, d))],
                key=lambda p: os.path.getmtime(p),
                reverse=True,
            )
            if not cands:
                return {"ok": False, "message": "No candidates found"}
            chosen = None
            if symbol:
                safe_symbol = symbol.upper()
                for c in cands:
                    sc = os.path.join(c, "scorecard.json")
                    if not os.path.exists(sc):
                        continue
                    try:
                        with open(sc, "r", encoding="utf-8") as f:
                            meta = json.load(f) or {}
                        if str(meta.get("symbol", "")).upper() == safe_symbol:
                            chosen = c
                            break
                    except Exception:
                        continue
                if chosen is None:
                    return {"ok": False, "message": f"No candidate found for symbol {symbol}"}
                reg.set_canary(chosen, symbol=symbol)
                return {"ok": True, "message": f"Canary set for {symbol}: {chosen}"}
            reg.set_canary(cands[0])
            return {"ok": True, "message": f"Canary set to {cands[0]}"}

        if action == "promote_canary":
            symbol = str(payload.get("symbol") or "").strip()
            reg.promote_canary_to_champion(symbol=symbol or None)
            return {"ok": True, "message": f"Canary promoted to champion{f' for {symbol}' if symbol else ''}"}

        if action == "promote_canary_force":
            symbol = str(payload.get("symbol") or "").strip()
            reg.promote_canary_to_champion(symbol=symbol or None, force=True)
            return {"ok": True, "message": f"Canary force-promoted to champion{f' for {symbol}' if symbol else ''}"}

        if action == "rollback_canary":
            symbol = str(payload.get("symbol") or "").strip()
            reg.rollback_to_champion(symbol=symbol or None)
            return {"ok": True, "message": f"Canary rolled back to champion{f' for {symbol}' if symbol else ''}"}

        if action == "set_timeframe":
            tf = str(payload.get("timeframe", "")).strip().upper()
            valid_tf = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
            if tf not in valid_tf:
                return {"ok": False, "message": f"Invalid timeframe '{tf}'. Use: {', '.join(sorted(valid_tf))}"}
            cfg_path = os.path.join(ROOT, "config.yaml")
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("trading", {})["timeframe"] = tf
                cfg.setdefault("drl", {})["interval"] = tf
                cfg.setdefault("training", {})["lstm_interval"] = tf
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                return {"ok": True, "message": f"Timeframe set to {tf} in config.yaml (trading, drl, lstm). Restart training to use."}
            except Exception as exc:
                return {"ok": False, "message": f"Failed to update config: {exc}"}

        if action == "set_period":
            period = str(payload.get("period", "")).strip()
            if not period:
                return {"ok": False, "message": "Missing 'period' parameter (e.g. '30d', '90d', '180d')"}
            cfg_path = os.path.join(ROOT, "config.yaml")
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("drl", {})["period"] = period
                cfg.setdefault("drl", {})["eval_period"] = period
                cfg.setdefault("training", {})["lstm_period"] = period
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                return {"ok": True, "message": f"Training period set to {period}. Restart training to use."}
            except Exception as exc:
                return {"ok": False, "message": f"Failed to update config: {exc}"}

        if action == "set_timesteps":
            ts = int(payload.get("timesteps", 0))
            if ts < 1000:
                return {"ok": False, "message": "Timesteps must be >= 1000"}
            cfg_path = os.path.join(ROOT, "config.yaml")
            try:
                import yaml
                with open(cfg_path, "r", encoding="utf-8") as f:
                    cfg = yaml.safe_load(f) or {}
                cfg.setdefault("drl", {})["total_timesteps"] = ts
                with open(cfg_path, "w", encoding="utf-8") as f:
                    yaml.safe_dump(cfg, f, default_flow_style=False, sort_keys=False)
                return {"ok": True, "message": f"PPO timesteps set to {ts:,}. Restart training to use."}
            except Exception as exc:
                return {"ok": False, "message": f"Failed to update config: {exc}"}

        if action == "force_ingest":
            # Delete cached data to force fresh ingestion on next training
            cache_dir = os.path.join(ROOT, "data", "dukascopy")
            deleted = 0
            if os.path.isdir(cache_dir):
                for fn in os.listdir(cache_dir):
                    fp = os.path.join(cache_dir, fn)
                    if os.path.isfile(fp):
                        try:
                            os.remove(fp)
                            deleted += 1
                        except Exception:
                            pass
            return {"ok": True, "message": f"Cleared {deleted} cached data files. Next training will fetch fresh MT5 data."}

        if action == "start_cycle_with_tf":
            tf = str(payload.get("timeframe", "")).strip().upper()
            valid_tf = {"M1", "M5", "M15", "M30", "H1", "H4", "D1"}
            if tf not in valid_tf:
                return {"ok": False, "message": f"Invalid timeframe '{tf}'."}
            if _is_running("tools/champion_cycle.py") or _is_running("training/train_drl.py") or _is_running("training/train_lstm.py"):
                return {"ok": False, "message": "Training already running. Stop it first."}
            env_overrides = {
                "AGI_DRL_INTERVAL": tf,
                "AGI_LSTM_INTERVAL": tf,
                "AGI_DREAMER_INTERVAL": tf,
            }
            pid = _spawn([_venv_python(), "tools/champion_cycle.py"], "champion_cycle_stdout.log", "champion_cycle_stderr.log", env=env_overrides)
            time.sleep(1.2)
            if not _is_running("tools/champion_cycle.py"):
                tail = _tail_text(os.path.join(LOG_DIR, "champion_cycle_stderr.log"))
                return {"ok": False, "message": f"Cycle failed to start on {tf}. {tail}".strip()}
            return {"ok": True, "message": f"Champion cycle started on {tf} timeframe, pid={pid}"}

        if action == "start_hft":
            if _is_running("AGI_MODE_TAG=hft") or _is_running("start_hft"):
                return {"ok": True, "message": "HFT server already running"}
            hft_config = os.path.join(ROOT, "config_hft.yaml")
            env_overrides = {
                "AGI_CONFIG": hft_config,
                "AGI_MODE_TAG": "hft",
                "AGI_LOOP_SEC": "5",
                "AGI_HEARTBEAT_SEC": "300",
                "AGI_SYMBOL_CARD_SEC": "60",
                "AGI_TRADE_LEARN_SEC": "300",
            }
            pid = _spawn([_venv_python(), "-m", "Python.Server_AGI", "--live"], "server_hft_stdout.log", "server_hft_stderr.log", env=env_overrides)
            time.sleep(1.2)
            return {"ok": True, "message": f"HFT scalping server started pid={pid}"}

        if action == "stop_hft":
            # Kill HFT server by looking for the mode tag in environment
            ids = []
            try:
                import subprocess as _sp
                out = _sp.check_output(
                    ["wmic", "process", "get", "ProcessId,CommandLine"],
                    creationflags=getattr(_sp, "CREATE_NO_WINDOW", 0),
                    stderr=_sp.DEVNULL, text=True,
                )
                for line in out.strip().splitlines():
                    if "Server_AGI" in line and "hft" in line.lower():
                        parts = line.strip().split()
                        for p in parts:
                            if p.isdigit():
                                try:
                                    os.kill(int(p), 9)
                                    ids.append(int(p))
                                except Exception:
                                    pass
            except Exception:
                pass
            # Also kill by log file pattern
            ids2 = _kill_by_token("server_hft")
            ids.extend(ids2)
            lock_path = os.path.join(ROOT, ".tmp", "server_agi_hft.lock")
            if os.path.exists(lock_path):
                try:
                    os.remove(lock_path)
                except Exception:
                    pass
            return {"ok": True, "message": f"HFT server stopped pids={ids}"}

        if action == "run_hft_cycle":
            if _is_running("start_hft_cycle"):
                return {"ok": True, "message": "HFT training cycle already running"}
            hft_config = os.path.join(ROOT, "config_hft.yaml")
            env_overrides = {
                "AGI_CONFIG": hft_config,
                "AGI_MODE_TAG": "hft",
            }
            pid = _spawn([_venv_python(), "tools/champion_cycle.py"], "hft_cycle_stdout.log", "hft_cycle_stderr.log", env=env_overrides)
            time.sleep(1.2)
            return {"ok": True, "message": f"HFT training cycle started pid={pid}"}

    except Exception as exc:
        return {"ok": False, "message": str(exc)}

    return {"ok": False, "message": f"Unknown action: {action}"}


def _load_html(path):
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as exc:
        return f"<html><body><h1>UI asset missing</h1><pre>{exc}</pre></body></html>"


async def index(_request):
    html = _load_html(UI_HTML_PATH)
    # Inject current status data directly into HTML so page renders
    # even if fetch/WebSocket are blocked by proxy or network
    try:
        status_data = json.dumps(read_status(refresh_if_booting=False), ensure_ascii=False)
        inject_script = (
            '\n<script>'
            'try{render(' + status_data + ');}catch(e){console.warn("SSR render",e);}'
            'if(!window.__jsonpPoll){window.__jsonpPoll=true;'
            'window.__renderJSONP=function(d){try{render(d);}catch(e){}};'
            'setInterval(function(){'
            'var s=document.createElement("script");'
            's.src="/api/jsonp?cb=__renderJSONP&_="+Date.now();'
            's.onerror=function(){try{this.remove();}catch(e){}};'
            's.onload=function(){try{this.remove();}catch(e){}};'
            'document.body.appendChild(s);'
            '},4000);}'
            '</script>\n'
        )
        html = html.replace('</body>', inject_script + '</body>', 1)
    except Exception:
        pass
    return web.Response(text=html, content_type="text/html")


async def mini_app(_request):
    return web.Response(text=_load_html(MINI_UI_HTML_PATH), content_type="text/html")


async def react_app(_request):
    """Serve the React SPA index.html from frontend/dist/."""
    index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    if os.path.isfile(index_path):
        return web.FileResponse(index_path)
    return web.Response(text="React build not found. Run npm run build in frontend/.", status=404)


async def react_app_static(request):
    """Serve static files under /app/ with SPA fallback to index.html."""
    rel_path = request.match_info.get("path", "")
    file_path = os.path.join(FRONTEND_DIST_DIR, rel_path.replace("/", os.sep))
    if rel_path and os.path.isfile(file_path):
        return web.FileResponse(file_path)
    # SPA fallback: serve index.html for any unknown path under /app/
    index_path = os.path.join(FRONTEND_DIST_DIR, "index.html")
    if os.path.isfile(index_path):
        return web.FileResponse(index_path)
    return web.Response(text="React build not found. Run npm run build in frontend/.", status=404)


async def api_status(_request):
    return web.json_response(read_status(refresh_if_booting=False))


async def api_jsonp(request):
    """JSONP fallback for browsers where fetch/WebSocket are blocked."""
    import re as _re
    cb = request.query.get("cb", "render")
    if not _re.match(r'^[a-zA-Z_][a-zA-Z0-9_.]*$', cb):
        cb = "render"
    data = json.dumps(read_status(refresh_if_booting=False), ensure_ascii=False)
    return web.Response(text=f"{cb}({data});", content_type="application/javascript")


async def static_status(_request):
    """Multi-page server-rendered HTML dashboard organized by ticker."""
    page = _request.query.get("p", "overview")
    sel_sym = _request.query.get("s", "")
    d = read_status(refresh_if_booting=False)
    t = d.get("training", {})
    rows = t.get("symbol_stage_rows", [])
    server = d.get("server", {})
    registry = d.get("registry", {})
    gate = d.get("canary_gate", {})
    acct = d.get("account", {})
    vis = t.get("visual", {})
    tl = d.get("trade_learning", {})
    active = d.get("active_models", {})
    sym_perf = d.get("symbol_perf", [])
    all_symbols = [r.get("symbol", "") for r in rows] if rows else list(d.get("active_models", {}).get("symbols", {}).keys())
    stages = [("data_ingest", "MT5 Ingest"), ("features", "Features"), ("lstm", "LSTM Train"),
              ("dreamer", "Dreamer Train"), ("ppo", "PPO Train"), ("candidate", "Evaluate"),
              ("backtest", "Gate Check"), ("champion", "Canary/Champion"), ("trading", "Live Trade")]
    def _e(s):
        return str(s).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;").replace('"', "&quot;")
    def _cls(ok, t_str="good", f_str="warn"):
        return t_str if ok else f_str
    def _pill(label, value, tone=""):
        return f'<div class="status-chip {_e(tone)}"><span>{_e(label)}</span><strong>{_e(value)}</strong></div>'
    page_labels = {
        "overview": "Overview",
        "training": "Training",
        "performance": "Performance",
        "activity": "Activity",
        "control": "Control",
        "ticker": "Ticker detail",
    }
    page_blurb = {
        "overview": "Portfolio, pipeline, and registry at a glance.",
        "training": "Live model training status and queue progress.",
        "performance": "Realized trading performance and equity curves.",
        "activity": "Event intelligence and operational incidents.",
        "control": "Operator controls for runtime and training actions.",
        "ticker": "Per-symbol status and autonomous evolution trace.",
    }
    current_page_label = page_labels.get(page, "Overview")
    current_page_blurb = page_blurb.get(page, "Live production status.")
    last_refresh = d.get("timestamp_utc", "")[:19] or "pending"
    state_value = str(d.get("state", "booting") or "booting")
    state_tone = "good" if state_value == "live" else "warn" if state_value == "booting" else "bad"
    server_running = bool(server.get("running"))
    gate_ready = bool(gate.get("ready"))
    account_connected = bool(acct.get("connected"))
    CSS = """*{box-sizing:border-box;margin:0;padding:0}
body{background:#06101b;color:#edf4ff;font-family:Inter,Segoe UI,Arial,sans-serif;padding:28px 32px;max-width:1600px;margin:0 auto}
h1{font-size:36px;margin-bottom:6px;animation:fadeSlideDown .6s ease}
.sub{color:#8ea3c2;font-size:15px;margin-bottom:12px;animation:fadeSlideDown .6s ease .1s both}

/* Navigation tabs */
.nav{display:flex;gap:6px;flex-wrap:wrap;margin-bottom:24px;animation:fadeSlideDown .5s ease .15s both}
.nav a{padding:10px 22px;border-radius:12px;font-size:14px;font-weight:600;text-decoration:none;color:#8ea3c2;
  background:rgba(255,255,255,.04);border:1px solid rgba(255,255,255,.06);transition:all .25s ease}
.nav a:hover{background:rgba(79,214,255,.1);color:#4fd6ff;border-color:rgba(79,214,255,.2);transform:translateY(-2px)}
.nav a.active{background:rgba(79,214,255,.15);color:#4fd6ff;border-color:rgba(79,214,255,.3);box-shadow:0 0 16px rgba(79,214,255,.15)}
.nav .ticker-link{background:rgba(167,139,250,.08);border-color:rgba(167,139,250,.15)}
.nav .ticker-link:hover,.nav .ticker-link.active{background:rgba(167,139,250,.2);color:#a78bfa;border-color:rgba(167,139,250,.35);box-shadow:0 0 16px rgba(167,139,250,.15)}

.row{display:grid;grid-template-columns:1fr 1fr;gap:20px}
@media(max-width:1000px){.row{grid-template-columns:1fr}}

/* Card appearance + hover lift */
.card{background:rgba(10,18,30,.86);border:1px solid rgba(255,255,255,.08);border-radius:18px;padding:24px 26px;margin-bottom:18px;
  backdrop-filter:blur(12px);transition:transform .25s ease,border-color .3s ease,box-shadow .3s ease;
  animation:fadeSlideUp .5s ease both}
.card:hover{transform:translateY(-3px);border-color:rgba(79,214,255,.22);box-shadow:0 12px 40px rgba(0,0,0,.35),0 0 20px rgba(79,214,255,.06)}
.card:nth-child(1){animation-delay:.05s}.card:nth-child(2){animation-delay:.1s}.card:nth-child(3){animation-delay:.15s}
.card:nth-child(4){animation-delay:.2s}.card:nth-child(5){animation-delay:.25s}.card:nth-child(6){animation-delay:.3s}
.card h2{font-size:20px;margin-bottom:14px;color:#4fd6ff}
.card h3{font-size:15px;margin:14px 0 8px;color:#a78bfa}
.card.ticker-hero{border-color:rgba(167,139,250,.25);background:rgba(15,10,35,.9)}
.card.ticker-hero h2{color:#a78bfa;font-size:24px}

/* KPI grid */
.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:12px}
.kv{padding:14px 16px;background:rgba(255,255,255,.03);border-radius:12px;border:1px solid rgba(255,255,255,.06);
  transition:background .2s ease,transform .2s ease,border-color .2s ease}
.kv:hover{background:rgba(255,255,255,.06);transform:scale(1.03);border-color:rgba(79,214,255,.18)}
.kv .label{font-size:11px;color:#8ea3c2;text-transform:uppercase;letter-spacing:.1em;transition:color .2s}
.kv:hover .label{color:#4fd6ff}
.kv .val{font-size:20px;font-weight:700;margin-top:4px;transition:color .3s ease}
.good{color:#34d399}.warn{color:#fbbf24}.bad{color:#fb7185}.cyan{color:#4fd6ff}

/* Table rows */
table{width:100%;border-collapse:collapse;margin-top:12px}
th,td{text-align:left;padding:10px 12px;border-bottom:1px solid rgba(255,255,255,.06);font-size:14px;transition:background .15s}
tr:hover td{background:rgba(79,214,255,.04)}
th{color:#8ea3c2;font-size:11px;text-transform:uppercase;letter-spacing:.1em}

/* Badges */
.badge{display:inline-block;padding:4px 10px;border-radius:8px;font-size:12px;font-weight:600;transition:transform .15s,box-shadow .15s}
.badge:hover{transform:scale(1.12)}
.badge.done,.badge.live,.badge.ready,.badge.armed{background:rgba(52,211,153,.15);color:#34d399;box-shadow:0 0 8px rgba(52,211,153,.15)}
.badge.active{background:rgba(79,214,255,.15);color:#4fd6ff;box-shadow:0 0 12px rgba(79,214,255,.2);animation:glowPulse 2s ease-in-out infinite}
.badge.failed,.badge.paused{background:rgba(251,113,133,.15);color:#fb7185;box-shadow:0 0 8px rgba(251,113,133,.12)}
.badge.queued,.badge.waiting{background:rgba(255,255,255,.06);color:#8ea3c2}
.badge.testing,.badge.partial{background:rgba(251,191,36,.15);color:#fbbf24;animation:glowAmber 2s ease-in-out infinite}

/* Progress bars - animated fill with ticker label */
.bar{height:18px;background:rgba(255,255,255,.08);border-radius:99px;overflow:hidden;margin-top:6px;position:relative}
.bf{height:100%;border-radius:99px;background:linear-gradient(90deg,#4fd6ff,#a78bfa);animation:barGrow .8s ease both;position:relative;min-width:0}
.bf::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.25),transparent);animation:shimmer 2s ease-in-out infinite}
.bf.done,.bf.live{background:linear-gradient(90deg,#1fd79b,#34d399)}
.bf.failed{background:#fb7185}
.bf.active{background:linear-gradient(90deg,#4fd6ff,#a78bfa)}
.bar-label{position:absolute;left:8px;top:50%;transform:translateY(-50%);font-size:10px;font-weight:700;color:#fff;z-index:1;text-shadow:0 1px 3px rgba(0,0,0,.6);white-space:nowrap}
.bar-pct{position:absolute;right:8px;top:50%;transform:translateY(-50%);font-size:10px;font-weight:600;color:#edf4ff;z-index:1;text-shadow:0 1px 3px rgba(0,0,0,.6)}

/* Large progress bars */
.pbar{height:10px;background:rgba(255,255,255,.06);border-radius:99px;overflow:hidden;margin-top:8px}
.pfill{height:100%;border-radius:99px;animation:barGrow 1s ease both;position:relative}
.pfill::after{content:'';position:absolute;top:0;left:0;right:0;bottom:0;background:linear-gradient(90deg,transparent,rgba(255,255,255,.3),transparent);animation:shimmer 2.5s ease-in-out infinite}

/* Equity chart bars */
.eq-bar{transition:height .4s ease;animation:barRise .6s ease both}

/* Incident feed rows */
.feed-row{transition:background .15s;border-radius:8px;padding:6px 10px !important;font-size:13px !important}
.feed-row:hover{background:rgba(79,214,255,.06)}

/* Live indicator */
.live-dot{display:inline-block;width:10px;height:10px;border-radius:50%;background:#34d399;animation:breathe 2s ease-in-out infinite;margin-right:8px;box-shadow:0 0 8px rgba(52,211,153,.5)}

/* Stage pipeline vertical for ticker page */
.stage-vert{display:flex;flex-direction:column;gap:8px}
.stage-step{display:flex;align-items:center;gap:14px;padding:14px 18px;background:rgba(255,255,255,.02);border-radius:14px;border:1px solid rgba(255,255,255,.05);transition:all .25s}
.stage-step:hover{background:rgba(255,255,255,.05);border-color:rgba(79,214,255,.15)}
.stage-num{width:36px;height:36px;border-radius:50%;display:flex;align-items:center;justify-content:center;font-weight:700;font-size:14px;flex-shrink:0}
.stage-num.done{background:rgba(52,211,153,.2);color:#34d399;border:2px solid #34d399}
.stage-num.active{background:rgba(79,214,255,.2);color:#4fd6ff;border:2px solid #4fd6ff;animation:glowPulse 2s ease-in-out infinite}
.stage-num.failed{background:rgba(251,113,133,.2);color:#fb7185;border:2px solid #fb7185}
.stage-num.queued{background:rgba(255,255,255,.05);color:#8ea3c2;border:2px solid rgba(255,255,255,.15)}
.stage-info{flex:1}
.stage-name{font-size:15px;font-weight:600}
.stage-detail{font-size:12px;color:#8ea3c2;margin-top:2px}

/* Keyframes */
@keyframes fadeSlideUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
@keyframes fadeSlideDown{from{opacity:0;transform:translateY(-10px)}to{opacity:1;transform:translateY(0)}}
@keyframes barGrow{from{width:0 !important}to{}}
@keyframes shimmer{0%{transform:translateX(-100%)}100%{transform:translateX(100%)}}
@keyframes breathe{0%,100%{opacity:1;box-shadow:0 0 8px rgba(52,211,153,.5)}50%{opacity:.6;box-shadow:0 0 16px rgba(52,211,153,.8)}}
@keyframes glowPulse{0%,100%{box-shadow:0 0 8px rgba(79,214,255,.15)}50%{box-shadow:0 0 18px rgba(79,214,255,.35)}}
@keyframes glowAmber{0%,100%{box-shadow:0 0 6px rgba(251,191,36,.1)}50%{box-shadow:0 0 14px rgba(251,191,36,.3)}}
@keyframes spin{to{transform:rotate(360deg)}}
@keyframes barRise{from{height:0}}

/* ── Animated background grid ── */
body::before{content:'';position:fixed;inset:0;z-index:-2;
  background:linear-gradient(rgba(79,214,255,.025) 1px,transparent 1px),
             linear-gradient(90deg,rgba(79,214,255,.025) 1px,transparent 1px);
  background-size:44px 44px;animation:gridDrift 30s linear infinite;pointer-events:none}
@keyframes gridDrift{to{background-position:44px 44px}}
body::after{content:'';position:fixed;inset:0;z-index:-1;
  background:radial-gradient(ellipse at 20% 40%,rgba(79,214,255,.04) 0%,transparent 55%),
             radial-gradient(ellipse at 80% 20%,rgba(167,139,250,.04) 0%,transparent 55%),
             radial-gradient(ellipse at 50% 90%,rgba(52,211,153,.03) 0%,transparent 50%);
  pointer-events:none}

/* ── Floating particles ── */
.particle{position:fixed;width:3px;height:3px;border-radius:50%;opacity:0;pointer-events:none;
  animation:particleFloat 8s ease-in-out infinite}
@keyframes particleFloat{
  0%{opacity:0;transform:translateY(100vh) scale(0)}
  20%{opacity:.6}60%{opacity:.4}
  100%{opacity:0;transform:translateY(-10vh) scale(1.2)}}

/* ── Enhanced active bar effects ── */
.bf.active{background:linear-gradient(90deg,#4fd6ff,#a78bfa,#4fd6ff);background-size:300% 100%;
  animation:barFlow 2.5s ease-in-out infinite;box-shadow:0 0 8px rgba(79,214,255,.3)}
@keyframes barFlow{0%{background-position:100% 0}100%{background-position:-100% 0}}
.pfill{transition:width .8s cubic-bezier(.4,0,.2,1)}

/* ── Glow pulse on active cards ── */
.card:has(.badge.active){border-color:rgba(79,214,255,.2);
  box-shadow:0 0 24px rgba(79,214,255,.06),0 4px 20px rgba(0,0,0,.2)}

/* ── Spinning indicator for active badge ── */
.badge.active::before{content:'';display:inline-block;width:8px;height:8px;border:2px solid transparent;
  border-top-color:currentColor;border-radius:50%;margin-right:5px;vertical-align:middle;
  animation:spin .8s linear infinite}

/* ── Value change flash ── */
.val-flash{animation:valFlash .6s ease}
@keyframes valFlash{0%{text-shadow:0 0 12px currentColor}100%{text-shadow:none}}

/* ── Smooth number counter feel ── */
.kv .val{transition:color .4s ease,opacity .15s}

/* ── Connection status bar ── */
#wsStatus{transition:background .3s;border-radius:8px;padding:8px 14px;
  background:rgba(255,255,255,.02);border:1px solid rgba(255,255,255,.04)}
.conn-live{color:#34d399}.conn-poll{color:#fbbf24}.conn-off{color:#fb7185}

/* Production shell overrides */
body{padding:22px 24px 36px;max-width:1720px;line-height:1.5;text-rendering:optimizeLegibility}
.dashboard-shell{display:grid;gap:18px}
.page-header{display:grid;grid-template-columns:minmax(0,1.18fr) minmax(320px,.82fr);gap:14px;padding:24px;border:1px solid rgba(255,255,255,.08);border-radius:26px;
  background:linear-gradient(180deg,rgba(12,20,34,.95),rgba(7,12,20,.9));box-shadow:0 18px 38px rgba(0,0,0,.28);backdrop-filter:blur(14px)}
.page-copy{display:grid;gap:8px}
.eyebrow{font-size:11px;letter-spacing:.22em;text-transform:uppercase;color:#8ddfff}
.page-header h1{font-size:clamp(30px,4vw,48px);line-height:1.02;margin:0;letter-spacing:-.05em}
.hero-meta{display:flex;flex-wrap:wrap;gap:10px;color:#8ea3c2;font-size:12px}
.status-strip{display:grid;grid-template-columns:repeat(5,minmax(0,1fr));gap:10px}
.status-chip{padding:12px 14px;border-radius:16px;border:1px solid rgba(255,255,255,.08);background:rgba(255,255,255,.03);display:grid;gap:3px}
.status-chip span{font-size:10px;text-transform:uppercase;letter-spacing:.16em;color:#8ea3c2}
.status-chip strong{font-size:15px;font-weight:700}
.status-chip.good strong{color:#34d399}
.status-chip.warn strong{color:#fbbf24}
.status-chip.bad strong{color:#fb7185}
.status-chip.cyan strong{color:#4fd6ff}
.nav{position:sticky;top:12px;z-index:30;backdrop-filter:blur(12px);background:rgba(5,10,18,.66);padding:10px;border-radius:18px;border:1px solid rgba(255,255,255,.06)}
.nav a{border-radius:14px}
.card{padding:22px 24px}
.card h2,.card h3{display:flex;align-items:center;gap:10px;flex-wrap:wrap}
.table-shell{overflow-x:auto;border:1px solid rgba(255,255,255,.06);border-radius:18px;background:rgba(255,255,255,.02);box-shadow:inset 0 1px 0 rgba(255,255,255,.03)}
.table-shell table{min-width:760px;margin:0}
.table-shell th{position:sticky;top:0;background:rgba(10,18,30,.95);backdrop-filter:blur(8px)}
table{width:100%;border-collapse:separate;border-spacing:0}
th,td{border-bottom:1px solid rgba(255,255,255,.06)}
td:first-child,th:first-child{padding-left:14px}
td:last-child,th:last-child{padding-right:14px}
@media(max-width:1200px){.page-header{grid-template-columns:1fr}.status-strip{grid-template-columns:repeat(2,minmax(0,1fr))}}
@media(max-width:900px){body{padding:16px}.nav{position:static}.row,.laneGrid,.traceMetaGrid,.stats,.status-strip{grid-template-columns:1fr}.card,.page-header{padding:18px}.table-shell table{min-width:640px}}
@media(max-width:640px){h1,.page-header h1{font-size:30px}.sub{font-size:13px}.nav a{padding:8px 12px;font-size:13px}}"""

    # === Build navigation ===
    def _nav_link(p_name, label, is_ticker=False):
        active = " active" if (page == p_name or (p_name == "ticker" and page == "ticker" and sel_sym == label)) else ""
        cls = "ticker-link" if is_ticker else ""
        if is_ticker:
            return f'<a class="{cls}{active}" href="/static?p=ticker&s={_e(label)}">{_e(label)}</a>'
        return f'<a class="{cls}{active}" href="/static?p={_e(p_name)}">{_e(label)}</a>'

    h = [f'<!DOCTYPE html><html><head><meta charset="utf-8"><title>Cautious Giggle</title>',
         '<meta http-equiv="refresh" content="8" id="metaRefresh">',
         f'<style>{CSS}</style></head><body>',
         '<div id="bgParticles" style="position:fixed;inset:0;pointer-events:none;z-index:-1;overflow:hidden"></div>',
         '<main class="dashboard-shell">',
         '<section class="page-header">',
         '<div class="page-copy">',
         '<div class="eyebrow">Live status dashboard</div>',
         '<h1>Cautious Giggle</h1>',
         f'<div class="sub">{_e(current_page_blurb)}</div>',
         f'<div class="hero-meta"><span><span class="live-dot"></span>Auto-refresh every 8s</span><span>Updated { _e(last_refresh) }</span><span>{_e(current_page_label)}</span></div>',
         '</div>',
         '<div class="status-strip">',
         _pill("State", state_value, state_tone),
         _pill("Server", "Running" if server_running else "Stopped", "good" if server_running else "warn"),
         _pill("Gate", "Ready" if gate_ready else "Hold", "good" if gate_ready else "warn"),
         _pill("Account", "Connected" if account_connected else "Offline", "good" if account_connected else "warn"),
         _pill("Champion", registry.get("champion") or "none", "good" if registry.get("champion") else "warn"),
         '</div></section>']

    # Navigation bar
    h.append('<div class="nav">')
    h.append(_nav_link("overview", "Overview"))
    h.append(_nav_link("training", "Training"))
    h.append(_nav_link("performance", "Performance"))
    h.append(_nav_link("activity", "Activity"))
    h.append(_nav_link("control", "Control"))
    h.append('<span style="width:2px;background:rgba(255,255,255,.1);margin:0 8px;border-radius:1px"></span>')
    for sym in all_symbols:
        h.append(_nav_link("ticker", sym, is_ticker=True))
    h.append('</div>')

    # Build corrected status from symbol_stage_rows (overrides stale queue)
    _corrected = {}
    for _row in rows:
        _sym = _row.get("symbol", "")
        for _mk in ("lstm", "dreamer", "ppo"):
            _sg = _row.get(_mk, {})
            _corrected.setdefault(_mk, {})[_sym] = _sg

    # ============================================================
    # PAGE: OVERVIEW
    # ============================================================
    if page == "overview":
        # System Overview
        h.append('<div class="card"><h2>System Overview</h2><div class="grid">')
        srv = server.get("running", False)
        h.append(f'<div class="kv"><div class="label">State</div><div class="val good" data-v="state">{_e(d.get("state","?"))}</div></div>')
        h.append(f'<div class="kv"><div class="label">Server</div><div class="val {_cls(srv)}" data-v="server.running" data-fmt="bool">{"Running" if srv else "Stopped"}</div></div>')
        h.append(f'<div class="kv"><div class="label">Champion</div><div class="val" data-v="registry.champion">{_e(registry.get("champion") or "none")}</div></div>')
        h.append(f'<div class="kv"><div class="label">Canary Gate</div><div class="val {_cls(gate.get("ready"))}" data-v="canary_gate.ready" data-fmt="gate">{"Ready" if gate.get("ready") else "Hold"}</div></div>')
        if acct.get("connected"):
            h.append(f'<div class="kv"><div class="label">Balance</div><div class="val" data-v="account.balance" data-fmt="usd">${acct.get("balance",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Equity</div><div class="val" data-v="account.equity" data-fmt="usd">${acct.get("equity",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Profit</div><div class="val {_cls(acct.get("profit",0)>=0)}" data-v="account.profit" data-fmt="usd">${acct.get("profit",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Open Positions</div><div class="val" data-v="account.open_positions" data-fmt="int">{acct.get("open_positions",0)}</div></div>')
        h.append('</div></div>')

        # 9-Stage Pipeline
        if rows:
            h.append('<div class="card"><h2>9-Stage Pipeline</h2>')
            h.append('<div class="table-shell"><table><tr><th>Symbol</th>')
            for _, lbl in stages:
                h.append(f'<th>{_e(lbl)}</th>')
            h.append('</tr>')
            for row in rows:
                sym_name = row.get("symbol", "?")
                h.append(f'<tr><td style="font-weight:600"><a href="/static?p=ticker&s={_e(sym_name)}" style="color:#a78bfa;text-decoration:none">{_e(sym_name)}</a></td>')
                for key, _ in stages:
                    sg = row.get(key, {})
                    st = sg.get("state", "queued")
                    pct = int(sg.get("progress_pct", 0) or 0)
                    det = str(sg.get("detail", "") or "")[:40]
                    h.append(f'<td><span class="badge {_e(st)}" data-tb="{_e(key)}.{_e(sym_name)}">{_e(st)}</span>')
                    h.append(f'<div style="font-size:9px;color:#8ea3c2;margin-top:2px" data-sd="{_e(key)}.{_e(sym_name)}">{_e(det) if det and det != chr(8212) else ""}</div>')
                    h.append(f'<div class="bar"><span class="bar-label">{_e(sym_name)}</span><div class="bf {_e(st)}" style="width:{pct}%"></div><span class="bar-pct">{pct}%</span></div></td>')
                h.append('</tr>')
            h.append('</table></div></div>')

        # Ticker quick cards
        if all_symbols:
            h.append('<div class="row">')
            for sym in all_symbols:
                sym_row = next((r for r in rows if r.get("symbol") == sym), {})
                sym_active_info = active.get("symbols", {}).get(sym, {})
                # Count done/total stages
                n_done_s = sum(1 for k, _ in stages if sym_row.get(k, {}).get("state") == "done")
                n_active_s = sum(1 for k, _ in stages if sym_row.get(k, {}).get("state") == "active")
                # Sym perf
                sp = next((s for s in sym_perf if s.get("symbol") == sym), {})
                h.append(f'<a href="/static?p=ticker&s={_e(sym)}" style="text-decoration:none;color:inherit">')
                h.append(f'<div class="card ticker-hero" style="cursor:pointer">')
                h.append(f'<h2>{_e(sym)}</h2>')
                h.append('<div class="grid">')
                h.append(f'<div class="kv"><div class="label">Pipeline</div><div class="val">{n_done_s}/{len(stages)} done</div></div>')
                if n_active_s:
                    h.append(f'<div class="kv"><div class="label">Active Stages</div><div class="val cyan">{n_active_s}</div></div>')
                champ = sym_active_info.get("champion") or "none"
                h.append(f'<div class="kv"><div class="label">Champion</div><div class="val">{_e(champ[:20])}</div></div>')
                if sp:
                    spnl = sp.get("pnl", 0)
                    h.append(f'<div class="kv"><div class="label">7d PnL</div><div class="val {_cls(spnl>=0)}">${spnl:,.2f}</div></div>')
                h.append('</div></div></a>')
            h.append('</div>')

        # Model Registry summary
        h.append('<div class="card"><h2>Model Registry</h2><div class="grid">')
        h.append(f'<div class="kv"><div class="label">Global Champion</div><div class="val">{_e(active.get("champion") or "none")}</div></div>')
        h.append(f'<div class="kv"><div class="label">Global Canary</div><div class="val">{_e(active.get("canary") or "none")}</div></div>')
        h.append('</div>')
        syms_reg = active.get("symbols", {})
        if syms_reg:
            h.append('<div class="table-shell"><table><tr><th>Symbol</th><th>Champion</th><th>Canary</th><th>Canary State</th></tr>')
            for sym, info in syms_reg.items():
                cs = info.get("canary_state", {})
                canary_val = info.get("canary") or "none"
                if canary_val != "none" and len(canary_val) > 30:
                    canary_val = "..." + canary_val[-25:]
                h.append(f'<tr><td style="font-weight:600"><a href="/static?p=ticker&s={_e(sym)}" style="color:#a78bfa;text-decoration:none">{_e(sym)}</a></td>')
                h.append(f'<td>{_e(info.get("champion") or "none")}</td>')
                h.append(f'<td style="font-size:10px">{_e(canary_val)}</td>')
                h.append(f'<td><span class="badge {"done" if cs.get("passed") else "warn"}">{_e(cs.get("reason",""))}</span></td></tr>')
            h.append('</table></div>')
        h.append('</div>')

    # ============================================================
    # PAGE: TICKER DETAIL (per-symbol)
    # ============================================================
    elif page == "ticker" and sel_sym:
        sym_row = next((r for r in rows if r.get("symbol") == sel_sym), {})
        sym_active_info = active.get("symbols", {}).get(sel_sym, {})
        sp = next((s for s in sym_perf if s.get("symbol") == sel_sym), {})
        tl_sym = next((b for b in tl.get("best_symbols", []) if b.get("symbol") == sel_sym), {})

        # Hero card
        h.append(f'<div class="card ticker-hero"><h2>{_e(sel_sym)}</h2>')
        h.append('<div class="grid">')
        champ = sym_active_info.get("champion") or "none"
        canary = sym_active_info.get("canary") or "none"
        cs = sym_active_info.get("canary_state", {})
        h.append(f'<div class="kv"><div class="label">Champion</div><div class="val">{_e(champ[:30])}</div></div>')
        h.append(f'<div class="kv"><div class="label">Canary</div><div class="val">{_e(canary[:30])}</div></div>')
        h.append(f'<div class="kv"><div class="label">Canary Gate</div><div class="val {_cls(cs.get("passed"))}">{_e(cs.get("reason","pending"))}</div></div>')
        if sp:
            h.append(f'<div class="kv"><div class="label">7d Trades</div><div class="val">{sp.get("trades",0)}</div></div>')
            spnl = sp.get("pnl", 0)
            h.append(f'<div class="kv"><div class="label">7d PnL</div><div class="val {_cls(spnl>=0)}">${spnl:,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">7d Win Rate</div><div class="val {_cls(sp.get("win_rate",0)>40)}">{sp.get("win_rate",0):.1f}%</div></div>')
        if tl_sym:
            h.append(f'<div class="kv"><div class="label">All-time Trades</div><div class="val">{tl_sym.get("trades",0)}</div></div>')
            h.append(f'<div class="kv"><div class="label">Expectancy</div><div class="val {_cls(tl_sym.get("expectancy",0)>0)}">${tl_sym.get("expectancy",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Profit Factor</div><div class="val {_cls(tl_sym.get("profit_factor",0)>1)}">{tl_sym.get("profit_factor",0):.2f}</div></div>')
        h.append('</div></div>')

        # Pipeline stages - vertical layout with descriptions
        stage_desc = {
            "data_ingest": "Pull live M5 candles from MetaTrader 5",
            "features": "Engineer 150 technical features from raw OHLCV",
            "lstm": "Train sequence memory model on price patterns",
            "dreamer": "Train world-model for future state prediction",
            "ppo": "Train RL policy via Proximal Policy Optimization",
            "candidate": "Evaluate candidate vs current champion via backtest",
            "backtest": "Gate check: Sharpe, drawdown, return thresholds",
            "champion": "Deploy as canary, monitor live PnL for promotion",
            "trading": "Live autonomous trading with risk supervisor",
        }
        if sym_row:
            h.append(f'<div class="card"><h2>Autonomous Evolution Pipeline <button class="ibtn cycle" onclick="ctrlBtn(\'run_cycle\')">Run Full Cycle</button> <button class="ibtn rerun" onclick="ctrlBtn(\'force_ingest\');setTimeout(function(){{ctrlBtn(\'run_cycle\')}},1000)">Fresh Data + Cycle</button></h2>')
            h.append(f'<div style="color:#8ea3c2;font-size:12px;margin-bottom:14px">Full training-to-trade cycle for <span style="color:#a78bfa;font-weight:700">{_e(sel_sym)}</span></div>')
            h.append('<div class="stage-vert">')
            for idx, (key, lbl) in enumerate(stages, 1):
                sg = sym_row.get(key, {})
                st = sg.get("state", "queued")
                pct = int(sg.get("progress_pct", 0) or 0)
                det = str(sg.get("detail", "") or "")[:60]
                desc = stage_desc.get(key, "")
                h.append(f'<div class="stage-step" style="animation:fadeSlideUp .4s ease {round(idx*0.06,2)}s both">')
                h.append(f'<div class="stage-num {_e(st)}">{idx}</div>')
                h.append(f'<div class="stage-info"><div class="stage-name">{_e(lbl)} <span class="badge {_e(st)}">{_e(st)}</span></div>')
                h.append(f'<div class="stage-detail">{_e(desc)}</div>')
                if det and det != "\u2014":
                    h.append(f'<div class="stage-detail" style="color:#4fd6ff;margin-top:2px" data-sd="{_e(key)}.{_e(sel_sym)}">{_e(det)}</div>')
                else:
                    h.append(f'<div class="stage-detail" style="color:#4fd6ff;margin-top:2px;display:none" data-sd="{_e(key)}.{_e(sel_sym)}"></div>')
                h.append(f'<div class="bar" style="margin-top:4px"><span class="bar-label">{_e(sel_sym)}</span><div class="bf {_e(st)}" style="width:{pct}%"></div><span class="bar-pct">{pct}%</span></div>')
                h.append('</div></div>')
            h.append('</div></div>')

        # Training detail per model for this symbol
        h.append('<div class="row">')
        for mk, ml, color in [("lstm", "LSTM", "#4fd6ff"), ("ppo", "PPO", "#a78bfa"), ("dreamer", "DreamerV3", "#34d399")]:
            mv = vis.get(mk, {})
            corr_item = _corrected.get(mk, {}).get(sel_sym, {})
            qi = next((q for q in mv.get("queue", []) if q.get("symbol") == sel_sym), {})
            if not corr_item and not qi:
                continue
            st = corr_item.get("state") or qi.get("status", "queued")
            pct = corr_item.get("progress_pct") or qi.get("progress_pct", 0) or 0
            _act = "start_lstm" if mk == "lstm" else "start_drl"
            _stp = "stop_lstm" if mk == "lstm" else "stop_drl"
            h.append(f'<div class="card"><h2 style="color:{color}">{ml}')
            if mk in ("lstm", "ppo"):
                h.append(f'<button class="ibtn start" onclick="ctrlBtn(\'{_act}\')">Start</button>')
                h.append(f'<button class="ibtn rerun" onclick="ctrlBtn(\'{_stp}\');setTimeout(function(){{ctrlBtn(\'{_act}\')}},2000)">Re-run</button>')
                h.append(f'<button class="ibtn stop" onclick="ctrlBtn(\'{_stp}\')">Stop</button>')
            h.append('</h2>')
            h.append('<div class="grid">')
            h.append(f'<div class="kv"><div class="label">Status</div><div class="val"><span class="badge {_e(st)}" data-tb="{_e(mk)}.{_e(sel_sym)}">{_e(st)}</span></div></div>')
            h.append(f'<div class="kv"><div class="label">Progress</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.pct">{int(pct)}%</div></div>')
            if mk == "lstm":
                h.append(f'<div class="kv"><div class="label">Epoch</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.epoch">{qi.get("epoch","?")}/{qi.get("epochs_total","?")}</div></div>')
                h.append(f'<div class="kv"><div class="label">Loss</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.loss">{qi.get("loss","?")}</div></div>')
                h.append(f'<div class="kv"><div class="label">Accuracy</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.acc">{qi.get("acc","?")}%</div></div>')
            elif mk == "ppo":
                h.append(f'<div class="kv"><div class="label">Timesteps</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.timesteps">{qi.get("current_timesteps",0):,}/{qi.get("target_timesteps",0):,}</div></div>')
            elif mk == "dreamer":
                h.append(f'<div class="kv"><div class="label">Steps</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.steps">{qi.get("steps",0):,}</div></div>')
                h.append(f'<div class="kv"><div class="label">Detail</div><div class="val" data-tg="{_e(mk)}.{_e(sel_sym)}.detail">{_e(qi.get("detail",""))}</div></div>')
            h.append('</div>')
            h.append(f'<div class="pbar" style="position:relative"><span class="bar-label">{_e(sel_sym)}</span><div class="pfill" style="width:{int(pct)}%;background:{color}"></div><span class="bar-pct">{int(pct)}%</span></div>')
            h.append('</div>')
        h.append('</div>')

        # Event intel for this symbol
        ei = d.get("event_intel", {})
        sym_ei = ei.get("by_symbol", {}).get(sel_sym, {})
        if sym_ei:
            regime = sym_ei.get("regime", "normal")
            h.append(f'<div class="card"><h2>Event Intelligence</h2><div class="grid">')
            h.append(f'<div class="kv"><div class="label">Regime</div><div class="val {_cls(regime=="normal")}">{_e(regime)}</div></div>')
            h.append('</div></div>')

        # Incidents for this symbol
        incidents = d.get("incidents", [])
        sym_incidents = [i for i in incidents if i.get("symbol") == sel_sym]
        if sym_incidents:
            h.append(f'<div class="card"><h2>Activity Feed</h2>')
            h.append('<div style="max-height:300px;overflow-y:auto">')
            sev_cls = {"critical": "bad", "warning": "warn", "info": "cyan", "activity": "good"}
            for inc_idx, inc in enumerate(sym_incidents[:20]):
                sev = inc.get("severity", "info")
                ts_raw = inc.get("ts", "")
                ts_short = ts_raw[11:19] if len(ts_raw) > 19 else ts_raw
                summ = inc.get("summary", "")
                h.append(f'<div class="feed-row" style="display:flex;gap:8px;align-items:baseline;border-bottom:1px solid rgba(255,255,255,.04);animation:fadeSlideUp .4s ease {round(inc_idx*0.04,2)}s both">')
                h.append(f'<span style="color:#8ea3c2;min-width:56px;font-size:10px">{_e(ts_short)}</span>')
                h.append(f'<span class="badge {_e(sev)}" style="min-width:52px;text-align:center">{_e(sev)}</span>')
                h.append(f'<span>{_e(summ[:120])}</span>')
                h.append('</div>')
            h.append('</div></div>')

    # ============================================================
    # PAGE: TRAINING
    # ============================================================
    elif page == "training":
        for mk, ml, color in [("lstm", "LSTM (Memory Forge)", "#4fd6ff"),
                               ("ppo", "PPO (Tension Chamber)", "#a78bfa"),
                               ("dreamer", "DreamerV3 (Future Prism)", "#34d399")]:
            mv = vis.get(mk, {})
            if not mv:
                continue
            corr = _corrected.get(mk, {})
            n_done = sum(1 for v in corr.values() if v.get("state") == "done")
            n_active = sum(1 for v in corr.values() if v.get("state") == "active")
            n_failed = sum(1 for v in corr.values() if v.get("state") == "failed")
            n_total = len(corr) or mv.get("summary", {}).get("total_symbols", 0)
            comp_pct = (n_done / n_total * 100) if n_total else 0
            queue = mv.get("queue", [])
            _act = "start_lstm" if mk == "lstm" else "start_drl"
            _stp = "stop_lstm" if mk == "lstm" else "stop_drl"
            h.append(f'<div class="card"><h2 style="color:{color}">{ml}')
            if mk in ("lstm", "ppo"):
                h.append(f'<button class="ibtn start" onclick="ctrlBtn(\'{_act}\')">Start</button>')
                h.append(f'<button class="ibtn rerun" onclick="ctrlBtn(\'{_stp}\');setTimeout(function(){{ctrlBtn(\'{_act}\')}},2000)">Re-run</button>')
                h.append(f'<button class="ibtn stop" onclick="ctrlBtn(\'{_stp}\')">Stop</button>')
            h.append('</h2>')
            h.append('<div class="grid">')
            h.append(f'<div class="kv"><div class="label">Completion</div><div class="val" data-tg="{mk}.completion">{comp_pct:.0f}%</div></div>')
            h.append(f'<div class="kv"><div class="label">Completed</div><div class="val" data-tg="{mk}.completed">{n_done}/{n_total}</div></div>')
            h.append(f'<div class="kv"><div class="label">Active</div><div class="val good" data-tg="{mk}.active">{n_active}</div></div>')
            h.append(f'<div class="kv"><div class="label">Failed</div><div class="val bad" data-tg="{mk}.failed">{n_failed}</div></div>')
            if mk == "lstm":
                h.append(f'<div class="kv"><div class="label">Epochs</div><div class="val" data-tg="{mk}.epochs">{mv.get("epochs_total","—")}</div></div>')
            if mk == "ppo":
                h.append(f'<div class="kv"><div class="label">Timesteps</div><div class="val" data-tg="{mk}.timesteps">{mv.get("target_timesteps",0):,}</div></div>')
                h.append(f'<div class="kv"><div class="label">Current</div><div class="val" data-tg="{mk}.current_ts">{mv.get("current_timesteps",0):,}</div></div>')
            if mk == "dreamer":
                h.append(f'<div class="kv"><div class="label">Steps</div><div class="val" data-tg="{mk}.steps">{mv.get("steps",0):,}</div></div>')
            h.append('</div>')
            # Per-symbol progress bars with ticker labels (replaces old summary bar)
            if queue:
                for qi in queue:
                    qsym = qi.get("symbol", "")
                    corr_item = corr.get(qsym, {})
                    qs = corr_item.get("state") or qi.get("status", "queued")
                    qpct = corr_item.get("progress_pct") or qi.get("progress_pct", 0) or 0
                    qdet = ""
                    if mk == "lstm":
                        qdet = f'epoch {qi.get("epoch","?")}/{qi.get("epochs_total","?")} loss={qi.get("loss","?")} acc={qi.get("acc","?")}%'
                    elif mk == "ppo":
                        qdet = f'{qi.get("current_timesteps",0):,}/{qi.get("target_timesteps",0):,}'
                    elif mk == "dreamer":
                        qdet = qi.get("detail", "") or f'steps={qi.get("steps",0)}'
                    h.append(f'<div style="margin-top:10px">')
                    h.append(f'<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:4px">')
                    h.append(f'<a href="/static?p=ticker&s={_e(qsym)}" style="color:#a78bfa;text-decoration:none;font-weight:700;font-size:14px">{_e(qsym)}</a>')
                    h.append(f'<span><span class="badge {_e(qs)}" data-tb="{_e(mk)}.{_e(qsym)}">{_e(qs)}</span> <span style="color:#8ea3c2;font-size:11px;margin-left:6px" data-td="{_e(mk)}.{_e(qsym)}">{_e(qdet)}</span></span>')
                    h.append(f'</div>')
                    h.append(f'<div class="bar"><span class="bar-label">{_e(qsym)}</span><div class="bf {_e(qs)}" style="width:{int(qpct)}%"></div><span class="bar-pct">{int(qpct)}%</span></div>')
                    h.append(f'</div>')
                h.append(f'<div class="pbar" style="margin-top:14px;position:relative"><span class="bar-label">Overall</span><div class="pfill" style="width:{comp_pct}%;background:{color}"></div><span class="bar-pct">{comp_pct:.0f}%</span></div>')
            else:
                h.append(f'<div class="pbar" style="position:relative"><span class="bar-label">Overall</span><div class="pfill" style="width:{comp_pct}%;background:{color}"></div><span class="bar-pct">{comp_pct:.0f}%</span></div>')
            h.append('</div>')

    # ============================================================
    # PAGE: PERFORMANCE
    # ============================================================
    elif page == "performance":
        # Account overview
        if acct.get("connected"):
            h.append('<div class="card"><h2>Account</h2><div class="grid">')
            h.append(f'<div class="kv"><div class="label">Balance</div><div class="val" data-v="account.balance" data-fmt="usd">${acct.get("balance",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Equity</div><div class="val" data-v="account.equity" data-fmt="usd">${acct.get("equity",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Profit</div><div class="val {_cls(acct.get("profit",0)>=0)}" data-v="account.profit" data-fmt="usd">${acct.get("profit",0):,.2f}</div></div>')
            h.append(f'<div class="kv"><div class="label">Open Positions</div><div class="val" data-v="account.open_positions" data-fmt="int">{acct.get("open_positions",0)}</div></div>')
            h.append('</div></div>')

        # Trade Learning
        if tl.get("available"):
            h.append('<div class="card"><h2>Trade Learning</h2><div class="grid">')
            h.append(f'<div class="kv"><div class="label">Total Trades</div><div class="val">{tl.get("trades",0):,}</div></div>')
            wr = tl.get("win_rate", 0)
            h.append(f'<div class="kv"><div class="label">Win Rate</div><div class="val {_cls(wr>40)}">{wr:.1f}%</div></div>')
            exp = tl.get("expectancy", 0)
            h.append(f'<div class="kv"><div class="label">Expectancy</div><div class="val {_cls(exp>0)}">${exp:,.2f}</div></div>')
            pf = tl.get("profit_factor", 0)
            h.append(f'<div class="kv"><div class="label">Profit Factor</div><div class="val {_cls(pf>1)}">{pf:.2f}</div></div>')
            pnl = tl.get("total_pnl", 0)
            h.append(f'<div class="kv"><div class="label">Total PnL</div><div class="val {_cls(pnl>0)}">${pnl:,.2f}</div></div>')
            h.append('</div>')
            # Per-symbol performance table
            best = tl.get("best_symbols", [])
            if best:
                h.append('<h3>Per-Symbol Performance</h3>')
                h.append('<div class="table-shell"><table><tr><th>Symbol</th><th>Trades</th><th>Win Rate</th><th>PnL</th><th>Expectancy</th><th>P/F</th></tr>')
                for bs in best[:10]:
                    bpnl = bs.get("total_pnl", 0)
                    h.append(f'<tr><td style="font-weight:600"><a href="/static?p=ticker&s={_e(bs.get("symbol",""))}" style="color:#a78bfa;text-decoration:none">{_e(bs.get("symbol","?"))}</a></td>')
                    h.append(f'<td>{bs.get("trades",0)}</td>')
                    h.append(f'<td class="{_cls(bs.get("win_rate",0)>40)}">{bs.get("win_rate",0):.1f}%</td>')
                    h.append(f'<td class="{_cls(bpnl>0)}">${bpnl:,.2f}</td>')
                    h.append(f'<td>${bs.get("expectancy",0):,.2f}</td>')
                    h.append(f'<td>{bs.get("profit_factor",0):.2f}</td></tr>')
                h.append('</table></div>')
            h.append('</div>')

        # 7-Day Symbol Performance
        if sym_perf:
            h.append('<div class="card"><h2>7-Day Symbol Performance</h2>')
            h.append('<div class="table-shell"><table><tr><th>Symbol</th><th>Trades</th><th>Wins</th><th>Win Rate</th><th>PnL</th></tr>')
            for sp in sym_perf[:10]:
                spnl = sp.get("pnl", 0)
                h.append(f'<tr><td style="font-weight:600"><a href="/static?p=ticker&s={_e(sp.get("symbol",""))}" style="color:#a78bfa;text-decoration:none">{_e(sp.get("symbol","?"))}</a></td>')
                h.append(f'<td>{sp.get("trades",0)}</td><td>{sp.get("wins",0)}</td>')
                h.append(f'<td class="{_cls(sp.get("win_rate",0)>40)}">{sp.get("win_rate",0):.1f}%</td>')
                h.append(f'<td class="{_cls(spnl>0)}">${spnl:,.2f}</td></tr>')
            h.append('</table></div></div>')

        # Equity + PnL charts
        eq_chart = d.get("charts", {}).get("equity_curve", {})
        eq_vals = eq_chart.get("values", []) if isinstance(eq_chart, dict) else []
        spnl_chart = d.get("charts", {}).get("symbol_pnl", {})
        if eq_vals or spnl_chart:
            h.append('<div class="row">')
            if eq_vals:
                h.append('<div class="card"><h2>Equity Curve</h2>')
                mn, mx = min(eq_vals), max(eq_vals)
                rng = mx - mn if mx != mn else 1
                h.append('<div style="display:flex;align-items:flex-end;gap:1px;height:100px">')
                step = max(1, len(eq_vals) // 60)
                sampled = eq_vals[::step][-60:]
                for idx, v in enumerate(sampled):
                    pct = max(2, int(((v - mn) / rng) * 100))
                    color = "#34d399" if v >= sampled[0] else "#fb7185"
                    delay = round(idx * 0.015, 3)
                    h.append(f'<div class="eq-bar" style="flex:1;min-width:2px;height:{pct}%;background:{color};border-radius:2px 2px 0 0;animation-delay:{delay}s"></div>')
                h.append('</div>')
                h.append(f'<div style="display:flex;justify-content:space-between;font-size:10px;color:#8ea3c2;margin-top:4px"><span>${mn:,.2f}</span><span>${mx:,.2f}</span></div>')
                h.append('</div>')
            if isinstance(spnl_chart, dict) and spnl_chart.get("values"):
                labels = spnl_chart.get("labels", [])
                values = spnl_chart.get("values", [])
                h.append('<div class="card"><h2>Symbol PnL</h2>')
                mx_abs = max(abs(v) for v in values) if values else 1
                for i, (lbl, val) in enumerate(zip(labels, values)):
                    pct = int(abs(val) / mx_abs * 100) if mx_abs else 0
                    color = "#34d399" if val >= 0 else "#fb7185"
                    delay = round(i * 0.12, 2)
                    h.append(f'<div style="display:flex;align-items:center;gap:8px;margin-bottom:6px;animation:fadeSlideUp .5s ease {delay}s both">')
                    h.append(f'<span style="min-width:70px;font-size:11px;font-weight:600"><a href="/static?p=ticker&s={_e(lbl)}" style="color:#a78bfa;text-decoration:none">{_e(lbl)}</a></span>')
                    h.append(f'<div style="flex:1;height:14px;background:rgba(255,255,255,.04);border-radius:4px;overflow:hidden">')
                    h.append(f'<div style="height:100%;width:{pct}%;background:{color};border-radius:4px;animation:barGrow .8s ease {delay}s both"></div></div>')
                    h.append(f'<span style="min-width:80px;text-align:right;font-size:11px;color:{color}">${val:,.2f}</span>')
                    h.append('</div>')
                h.append('</div>')
            h.append('</div>')

    # ============================================================
    # PAGE: ACTIVITY
    # ============================================================
    elif page == "activity":
        # Event Intelligence
        ei = d.get("event_intel", {})
        if ei.get("enabled"):
            es = ei.get("summary", {})
            by_sym = ei.get("by_symbol", {})
            h.append('<div class="card"><h2>Event Intelligence</h2><div class="grid">')
            h.append(f'<div class="kv"><div class="label">Upcoming 24h</div><div class="val">{es.get("upcoming_24h",0)}</div></div>')
            h.append(f'<div class="kv"><div class="label">Active Window</div><div class="val">{es.get("active_window",0)}</div></div>')
            h.append(f'<div class="kv"><div class="label">High Impact 24h</div><div class="val {_cls(es.get("high_upcoming_24h",0)==0)}">{es.get("high_upcoming_24h",0)}</div></div>')
            for sym, info in by_sym.items():
                regime = info.get("regime", "normal")
                h.append(f'<div class="kv"><div class="label"><a href="/static?p=ticker&s={_e(sym)}" style="color:#8ea3c2;text-decoration:none">{_e(sym)}</a></div><div class="val {_cls(regime=="normal")}">{_e(regime)}</div></div>')
            h.append('</div></div>')

        # Live Incident Feed
        incidents = d.get("incidents", [])
        if incidents:
            h.append('<div class="card"><h2>Live Activity Feed</h2>')
            h.append('<div style="max-height:500px;overflow-y:auto">')
            sev_cls = {"critical": "bad", "warning": "warn", "info": "cyan", "activity": "good"}
            for inc_idx, inc in enumerate(incidents[:40]):
                sev = inc.get("severity", "info")
                ts_raw = inc.get("ts", "")
                ts_short = ts_raw[11:19] if len(ts_raw) > 19 else ts_raw
                sym = inc.get("symbol", "")
                summ = inc.get("summary", "")
                h.append(f'<div class="feed-row" style="display:flex;gap:8px;align-items:baseline;border-bottom:1px solid rgba(255,255,255,.04);animation:fadeSlideUp .4s ease {round(inc_idx*0.04,2)}s both">')
                h.append(f'<span style="color:#8ea3c2;min-width:56px;font-size:10px">{_e(ts_short)}</span>')
                h.append(f'<span class="badge {_e(sev)}" style="min-width:52px;text-align:center">{_e(sev)}</span>')
                if sym:
                    h.append(f'<a href="/static?p=ticker&s={_e(sym)}" style="font-weight:600;min-width:70px;color:#a78bfa;text-decoration:none">{_e(sym)}</a>')
                h.append(f'<span style="color:#edf4ff">{_e(summ[:120])}</span>')
                h.append('</div>')
            h.append('</div></div>')

    # ============================================================
    # PAGE: CONTROL PANEL
    # ============================================================
    if page == "control":
        # Read current config
        try:
            import yaml as _yaml
            with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as _cf:
                _cfg = _yaml.safe_load(_cf) or {}
        except Exception:
            _cfg = {}
        cur_tf = _cfg.get("trading", {}).get("timeframe", "M5")
        cur_period = _cfg.get("drl", {}).get("period", "90d")
        cur_ts = _cfg.get("drl", {}).get("total_timesteps", 500000)
        cur_lstm_epochs = _cfg.get("training", {}).get("lstm_epochs", 20)

        # JS for control actions
        h.append("""<script>
function ctrlAction(action, extra){
  var payload = Object.assign({action:action}, extra||{});
  var btn = event.target; btn.disabled=true; btn.style.opacity='0.5';
  var status = document.getElementById('ctrl-status');
  status.textContent = 'Sending: ' + action + '...';
  status.style.color = '#fbbf24';
  fetch('/api/control', {method:'POST', headers:{'Content-Type':'application/json'}, body:JSON.stringify(payload)})
    .then(function(r){return r.json()})
    .then(function(d){
      status.textContent = (d.ok?'OK':'FAIL') + ': ' + d.message;
      status.style.color = d.ok?'#34d399':'#fb7185';
      btn.disabled=false; btn.style.opacity='1';
    })
    .catch(function(e){
      status.textContent = 'Error: ' + e;
      status.style.color = '#fb7185';
      btn.disabled=false; btn.style.opacity='1';
    });
}
function setTimeframe(){
  var tf = document.getElementById('sel-tf').value;
  ctrlAction('set_timeframe', {timeframe: tf});
}
function setPeriod(){
  var p = document.getElementById('sel-period').value;
  ctrlAction('set_period', {period: p});
}
function setTimesteps(){
  var ts = parseInt(document.getElementById('inp-ts').value);
  ctrlAction('set_timesteps', {timesteps: ts});
}
function startCycleWithTF(){
  var tf = document.getElementById('sel-cycle-tf').value;
  ctrlAction('start_cycle_with_tf', {timeframe: tf});
}
function startDRL(){
  var ts = parseInt(document.getElementById('inp-ts').value || 150000);
  ctrlAction('start_drl', {timesteps: ts});
}
</script>""")

        # Status bar
        h.append('<div id="ctrl-status" style="background:rgba(79,214,255,.08);border:1px solid rgba(79,214,255,.2);border-radius:12px;padding:12px 18px;margin-bottom:18px;font-size:14px;color:#8ea3c2;animation:fadeSlideDown .4s ease both">Ready for commands</div>')

        # CSS for control buttons
        h.append("""<style>
.ctrl-btn{padding:12px 24px;border-radius:12px;font-size:14px;font-weight:700;border:none;cursor:pointer;
  transition:all .2s ease;text-transform:uppercase;letter-spacing:.05em}
.ctrl-btn:hover{transform:translateY(-2px);box-shadow:0 8px 24px rgba(0,0,0,.3)}
.ctrl-btn:active{transform:translateY(0)}
.ctrl-btn.green{background:linear-gradient(135deg,#1a7a4c,#34d399);color:#fff}
.ctrl-btn.green:hover{box-shadow:0 8px 24px rgba(52,211,153,.3)}
.ctrl-btn.cyan{background:linear-gradient(135deg,#1a6a8a,#4fd6ff);color:#fff}
.ctrl-btn.cyan:hover{box-shadow:0 8px 24px rgba(79,214,255,.3)}
.ctrl-btn.purple{background:linear-gradient(135deg,#5b3d99,#a78bfa);color:#fff}
.ctrl-btn.purple:hover{box-shadow:0 8px 24px rgba(167,139,250,.3)}
.ctrl-btn.red{background:linear-gradient(135deg,#8a1a2e,#fb7185);color:#fff}
.ctrl-btn.red:hover{box-shadow:0 8px 24px rgba(251,113,133,.3)}
.ctrl-btn.amber{background:linear-gradient(135deg,#8a6a1a,#fbbf24);color:#000}
.ctrl-btn.amber:hover{box-shadow:0 8px 24px rgba(251,191,36,.3)}
.ctrl-select{background:rgba(255,255,255,.06);color:#edf4ff;border:1px solid rgba(255,255,255,.15);border-radius:10px;
  padding:10px 16px;font-size:15px;font-weight:600;outline:none;cursor:pointer;min-width:120px}
.ctrl-select:focus{border-color:#4fd6ff}
.ctrl-select option{background:#0a121e;color:#edf4ff}
.ctrl-input{background:rgba(255,255,255,.06);color:#edf4ff;border:1px solid rgba(255,255,255,.15);border-radius:10px;
  padding:10px 16px;font-size:15px;font-weight:600;outline:none;width:140px}
.ctrl-input:focus{border-color:#4fd6ff}
.ctrl-group{display:flex;align-items:center;gap:12px;flex-wrap:wrap}
</style>""")

        # Current Config card
        h.append('<div class="card"><h2>Current Configuration</h2><div class="grid">')
        h.append(f'<div class="kv"><div class="label">Timeframe</div><div class="val cyan">{_e(cur_tf)}</div></div>')
        h.append(f'<div class="kv"><div class="label">Training Period</div><div class="val">{_e(cur_period)}</div></div>')
        h.append(f'<div class="kv"><div class="label">PPO Timesteps</div><div class="val">{cur_ts:,}</div></div>')
        h.append(f'<div class="kv"><div class="label">LSTM Epochs</div><div class="val">{cur_lstm_epochs}</div></div>')
        h.append(f'<div class="kv"><div class="label">Data Source</div><div class="val">{_e(_cfg.get("data",{}).get("source","mt5"))}</div></div>')
        h.append(f'<div class="kv"><div class="label">Features</div><div class="val">{_e(_cfg.get("drl",{}).get("feature_version","?"))}</div></div>')
        h.append('</div></div>')

        # Timeframe + Period + Timesteps controls
        tfs = ["M1", "M5", "M15", "M30", "H1", "H4", "D1"]
        periods = ["30d", "60d", "90d", "120d", "180d", "365d"]

        h.append('<div class="row">')

        h.append('<div class="card"><h2>Timeframe & Data</h2>')
        h.append('<div style="display:flex;flex-direction:column;gap:16px">')
        # Timeframe selector
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Timeframe</label>')
        h.append(f'<select id="sel-tf" class="ctrl-select">')
        for tf in tfs:
            sel = ' selected' if tf == cur_tf else ''
            h.append(f'<option value="{tf}"{sel}>{tf}</option>')
        h.append('</select>')
        h.append('<button class="ctrl-btn cyan" onclick="setTimeframe()">Set Timeframe</button></div>')
        # Period selector
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Period</label>')
        h.append(f'<select id="sel-period" class="ctrl-select">')
        for p in periods:
            sel = ' selected' if p == cur_period else ''
            h.append(f'<option value="{p}"{sel}>{p}</option>')
        h.append('</select>')
        h.append('<button class="ctrl-btn cyan" onclick="setPeriod()">Set Period</button></div>')
        # Timesteps
        h.append(f'<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">PPO Steps</label>')
        h.append(f'<input id="inp-ts" class="ctrl-input" type="number" value="{cur_ts}" step="10000" min="1000">')
        h.append('<button class="ctrl-btn cyan" onclick="setTimesteps()">Set Timesteps</button></div>')
        # Force fresh data
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Data Cache</label>')
        h.append('<button class="ctrl-btn amber" onclick="ctrlAction(\'force_ingest\')">Clear Cache &amp; Force Fresh Data</button></div>')
        h.append('</div></div>')

        # Training Controls
        h.append('<div class="card"><h2>Training Controls</h2>')
        h.append('<div style="display:flex;flex-direction:column;gap:16px">')
        # Quick cycle with timeframe
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Quick Cycle</label>')
        h.append(f'<select id="sel-cycle-tf" class="ctrl-select">')
        for tf in tfs:
            sel = ' selected' if tf == cur_tf else ''
            h.append(f'<option value="{tf}"{sel}>{tf}</option>')
        h.append('</select>')
        h.append('<button class="ctrl-btn green" onclick="startCycleWithTF()">Start Full Cycle</button></div>')
        # Individual training
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">LSTM</label>')
        h.append('<button class="ctrl-btn green" onclick="ctrlAction(\'start_lstm\')">Start LSTM</button>')
        h.append('<button class="ctrl-btn red" onclick="ctrlAction(\'stop_lstm\')">Stop LSTM</button></div>')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">PPO</label>')
        h.append('<button class="ctrl-btn green" onclick="startDRL()">Start PPO</button>')
        h.append('<button class="ctrl-btn red" onclick="ctrlAction(\'stop_drl\')">Stop PPO</button></div>')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Full Cycle</label>')
        h.append('<button class="ctrl-btn green" onclick="ctrlAction(\'run_cycle\')">Run Champion Cycle</button></div>')
        h.append('</div></div>')

        # HFT Controls
        h.append('<div class="card"><h2 style="color:#fbbf24">HFT Scalper (M1 High-Frequency)</h2>')
        h.append('<div style="color:#8ea3c2;font-size:12px;margin-bottom:14px">High-frequency M1 scalping mode — 5-second decision loop, aggressive reward weights, tight risk management</div>')
        h.append('<div style="display:flex;flex-direction:column;gap:12px">')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">HFT Server</label>')
        h.append('<button class="ctrl-btn green" onclick="ctrlAction(\'start_hft\')">Start HFT</button>')
        h.append('<button class="ctrl-btn red" onclick="ctrlAction(\'stop_hft\')">Stop HFT</button></div>')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">HFT Training</label>')
        h.append('<button class="ctrl-btn green" onclick="ctrlAction(\'run_hft_cycle\')">Train HFT Models (M1)</button></div>')
        h.append('</div></div>')

        h.append('</div>')  # end row

        # Model Management
        h.append('<div class="row">')
        # Per-symbol canary controls
        h.append('<div class="card"><h2>Model Promotion</h2>')
        h.append('<div style="display:flex;flex-direction:column;gap:16px">')
        for sym in all_symbols:
            sym_info = active.get("symbols", {}).get(sym, {})
            champ = sym_info.get("champion") or "none"
            canary = sym_info.get("canary") or "none"
            h.append(f'<div style="background:rgba(255,255,255,.03);border-radius:12px;padding:14px 18px;border:1px solid rgba(255,255,255,.06)">')
            h.append(f'<div style="font-weight:700;font-size:16px;color:#a78bfa;margin-bottom:8px">{_e(sym)}</div>')
            h.append(f'<div style="font-size:12px;color:#8ea3c2;margin-bottom:10px">Champion: {_e(str(champ)[:40])} | Canary: {_e(str(canary)[:40])}</div>')
            h.append(f'<div class="ctrl-group">')
            h.append(f'<button class="ctrl-btn purple" onclick="ctrlAction(\'set_canary_latest\',{{symbol:\'{_e(sym)}\'}})">Set Canary</button>')
            h.append(f'<button class="ctrl-btn green" onclick="ctrlAction(\'promote_canary\',{{symbol:\'{_e(sym)}\'}})">Promote</button>')
            h.append(f'<button class="ctrl-btn amber" onclick="ctrlAction(\'promote_canary_force\',{{symbol:\'{_e(sym)}\'}})">Force Promote</button>')
            h.append(f'<button class="ctrl-btn red" onclick="ctrlAction(\'rollback_canary\',{{symbol:\'{_e(sym)}\'}})">Rollback</button>')
            h.append('</div></div>')
        h.append('</div></div>')

        # Server controls
        h.append('<div class="card"><h2>System Controls</h2>')
        h.append('<div style="display:flex;flex-direction:column;gap:16px">')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Server AGI</label>')
        h.append('<button class="ctrl-btn green" onclick="ctrlAction(\'restart_server\')">Restart Server</button></div>')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Trade Memory</label>')
        h.append('<button class="ctrl-btn cyan" onclick="ctrlAction(\'rebuild_trade_memory\')">Rebuild Trade Memory</button></div>')
        h.append('<div class="ctrl-group"><label style="min-width:100px;color:#8ea3c2;font-size:12px;text-transform:uppercase">Processes</label>')
        h.append('<button class="ctrl-btn red" onclick="ctrlAction(\'normalize_owners\')">Kill Duplicate Processes</button></div>')
        h.append('</div></div>')

        h.append('</div>')  # end row

    h.append('</main>')

    # Footer
    h.append(f'<div id="wsStatus" style="color:#8ea3c2;font-size:11px;margin-top:16px;padding-bottom:20px">')
    h.append(f'<span class="live-dot"></span>')
    h.append(f'<span id="connLabel">Connecting...</span></div>')

    # Inline control button styles + JS (available on all pages)
    h.append("""<style>
.ibtn{display:inline-block;padding:6px 14px;border-radius:8px;font-size:11px;font-weight:700;border:none;cursor:pointer;
  transition:all .2s;text-transform:uppercase;letter-spacing:.04em;vertical-align:middle;margin-left:6px}
.ibtn:hover{transform:translateY(-1px);box-shadow:0 4px 12px rgba(0,0,0,.3)}
.ibtn.start{background:linear-gradient(135deg,#1a7a4c,#34d399);color:#fff}
.ibtn.stop{background:linear-gradient(135deg,#8a1a2e,#fb7185);color:#fff}
.ibtn.rerun{background:linear-gradient(135deg,#1a6a8a,#4fd6ff);color:#fff}
.ibtn.cycle{background:linear-gradient(135deg,#5b3d99,#a78bfa);color:#fff}
.ctrl-toast{position:fixed;top:20px;right:20px;padding:12px 22px;border-radius:12px;font-size:13px;font-weight:600;
  z-index:9999;animation:toastIn .4s ease both;max-width:400px;backdrop-filter:blur(8px)}
.ctrl-toast.ok{background:rgba(52,211,153,.2);color:#34d399;border:1px solid rgba(52,211,153,.3)}
.ctrl-toast.err{background:rgba(251,113,133,.2);color:#fb7185;border:1px solid rgba(251,113,133,.3)}
@keyframes toastIn{from{opacity:0;transform:translateY(-20px)}to{opacity:1;transform:translateY(0)}}
</style>
<script>
function ctrlBtn(action, extra){
  var btn=event.target;btn.disabled=true;btn.style.opacity='.5';
  var payload=Object.assign({action:action},extra||{});
  fetch('/api/control',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(payload)})
    .then(function(r){return r.json()})
    .then(function(d){
      var t=document.createElement('div');
      t.className='ctrl-toast '+(d.ok?'ok':'err');
      t.textContent=(d.ok?'OK':'FAIL')+': '+d.message;
      document.body.appendChild(t);
      setTimeout(function(){t.style.opacity='0';setTimeout(function(){t.remove()},400)},4000);
      btn.disabled=false;btn.style.opacity='1';
    })
    .catch(function(e){btn.disabled=false;btn.style.opacity='1';});
}
</script>""")

    # WebSocket + fetch polling — real-time JSON updates, NO page refresh
    h.append("""<script>
(function(){
  var ws, reconnDelay=1000, pollTimer=null, wsConnected=false, lastDataTs=0;
  var connLabel=document.getElementById('connLabel');
  window.__liveData=null;

  /* ── Disable meta-refresh once JS is running ── */
  var metaRef=document.getElementById('metaRefresh');
  if(metaRef) metaRef.remove();

  /* ── Generate ambient particles ── */
  (function(){
    var c=document.getElementById('bgParticles');if(!c)return;
    var colors=['#4fd6ff','#a78bfa','#34d399','#fbbf24'];
    for(var i=0;i<20;i++){
      var p=document.createElement('div');p.className='particle';
      p.style.left=Math.random()*100+'%';
      p.style.background=colors[Math.floor(Math.random()*colors.length)];
      p.style.animationDelay=Math.random()*8+'s';
      p.style.animationDuration=(6+Math.random()*6)+'s';
      p.style.width=p.style.height=(1+Math.random()*3)+'px';
      c.appendChild(p);
    }
  })();

  function setConn(text,color,cls){
    if(connLabel){connLabel.textContent=text;connLabel.className=cls||'';}
    var dot=document.querySelector('#wsStatus .live-dot');
    if(dot) dot.style.background=color||'#34d399';
  }

  /* ── Deep-get a value from nested object by dot path ── */
  function _get(obj,path){
    var parts=path.split('.');
    for(var i=0;i<parts.length;i++){
      if(!obj) return undefined;
      var k=parts[i], m=k.match(/^(.+)\[(\d+)\]$/);
      if(m){obj=obj[m[1]];if(Array.isArray(obj))obj=obj[parseInt(m[2])];} else obj=obj[k];
    }
    return obj;
  }

  /* ── Update all elements from live JSON ── */
  function applyJSON(d){
    window.__liveData=d;
    /* Update timestamp in subtitle */
    var sub=document.querySelector('.sub');
    if(sub){
      var ts=(d.timestamp_utc||'').substring(0,19);
      var html=sub.innerHTML;
      var idx=html.lastIndexOf('bull;');
      if(idx>-1) sub.innerHTML=html.substring(0,idx+5)+' '+ts;
    }

    /* Update data-v bound elements */
    var els=document.querySelectorAll('[data-v]');
    for(var i=0;i<els.length;i++){
      var el=els[i], val=_get(d,el.getAttribute('data-v'));
      if(val===undefined) continue;
      var fmt=el.getAttribute('data-fmt'), txt;
      if(fmt==='bool') txt=val?'Running':'Stopped';
      else if(fmt==='gate') txt=val?'Ready':'Hold';
      else if(fmt==='usd') txt='$'+Number(val).toLocaleString(undefined,{minimumFractionDigits:2,maximumFractionDigits:2});
      else if(fmt==='pct') txt=Math.round(val)+'%';
      else if(fmt==='int') txt=Number(val).toLocaleString();
      else txt=String(val);
      if(el.textContent!==txt){el.textContent=txt;el.classList.remove('val-flash');void el.offsetWidth;el.classList.add('val-flash');}
    }

    /* ── Live-update training progress bars by ticker label ── */
    var t=d.training||{}, rows=t.symbol_stage_rows||[], vis=t.visual||{};

    /* Build lookup: symbol -> stage -> {state, progress_pct} */
    var symStages={};
    for(var r=0;r<rows.length;r++){
      var row=rows[r], sym=row.symbol||'';
      symStages[sym]=row;
    }

    /* Build lookup: model -> symbol -> {state, progress_pct} from visual queue */
    var modelQ={};
    ['lstm','ppo','dreamer'].forEach(function(mk){
      var mv=vis[mk]||{}, q=mv.queue||[];
      modelQ[mk]={};
      for(var j=0;j<q.length;j++){
        var qi=q[j]; modelQ[mk][qi.symbol||'']=qi;
      }
    });

    /* Walk all progress bars: find .bar-label text to identify ticker */
    var allBars=document.querySelectorAll('.bar, .pbar');
    for(var b=0;b<allBars.length;b++){
      var bar=allBars[b];
      var lbl=bar.querySelector('.bar-label');
      if(!lbl) continue;
      var sym=lbl.textContent.trim();
      if(!sym || sym==='Overall') continue;

      /* Find the parent context to determine which stage/model this bar belongs to */
      var card=bar.closest('.card');
      if(!card) continue;
      var h2=card.querySelector('h2');
      if(!h2) continue;
      var title=h2.textContent||'';

      var newPct=null, newState=null;

      /* Match by card title to find the right data */
      if(title.indexOf('LSTM')>-1){
        var corr=(symStages[sym]||{}).lstm||{};
        var qi=modelQ.lstm[sym]||{};
        newPct=corr.progress_pct||qi.progress_pct||0;
        newState=corr.state||qi.status||'queued';
      } else if(title.indexOf('PPO')>-1){
        var corr=(symStages[sym]||{}).ppo||{};
        var qi=modelQ.ppo[sym]||{};
        newPct=corr.progress_pct||qi.progress_pct||0;
        newState=corr.state||qi.status||'queued';
      } else if(title.indexOf('Dreamer')>-1||title.indexOf('DreamerV3')>-1){
        var corr=(symStages[sym]||{}).dreamer||{};
        var qi=modelQ.dreamer[sym]||{};
        newPct=corr.progress_pct||qi.progress_pct||0;
        newState=corr.state||qi.status||'queued';
      } else if(title.indexOf('Pipeline')>-1||title.indexOf('9-Stage')>-1){
        /* Pipeline bars - find stage by row context */
        var td=bar.closest('td');
        if(td){
          var tr=td.closest('tr');
          var cellIdx=Array.prototype.indexOf.call(tr.children, td);
          var stages=['data_ingest','features','lstm','dreamer','ppo','candidate','backtest','champion','trading'];
          var stageKey=stages[cellIdx-1];
          if(stageKey && symStages[sym]){
            var sg=symStages[sym][stageKey]||{};
            newPct=sg.progress_pct||0;
            newState=sg.state||'queued';
          }
        }
        /* Vertical pipeline (ticker page) */
        var step=bar.closest('.stage-step');
        if(step){
          var stepName=step.querySelector('.stage-name');
          if(stepName){
            var sn=stepName.textContent.toLowerCase();
            var stageMap={'mt5':'data_ingest','feature':'features','lstm':'lstm','dreamer':'dreamer',
              'ppo':'ppo','evaluat':'candidate','gate':'backtest','canary':'champion','champion':'champion','live':'trading','trade':'trading'};
            for(var sk in stageMap){
              if(sn.indexOf(sk)>-1 && symStages[sym]){
                var sg=symStages[sym][stageMap[sk]]||{};
                newPct=sg.progress_pct||0;
                newState=sg.state||'queued';
                break;
              }
            }
          }
        }
      }

      if(newPct!==null){
        newPct=Math.round(newPct);
        var fill=bar.querySelector('.bf,.pfill');
        if(fill){
          var curW=parseInt(fill.style.width)||0;
          if(curW!==newPct){
            fill.style.transition='width .8s ease';
            fill.style.width=newPct+'%';
          }
          if(newState){
            fill.className=fill.className.replace(/\b(done|active|failed|queued|waiting|live)\b/g,'').trim()+' '+newState;
          }
        }
        var pctEl=bar.querySelector('.bar-pct');
        if(pctEl && pctEl.textContent!==newPct+'%') pctEl.textContent=newPct+'%';
      }
    }

    /* Update badge states */
    var allBadges=document.querySelectorAll('.badge');
    /* Update badges inside stage-steps for ticker page */
    var stageSteps=document.querySelectorAll('.stage-step');
    for(var s=0;s<stageSteps.length;s++){
      var step=stageSteps[s];
      var nameEl=step.querySelector('.stage-name');
      var barEl=step.querySelector('.bar');
      if(!nameEl||!barEl) continue;
      var lblEl=barEl.querySelector('.bar-label');
      if(!lblEl) continue;
      var sym=lblEl.textContent.trim();
      var sn=nameEl.textContent.toLowerCase();
      var stageMap={'mt5':'data_ingest','feature':'features','lstm':'lstm','dreamer':'dreamer',
        'ppo':'ppo','evaluat':'candidate','gate':'backtest','canary':'champion','champion':'champion','live':'trading','trade':'trading'};
      for(var sk in stageMap){
        if(sn.indexOf(sk)>-1 && symStages[sym]){
          var sg=symStages[sym][stageMap[sk]]||{};
          var badge=nameEl.querySelector('.badge');
          var numEl=step.querySelector('.stage-num');
          if(badge && sg.state && badge.textContent!==sg.state){
            badge.textContent=sg.state;
            badge.className='badge '+sg.state;
          }
          if(numEl && sg.state){
            numEl.className='stage-num '+sg.state;
          }
          break;
        }
      }
    }

    /* ── Live-update training detail text [data-td] ── */
    var tdEls=document.querySelectorAll('[data-td]');
    for(var i=0;i<tdEls.length;i++){
      var el=tdEls[i], key=el.getAttribute('data-td');
      var parts=key.split('.'); /* mk.sym */
      if(parts.length<2) continue;
      var mk=parts[0], sym=parts[1];
      var qi=(modelQ[mk]||{})[sym]||{};
      var corr=(symStages[sym]||{})[mk]||{};
      var txt='';
      if(mk==='lstm'){
        var ep=qi.epoch||'?', ept=qi.epochs_total||'?', lo=qi.loss||'?', ac=qi.acc||'?';
        txt='epoch '+ep+'/'+ept+' loss='+lo+' acc='+ac+'%';
      } else if(mk==='ppo'){
        var cs=qi.current_timesteps||0, ts=qi.target_timesteps||0;
        txt=cs.toLocaleString()+'/'+ts.toLocaleString();
      } else if(mk==='dreamer'){
        txt=qi.detail||('steps='+(qi.steps||0));
      }
      if(txt && el.textContent!==txt) el.textContent=txt;
    }

    /* ── Live-update training badges [data-tb] ── */
    var tbEls=document.querySelectorAll('[data-tb]');
    for(var i=0;i<tbEls.length;i++){
      var el=tbEls[i], key=el.getAttribute('data-tb');
      var parts=key.split('.');
      if(parts.length<2) continue;
      var mk=parts[0], sym=parts[1];
      var corr=(symStages[sym]||{})[mk]||{};
      var qi=(modelQ[mk]||{})[sym]||{};
      var st=corr.state||qi.status||'';
      if(st && el.textContent!==st){
        el.textContent=st;
        el.className='badge '+st;
      }
    }

    /* ── Live-update stage detail text [data-sd] ── */
    var sdEls=document.querySelectorAll('[data-sd]');
    for(var i=0;i<sdEls.length;i++){
      var el=sdEls[i], key=el.getAttribute('data-sd');
      var parts=key.split('.');
      if(parts.length<2) continue;
      var stageKey=parts[0], sym=parts[1];
      var sg=(symStages[sym]||{})[stageKey]||{};
      var det=sg.detail||'';
      if(det){
        el.textContent=det;
        el.style.display='';
      }
    }

    /* ── Live-update training grid stats [data-tg] ── */
    var tgEls=document.querySelectorAll('[data-tg]');
    for(var i=0;i<tgEls.length;i++){
      var el=tgEls[i], key=el.getAttribute('data-tg');
      var parts=key.split('.');
      if(parts.length<2) continue;
      var mk=parts[0];

      /* Per-symbol fields: mk.sym.field (e.g., lstm.BTCUSDm.epoch) */
      if(parts.length>=3){
        var sym=parts[1], field=parts[2];
        var qi=(modelQ[mk]||{})[sym]||{};
        var corr=(symStages[sym]||{})[mk]||{};
        var txt='';
        if(field==='pct') txt=Math.round(corr.progress_pct||qi.progress_pct||0)+'%';
        else if(field==='epoch') txt=(qi.epoch||'?')+'/'+(qi.epochs_total||'?');
        else if(field==='loss') txt=String(qi.loss||'?');
        else if(field==='acc') txt=(qi.acc||'?')+'%';
        else if(field==='timesteps') txt=(qi.current_timesteps||0).toLocaleString()+'/'+(qi.target_timesteps||0).toLocaleString();
        else if(field==='steps') txt=(qi.steps||0).toLocaleString();
        else if(field==='detail') txt=qi.detail||'';
        if(txt && el.textContent!==txt) el.textContent=txt;
        continue;
      }

      /* Global model stats: mk.stat (e.g., lstm.completion) */
      var stat=parts[1];
      var mv=vis[mk]||{}, summ=mv.summary||{}, q=mv.queue||[];
      var corr_all={};
      for(var sym in symStages){
        var sg=(symStages[sym]||{})[mk];
        if(sg) corr_all[sym]=sg;
      }
      var nDone=0, nActive=0, nFailed=0, nTotal=Object.keys(corr_all).length||summ.total_symbols||0;
      for(var sym in corr_all){
        if(corr_all[sym].state==='done') nDone++;
        else if(corr_all[sym].state==='active') nActive++;
        else if(corr_all[sym].state==='failed') nFailed++;
      }
      var compPct=nTotal>0?Math.round(nDone/nTotal*100):0;
      var txt='';
      if(stat==='completion') txt=compPct+'%';
      else if(stat==='completed') txt=nDone+'/'+nTotal;
      else if(stat==='active') txt=String(nActive);
      else if(stat==='failed') txt=String(nFailed);
      else if(stat==='epochs') txt=String(mv.epochs_total||'—');
      else if(stat==='timesteps') txt=(mv.target_timesteps||0).toLocaleString();
      else if(stat==='current_ts') txt=(mv.current_timesteps||0).toLocaleString();
      else if(stat==='steps') txt=(mv.steps||0).toLocaleString();
      if(txt && el.textContent!==txt) el.textContent=txt;
    }

    /* ── Live-update overall completion bars (.pbar with Overall label) ── */
    var pBars=document.querySelectorAll('.pbar');
    for(var b=0;b<pBars.length;b++){
      var bar=pBars[b];
      var lbl=bar.querySelector('.bar-label');
      if(!lbl || lbl.textContent.trim()!=='Overall') continue;
      var card=bar.closest('.card');
      if(!card) continue;
      var h2=card.querySelector('h2');
      if(!h2) continue;
      var title=h2.textContent||'';
      var mk=null;
      if(title.indexOf('LSTM')>-1) mk='lstm';
      else if(title.indexOf('PPO')>-1) mk='ppo';
      else if(title.indexOf('Dreamer')>-1||title.indexOf('DreamerV3')>-1) mk='dreamer';
      if(!mk) continue;
      var corr_all={};
      for(var sym in symStages){
        var sg=(symStages[sym]||{})[mk];
        if(sg) corr_all[sym]=sg;
      }
      var nDone=0, nTotal=Object.keys(corr_all).length;
      for(var sym in corr_all){ if(corr_all[sym].state==='done') nDone++; }
      var compPct=nTotal>0?Math.round(nDone/nTotal*100):0;
      var fill=bar.querySelector('.pfill');
      if(fill){
        var curW=parseInt(fill.style.width)||0;
        if(curW!==compPct){
          fill.style.transition='width .8s ease';
          fill.style.width=compPct+'%';
        }
      }
      var pctEl=bar.querySelector('.bar-pct');
      if(pctEl && pctEl.textContent!==compPct+'%') pctEl.textContent=compPct+'%';
    }
  }

  /* ── Fetch-based polling fallback ── */
  function fetchPoll(){
    fetch('/api/status').then(function(r){return r.json()}).then(function(d){
      applyJSON(d);
      lastDataTs=Date.now();
      if(!wsConnected) setConn('Polling','#fbbf24','conn-poll');
    }).catch(function(){});
  }

  function startPollFallback(){
    if(pollTimer) return;
    pollTimer=setInterval(function(){
      if(!wsConnected || (Date.now()-lastDataTs)>6000) fetchPoll();
    },3000);
  }

  function connectWS(){
    try{
      var proto=location.protocol==='https:'?'wss':'ws';
      ws=new WebSocket(proto+'://'+location.host+'/ws');
    }catch(e){
      setConn('WS unavailable — polling','#fbbf24','conn-poll');
      startPollFallback();
      return;
    }
    ws.onopen=function(){
      wsConnected=true;
      setConn('Live','#34d399','conn-live');
      reconnDelay=1000;
    };
    ws.onmessage=function(ev){
      try{
        var d=JSON.parse(ev.data);
        lastDataTs=Date.now();
        applyJSON(d);
      }catch(e){}
    };
    ws.onerror=function(){try{ws.close();}catch(e){}};
    ws.onclose=function(){
      ws=null;wsConnected=false;
      setConn('Reconnecting...','#fbbf24','conn-poll');
      reconnDelay=Math.min(reconnDelay*1.5,15000);
      setTimeout(connectWS,reconnDelay);
    };
  }

  connectWS();
  startPollFallback();
  setConn('Connecting...','#fbbf24','conn-poll');
})();
</script>""")

    h.append('</body></html>')
    return web.Response(text="".join(h), content_type="text/html")


async def api_control(request):
    data = await request.json()
    action = str(data.get("action", "")).strip()
    result = control_action(action, data)
    alerter = request.app.get("alerter")
    if alerter and result.get("ok"):
        alerter.alert(f"UI control executed: {action} | {result.get('message')}")
    return web.json_response(result)

async def api_patterns(request):
    lib = _ui_pattern_library()
    items = []
    for k, v in lib.items():
        if isinstance(v, dict):
            items.append({**v, 'pattern_name': k})
        else:
            items.append({'pattern_name': k, 'details': v})
    return web.json_response(items)

async def api_performance(request):
    from Python.perpetual_improvement import export_perpetual_improvement_state
    try:
        return web.json_response(export_perpetual_improvement_state())
    except Exception:
        return web.json_response({})

async def api_frontend(request):
    # Serve lightweight static frontend (status.html) if present
    S = os.path.join(ROOT, 'frontend', 'status.html')
    if os.path.exists(S):
        with open(S, 'r', encoding='utf-8') as f:
            html = f.read()
        return web.Response(text=html, content_type='text/html')
    return web.Response(text='<html><body>Frontend MVP not found</body></html>', content_type='text/html')


async def ws_status(request):
    ws = web.WebSocketResponse(heartbeat=20)
    await ws.prepare(request)
    try:
        while not ws.closed:
            await ws.send_str(json.dumps(read_status(refresh_if_booting=False), ensure_ascii=False))
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
            d = read_status(refresh_if_booting=False)
            if d.get("state") == "booting":
                await asyncio.sleep(1)
                continue
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


async def telegram_card_sync_loop(app):
    alerter = app.get("alerter")
    while True:
        try:
            status = read_status(refresh_if_booting=False)
            if status.get("state") != "booting":
                await asyncio.to_thread(_sync_dashboard_cards, alerter, status)
                await asyncio.to_thread(alerter.retry_pending_cards)
        except Exception:
            pass
        await asyncio.sleep(TELEGRAM_CARD_SYNC_SECONDS)


# ── Telegram Bot: command polling & mini-app attachment ──────────────

def _tg_api(token, method, payload, timeout=10):
    """Direct Telegram Bot API call (used for long-poll with custom timeout)."""
    import requests as _req
    url = f"https://api.telegram.org/bot{token}/{method}"
    body = dict(payload or {})
    try:
        resp = _req.post(url, json=body, timeout=timeout)
        if resp.ok:
            data = resp.json()
            if data.get("ok"):
                return data.get("result")
    except Exception:
        pass
    return None


def _bot_start_response():
    return (
        "<b>Cautious Giggle Trading Bot</b>\n\n"
        "Your AGI trading system is connected.\n\n"
        "<b>Commands</b>\n"
        "/status — System status\n"
        "/balance — Account balance\n"
        "/signals — Current trading signals\n"
        "/training — Training pipeline\n"
        "/help — Show this help\n\n"
        "Tap the <b>Dashboard</b> button to open the full mini app."
    )


def _bot_help_response():
    return _bot_start_response()


def _bot_status_response():
    try:
        status = read_status(refresh_if_booting=False)
        server = status.get("server", {})
        training = status.get("training", {})
        gate = status.get("canary_gate", {})
        symbols = ", ".join(training.get("configured_symbols", [])) or "none"
        return (
            f"<b>System Status</b>\n"
            f"Runtime: {'RUNNING' if server.get('running') else 'STOPPED'}\n"
            f"Training cycle: {'ACTIVE' if training.get('cycle_running') else 'IDLE'}\n"
            f"Canary gate: {'READY' if gate.get('ready') else 'HOLD'}\n"
            f"Symbols: {symbols}"
        )
    except Exception as exc:
        return f"<b>Status unavailable</b>\n{str(exc)[:200]}"


def _bot_balance_response():
    try:
        status = read_status(refresh_if_booting=False)
        account = status.get("account", {})
        bal = float(account.get("balance", 0) or 0)
        eq = float(account.get("equity", 0) or 0)
        fm = float(account.get("free_margin", 0) or 0)
        op = int(account.get("open_positions", 0) or 0)
        return (
            f"<b>Account</b>\n"
            f"Balance: ${bal:.2f}\n"
            f"Equity: ${eq:.2f}\n"
            f"Free margin: ${fm:.2f}\n"
            f"Open positions: {op}"
        )
    except Exception as exc:
        return f"<b>Balance unavailable</b>\n{str(exc)[:200]}"


def _bot_signals_response():
    try:
        status = read_status(refresh_if_booting=False)
        rows = (status.get("training", {}).get("symbol_lane_rows") or [])[:6]
        if not rows:
            return "<b>Signals</b>\nNo signal lanes active."
        lines = ["<b>Current Signals</b>"]
        for row in rows:
            sym = row.get("symbol", "?")
            decision = row.get("decision", {})
            regime = decision.get("regime", "-")
            final_t = float(decision.get("final_target", 0) or 0)
            ppo_t = float(decision.get("ppo_target", 0) or 0)
            dreamer_t = float(decision.get("dreamer_target", 0) or 0)
            lines.append(
                f"\n<b>{sym}</b>  regime={regime}  "
                f"final={final_t:.3f}  PPO={ppo_t:.3f}  Dreamer={dreamer_t:.3f}"
            )
        return "\n".join(lines)
    except Exception as exc:
        return f"<b>Signals unavailable</b>\n{str(exc)[:200]}"


def _bot_training_response():
    try:
        status = read_status(refresh_if_booting=False)
        training = status.get("training", {})
        visual = training.get("visual", {})
        lstm = visual.get("lstm", {})
        ppo = visual.get("ppo", {})
        dreamer = visual.get("dreamer", {})
        return (
            f"<b>Training Pipeline</b>\n"
            f"Cycle: {'RUNNING' if training.get('cycle_running') else 'IDLE'}\n\n"
            f"LSTM: {lstm.get('current_symbol', '-')} ({lstm.get('state', 'idle')})\n"
            f"PPO: {ppo.get('current_symbol', '-')} ({ppo.get('state', 'idle')})\n"
            f"Dreamer: {dreamer.get('current_symbol', '-')} ({dreamer.get('state', 'idle')})"
        )
    except Exception as exc:
        return f"<b>Training unavailable</b>\n{str(exc)[:200]}"


_BOT_COMMANDS = {
    "/start": _bot_start_response,
    "/help": _bot_help_response,
    "/status": _bot_status_response,
    "/balance": _bot_balance_response,
    "/signals": _bot_signals_response,
    "/training": _bot_training_response,
}


async def _telegram_bot_setup(alerter):
    """Register bot commands and attach the mini-app menu button."""
    if not alerter._configured():
        return
    token = alerter.token
    commands = [
        {"command": "start", "description": "Start the bot and show mini app"},
        {"command": "status", "description": "Get current system status"},
        {"command": "balance", "description": "Show account balance"},
        {"command": "signals", "description": "Show current trading signals"},
        {"command": "training", "description": "Show training pipeline status"},
        {"command": "help", "description": "Show available commands"},
    ]
    await asyncio.to_thread(_tg_api, token, "setMyCommands", {"commands": commands})

    mini_app_url = os.environ.get("AGI_MINI_APP_URL", "").strip()
    if mini_app_url:
        await asyncio.to_thread(
            _tg_api, token, "setChatMenuButton",
            {
                "menu_button": {
                    "type": "web_app",
                    "text": "Dashboard",
                    "web_app": {"url": mini_app_url},
                }
            },
        )


async def telegram_bot_polling_loop(app):
    """Long-poll for incoming Telegram bot commands and respond."""
    alerter = app.get("alerter")
    if not alerter or not alerter._configured():
        return
    token = alerter.token
    offset = 0
    while True:
        try:
            updates = await asyncio.to_thread(
                _tg_api, token, "getUpdates",
                {"offset": offset, "timeout": 25, "allowed_updates": ["message"]},
                timeout=35,
            )
            if not updates:
                await asyncio.sleep(1)
                continue
            for upd in updates:
                offset = max(offset, int(upd.get("update_id", 0)) + 1)
                msg = upd.get("message") or {}
                text = (msg.get("text") or "").strip()
                chat_id = (msg.get("chat") or {}).get("id")
                if not chat_id or not text:
                    continue
                cmd = text.split()[0].lower().split("@")[0]
                handler = _BOT_COMMANDS.get(cmd)
                if not handler:
                    continue
                reply = handler()
                if not reply:
                    continue
                payload = {"chat_id": chat_id, "text": reply, "parse_mode": "HTML"}
                mini_url = os.environ.get("AGI_MINI_APP_URL", "").strip()
                if cmd == "/start" and mini_url:
                    payload["reply_markup"] = json.dumps({
                        "inline_keyboard": [[{
                            "text": "Open Dashboard",
                            "web_app": {"url": mini_url},
                        }]]
                    })
                await asyncio.to_thread(_tg_api, token, "sendMessage", payload)
        except Exception:
            await asyncio.sleep(5)


async def on_startup(app):
    app["alerter"] = _build_alerter()
    app["status_task"] = asyncio.create_task(status_refresh_loop())
    app["notify_task"] = asyncio.create_task(notify_loop(app))
    app["telegram_task"] = asyncio.create_task(telegram_card_sync_loop(app))
    await _telegram_bot_setup(app["alerter"])
    app["telegram_bot_task"] = asyncio.create_task(telegram_bot_polling_loop(app))


async def status_refresh_loop():
    global STATUS_CACHE, _STATUS_REFRESH_TASK, _STATUS_REFRESH_STARTED_AT, _STATUS_REFRESH_DEGRADED
    loop = asyncio.get_running_loop()
    while True:
        if _STATUS_REFRESH_TASK is None:
            _STATUS_REFRESH_STARTED_AT = loop.time()
            _STATUS_REFRESH_DEGRADED = False
            _STATUS_REFRESH_TASK = asyncio.create_task(asyncio.to_thread(_collect_status))
        elif _STATUS_REFRESH_TASK.done():
            try:
                STATUS_CACHE = _STATUS_REFRESH_TASK.result()
            except Exception as exc:
                STATUS_CACHE = _collect_status_fast(state="degraded", error=str(exc))
            finally:
                _STATUS_REFRESH_TASK = None
                _STATUS_REFRESH_STARTED_AT = None
                _STATUS_REFRESH_DEGRADED = False
        elif _STATUS_REFRESH_STARTED_AT is not None:
            elapsed = loop.time() - _STATUS_REFRESH_STARTED_AT
            if elapsed > 45 and not _STATUS_REFRESH_DEGRADED:
                STATUS_CACHE = _collect_status_fast(state="degraded", error="status refresh timed out after 45s")
                _STATUS_REFRESH_DEGRADED = True
        await asyncio.sleep(4)


def _kill_port(port: int) -> None:
    """Kill any process occupying *port* so the server can bind cleanly."""
    try:
        out = subprocess.check_output(
            ["netstat", "-ano"], text=True, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
        for line in out.splitlines():
            if f":{port}" in line and "LISTENING" in line:
                parts = line.split()
                pid = int(parts[-1])
                if pid > 0 and pid != os.getpid():
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                        capture_output=True,
                    )
    except Exception:
        pass


def run(host="127.0.0.1", port=8088):
    _kill_port(int(port))
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/mini", mini_app)
    app.router.add_get("/api/status", api_status)
    app.router.add_get("/api/jsonp", api_jsonp)
    app.router.add_get("/static", static_status)
    app.router.add_post("/api/control", api_control)
    app.router.add_get("/ws", ws_status)
    app.on_startup.append(on_startup)
    app.router.add_get("/frontend/status", api_frontend)
    app.router.add_get("/api/patterns", api_patterns)
    app.router.add_get("/api/perf", api_performance)
    app.router.add_get("/app", react_app)
    app.router.add_get("/app/{path:.*}", react_app_static)
    web.run_app(app, host=host, port=int(port))


if __name__ == "__main__":
    run(host=os.environ.get("AGI_UI_HOST", "0.0.0.0"), port=int(os.environ.get("AGI_UI_PORT", "8088")))
