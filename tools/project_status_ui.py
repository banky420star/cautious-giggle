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
    if not os.path.exists(path):
        return {
            "available": False,
            "trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "generated_at_utc": None,
            "best_symbols": [],
            "worst_symbols": [],
        }
    try:
        with open(path, "r", encoding="utf-8") as f:
            d = json.load(f)
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
        }
    except Exception:
        return {
            "available": False,
            "trades": 0,
            "win_rate": 0.0,
            "expectancy": 0.0,
            "profit_factor": 0.0,
            "total_pnl": 0.0,
            "generated_at_utc": None,
            "best_symbols": [],
            "worst_symbols": [],
        }


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
                    "updated_utc": None,
                },
            )
            item["status"] = "failed"
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

        queue.append(
            {
                "symbol": sym,
                "status": status,
                "epoch": epoch,
                "epochs_total": total,
                "progress_pct": pct,
                "loss": item.get("loss"),
                "acc": item.get("acc"),
                "updated_utc": item.get("updated_utc"),
            }
        )

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


def _build_training_visuals(lstm_lines, ppo_lines, dreamer_lines, lstm_running: bool, drl_running: bool, dreamer_running: bool) -> dict:
    lstm = _build_lstm_visual(lstm_lines, lstm_running)
    ppo = _build_ppo_visual(ppo_lines, drl_running)
    dreamer = _build_dreamer_visual(dreamer_lines, dreamer_running)
    lstm_active = _as_int((lstm.get("summary") or {}).get("active_symbols"), 0)
    ppo_active = _as_int((ppo.get("summary") or {}).get("active_symbols"), 0)
    dreamer_active = _as_int((dreamer.get("summary") or {}).get("active_symbols"), 0)
    if lstm_running and drl_running:
        active_stage = "parallel"
        active_label = "LSTM and PPO running"
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
        if lstm_state in {"active", "partial", "done"}:
            epoch = _as_int(lstm_item.get("epoch"), 0)
            total = _as_int(lstm_item.get("epochs_total"), 0)
            if total > 0:
                lstm_detail = f"epoch {epoch}/{total}"
        if lstm_state == "failed":
            lstm_detail = "training failed"

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
    cycle_ppo_re = re.compile(r"Cycle step:\s*train PPO candidate for\s+([A-Za-z0-9_]+)", re.IGNORECASE)
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
    }


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
    cards["configured"] = configured
    cards["delivery_target"] = "both" if configured else "dashboard_only"
    return cards


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
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "repo_root": ROOT,
        "active_models": active,
        "registry": _registry_summary(active),
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
    return web.Response(text=_load_html(UI_HTML_PATH), content_type="text/html")


async def mini_app(_request):
    return web.Response(text=_load_html(MINI_UI_HTML_PATH), content_type="text/html")


async def api_status(_request):
    return web.json_response(read_status(refresh_if_booting=False))


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


async def on_startup(app):
    app["alerter"] = _build_alerter()
    app["status_task"] = asyncio.create_task(status_refresh_loop())
    app["notify_task"] = asyncio.create_task(notify_loop(app))
    app["telegram_task"] = asyncio.create_task(telegram_card_sync_loop(app))


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


def run(host="127.0.0.1", port=8088):
    app = web.Application()
    app.router.add_get("/", index)
    app.router.add_get("/mini", mini_app)
    app.router.add_get("/api/status", api_status)
    app.router.add_post("/api/control", api_control)
    app.router.add_get("/ws", ws_status)
    app.on_startup.append(on_startup)
    web.run_app(app, host=host, port=int(port))


if __name__ == "__main__":
    run(host=os.environ.get("AGI_UI_HOST", "127.0.0.1"), port=int(os.environ.get("AGI_UI_PORT", "8088")))
