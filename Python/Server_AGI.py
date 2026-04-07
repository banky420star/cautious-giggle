import atexit
import datetime
import json
import os
import signal
import subprocess
import threading
import time

try:
    import MetaTrader5 as mt5
except Exception as exc:
    _MT5_IMPORT_ERROR = exc

    class _MissingMetaTrader5:
        def __getattr__(self, name):
            raise RuntimeError(
                "MetaTrader5 is required for live runtime operations and is unavailable in this environment."
            ) from _MT5_IMPORT_ERROR

    mt5 = _MissingMetaTrader5()

import pandas as pd
from loguru import logger

from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols
from Python.perpetual_improvement import get_perpetual_improvement_system

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Auto-load .env so Server_AGI works regardless of how it's launched
_env_path = os.path.join(BASE_DIR, ".env")
if os.path.exists(_env_path):
    with open(_env_path, "r", encoding="utf-8") as _ef:
        for _line in _ef:
            _line = _line.strip()
            if _line and not _line.startswith("#") and "=" in _line:
                _k, _v = _line.split("=", 1)
                os.environ.setdefault(_k.strip(), _v.strip())

# Session config — single source of truth written by launcher
_SESSION_PATH = os.path.join(BASE_DIR, "runtime", "session.json")

def _load_session() -> dict | None:
    """Load runtime session config if it exists."""
    try:
        if os.path.exists(_SESSION_PATH):
            with open(_SESSION_PATH, "r", encoding="utf-8") as f:
                return json.load(f)
    except (json.JSONDecodeError, OSError):
        pass
    return None

_MODE_TAG = os.environ.get("AGI_MODE_TAG", "")  # e.g. "hft" for HFT mode
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, f"server_agi{'_' + _MODE_TAG if _MODE_TAG else ''}.lock")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, f"server{'_' + _MODE_TAG if _MODE_TAG else ''}.log")
AUDIT_LOG = os.path.join(LOG_DIR, f"audit_events{'_' + _MODE_TAG if _MODE_TAG else ''}.jsonl")
TRADE_EVENTS_LOG = os.path.join(LOG_DIR, f"trade_events{'_' + _MODE_TAG if _MODE_TAG else ''}.jsonl")
ACTIVE_MODELS_PATH = os.path.join(BASE_DIR, "models", "registry", "active.json")

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(SERVER_LOG, rotation="10 MB", level="INFO")

_shutdown_flag = threading.Event()

SYMBOL_EXECUTION_PROFILES = {
    "BTCUSDm": {
        "ppo_weight": 0.65,
        "dreamer_weight": 0.25,
        "agi_weight": 0.10,
        "min_trade_threshold": 0.10,
        "max_abs_target": 0.75,
        "cooldown_sec": 30,
        "rebalance_min_delta_exposure": 0.12,
        "memory_min_trades": 8,
        "memory_penalty_scale": 0.45,
        "memory_profit_factor_floor": 0.95,
        "memory_expectancy_floor": -0.10,
        "memory_max_recent_loss_streak": 3,
    },
    "XAUUSDm": {
        "ppo_weight": 0.50,
        "dreamer_weight": 0.20,
        "agi_weight": 0.30,
        "min_trade_threshold": 0.12,
        "max_abs_target": 0.45,
        "cooldown_sec": 45,
        "rebalance_min_delta_exposure": 0.10,
        "memory_min_trades": 8,
        "memory_penalty_scale": 0.35,
        "memory_profit_factor_floor": 1.00,
        "memory_expectancy_floor": 0.00,
        "memory_max_recent_loss_streak": 2,
    },
}


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_default(v):
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    return str(v)


_JSONL_MAX_BYTES = 10 * 1024 * 1024  # 10 MB per JSONL file before rotation


_JSONL_MAX_GENERATIONS = 5


def _rotate_jsonl_if_needed(path: str) -> None:
    """Rotate path -> path.1 -> ... -> path.N, keeping _JSONL_MAX_GENERATIONS backups."""
    try:
        if os.path.exists(path) and os.path.getsize(path) >= _JSONL_MAX_BYTES:
            # Shift existing backups: .4 -> .5, .3 -> .4, ... .1 -> .2
            for i in range(_JSONL_MAX_GENERATIONS, 1, -1):
                src = f"{path}.{i - 1}"
                dst = f"{path}.{i}"
                if os.path.exists(src):
                    if os.path.exists(dst):
                        os.remove(dst)
                    os.rename(src, dst)
            # Rotate current -> .1
            os.rename(path, f"{path}.1")
    except Exception:
        pass  # Never let rotation failure block a write


def _append_jsonl(path: str, row: dict):
    _rotate_jsonl_if_needed(path)
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps(row, ensure_ascii=True, default=_json_default) + "\n")


def _append_audit(event: str, payload: dict):
    _append_jsonl(
        AUDIT_LOG,
        {
            "ts": _utc_now().isoformat(timespec="microseconds"),
            "event": event,
            "payload": payload,
        },
    )


def _append_trade_event(event: str, payload: dict):
    row = {
        "ts": _utc_now().isoformat(timespec="microseconds"),
        "event": event,
        "payload": payload,
    }
    _append_jsonl(TRADE_EVENTS_LOG, row)
    _append_audit(event, payload)


_PERPETUAL_IMPROVEMENT = get_perpetual_improvement_system()

def _load_runtime_components():
    from Python.agi_brain import SmartAGI
    from Python.event_intel import EventIntel
    from Python.hybrid_brain import HybridBrain
    from Python.mt5_executor import MT5Executor
    from Python.pattern_recognition import PatternRecognitionSystem
    from Python.risk_engine import RiskEngine
    from Python.risk_supervisor import RiskSupervisor
    from Python.trade_learning import build_trade_learning
    from alerts.telegram_alerts import TelegramAlerter

    return {
        "SmartAGI": SmartAGI,
        "EventIntel": EventIntel,
        "HybridBrain": HybridBrain,
        "MT5Executor": MT5Executor,
        "PatternRecognitionSystem": PatternRecognitionSystem,
        "RiskEngine": RiskEngine,
        "RiskSupervisor": RiskSupervisor,
        "build_trade_learning": build_trade_learning,
        "TelegramAlerter": TelegramAlerter,
    }



def _read_active_models():
    if not os.path.exists(ACTIVE_MODELS_PATH):
        return {"champion": None, "canary": None}
    try:
        with open(ACTIVE_MODELS_PATH, "r", encoding="utf-8") as f:
            d = json.load(f)
        if not isinstance(d, dict):
            return {"champion": None, "canary": None}
        return {"champion": d.get("champion"), "canary": d.get("canary")}
    except Exception:
        return {"champion": None, "canary": None}


def _training_state():
    out = {
        "lstm_running": False,
        "drl_running": False,
        "cycle_running": False,
        "lstm_symbol": None,
        "lstm_epoch": None,
        "lstm_epochs_total": None,
        "lstm_score": None,
        "drl_symbol": None,
        "drl_score": None,
    }
    try:
        cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object CommandLine | ConvertTo-Json -Depth 3"
        )
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=6)
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
        lines = [str((r or {}).get("CommandLine") or "").lower().replace("\\", "/") for r in (rows or [])]
        out["lstm_running"] = any("training/train_lstm.py" in x for x in lines)
        out["drl_running"] = any("training/train_drl.py" in x for x in lines)
        out["cycle_running"] = any(("tools/champion_cycle.py" in x or "tools/champion_cycle_loop.py" in x) for x in lines)
    except Exception:
        pass

    try:
        lstm_lines = []
        lstm_log = os.path.join(LOG_DIR, "lstm_training.log")
        if os.path.exists(lstm_log):
            with open(lstm_log, "r", encoding="utf-8", errors="replace") as f:
                lstm_lines = [x.rstrip("\n") for x in f.readlines()[-40:]]
        for line in reversed(lstm_lines):
            if " | epoch " in line and " | loss " in line:
                parts = line.split("|")
                if len(parts) >= 4:
                    left = parts[1].strip()
                    ep = parts[2].strip().replace("epoch", "").strip()
                    score = parts[3].strip()
                    sym = left.split()[-1]
                    out["lstm_symbol"] = sym
                    if "/" in ep:
                        a, b = ep.split("/", 1)
                        out["lstm_epoch"] = a.strip()
                        out["lstm_epochs_total"] = b.strip()
                    if "acc" in score.lower():
                        out["lstm_score"] = score
                    break
    except Exception:
        pass

    try:
        ppo_lines = []
        ppo_log = os.path.join(LOG_DIR, "ppo_training.log")
        if os.path.exists(ppo_log):
            with open(ppo_log, "r", encoding="utf-8", errors="replace") as f:
                ppo_lines = [x.rstrip("\n") for x in f.readlines()[-80:]]
        for line in reversed(ppo_lines):
            if "DRL Training | symbols=" in line:
                idx = line.find("symbols=")
                if idx >= 0:
                    chunk = line[idx + len("symbols=") :]
                    if "[" in chunk and "]" in chunk:
                        s = chunk[chunk.find("[") + 1 : chunk.find("]")]
                        first = s.split(",")[0].strip().strip("'\"")
                        out["drl_symbol"] = first or None
            if "best_score=" in line:
                out["drl_score"] = line.split("best_score=", 1)[1].strip()
                if out["drl_symbol"] is not None:
                    break
    except Exception:
        pass

    return out


def _runtime_owner_health():
    out = {"ok": True, "issues": []}
    try:
        cmd = (
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" | "
            "Select-Object ProcessId,ParentProcessId,ExecutablePath,CommandLine | ConvertTo-Json -Depth 4"
        )
        raw = subprocess.check_output(["powershell", "-NoProfile", "-Command", cmd], text=True, timeout=6)
        rows = json.loads(raw)
        if isinstance(rows, dict):
            rows = [rows]
    except Exception:
        return out

    tokens = [
        ("server", "python.server_agi"),
        ("ui", "tools/project_status_ui.py"),
        ("cycle", "tools/champion_cycle.py"),
        ("train_lstm", "training/train_lstm.py"),
        ("train_drl", "training/train_drl.py"),
    ]
    try:
        cfg = _load_cfg(live=True)
        max_parallel_roots = max(1, len(resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)))
    except Exception:
        max_parallel_roots = max(1, len(DEFAULT_TRADING_SYMBOLS))
    parallel_roles = {"train_lstm", "train_drl"}

    for role, token in tokens:
        matches = []
        for r in rows or []:
            cmdline = str((r or {}).get("CommandLine") or "").lower().replace("\\", "/")
            if token in cmdline:
                matches.append(
                    {
                        "pid": int((r or {}).get("ProcessId") or 0),
                        "ppid": int((r or {}).get("ParentProcessId") or 0),
                        "exe": str((r or {}).get("ExecutablePath") or ""),
                    }
                )
        if not matches:
            continue

        pid_set = {m["pid"] for m in matches}
        roots = [m for m in matches if m["ppid"] not in pid_set]
        exe_paths = sorted({m["exe"].lower() for m in matches if m["exe"]})

        # Windows venv redirector chain: venv launcher roots the tree and the base
        # interpreter appears only as a child process for the same role token.
        allowed_paths = {
            "users\\administrator\\desktop\\python.exe",
            ".venv312\\scripts\\python.exe",
            ".venv\\scripts\\python.exe",
        }
        if len(roots) == 1 and exe_paths and all(any(token in p for token in allowed_paths) for p in exe_paths):
            non_root_children_ok = True
            for m in matches:
                if m["pid"] != (roots[0]["pid"] if roots else 0) and m["ppid"] not in pid_set:
                    non_root_children_ok = False
                    break
            if non_root_children_ok:
                continue

        if len(roots) > 1 and role in parallel_roles and len(roots) <= max_parallel_roots:
            continue

        if len(roots) > 1:
            out["ok"] = False
            out["issues"].append(
                {
                    "role": role,
                    "type": "multiple_root_owners",
                    "root_pids": [m["pid"] for m in roots],
                    "exe_paths": exe_paths,
                }
            )
        elif len(exe_paths) > 1:
            out["ok"] = False
            out["issues"].append(
                {
                    "role": role,
                    "type": "mixed_executables",
                    "root_pids": [m["pid"] for m in roots] or [matches[0]["pid"]],
                    "exe_paths": exe_paths,
                }
            )
    return out


def _is_lock_pid_alive(pid: int) -> bool:
    """Check if a PID is alive.  Uses OpenProcess on Windows for reliability."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _acquire_single_instance_lock():
    os.makedirs(LOCK_DIR, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        # Lock file exists -- check if the owning process is still alive.
        try:
            existing_pid = int(
                (open(LOCK_PATH, "r", encoding="utf-8").read() or "0").strip()
            )
        except Exception:
            existing_pid = 0

        if existing_pid and _is_lock_pid_alive(existing_pid):
            # Genuinely running -- refuse.
            return False

        # Stale lock from a dead process -- remove and re-acquire.
        logger.warning(
            "Removing stale lock {} (PID {} is dead)", LOCK_PATH, existing_pid
        )
        try:
            os.remove(LOCK_PATH)
        except OSError:
            return False

        try:
            fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            os.write(fd, str(os.getpid()).encode("utf-8"))
            os.close(fd)
        except FileExistsError:
            # Another process raced us -- let them win.
            return False

    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
                # Only remove if we still own it.
                with open(LOCK_PATH, "r", encoding="utf-8") as fh:
                    if fh.read().strip() == str(os.getpid()):
                        os.remove(LOCK_PATH)
        except Exception:
            pass

    atexit.register(_cleanup_lock)
    return True


def _load_cfg(live: bool = False):
    return load_project_config(BASE_DIR, live_mode=bool(live))


def _resolve_env_ref(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1])
    return v


def _load_telegram_cfg(cfg):
    tcfg = cfg.get("telegram", {}) or {}
    token = os.environ.get("TELEGRAM_TOKEN") or _resolve_env_ref(tcfg.get("token"))
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or _resolve_env_ref(tcfg.get("chat_id"))

    if token in ("", "YOUR_BOT_TOKEN_HERE"):
        token = None
    if chat_id in ("", "YOUR_CHAT_ID_HERE"):
        chat_id = None

    return token, chat_id


def _init_mt5(cfg):
    mt5_cfg = cfg.get("mt5", {})
    login = int(os.environ.get("MT5_LOGIN", _resolve_env_ref(mt5_cfg.get("login", 0))) or 0)
    password = os.environ.get("MT5_PASSWORD") or _resolve_env_ref(mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER") or _resolve_env_ref(mt5_cfg.get("server", ""))

    if login and password and server:
        return mt5.initialize(login=login, password=password, server=server)
    return mt5.initialize()


def _to_mt5_timeframe(tf: str):
    mapping = {
        "M1": mt5.TIMEFRAME_M1,
        "M5": mt5.TIMEFRAME_M5,
        "M15": mt5.TIMEFRAME_M15,
        "M30": mt5.TIMEFRAME_M30,
        "H1": mt5.TIMEFRAME_H1,
        "H4": mt5.TIMEFRAME_H4,
    }
    return mapping.get((tf or "M5").upper(), mt5.TIMEFRAME_M5)


def _clip(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, float(v)))


def _symbol_profile(symbol: str, cfg: dict | None = None) -> dict:
    base = dict(SYMBOL_EXECUTION_PROFILES.get(str(symbol), SYMBOL_EXECUTION_PROFILES["BTCUSDm"]))
    trading_cfg = (cfg or {}).get("trading", {}) if isinstance(cfg, dict) else {}
    symbol_profiles = trading_cfg.get("symbol_profiles", {}) or {}
    raw = symbol_profiles.get(str(symbol), {}) if isinstance(symbol_profiles, dict) else {}
    if isinstance(raw, dict):
        if "ppo_weight" in raw:
            base["ppo_weight"] = float(raw["ppo_weight"])
        if "dreamer_weight" in raw:
            base["dreamer_weight"] = float(raw["dreamer_weight"])
        if "agi_context_weight" in raw:
            base["agi_weight"] = float(raw["agi_context_weight"])
        if "min_actionable_exposure" in raw:
            base["min_trade_threshold"] = float(raw["min_actionable_exposure"])
        if "max_policy_exposure" in raw:
            base["max_abs_target"] = float(raw["max_policy_exposure"])
        if "min_trade_interval_sec" in raw:
            base["cooldown_sec"] = int(raw["min_trade_interval_sec"])
        if "rebalance_min_delta_exposure" in raw:
            base["rebalance_min_delta_exposure"] = float(raw["rebalance_min_delta_exposure"])
        if "memory_min_trades" in raw:
            base["memory_min_trades"] = int(raw["memory_min_trades"])
        if "memory_penalty_scale" in raw:
            base["memory_penalty_scale"] = float(raw["memory_penalty_scale"])
        if "memory_profit_factor_floor" in raw:
            base["memory_profit_factor_floor"] = float(raw["memory_profit_factor_floor"])
        if "memory_expectancy_floor" in raw:
            base["memory_expectancy_floor"] = float(raw["memory_expectancy_floor"])
        if "memory_max_recent_loss_streak" in raw:
            base["memory_max_recent_loss_streak"] = int(raw["memory_max_recent_loss_streak"])
    return base


def _blend_symbol_decision(
    symbol: str,
    agi_meta: dict | None,
    ppo_meta: dict | None,
    dreamer_meta: dict | None,
    trade_memory: dict | None = None,
    cfg: dict | None = None,
) -> dict:
    profile = _symbol_profile(symbol, cfg=cfg)

    ppo_target = float((ppo_meta or {}).get("target", 0.0) or 0.0)
    dreamer_target = float((dreamer_meta or {}).get("target", 0.0) or 0.0)

    agi_conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
    agi_risk = float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0)
    agi_bias = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)

    # Redistribute weights from absent models to AGI so the 150-feature
    # LSTM signal can drive real decisions before a PPO champion is promoted.
    # When both PPO and Dreamer are present, weights are used as configured.
    ppo_w = float(profile["ppo_weight"]) if ppo_meta is not None else 0.0
    dreamer_w = float(profile["dreamer_weight"]) if dreamer_meta is not None else 0.0
    agi_w = max(float(profile["agi_weight"]), 1.0 - ppo_w - dreamer_w)

    raw = ppo_w * ppo_target + dreamer_w * dreamer_target + agi_w * agi_bias

    adjusted = raw * agi_risk
    adjusted = _clip(adjusted, -float(profile["max_abs_target"]), float(profile["max_abs_target"]))

    memory = trade_memory or {}
    memory_scale = 1.0
    memory_reason = "neutral"
    memory_min_trades = int(profile.get("memory_min_trades", 0) or 0)
    memory_penalty_scale = _clip(float(profile.get("memory_penalty_scale", 1.0) or 1.0), 0.0, 1.0)
    if memory_min_trades > 0 and int(memory.get("trades", 0) or 0) >= memory_min_trades:
        profit_factor = float(memory.get("profit_factor", 1.0) or 1.0)
        expectancy = float(memory.get("expectancy", 0.0) or 0.0)
        recent_loss_streak = int(memory.get("recent_loss_streak", 0) or 0)
        profit_factor_floor = float(profile.get("memory_profit_factor_floor", 0.0) or 0.0)
        expectancy_floor = float(profile.get("memory_expectancy_floor", -999.0) or -999.0)
        max_recent_loss_streak = int(profile.get("memory_max_recent_loss_streak", 999) or 999)
        if (
            profit_factor < profit_factor_floor
            or expectancy < expectancy_floor
            or recent_loss_streak > max_recent_loss_streak
        ):
            memory_scale = memory_penalty_scale
            memory_reason = f"pf={profit_factor:.2f} exp={expectancy:.2f} streak={recent_loss_streak}"

    adjusted *= memory_scale

    if abs(adjusted) < float(profile["min_trade_threshold"]):
        adjusted = 0.0

    return {
        "target": adjusted,
        "raw_target": raw,
        "agi_confidence": agi_conf,
        "agi_risk_scalar": agi_risk,
        "ppo_target": ppo_target,
        "dreamer_target": dreamer_target,
        "agi_bias": agi_bias,
        "agi_direction": str((agi_meta or {}).get("direction", (agi_meta or {}).get("signal", "HOLD")) or "HOLD"),
        "agi_feature_version": str((agi_meta or {}).get("feature_version", "unknown") or "unknown"),
        "ppo_weight_used": round(ppo_w, 3),
        "dreamer_weight_used": round(dreamer_w, 3),
        "agi_weight_used": round(agi_w, 3),
        "memory_scale": round(memory_scale, 3),
        "memory_reason": memory_reason,
        "profile": profile,
    }


def _low_volatility_memory_base(trade_memory: dict | None) -> float:
    memory = trade_memory or {}
    min_trades = int(os.environ.get("AGI_LOW_VOL_MIN_TRADES", "20") or 20)
    min_profit_factor = float(os.environ.get("AGI_LOW_VOL_MIN_PROFIT_FACTOR", "1.15") or 1.15)
    min_expectancy = float(os.environ.get("AGI_LOW_VOL_MIN_EXPECTANCY", "0.0") or 0.0)
    max_recent_loss_streak = int(os.environ.get("AGI_LOW_VOL_MAX_RECENT_LOSS_STREAK", "3") or 3)

    trades = int(memory.get("trades", 0) or 0)
    profit_factor = float(memory.get("profit_factor", 0.0) or 0.0)
    expectancy = float(memory.get("expectancy", 0.0) or 0.0)
    recent_loss_streak = int(memory.get("recent_loss_streak", 0) or 0)

    if trades < min_trades:
        return 0.0
    if profit_factor < min_profit_factor:
        return 0.0
    if expectancy < min_expectancy:
        return 0.0
    if recent_loss_streak > max_recent_loss_streak:
        return 0.0
    return 1.0


def _fetch_symbol_df(symbol: str, timeframe, bars=220):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None or len(rates) < 80:
        return None

    df = pd.DataFrame(rates)
    if df.empty:
        return None

    df["time"] = pd.to_datetime(df["time"], unit="s")
    df = df.rename(columns={"tick_volume": "volume"})
    keep = ["time", "open", "high", "low", "close", "volume"]
    for k in keep:
        if k not in df.columns:
            return None

    out = df[keep].copy()
    out["symbol"] = symbol
    return out


def _account_snapshot():
    info = mt5.account_info()
    positions = mt5.positions_get() or []
    floating = sum(float(getattr(p, "profit", 0.0)) for p in positions)

    pnl_today = 0.0
    try:
        now_utc = _utc_now()
        day_start = now_utc.replace(hour=0, minute=0, second=0, microsecond=0)
        deals = mt5.history_deals_get(day_start, now_utc)
        for d in deals or []:
            if int(getattr(d, "entry", -1)) == int(mt5.DEAL_ENTRY_OUT):
                pnl_today += float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))
    except Exception:
        pass

    return {
        "balance": None if info is None else float(info.balance),
        "equity": None if info is None else float(info.equity),
        "free_margin": None if info is None else float(info.margin_free),
        "pnl_today": float(pnl_today),
        "floating": float(floating),
        "open_positions": len(positions),
    }


def _expected_usd(symbol: str, side: str, entry: float, tp: float, sl: float, lots: float):
    try:
        info = mt5.symbol_info(symbol)
        tick_size = float(getattr(info, "trade_tick_size", 0.0) or 0.0)
        tick_value = float(getattr(info, "trade_tick_value", 0.0) or 0.0)
        if tick_size <= 0 or tick_value <= 0:
            return None, None
        usd_per_price = tick_value / tick_size
        if str(side).upper() == "BUY":
            tp_outcome = (float(tp) - float(entry)) * usd_per_price * float(lots)
            sl_outcome = (float(sl) - float(entry)) * usd_per_price * float(lots)
        else:
            tp_outcome = (float(entry) - float(tp)) * usd_per_price * float(lots)
            sl_outcome = (float(entry) - float(sl)) * usd_per_price * float(lots)
        return tp_outcome, sl_outcome
    except Exception:
        return None, None


def _tick_spread_bps(symbol: str) -> float | None:
    try:
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        bid = float(getattr(tick, "bid", 0.0) or 0.0)
        ask = float(getattr(tick, "ask", 0.0) or 0.0)
        mid = (bid + ask) * 0.5
        if bid <= 0.0 or ask <= 0.0 or mid <= 0.0:
            return None
        return float(((ask - bid) / mid) * 10000.0)
    except Exception:
        return None


def _position_exposure_state(symbol: str, max_lots: float) -> tuple[float, float, int, int]:
    positions = mt5.positions_get() or []
    symbol_positions = [p for p in positions if str(getattr(p, "symbol", "")) == str(symbol)]
    current_symbol_exposure = 0.0
    total_exposure = 0.0
    for pos in positions:
        volume = float(getattr(pos, "volume", 0.0) or 0.0)
        side = -1.0 if int(getattr(pos, "type", 0)) == int(mt5.ORDER_TYPE_SELL) else 1.0
        exp = side * (volume / max(max_lots, 1e-8))
        total_exposure += abs(exp)
        if str(getattr(pos, "symbol", "")) == str(symbol):
            current_symbol_exposure += exp
    return float(current_symbol_exposure), float(total_exposure), len(symbol_positions), len(positions)


def _scan_trade_events(alerter, risk, known_open_tickets, seen_closed_deals, last_deal_check):
    now_utc = _utc_now()
    closed_events = []

    positions = mt5.positions_get() or []
    current_open = {int(p.ticket): p for p in positions}
    current_tickets = set(current_open.keys())

    new_tickets = sorted(current_tickets - known_open_tickets)
    for ticket in new_tickets:
        p = current_open[ticket]
        side = "BUY" if int(getattr(p, "type", 0)) == int(mt5.ORDER_TYPE_BUY) else "SELL"
        payload = {
            "ticket": ticket,
            "symbol": str(getattr(p, "symbol", "?")),
            "side": side,
            "volume": float(getattr(p, "volume", 0.0)),
            "open_price": float(getattr(p, "price_open", 0.0)),
            "sl": float(getattr(p, "sl", 0.0) or 0.0),
            "tp": float(getattr(p, "tp", 0.0) or 0.0),
        }
        _append_trade_event("trade_open", payload)

        snap = _account_snapshot()
        alerter.trade(
            symbol=payload["symbol"],
            action=side,
            exposure=payload["volume"],
            confidence=1.0,
            balance=0.0 if snap["balance"] is None else snap["balance"],
            equity=0.0 if snap["equity"] is None else snap["equity"],
            free_margin=0.0 if snap["free_margin"] is None else snap["free_margin"],
        )

    removed_tickets = sorted(known_open_tickets - current_tickets)
    for ticket in removed_tickets:
        _append_trade_event("position_removed", {"ticket": ticket})

    try:
        deals = mt5.history_deals_get(last_deal_check, now_utc) or []
    except Exception:
        deals = []

    for d in deals:
        try:
            if int(getattr(d, "entry", -1)) != int(mt5.DEAL_ENTRY_OUT):
                continue
            deal_id = int(getattr(d, "deal", 0))
            if deal_id <= 0 or deal_id in seen_closed_deals:
                continue
            seen_closed_deals.add(deal_id)

            pnl = float(getattr(d, "profit", 0.0) + getattr(d, "commission", 0.0) + getattr(d, "swap", 0.0))
            payload = {
                "deal_id": deal_id,
                "ticket": int(getattr(d, "position_id", 0) or 0),
                "symbol": str(getattr(d, "symbol", "?")),
                "volume": float(getattr(d, "volume", 0.0)),
                "price": float(getattr(d, "price", 0.0)),
                "profit": pnl,
                "comment": str(getattr(d, "comment", "")),
            }
            _append_trade_event("trade_closed", payload)
            closed_events.append(payload)
            try:
                risk.record_trade_result(payload["symbol"], pnl)
            except Exception as exc:
                logger.error(f"RISK_UPDATE_FAILED symbol={payload['symbol']} pnl={pnl} error={exc}")
            alerter.trade_closed(
                symbol=payload["symbol"],
                ticket=payload["ticket"],
                pnl=pnl,
                volume=payload["volume"],
                price=payload["price"],
                reason=payload.get("comment"),
                deal_id=deal_id,
            )
        except Exception:
            continue

    if len(seen_closed_deals) > 20000:
        seen_closed_deals = set(sorted(seen_closed_deals)[-10000:])

    return current_tickets, seen_closed_deals, now_utc - datetime.timedelta(seconds=3), closed_events


def main(live=False):
    if not _acquire_single_instance_lock():
        raise RuntimeError("Server_AGI is already running (lock file exists)")

    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    cfg = _load_cfg(live=live)

    # ── Session config (written by launcher) ──────────────────────────────
    session = _load_session()
    cold_start = False
    if session:
        logger.info(
            "SESSION mode=%s equity=%.2f max_lots=%.2f cold_start=%s",
            session.get("mode", "?"),
            session.get("start_equity", 0), session.get("max_lots", 0),
            session.get("cold_start", False),
        )
        cold_start = bool(session.get("cold_start", False))
        # Override YAML config with session values
        risk_cfg = cfg.setdefault("risk", {})
        risk_cfg["max_lots"] = session.get("max_lots", risk_cfg.get("max_lots", 1.0))
        sup_cfg = risk_cfg.setdefault("supervisor", {})
        sup_cfg["max_drawdown_pct"] = session.get("max_dd_pct", sup_cfg.get("max_drawdown_pct", 12.0))
    else:
        logger.warning("No session config at %s — using YAML defaults", _SESSION_PATH)

    ok = _init_mt5(cfg)
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    runtime = _load_runtime_components()
    risk = runtime["RiskEngine"](cfg=cfg)
    restored = runtime["RiskEngine"].load_state(cfg=cfg)
    if restored is not None:
        # Preserve config-driven limits from fresh instance, restore runtime counters
        restored.max_daily_loss = risk.max_daily_loss
        restored.max_daily_trades = risk.max_daily_trades
        restored.max_daily_trades_per_symbol = risk.max_daily_trades_per_symbol
        restored.max_daily_losing_trades_per_symbol = risk.max_daily_losing_trades_per_symbol
        restored.max_lots = risk.max_lots
        restored.default_symbol_profile = risk.default_symbol_profile
        restored.symbol_profiles = risk.symbol_profiles
        risk = restored
        logger.warning("Risk engine state restored from disk")

    # Apply session equity as peak if available and risk engine has no valid peak
    if session and session.get("start_equity"):
        if not risk.peak_equity or risk.peak_equity <= 0:
            risk.peak_equity = session["start_equity"]
            risk.current_dd = 0.0
            logger.info("RISK_BASELINE set peak_equity=%.2f from session config", risk.peak_equity)

    supervisor = runtime["RiskSupervisor"](cfg)
    executor = runtime["MT5Executor"](risk)
    brain = runtime["HybridBrain"](risk, executor)
    agi = runtime["SmartAGI"]()
    pattern_system = runtime["PatternRecognitionSystem"](log_dir=LOG_DIR)

    trading_cfg = cfg.get("trading", {})
    symbols = resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)
    timeframe = _to_mt5_timeframe(trading_cfg.get("timeframe", "M5"))
    max_lots = float(cfg.get("risk", {}).get("max_lots", 1.0))

    token, chat_id = _load_telegram_cfg(cfg)
    alerter = runtime["TelegramAlerter"](token, chat_id)
    event_intel = runtime["EventIntel"](cfg, LOG_DIR)

    alerter.online("Trading engine initialized")

    def _notify_offline():
        try:
            snap = _account_snapshot()
            alerter.offline(
                f"Balance={0.0 if snap['balance'] is None else snap['balance']:.2f} | "
                f"Equity={0.0 if snap['equity'] is None else snap['equity']:.2f} | "
                f"Open={int(snap['open_positions'])}"
            )
        except Exception:
            alerter.offline("Runtime exited")

    atexit.register(_notify_offline)

    def _handle_shutdown_signal(signum, frame):
        logger.warning(f"Received signal {signum}, initiating graceful shutdown")
        _shutdown_flag.set()

    signal.signal(signal.SIGTERM, _handle_shutdown_signal)
    signal.signal(signal.SIGINT, _handle_shutdown_signal)

    known_open_tickets = set()
    seen_closed_deals = set()
    last_deal_check = _utc_now() - datetime.timedelta(minutes=30)

    start_time = time.time()
    heartbeat_sec = int(os.environ.get("AGI_HEARTBEAT_SEC", "600"))
    symbol_card_sec = int(os.environ.get("AGI_SYMBOL_CARD_SEC", "90"))
    learning_sec = int(os.environ.get("AGI_TRADE_LEARN_SEC", "600"))
    loop_sleep_sec = int(os.environ.get("AGI_LOOP_SEC", "20"))
    last_heartbeat = 0.0
    last_symbol_cards = 0.0
    last_learning = 0.0
    last_models = {"champion": None, "canary": None}
    last_owner_issue_key = None
    last_owner_issue_time = 0.0
    last_daily_profit_date = None
    last_symbol_state = {str(s): {} for s in symbols}
    last_closed_by_symbol = {}
    trade_learning_by_symbol = {}
    cold_start_cycles = 0
    COLD_START_WARMUP = 3  # skip first N cycles on cold start

    while not _shutdown_flag.is_set():
        now = time.time()

        if now - last_heartbeat >= max(15, heartbeat_sec):
            uptime = int(now - start_time)
            acc = mt5.account_info()
            if acc:
                risk.update_equity(float(acc.equity))

            snap = _account_snapshot()
            tr_state = _training_state()
            models = _read_active_models()
            event_state = {}
            try:
                p = os.path.join(LOG_DIR, "event_intel_state.json")
                if os.path.exists(p):
                    with open(p, "r", encoding="utf-8") as f:
                        event_state = json.load(f)
            except Exception:
                event_state = {}
            alerter.heartbeat_full(
                uptime=str(uptime) + " sec",
                mt5_connected=(mt5.terminal_info() is not None),
                trading_enabled=not risk.halt,
                snapshot=snap,
                training=tr_state,
                models=models,
                event_intel=event_state,
            )
            alerter.snapshot(
                balance=0.0 if snap["balance"] is None else snap["balance"],
                equity=0.0 if snap["equity"] is None else snap["equity"],
                pnl_today=snap["pnl_today"],
                floating=snap["floating"],
                open_positions=snap["open_positions"],
            )
            _append_audit("snapshot", snap)
            if models.get("champion") != last_models.get("champion"):
                alerter.model(f"Champion changed: {models.get('champion') or 'none'}")
            if models.get("canary") != last_models.get("canary"):
                alerter.model(f"Canary changed: {models.get('canary') or 'none'}")
            last_models = models
            owner_health = _runtime_owner_health()
            _append_audit("runtime_owner_health", owner_health)
            if not owner_health.get("ok", True):
                issue_key = json.dumps(owner_health.get("issues", []), sort_keys=True, ensure_ascii=True)
                if issue_key != last_owner_issue_key or (now - last_owner_issue_time) > 1800:
                    lines = []
                    for it in owner_health.get("issues", []):
                        lines.append(
                            f"{it.get('role')}: {it.get('type')} | roots={it.get('root_pids')} | exe={it.get('exe_paths')}"
                        )
                    alerter.alert("RUNTIME OWNERSHIP WARNING\n" + "\n".join(lines))
                    last_owner_issue_key = issue_key
                    last_owner_issue_time = now
            last_heartbeat = now

        # Observe-only event intelligence (calendar/news/websocket).
        try:
            event_state = event_intel.tick(symbols)
            _append_audit("event_intel", event_state.get("summary", {}))
            for msg in event_intel.pop_alerts():
                alerter.alert(msg)
        except Exception as e:
            logger.warning(f"event_intel tick failed: {e}")

        if now - last_learning >= max(120, learning_sec):
            try:
                learn = runtime["build_trade_learning"](
                    log_dir=LOG_DIR,
                    out_dir=os.path.join(LOG_DIR, "learning"),
                    lookback_days=int(os.environ.get("AGI_TRADE_LEARN_DAYS", "30")),
                )
                _append_audit(
                    "trade_learning",
                    {
                        "trades": int(learn.get("trades", 0)),
                        "win_rate": float(learn.get("win_rate", 0.0)),
                        "expectancy": float(learn.get("expectancy", 0.0)),
                        "profit_factor": float(learn.get("profit_factor", 0.0)),
                        "total_pnl": float(learn.get("total_pnl", 0.0)),
                    },
                )
                try:
                    day = _utc_now().date().isoformat()
                    daily_hour = int(os.environ.get("AGI_DAILY_PROFIT_HOUR_UTC", "0"))
                    if _utc_now().hour >= max(0, min(23, daily_hour)) and day != last_daily_profit_date:
                        alerter.profitability_daily(learn)
                        last_daily_profit_date = day
                except Exception:
                    pass
                trade_learning_by_symbol = {
                    str((row or {}).get("symbol", "")): dict(row or {})
                    for row in (learn.get("by_symbol", []) if isinstance(learn, dict) else [])
                    if isinstance(row, dict) and row.get("symbol")
                }
                try:
                    for row in learn.get("by_symbol", []) if isinstance(learn, dict) else []:
                        if not isinstance(row, dict):
                            continue
                        symbol_name = str((row or {}).get("symbol") or "").strip()
                        if not symbol_name:
                            continue
                        score = float(row.get("profit_factor", 1.0) or 1.0)
                        _PERPETUAL_IMPROVEMENT.record_model_performance("trade_learning", symbol_name, score)
                except Exception:
                    pass
            except Exception as e:
                logger.warning(f"trade learning update failed: {e}")
            last_learning = now

        # Cold start: skip trading for first N cycles to let signals stabilise
        if cold_start and cold_start_cycles < COLD_START_WARMUP:
            cold_start_cycles += 1
            logger.info("COLD_START cycle %d/%d — warming up, no trades", cold_start_cycles, COLD_START_WARMUP)
            if cold_start_cycles >= COLD_START_WARMUP:
                cold_start = False
                # Clear cold_start in session file
                try:
                    s = _load_session()
                    if s and s.get("cold_start"):
                        s["cold_start"] = False
                        with open(_SESSION_PATH, "w", encoding="utf-8") as f:
                            json.dump(s, f, indent=4)
                except Exception:
                    pass
                logger.info("COLD_START complete — trading enabled")
            _shutdown_flag.wait(loop_sleep_sec)
            continue

        for symbol in symbols:
            try:
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue

                agi_meta = agi.predict(df, production=True)
                conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
                try:
                    pattern_system.detect_and_log(symbol, df)
                except Exception:
                    pass
                trade_memory = trade_learning_by_symbol.get(str(symbol), {})

                ppo_meta = brain.predict_ppo_action(symbol, df)
                dreamer_meta = brain.predict_dreamer_action(symbol, df)
                decision = _blend_symbol_decision(symbol, agi_meta, ppo_meta, dreamer_meta, trade_memory=trade_memory, cfg=cfg)
                exposure = float(decision["target"])

                # Derive regime from blended signal, not LSTM alone.
                # The LSTM can inform but shouldn't veto the whole pipeline.
                if abs(exposure) < 0.05:
                    regime = "HOLD"
                elif exposure > 0:
                    regime = "BUY"
                else:
                    regime = "SELL"

                # PPO diagnostics — always present even when PPO returns None
                ppo_diag = {}
                if ppo_meta and ppo_meta.get("ppo_diag"):
                    ppo_diag = ppo_meta["ppo_diag"]
                elif ppo_meta is None:
                    ppo_diag = getattr(brain, "_last_ppo_diag_by_symbol", {}).get(str(symbol), {})
                ppo_skip = str(ppo_diag.get("ppo_skip_reason") or "")

                logger.info(
                    "DECISION %s | regime=%s conf=%.4f risk=%.4f agi_bias=%.4f ppo=%.4f dreamer=%.4f raw=%.4f final=%.4f ppo_skip=%s"
                    % (
                        symbol,
                        regime,
                        float((agi_meta or {}).get("confidence", 0.0) or 0.0),
                        float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                        float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        float(decision["raw_target"]),
                        float(decision["target"]),
                        ppo_skip or "none",
                    )
                )
                _append_audit(
                    "signal",
                    {
                        "symbol": symbol,
                        "regime": regime,
                        "confidence": conf,
                        "risk_scalar": float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                        "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        "ppo_exposure": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        "dreamer_exposure": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        "raw_target": float(decision["raw_target"]),
                        "exposure": float(exposure),
                        "decision_profile": dict(decision["profile"]),
                        "trade_memory": {
                            "trades": int(trade_memory.get("trades", 0) or 0),
                            "expectancy": float(trade_memory.get("expectancy", 0.0) or 0.0),
                            "profit_factor": float(trade_memory.get("profit_factor", 0.0) or 0.0),
                            "recent_loss_streak": int(trade_memory.get("recent_loss_streak", 0) or 0),
                        },
                        "ppo_diag": ppo_diag if ppo_diag else None,
                    },
                )
                sym_state = last_symbol_state.setdefault(str(symbol), {})
                sym_state["signal"] = regime
                sym_state["regime"] = regime
                sym_state["confidence"] = conf
                sym_state["risk_scalar"] = float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0)
                sym_state["trend_bias"] = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)
                sym_state["ppo_exposure"] = float((ppo_meta or {}).get("target", 0.0) or 0.0)
                sym_state["dreamer_exposure"] = float((dreamer_meta or {}).get("target", 0.0) or 0.0)
                sym_state["raw_target"] = float(decision["raw_target"])
                sym_state["blend_exposure"] = float(exposure)

                # ── Phase A3: Signal collapse exit ──────────────────────
                trade_profile = _symbol_profile(symbol, cfg=cfg)
                if trade_profile.get("signal_collapse_exit", False):
                    _longs, _shorts = executor.get_positions(symbol)
                    pos_dir = 0
                    if _longs and not _shorts:
                        pos_dir = 1
                    elif _shorts and not _longs:
                        pos_dir = -1

                    if pos_dir != 0:
                        sig_dir = 1 if exposure > 0.05 else (-1 if exposure < -0.05 else 0)
                        collapse_cycles = int(trade_profile.get("signal_collapse_cycles", 2))

                        if sig_dir != 0 and sig_dir != pos_dir:
                            sym_state["signal_flip_count"] = sym_state.get("signal_flip_count", 0) + 1
                        else:
                            sym_state["signal_flip_count"] = 0

                        if sym_state.get("signal_flip_count", 0) >= collapse_cycles:
                            logger.info("SIGNAL_COLLAPSE %s | flip_count=%d >= %d, forcing flat",
                                        symbol, sym_state["signal_flip_count"], collapse_cycles)
                            exposure = 0.0
                            regime = "HOLD"
                            sym_state["signal_flip_count"] = 0
                    else:
                        sym_state["signal_flip_count"] = 0

                # ── Phase A4: Regime flip exit (losing positions only) ──
                if trade_profile.get("regime_flip_exit_losing", False):
                    prev_regime = sym_state.get("last_regime", "HOLD")
                    if prev_regime != regime and prev_regime in ("BUY", "SELL"):
                        _positions = mt5.positions_get(symbol=symbol) or []
                        total_floating = sum(float(getattr(p, "profit", 0.0)) for p in _positions)
                        if total_floating < 0 and len(_positions) > 0:
                            logger.info("REGIME_FLIP_EXIT %s | prev=%s new=%s floating=%.2f, forcing flat",
                                        symbol, prev_regime, regime, total_floating)
                            exposure = 0.0
                            regime = "HOLD"

                sym_state["last_regime"] = regime

                # ── Phase C: Pyramid winners ────────────────────────────
                if trade_profile.get("pyramid_enabled", False):
                    _longs, _shorts = executor.get_positions(symbol)
                    _positions_list = _longs if _longs else (_shorts if _shorts else [])

                    if _positions_list:
                        _pos_dir = 1 if _longs else -1
                        _signal_same_side = (exposure > 0.05 and _pos_dir > 0) or (exposure < -0.05 and _pos_dir < 0)

                        if _signal_same_side:
                            _info = mt5.symbol_info(symbol)
                            _tick = mt5.symbol_info_tick(symbol)
                            if _info and _tick:
                                _point = float(_info.point) if _info.point else 0.0001
                                _p0 = _positions_list[0]
                                _cur_price = _tick.bid if _p0.type == 0 else _tick.ask  # 0=BUY
                                _profit_pts = ((_cur_price - _p0.price_open) / _point
                                               if _p0.type == 0
                                               else (_p0.price_open - _cur_price) / _point)

                                _min_profit = int(trade_profile.get("pyramid_min_profit_points", 50))
                                _layers = int(sym_state.get("pyramid_layers", 0))
                                _max_layers = int(trade_profile.get("pyramid_max_layers", 2))

                                # Signal strength must have increased
                                _prev_signal = abs(float(sym_state.get("last_blended_exposure", 0.0)))
                                _curr_signal = abs(exposure)
                                _sig_increase = float(trade_profile.get("pyramid_signal_increase", 0.10))

                                # ATR spike check
                                _current_atr = executor._atr_points(symbol)
                                _prev_atr = sym_state.get("last_atr_points")
                                _atr_limit = float(trade_profile.get("pyramid_atr_spike_limit", 1.5))
                                _atr_ok = (_prev_atr is None or _current_atr is None or
                                           _current_atr <= _prev_atr * _atr_limit)

                                # Time between adds
                                import time as _time
                                _min_time = int(trade_profile.get("pyramid_min_time_between_sec", 120))
                                _last_add_ts = float(sym_state.get("pyramid_last_add_ts", 0))
                                _time_ok = (_time.time() - _last_add_ts) >= _min_time

                                if (_profit_pts >= _min_profit and
                                    _layers < _max_layers and
                                    _curr_signal >= _prev_signal + _sig_increase and
                                    _atr_ok and _time_ok):

                                    _decay = float(trade_profile.get("pyramid_volume_decay", 0.5))
                                    _last_vol = float(sym_state.get("pyramid_last_volume", _p0.volume))
                                    _add_vol = round(_last_vol * _decay, 2)
                                    if _add_vol >= 0.01:
                                        _add_exp = _add_vol / max(float(max_lots), 1e-8)
                                        if _pos_dir < 0:
                                            _add_exp = -_add_exp
                                        _current_lots = sum(p.volume for p in _longs) - sum(p.volume for p in _shorts)
                                        _current_exp = _current_lots / max(float(max_lots), 1e-8)
                                        _pyramid_target = _current_exp + _add_exp
                                        _max_abs = float(trade_profile.get("max_abs_target", 0.75))
                                        _pyramid_target = max(-_max_abs, min(_max_abs, _pyramid_target))

                                        logger.info(
                                            "PYRAMID %s | layer=%d profit_pts=%.0f signal=%.4f>%.4f+%.2f atr_ok=%s vol=%.2f",
                                            symbol, _layers + 1, _profit_pts, _curr_signal, _prev_signal,
                                            _sig_increase, _atr_ok, _add_vol
                                        )
                                        exposure = _pyramid_target
                                        sym_state["pyramid_layers"] = _layers + 1
                                        sym_state["pyramid_last_volume"] = _add_vol
                                        sym_state["pyramid_last_add_ts"] = _time.time()

                                if _current_atr is not None:
                                    sym_state["last_atr_points"] = _current_atr
                    else:
                        sym_state["pyramid_layers"] = 0
                        sym_state.pop("pyramid_last_volume", None)
                        sym_state.pop("pyramid_last_add_ts", None)

                sym_state["last_blended_exposure"] = exposure

                action_meta = ppo_meta or brain.get_last_action_meta(symbol=symbol)
                current_symbol_exposure, total_exposure, symbol_positions, total_positions = _position_exposure_state(
                    symbol, max_lots
                )
                supervisor_decision = supervisor.allow_trade(
                    symbol=symbol,
                    target_exposure=float(exposure),
                    confidence=conf,
                    spread_bps=_tick_spread_bps(symbol),
                    snapshot=snap,
                    symbol_positions=symbol_positions,
                    total_positions=total_positions,
                    current_symbol_exposure=current_symbol_exposure,
                    total_exposure=total_exposure,
                    drawdown_pct=float(risk.current_dd),
                )
                sym_state["risk_supervisor"] = {
                    "allowed": bool(supervisor_decision.allowed),
                    "reason": supervisor_decision.reason,
                    "current_symbol_exposure": float(current_symbol_exposure),
                    "total_exposure": float(total_exposure),
                }
                if abs(exposure) > 0.001:
                    logger.info(
                        "EXEC_TRACE %s | exposure=%.4f supervisor=%s reason=%s dd=%.2f positions=%d sym_positions=%d"
                        % (symbol, exposure, supervisor_decision.allowed, supervisor_decision.reason,
                           float(risk.current_dd), total_positions, symbol_positions)
                    )
                if supervisor_decision.allowed:
                    order_meta = brain.live_trade(
                        symbol,
                        exposure,
                        max_lots,
                        action_meta=action_meta,
                        execution_context={
                            "regime": regime,
                            "confidence": conf,
                            "target_exposure": float(exposure),
                            "raw_target": float(decision["raw_target"]),
                            "ppo_target": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                            "dreamer_target": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                            "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                            "agi_risk_scalar": float((agi_meta or {}).get("risk_scalar", 1.0) or 1.0),
                            "rebalance_min_delta_exposure": float(
                                (decision.get("profile") or {}).get("rebalance_min_delta_exposure", 0.0) or 0.0
                            ),
                            "min_hold_time_sec": int(trade_profile.get("min_hold_time_sec", 0) or 0),
                            "ignore_small_reduction": bool(trade_profile.get("ignore_small_reduction", False)),
                            "ignore_reduction_threshold": float(trade_profile.get("ignore_reduction_threshold", 0.15) or 0.15),
                        },
                    )
                    if order_meta and order_meta.get("executed"):
                        supervisor.mark_trade(symbol)
                else:
                    order_meta = None
                    _append_audit(
                        "risk_supervisor_block",
                        {
                            "symbol": symbol,
                            "target_exposure": float(exposure),
                            "reason": supervisor_decision.reason,
                            "confidence": conf,
                            "signal": regime,
                        },
                    )
                executor.manage_open_positions(symbol, signal_context={
                    "blended_exposure": float(exposure),
                    "regime": regime,
                    "ppo_target": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                    "dreamer_target": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                })
                if order_meta:
                    tp_outcome_usd, sl_outcome_usd = _expected_usd(
                        symbol=symbol,
                        side=str(order_meta.get("order_type", "BUY")),
                        entry=float(order_meta.get("entry_price", 0.0) or 0.0),
                        tp=float(order_meta.get("tp_price", 0.0) or 0.0),
                        sl=float(order_meta.get("sl_price", 0.0) or 0.0),
                        lots=float(order_meta.get("volume_lots", 0.0) or 0.0),
                    )
                    if tp_outcome_usd is not None:
                        order_meta["tp_outcome_usd"] = float(tp_outcome_usd)
                        order_meta["expected_profit_usd"] = float(tp_outcome_usd)
                    if sl_outcome_usd is not None:
                        order_meta["sl_outcome_usd"] = float(sl_outcome_usd)
                        order_meta["expected_loss_usd"] = float(sl_outcome_usd)
                    _append_audit("trade_action", dict(order_meta))
                    logger.info(
                        "ACTION %s | req=%s side=%s volume=%s target=%.4f ppo=%.4f dreamer=%.4f agi=%.4f magic=%s comment=%s ticket=%s retcode=%s TP=%s SL=%s"
                        % (
                            symbol,
                            order_meta.get("request_action"),
                            order_meta.get("order_type"),
                            order_meta.get("executed_lots", order_meta.get("volume_lots")),
                            float(order_meta.get("target_exposure", order_meta.get("exposure", 0.0)) or 0.0),
                            float(order_meta.get("ppo_target", 0.0) or 0.0),
                            float(order_meta.get("dreamer_target", 0.0) or 0.0),
                            float(order_meta.get("agi_bias", 0.0) or 0.0),
                            order_meta.get("magic"),
                            order_meta.get("comment"),
                            order_meta.get("ticket"),
                            order_meta.get("retcode"),
                            order_meta.get("tp_price"),
                            order_meta.get("sl_price"),
                        )
                    )
                    alerter.trade_action(symbol, order_meta)

                acc = mt5.account_info()
                if acc:
                    risk.update_equity(float(acc.equity))
            except Exception as exc:
                risk.record_error()
                alerter.alert(f"Execution loop error on {symbol}: {exc}")
                logger.exception(f"Execution loop error on {symbol}: {exc}")

        known_open_tickets, seen_closed_deals, last_deal_check, closed_events = _scan_trade_events(
            alerter,
            risk,
            known_open_tickets,
            seen_closed_deals,
            last_deal_check,
        )
        for c in closed_events:
            try:
                last_closed_by_symbol[str(c.get("symbol", "?"))] = c
            except Exception:
                pass

        if now - last_symbol_cards >= max(15, symbol_card_sec):
            for symbol in symbols:
                sym = str(symbol)
                sstate = dict(last_symbol_state.get(sym, {}))
                pos_rows = mt5.positions_get(symbol=sym) or []
                sstate["open_positions"] = len(pos_rows)
                sstate["floating_pnl"] = sum(float(getattr(p, "profit", 0.0)) for p in pos_rows)
                if pos_rows:
                    p0 = pos_rows[0]
                    p_side = "BUY" if int(getattr(p0, "type", 0)) == int(mt5.ORDER_TYPE_BUY) else "SELL"
                    p_vol = float(getattr(p0, "volume", 0.0) or 0.0)
                    p_entry = float(getattr(p0, "price_open", 0.0) or 0.0)
                    p_tp = float(getattr(p0, "tp", 0.0) or 0.0)
                    p_sl = float(getattr(p0, "sl", 0.0) or 0.0)
                    sstate["position_side"] = p_side
                    sstate["position_volume"] = p_vol
                    sstate["position_entry"] = p_entry
                    sstate["position_tp"] = p_tp
                    sstate["position_sl"] = p_sl
                    tpv, slv = _expected_usd(sym, p_side, p_entry, p_tp, p_sl, p_vol)
                    sstate["position_tp_value_usd"] = None if tpv is None else float(tpv)
                    sstate["position_sl_value_usd"] = None if slv is None else float(slv)
                else:
                    sstate["position_side"] = None
                    sstate["position_volume"] = None
                    sstate["position_entry"] = None
                    sstate["position_tp"] = None
                    sstate["position_sl"] = None
                    sstate["position_tp_value_usd"] = None
                    sstate["position_sl_value_usd"] = None
                sstate["last_closed"] = last_closed_by_symbol.get(sym)
                alerter.symbol_status(sym, sstate)
            last_symbol_cards = now

        time.sleep(max(5, loop_sleep_sec))

    # Graceful shutdown
    logger.warning("Shutdown flag set — exiting main loop")
    try:
        risk.save_state()
        logger.info("Risk engine state saved")
    except Exception as exc:
        logger.error(f"Failed to save risk engine state: {exc}")
    try:
        mt5.shutdown()
        logger.info("MT5 shutdown complete")
    except Exception:
        pass


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)
