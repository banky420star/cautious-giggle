import atexit
import datetime
import json
import os
import subprocess
import time

from Python.mt5_compat import mt5  # MetaTrader5 terminal or Exness API backend

import pandas as pd
from loguru import logger

from Python.model_memory import ModelMemory, market_context
from Python.indicator_council import council_vote, readings_mtf, votes_mtf
from Python import demo_exploration
from Python import trade_scorecard
from Python import auto_changes
from Python.trade_memory_scale import trade_memory_scale as _trade_memory_scale
from Python.trend_gate import TrendGate
from Python.config_utils import DEFAULT_TRADING_SYMBOLS, load_project_config, resolve_trading_symbols

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "server_agi.lock")
LOG_DIR = os.environ.get("AGI_LOG_DIR") or os.path.join(BASE_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
AUDIT_LOG = os.path.join(LOG_DIR, "audit_events.jsonl")
TRADE_EVENTS_LOG = os.path.join(LOG_DIR, "trade_events.jsonl")
ACTIVE_MODELS_PATH = os.path.join(BASE_DIR, "models", "registry", "active.json")

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(SERVER_LOG, rotation="10 MB", level="INFO")

SYMBOL_EXECUTION_PROFILES = {
    "BTCUSDm": {
        # Dreamer is not evaluated by any promotion gate, so it gets no live weight
        # by default. Override per symbol via trading.symbol_profiles.<sym>.dreamer_weight.
        "ppo_weight": 0.70,
        "dreamer_weight": 0.0,
        "agi_weight": 0.30,
        "council_weight": 0.35,
        "min_trade_threshold": 0.18,
        "max_abs_target": 1.00,
        "cooldown_sec": 30,
    },
    "XAUUSDm": {
        "ppo_weight": 0.60,
        "dreamer_weight": 0.0,
        "agi_weight": 0.40,
        "council_weight": 0.35,
        "min_trade_threshold": 0.12,
        "max_abs_target": 0.75,
        "cooldown_sec": 45,
    },
}


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_default(v):
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    return str(v)


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


def _recorded_close_deals(max_bytes: int = 20_000_000) -> set:
    """Deal ids of every trade_closed event already in the trade log."""
    out = set()
    try:
        with open(TRADE_EVENTS_LOG, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - max_bytes))
            for ln in f.read().decode("utf-8", "replace").splitlines():
                if '"trade_closed"' not in ln:
                    continue
                try:
                    did = int((json.loads(ln).get("payload") or {}).get("deal_id") or 0)
                except (ValueError, TypeError):
                    continue
                if did > 0:
                    out.add(did)
    except OSError:
        pass
    return out


def _append_trade_event(event: str, payload: dict):
    row = {
        "ts": _utc_now().isoformat(timespec="microseconds"),
        "event": event,
        "payload": payload,
    }
    _append_jsonl(TRADE_EVENTS_LOG, row)
    _append_audit(event, payload)


def _load_runtime_components():
    from Python.agi_brain import SmartAGI
    from Python.event_intel import EventIntel
    from Python.hybrid_brain import HybridBrain
    from Python.mt5_executor import MT5Executor
    from Python.risk_engine import RiskEngine
    from Python.risk_supervisor import RiskSupervisor
    from Python.trade_learning import build_trade_learning
    from alerts.telegram_alerts import TelegramAlerter

    return {
        "SmartAGI": SmartAGI,
        "EventIntel": EventIntel,
        "HybridBrain": HybridBrain,
        "MT5Executor": MT5Executor,
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


def _acquire_single_instance_lock():
    os.makedirs(LOCK_DIR, exist_ok=True)
    try:
        fd = os.open(LOCK_PATH, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
        os.write(fd, str(os.getpid()).encode("utf-8"))
        os.close(fd)
    except FileExistsError:
        # Take over a stale lock left by a crashed process.
        try:
            pid = int(open(LOCK_PATH).read().strip() or 0)
        except Exception:
            pid = 0
        alive = False
        if pid:
            try:
                os.kill(pid, 0)
                alive = True
            except Exception:
                alive = False
        if alive:
            return False
        try:
            os.remove(LOCK_PATH)
        except Exception:
            return False
        return _acquire_single_instance_lock()

    def _cleanup_lock():
        try:
            if os.path.exists(LOCK_PATH):
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


def _ensure_symbols_selected(symbols) -> None:
    for sym in symbols:
        try:
            if not mt5.symbol_select(str(sym), True):
                logger.warning(f"symbol_select({sym}) failed: {mt5.last_error()}")
        except Exception as exc:
            logger.warning(f"symbol_select({sym}) raised: {exc}")


def _ensure_mt5_connected(cfg, symbols) -> bool:
    """Re-initialize the MT5 terminal link if it dropped. Returns True when connected."""
    try:
        if mt5.terminal_info() is not None and mt5.account_info() is not None:
            return True
    except Exception:
        pass
    logger.warning("MT5 connection lost; re-initializing")
    try:
        mt5.shutdown()
    except Exception:
        pass
    try:
        if _init_mt5(cfg):
            _ensure_symbols_selected(symbols)
            return True
    except Exception as exc:
        logger.warning(f"MT5 re-init failed: {exc}")
    return False


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
    # Demo learning mode (set at startup only when the account is a demo): lower trade bar so the
    # bot trades more often and its memory/models learn from more real outcomes.
    scale = float(trading_cfg.get("_learning_threshold_scale", 1.0) or 1.0)
    if 0 < scale < 1:
        base["min_trade_threshold"] = float(base["min_trade_threshold"]) * scale
    return base


def _risk_scalar(agi_meta: dict | None) -> float:
    """risk_scalar from the AGI meta; a legitimate 0.0 (full block) must stay 0.0."""
    value = (agi_meta or {}).get("risk_scalar")
    if value is None:
        return 1.0
    try:
        return max(0.0, float(value))
    except Exception:
        return 1.0


class _CouncilOff(Exception):
    pass


def _smart_entry_settings(cfg: dict) -> dict:
    se = ((cfg or {}).get("trading", {}) or {}).get("smart_entry", {}) or {}
    return {"enabled": bool(se.get("enabled", False)), "wait_min": float(se.get("wait_minutes", 15)),
            "pullback_atr": dict(se.get("pullback_atr", {}) or {}), "default_atr": float(se.get("default_pullback_atr", 0.35)),
            "on_timeout": str(se.get("on_timeout", "market")).lower()}


def _plan_entry(symbol: str, exposure: float, executor, settings: dict) -> dict | None:
    """Pullback entry: instead of buying/selling at market, wait up to wait_min for price to come
    back k x ATR(M5); if it doesn't, enter at market when the wait runs out."""
    tick = mt5.symbol_info_tick(symbol)
    info = executor._symbol_info(symbol)
    atr_pts = executor._atr_points(symbol)
    if tick is None or info is None or not atr_pts:
        return None
    k = float(settings["pullback_atr"].get(str(symbol), settings["default_atr"]))
    side = 1 if exposure > 0 else -1
    ref = float(tick.ask) if side > 0 else float(tick.bid)
    atr = float(atr_pts) * float(info.point or 0.0)
    return {"side": side, "exposure": float(exposure), "ref": ref, "limit": ref - side * k * atr, "k": k,
            "created": time.time(), "expires": time.time() + settings["wait_min"] * 60.0}


_HTF_CACHE: dict = {}


def _htf_frames(symbol: str) -> dict:
    """Closed M15/H1/H4 bars from the broker (cached until a new bar can exist)."""
    out = {}
    for tf_name, minutes, const in (("M15", 15, "TIMEFRAME_M15"), ("H1", 60, "TIMEFRAME_H1"), ("H4", 240, "TIMEFRAME_H4")):
        key = (symbol, tf_name)
        bucket = int(time.time() // (minutes * 60))
        hit = _HTF_CACHE.get(key)
        if hit and hit[0] == bucket:
            out[tf_name] = hit[1]
            continue
        try:
            rates = mt5.copy_rates_from_pos(symbol, getattr(mt5, const), 1, 400)
            if rates is None or len(rates) < 60:
                continue
            df = pd.DataFrame(rates)
            df.index = pd.to_datetime(df["time"], unit="s", utc=True)
            df = df[["open", "high", "low", "close"]].astype(float)
            _HTF_CACHE[key] = (bucket, df)
            out[tf_name] = df
        except Exception:
            continue
    return out


_M15_CACHE: dict = {}


def _m15_long(symbol: str, bars: int = 2000):
    """Longer M15 history (with time and volume) for the external voters; cached per 15-min bar."""
    bucket = int(time.time() // 900)
    hit = _M15_CACHE.get(symbol)
    if hit and hit[0] == bucket:
        return hit[1]
    try:
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M15, 1, bars)
        if rates is None or len(rates) < 100:
            return None
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        _M15_CACHE[symbol] = (bucket, df)
        return df
    except Exception:
        return None


def _blend_symbol_decision(
    symbol: str,
    agi_meta: dict | None,
    ppo_meta: dict | None,
    dreamer_meta: dict | None,
    cfg: dict | None = None,
    memory_mult: dict | None = None,
    council_meta: dict | None = None,
) -> dict:
    profile = _symbol_profile(symbol, cfg=cfg)

    ppo_target = float((ppo_meta or {}).get("target", 0.0) or 0.0)
    dreamer_target = float((dreamer_meta or {}).get("target", 0.0) or 0.0)

    agi_conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
    agi_risk = _risk_scalar(agi_meta)
    agi_bias = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)

    trading_cfg = (cfg or {}).get("trading", {}) if isinstance(cfg, dict) else {}
    conf_threshold = float(trading_cfg.get("confidence_threshold", 0.0) or 0.0)
    if agi_conf < conf_threshold or bool((agi_meta or {}).get("trade_blocked", False)):
        agi_bias = 0.0

    # A missing model contributes 0 with its weight kept (exposure shrinks) rather than
    # handing its weight to the LSTM, which made targets swing whenever PPO abstained.
    # Set trading.redistribute_missing_weight: true to restore the old behaviour.
    ppo_w = float(profile["ppo_weight"]) if ppo_meta is not None else 0.0
    dreamer_w = float(profile["dreamer_weight"]) if dreamer_meta is not None else 0.0
    agi_w = float(profile["agi_weight"])
    if bool(trading_cfg.get("redistribute_missing_weight", False)):
        agi_w = max(agi_w, 1.0 - ppo_w - dreamer_w)

    # Model memory: scale each model's say by how well it has done in this market context.
    mm = memory_mult or {}
    ppo_w *= float((mm.get("ppo") or {}).get("mult", 1.0))
    dreamer_w *= float((mm.get("dreamer") or {}).get("mult", 1.0))
    agi_w *= float((mm.get("lstm") or {}).get("mult", 1.0))

    council_target = float((council_meta or {}).get("target", 0.0) or 0.0)
    council_w = float(profile.get("council_weight", 0.0)) if council_meta is not None else 0.0
    council_w *= float((mm.get("council") or {}).get("mult", 1.0))

    raw = ppo_w * ppo_target + dreamer_w * dreamer_target + agi_w * agi_bias + council_w * council_target

    # Weighted average over the voters that actually have a view this bar, so a model that is
    # missing (not trained yet) or abstaining (LSTM below its confidence bar) doesn't silently
    # shrink every signal below the trade threshold. trading.normalize_active_weights: false
    # restores the old "missing weight counts as zero" behaviour.
    active_w = ((ppo_w if ppo_meta is not None else 0.0) + (dreamer_w if dreamer_meta is not None else 0.0)
                + (agi_w if agi_bias != 0.0 else 0.0) + (council_w if council_meta is not None else 0.0))
    normalized = bool(trading_cfg.get("normalize_active_weights", False)) and active_w > 0
    if normalized:
        raw = raw / active_w

    adjusted = raw * agi_risk
    adjusted = _clip(adjusted, -float(profile["max_abs_target"]), float(profile["max_abs_target"]))

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
        "council_target": council_target,
        "council_weight_used": round(council_w, 3),
        "active_weight": round(active_w, 3),
        "weights_normalized": normalized,
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


_TF_SECONDS = {1: 60, 5: 300, 15: 900, 30: 1800, 60: 3600, 240: 14400, 1440: 86400}


_MOVE_FILTER_CACHE: dict = {}


def _move_filter_watch(symbol: str):
    """Watch-only "big move coming" filter (Python/move_filter.py). Recomputed once per new 5-minute bar;
    needs ~6000 M5 bars for its daily inputs (1000 bars disagreed with full history up to 30% of the time)."""
    from Python import move_filter
    probe = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, 1)
    if probe is None or len(probe) == 0:
        return None
    last = int(probe[-1]["time"])
    hit = _MOVE_FILTER_CACHE.get(symbol)
    if hit is not None and hit[0] == last:
        return hit[1]
    d = _fetch_symbol_df(symbol, mt5.TIMEFRAME_M5, bars=move_filter.HISTORY_BARS)
    out = move_filter.prob(symbol, d)
    if out is not None:
        out["bar_time"] = last
    _MOVE_FILTER_CACHE[symbol] = (last, out, d)
    return out


def _fetch_symbol_df(symbol: str, timeframe, bars=None):
    """Closed bars only (start_pos=1); None if the feed is missing or stale.

    Enough history is fetched for the H4/D1 features of ultimate_150 to be meaningful.
    """
    bars = int(bars or os.environ.get("AGI_LIVE_BARS", "1000"))
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, bars)
    if rates is None or len(rates) < 80:
        return None
    try:
        tf_sec = _TF_SECONDS.get(int(timeframe), 300)
        tick = mt5.symbol_info_tick(symbol)
        now_server = int(getattr(tick, "time", 0) or 0)
        last_bar = int(rates[-1]["time"])
        # Last closed bar must be within ~2 bars of the broker's latest tick time.
        if now_server > 0 and now_server - last_bar > 3 * tf_sec:
            logger.warning(f"{symbol}: stale bars (last closed bar {now_server - last_bar}s old); skipping")
            return None
    except Exception:
        pass

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


def _last_closed_bar_time(symbol: str, timeframe) -> int | None:
    try:
        rates = mt5.copy_rates_from_pos(symbol, timeframe, 1, 1)
        if rates is None or len(rates) == 0:
            return None
        return int(rates[-1]["time"])
    except Exception:
        return None


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
        "realized_today": float(pnl_today),
        # Realized + floating: what the daily-loss guard should compare against.
        "pnl_today_incl_floating": float(pnl_today + floating),
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


def _position_exposure_state(symbol: str, max_lots: float, executor=None) -> tuple[float, float, int, int]:
    positions = mt5.positions_get() or []
    if executor is not None and hasattr(executor, "_is_own_position"):
        positions = [p for p in positions if executor._is_own_position(p)]
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
        trade_scorecard.on_open(payload)  # who voted which way, and why, at entry

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
            card = trade_scorecard.on_close(payload)  # which voters called it right
            if card:
                right = [c["name"] for c in card["calls"] if c["call"] == "right"][:4]
                logger.info(f"SCORECARD {payload['symbol']} #{payload['ticket']} {card['outcome']} ${pnl:+.2f}: "
                            f"right = {', '.join(right) or 'nobody'}")
            try:
                risk.record_trade_result(payload["symbol"], pnl)
            except Exception:
                pass
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



def _stack_level(symbol, exposure, executor, stacking, state) -> int:
    """How many trades this market should hold in the signal's direction (demo stacking).

    Starts at 1. Adds one when every open trade on that side is in profit, the signal still
    points the same way and the last add was at least cooldown ago; drops back when the
    signal flips or goes flat, or a stacked trade is closed.
    """
    if not stacking.get("enabled") or exposure == 0.0:
        state.pop(symbol, None)
        return 1
    st = state.setdefault(symbol, {"n": 1, "t": 0.0, "side": 0})
    side = 1 if exposure > 0 else -1
    if st["side"] != side:
        st.update(n=1, t=time.time(), side=side)
    try:
        longs, shorts = executor.get_positions(symbol)
    except Exception:
        return int(st["n"])
    same, opp = (longs, shorts) if side > 0 else (shorts, longs)
    if opp:
        return int(st["n"])
    st["n"] = max(1, min(int(st["n"]), len(same) + 1))     # a stacked trade hit its stop or target
    if (len(same) >= st["n"] and st["n"] < int(stacking.get("max", 3))
            and time.time() - float(st["t"]) >= float(stacking.get("cooldown_s", 600))
            and same and all(float(getattr(p, "profit", 0.0) or 0.0) > 0 for p in same)):
        st["n"] += 1
        st["t"] = time.time()
        logger.info(f"STACK {symbol} | {len(same)} trade(s) in profit and the signal still agrees -> "
                    f"adding trade {st['n']} of {stacking.get('max')}")
    return int(st["n"])

def main(live=False):
    if not _acquire_single_instance_lock():
        raise RuntimeError("Server_AGI is already running (lock file exists)")

    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    cfg = _load_cfg(live=live)
    ok = _init_mt5(cfg)
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    runtime = _load_runtime_components()
    risk = runtime["RiskEngine"]()
    supervisor = runtime["RiskSupervisor"](cfg)
    executor = runtime["MT5Executor"](risk)
    brain = runtime["HybridBrain"](risk, executor)
    agi = runtime["SmartAGI"]()

    trading_cfg = cfg.get("trading", {})
    symbols = resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",), fallback=DEFAULT_TRADING_SYMBOLS)
    # markets that passed the exam through tools/graduation.py trade like the others
    try:
        if bool((((cfg.get("trading") or {}).get("graduation") or {}).get("enabled"))):
            with open(os.path.join(LOG_DIR, "graduation.json"), encoding="utf-8") as _gf:
                _gst = (json.load(_gf).get("symbols") or {})
            grads = [s_ for s_, v_ in _gst.items() if s_ not in symbols and (
                v_.get("status") == "graduated" or (v_.get("status") in ("training", "retrain_queued") and v_.get("was_live")))]
            if grads:
                symbols = list(symbols) + grads
                logger.info(f"graduated markets trading: {grads}")
    except FileNotFoundError:
        pass
    except Exception as exc:
        logger.warning(f"graduation state unreadable: {exc}")
    if hasattr(mt5, "hide_foreign"):
        mt5.hide_foreign = True  # trades opened by hand in Exness are the user's: the bot never touches them
        logger.info("manual trades are left alone: the bot only manages positions its own orders opened")
    _dl = ((cfg.get("trading", {}) or {}).get("demo_learning", {}) or {})
    if bool(_dl.get("enabled")) and bool(_dl.get("trade_all_markets")) and demo_exploration.demo_rules_apply(mt5, cfg):
        extra_ = [s_ for s_ in (((cfg.get("trading") or {}).get("news_trading") or {}).get("symbols") or []) if s_ not in symbols]
        if extra_:
            symbols = list(symbols) + extra_
            logger.info(f"demo: every market trades all day: {symbols} (no trained model yet -> indicator council decides)")
    # The brain was built from config's market list (BTC/XAU). Give it every market that trades so the
    # models trained for the newer markets are actually loaded and used.
    if set(symbols) != set(getattr(brain, "active_symbols", None) or []):
        brain.active_symbols = list(symbols)
        try:
            brain._load_ppo_from_registry()
            loaded_ = sorted((getattr(brain, "ppo_bundles_by_symbol", None) or {}).keys())
            logger.info(f"AI models loaded for: {loaded_ or 'none yet'}")
        except Exception as exc:
            logger.warning(f"model reload for all markets failed: {exc}")
    _stk = dict(_dl.get("stacking") or {}) if bool(_dl.get("enabled")) and demo_exploration.demo_rules_apply(mt5, cfg) else {}
    stacking = {"enabled": bool(_stk.get("enabled")), "max": int(_stk.get("max_per_symbol", 3)),
                "cooldown_s": float(_stk.get("cooldown_min", 10)) * 60.0}
    stack_state: dict = {}
    if stacking["enabled"]:
        logger.info(f"demo: trade stacking on: up to {stacking['max']} trades per market, adding when the open "
                    f"ones are in profit and the signal agrees (>= {stacking['cooldown_s'] / 60:.0f} min apart)")
    timeframe = _to_mt5_timeframe(trading_cfg.get("timeframe", "M5"))
    max_lots = float(cfg.get("risk", {}).get("max_lots", 1.0))
    # Demo learning mode may raise the risk limits, on demo accounts only (never carried to real money).
    _lr = (((cfg.get("trading", {}) or {}).get("demo_learning", {}) or {}).get("risk", {}) or {})
    if _lr and bool(((cfg.get("trading", {}) or {}).get("demo_learning", {}) or {}).get("enabled")) \
            and demo_exploration.demo_rules_apply(mt5, cfg):
        if "max_lots" in _lr:
            max_lots = float(_lr["max_lots"])
            risk.max_lots = max_lots
        if "risk_percent" in _lr:
            risk.risk_percent = float(_lr["risk_percent"])
        if "min_lot_max_risk_percent" in _lr:
            risk.min_lot_max_risk_percent = float(_lr["min_lot_max_risk_percent"])
        if "max_daily_loss" in _lr:
            risk.max_daily_loss = float(_lr["max_daily_loss"])
            supervisor.max_daily_loss = float(_lr["max_daily_loss"])
        if "max_drawdown_pct" in _lr and hasattr(supervisor, "max_drawdown_pct"):
            supervisor.max_drawdown_pct = float(_lr["max_drawdown_pct"])
        for k_ in ("max_open_positions", "max_positions_per_symbol"):
            if k_ in _lr:
                setattr(supervisor, k_, int(_lr[k_]))
        logger.info(f"demo learning risk: {risk.risk_percent}% per trade (min-lot ceiling {risk.min_lot_max_risk_percent}%), "
                    f"max_lots {max_lots}, daily loss stop ${risk.max_daily_loss:.0f}, "
                    f"drawdown halt {getattr(supervisor, 'max_drawdown_pct', '?')}%")
    _ensure_symbols_selected(symbols)

    token, chat_id = _load_telegram_cfg(cfg)
    alerter = runtime["TelegramAlerter"](token, chat_id)
    event_intel = runtime["EventIntel"](cfg, LOG_DIR)

    model_memory = ModelMemory.from_cfg(cfg, LOG_DIR)
    explore_cfg = demo_exploration.settings(cfg)
    explore_demo_ok = bool(explore_cfg["enabled"]) and demo_exploration.demo_rules_apply(mt5, cfg)
    smart_entry = _smart_entry_settings(cfg)
    pending_entries: dict = {}
    entry_skip: dict = {}  # symbol -> (side, until): pullback never came, don't chase that side for a while
    ext_voters = None
    try:
        from Python.external_voters import ExternalVoters
        _xv = ((cfg.get("trading") or {}).get("external_voters") or {})
        if bool(_xv.get("enabled", True)):
            ext_voters = ExternalVoters(_xv, logger)
            logger.info(f"external voters ON: {ext_voters.weights()} (your models vote in the council; "
                        f"ones that keep being wrong are ignored by the model memory)")
    except Exception as exc:
        logger.warning(f"external voters unavailable: {exc}")
    _xv_dumped = [0.0]
    council_readings: dict = {}
    _readings_dumped = [0.0]
    from Python.setup_gate import SetupGate
    setup_gate = SetupGate(((cfg.get("trading") or {}).get("setup_gate") or {}))
    trend_gate = TrendGate(((cfg.get("trading") or {}).get("trend_gate") or {}), mt5)
    _stops = (cfg.get("trading") or {}).get("stops") or {}
    executor.sl_atr_mult = float(_stops.get("sl_atr_mult", 2.0))
    executor.tp_atr_mult = float(_stops.get("tp_atr_mult", 3.2))
    logger.info(f"trend gate {'on' if trend_gate.enabled else 'off'} ({'+'.join(trend_gate.tfs)} EMA{trend_gate.span}); "
                f"stops {executor.sl_atr_mult}x ATR, targets {executor.tp_atr_mult}x ATR")
    _cg = ((cfg.get("trading") or {}).get("cost_gate") or {})
    cost_gate = {"enabled": bool(_cg.get("enabled", True)), "margin_bps": float(_cg.get("margin_bps", 0.2)),
                 "extra_cost_bps": float(_cg.get("extra_cost_bps", 0.0))}
    last_spread: dict = {}  # last good spread per market: a missing tick must not be scored as zero cost

    def _cost_blocked(sym: str):
        """(blocked, assessment): the market's own signal-follow record, net of spread, is not paying."""
        a = model_memory.assess(sym, "trade", "*")
        proven = a["n_all"] >= model_memory.min_samples
        return (cost_gate["enabled"] and proven and a["edge_bps"] <= cost_gate["margin_bps"]), a
    _gate_logged: dict = {}
    logger.info(f"setup gate {'ON' if setup_gate.enabled else 'OFF'}: unproven setups trade the smallest lot, "
                f"losing ones are blocked ({len(setup_gate.state['stats'])} scored setups loaded)")
    logger.info(f"smart entry {'ON' if smart_entry['enabled'] else 'OFF'} (pullback wait {smart_entry['wait_min']:.0f} min)")
    from Python.news_trader import NewsTrader
    news_trader = NewsTrader(mt5, executor, cfg, logger, _append_audit)
    news_trader.halted = lambda: bool(supervisor.halt_until is not None and supervisor.halt_until > _utc_now())
    if news_trader.enabled and not demo_exploration.demo_rules_apply(mt5, cfg):
        news_trader.enabled = False  # learning feature: demo, or live when same_on_live
    logger.info(f"news trading {'ON' if news_trader.enabled else 'OFF'}")
    learn_cfg = (cfg.get("trading", {}) or {}).get("demo_learning", {}) or {}
    if bool(learn_cfg.get("enabled", False)) and demo_exploration.demo_rules_apply(mt5, cfg):
        cfg.setdefault("trading", {})["_learning_threshold_scale"] = float(learn_cfg.get("threshold_scale", 0.5))
        logger.info(f"demo learning mode ON: trade threshold x{float(learn_cfg.get('threshold_scale', 0.5)):.2f}")
    else:
        logger.info("demo learning mode OFF")
    explore_cache: dict = {}
    council_cfg = ((cfg.get("trading", {}) or {}).get("council", {}) or {})
    # Council weights are learned from history replay (in-sample), so on a real-money account it
    # only votes when explicitly allowed.
    council_allowed = bool(council_cfg.get("enabled", True)) and (
        demo_exploration.demo_rules_apply(mt5, cfg) or bool(council_cfg.get("allow_live", False)))
    logger.info(f"indicator council {'ON' if council_allowed else 'OFF'}")
    if demo_exploration.same_on_live(cfg) and not demo_exploration.account_is_demo(mt5):
        logger.warning("LIVE account running with the DEMO settings (trading.demo_learning.same_on_live)")

    def _exploration_scale(sym: str) -> float:
        """Size scale for symbols running a demo-exploration canary (0 on a live account)."""
        hit = explore_cache.get(sym)
        if hit and time.time() - hit[0] < 60:
            return hit[1]
        scale = 1.0
        try:
            from Python.model_registry import ModelRegistry

            d = ModelRegistry().load_active_model(prefer_canary=True, symbol=sym)
            if demo_exploration.is_exploration(d):
                scale = float(explore_cfg["size_scale"]) if explore_demo_ok else 0.0
        except Exception:
            pass
        explore_cache[sym] = (time.time(), scale)
        return scale

    memory_on = bool(((cfg.get("trading", {}) or {}).get("model_memory", {}) or {}).get("enabled", True))
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

    known_open_tickets = set()
    # closes already recorded survive a restart: without this the last 30 minutes of closes were recorded
    # again on every restart (double-counted in trade learning, the scorecard and the risk engine's daily loss)
    seen_closed_deals = _recorded_close_deals()
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

    reconnect_backoff = 5.0
    # Decisions are made once per closed bar. Between bars only stops are managed;
    # recomputing an identical decision every loop added nothing but churn risk.
    last_decided_bar = {}
    every_loop = os.environ.get("AGI_DECIDE_EVERY_LOOP", "false").lower() == "true"
    while True:
        now = time.time()

        if not _ensure_mt5_connected(cfg, symbols):
            alerter.alert(f"MT5 disconnected; retrying in {int(reconnect_backoff)}s")
            time.sleep(reconnect_backoff)
            reconnect_backoff = min(300.0, reconnect_backoff * 2)
            continue
        reconnect_backoff = 5.0

        # Fresh account state every loop: the daily-loss guard must not run on a
        # snapshot that is up to a heartbeat interval old.
        snap = _account_snapshot()
        if snap.get("equity") is not None:
            risk.update_equity(float(snap["equity"]))
        guard_snap = dict(snap)
        guard_snap["pnl_today"] = float(snap.get("pnl_today_incl_floating", snap.get("pnl_today", 0.0)))

        if now - last_heartbeat >= max(15, heartbeat_sec):
            uptime = int(now - start_time)
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
            except Exception as e:
                logger.warning(f"trade learning update failed: {e}")
            last_learning = now

        # ---- news: prep for releases, trade the sweep, learn from the result
        try:
            news_trader.step()
            with open(os.path.join(LOG_DIR, "news_state.json"), "w", encoding="utf-8") as _f:
                json.dump({"t": time.time(), "enabled": news_trader.enabled, "events": news_trader.upcoming()}, _f)
        except Exception as exc:
            logger.warning(f"news step failed: {exc}")

        if ext_voters is not None and time.time() - _xv_dumped[0] > 30:
            _xv_dumped[0] = time.time()
            try:
                ext_voters.dump(os.path.join(LOG_DIR, "external_voters.json"))
            except Exception:
                pass
        try:
            if setup_gate.resolve(mt5):
                logger.info("setup scorecard updated from closed trades")
        except Exception as exc:
            logger.warning(f"setup scorecard update failed: {exc}")
        # ---- smart entry: fill pending pullback entries (or enter at market when the wait runs out)
        for psym, pe in list(pending_entries.items()):
            try:
                if supervisor.halt_until is not None and supervisor.halt_until > _utc_now():
                    pending_entries.pop(psym, None)
                    _append_audit("entry_cancelled", {"symbol": psym, "reason": "risk halt"})
                    continue
                if news_trader.blocks_regular(psym):
                    pending_entries.pop(psym, None)
                    _append_audit("entry_cancelled", {"symbol": psym, "reason": news_trader.blocks_regular(psym)})
                    continue
                if _cost_blocked(psym)[0]:
                    pending_entries.pop(psym, None)
                    _append_audit("entry_cancelled", {"symbol": psym, "reason": "cost gate"})
                    continue
                tk = mt5.symbol_info_tick(psym)
                if tk is None:
                    continue
                px = float(tk.ask) if pe["side"] > 0 else float(tk.bid)
                hit = px <= pe["limit"] if pe["side"] > 0 else px >= pe["limit"]
                if not hit and time.time() < pe["expires"]:
                    continue
                pending_entries.pop(psym, None)
                if not hit and smart_entry["on_timeout"] == "skip":
                    entry_skip[psym] = (pe["side"], time.time() + smart_entry["wait_min"] * 60.0)
                    _append_audit("entry_cancelled", {"symbol": psym, "reason": "pullback never came; not chasing"})
                    logger.info(f"ENTRY SKIPPED {psym} | the pullback to {pe['limit']:.5g} never came in "
                                f"{smart_entry['wait_min']:.0f} min (now {px:.5g}); not chasing the price")
                    continue
                om = brain.live_trade(psym, pe["exposure"], pe["max_lots"], action_meta=pe.get("action_meta"),
                                      execution_context=pe.get("ctx"))
                if om and om.get("executed"):
                    supervisor.mark_trade(psym)
                    if pe.get("gate"):
                        setup_gate.record_open(psym, pe["side"], *pe["gate"])
                    om = dict(om)
                    om["entry_mode"] = "pullback" if hit else "timeout_market"
                    om["entry_improvement_bps"] = round((pe["ref"] - px) * pe["side"] / pe["ref"] * 1e4, 2)
                    om["waited_s"] = round(time.time() - pe["created"])
                    _append_audit("trade_action", om)
                    logger.info(f"ENTRY {psym} | {om['entry_mode']} after {om['waited_s']}s | "
                                f"{'buy' if pe['side'] > 0 else 'sell'} {om.get('executed_lots', om.get('volume_lots'))} "
                                f"@ {px} vs {pe['ref']} at signal ({om['entry_improvement_bps']:+.2f} bps better)")
            except Exception as exc:
                pending_entries.pop(psym, None)
                logger.warning(f"smart entry failed for {psym}: {exc}")

        for symbol in symbols:
            try:
                bar_time = _last_closed_bar_time(symbol, timeframe)
                if not every_loop and bar_time is not None and last_decided_bar.get(symbol) == bar_time:
                    executor.manage_open_positions(symbol)
                    continue
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue
                # The bar is only marked decided once the decision has *settled*
                # (executed, deliberate no-op, or a persistent block). Failed orders and
                # transient blocks (spread spike, cooldown) are retried next loop.
                bar_settled = False

                agi_meta = agi.predict(df, production=True)
                conf = float((agi_meta or {}).get("confidence", 0.0) or 0.0)
                regime = str((agi_meta or {}).get("regime", (agi_meta or {}).get("signal", "UNKNOWN")))
                trade_memory = trade_learning_by_symbol.get(str(symbol), {})

                # compact_v3 PPO inputs: the filter's reading and its 6000-bar history (same bars as training's warm-up)
                try:
                    if not isinstance(getattr(brain, "live_aux", None), dict):
                        brain.live_aux = {}
                    _mf = _move_filter_watch(symbol)
                    _hit = _MOVE_FILTER_CACHE.get(symbol)
                    brain.live_aux[symbol] = {"mf_p": (_mf or {}).get("p"), "df": _hit[2] if _hit and len(_hit) > 2 else None}
                except Exception as e:
                    logger.warning(f"PPO live inputs: move filter unavailable for {symbol}: {e}")
                    if isinstance(getattr(brain, "live_aux", None), dict):
                        brain.live_aux[symbol] = {}
                ppo_meta = brain.predict_ppo_action(symbol, df)
                dreamer_meta = brain.predict_dreamer_action(symbol, df)
                ctx = market_context(df)
                mem_mult = model_memory.multipliers(symbol, ctx["key"]) if memory_on else {}
                council_meta = None
                ind_votes = {}
                try:
                    if not council_allowed:
                        raise _CouncilOff()
                    htf_ = _htf_frames(symbol)
                    vt = votes_mtf(df.tail(1500), htf_)
                    try:  # why each indicator voted as it did (war room); never blocks the decision
                        council_readings[str(symbol)] = {"t": time.time(), "readings": readings_mtf(df.tail(1500), htf_)}
                        if time.time() - _readings_dumped[0] > 20:
                            _readings_dumped[0] = time.time()
                            _tmp = os.path.join(LOG_DIR, "council_readings.json.tmp")
                            with open(_tmp, "w", encoding="utf-8") as _f:
                                json.dump(council_readings, _f)
                            os.replace(_tmp, os.path.join(LOG_DIR, "council_readings.json"))
                    except Exception as _e:
                        logger.debug(f"council readings skipped for {symbol}: {_e}")
                    ind_votes = {k: float(vt[k].iloc[-1]) for k in vt.columns}
                    if ext_voters is not None and ext_voters.enabled:
                        try:
                            ind_votes.update(ext_voters.votes(symbol, df, _m15_long(symbol)))
                        except Exception as e:
                            logger.warning(f"external voters failed for {symbol}: {e}")
                    names_ = list(ind_votes)
                    ind_assess = model_memory.multipliers(symbol, ctx["key"], names=[f"ind:{k}" for k in names_])
                    council_meta = council_vote(ind_votes, {k: ind_assess.get(f"ind:{k}") for k in names_},
                                                weights=(ext_voters.weights() if ext_voters is not None else None))
                    council_meta["votes"] = ind_votes
                    council_meta["assess"] = {k: ind_assess.get(f"ind:{k}") for k in names_}
                except _CouncilOff:
                    pass
                except Exception as e:
                    logger.warning(f"indicator council failed for {symbol}: {e}")
                decision = _blend_symbol_decision(symbol, agi_meta, ppo_meta, dreamer_meta, cfg=cfg,
                                                  memory_mult=mem_mult, council_meta=council_meta)
                explore_scale = _exploration_scale(symbol)
                if explore_scale != 1.0:
                    decision["target"] = float(decision["target"]) * explore_scale
                    decision["exploration_scale"] = explore_scale
                # trade memory -> size: the market's own record of real closed trades now reaches the decision.
                # It can only make the bot more careful (never bigger than normal).
                tm_scale, tm_why = _trade_memory_scale(trade_memory, cfg)
                if tm_scale != 1.0:
                    decision["target"] = float(decision["target"]) * tm_scale
                decision["trade_memory_scale"] = tm_scale
                decision["trade_memory_why"] = tm_why
                exposure = float(decision["target"])
                spread_now = _tick_spread_bps(symbol)
                if spread_now is not None:
                    last_spread[symbol] = spread_now
                cost_spread = spread_now if spread_now is not None else last_spread.get(symbol)
                # baseline for every model: the big-move filter's GO/STOP is recorded with each bar so every
                # model (LSTM, PPO, council, each indicator, external voters) is also graded per GO / STOP
                mf = None
                try:
                    mf = _move_filter_watch(symbol)
                    if mf is not None:
                        _append_audit("move_filter", {"symbol": symbol, **mf, "blend": float(exposure),
                                                      "watch_only": True})
                except Exception as e:
                    logger.warning(f"move filter failed for {symbol}: {e}")
                try:
                    scored = model_memory.observe(
                        symbol, pd.Timestamp(df["time"].iloc[-1]).tz_localize("UTC") if "time" in df.columns else df.index[-1],
                        float(df["close"].iloc[-1]), ctx,
                        {
                            "ppo": None if ppo_meta is None else float(ppo_meta.get("target", 0.0) or 0.0),
                            "lstm": None if agi_meta is None else float(agi_meta.get("trend_bias", 0.0) or 0.0),
                            "dreamer": None if dreamer_meta is None else float(dreamer_meta.get("target", 0.0) or 0.0),
                            "blend": float(exposure),
                            "council": None if council_meta is None else float(council_meta["target"]),
                            **{f"ind:{k}": v for k, v in ind_votes.items()},
                        },
                        cost_bps=None if cost_spread is None else cost_spread + cost_gate["extra_cost_bps"],
                        extra_ctx=None if mf is None else ["mf:go" if mf["go"] else "mf:stop"],
                    )
                    _append_audit("model_memory", {"symbol": symbol, "context": ctx,
                                                   "scored": [r for r in scored if not str(r.get("model", "")).startswith("ind:")],
                                                   "multipliers": mem_mult, "enabled": memory_on})
                    if council_meta is not None:
                        _append_audit("council", {"symbol": symbol, "context": ctx.get("key"),
                                                  "target": council_meta["target"], "up": council_meta["up"],
                                                  "down": council_meta["down"], "flat": council_meta["flat"],
                                                  "ignored": council_meta["ignored"], "votes": council_meta["votes"],
                                                  "assess": council_meta["assess"]})
                        auto_changes.observe_council(symbol, ctx.get("key"), council_meta["ignored"], council_meta["assess"])
                        auto_changes.observe_slow(setup_gate.table() if setup_gate is not None else None)
                except Exception as e:
                    logger.warning(f"model memory update failed for {symbol}: {e}")

                logger.info(
                    "DECISION %s | regime=%s conf=%.4f risk=%.4f agi_bias=%.4f ppo=%.4f dreamer=%.4f raw=%.4f final=%.4f"
                    % (
                        symbol,
                        regime,
                        float((agi_meta or {}).get("confidence", 0.0) or 0.0),
                        _risk_scalar(agi_meta),
                        float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        float(decision["raw_target"]),
                        float(decision["target"]),
                    )
                )
                _append_audit(
                    "signal",
                    {
                        "symbol": symbol,
                        "regime": regime,
                        "confidence": conf,
                        "risk_scalar": _risk_scalar(agi_meta),
                        "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                        # what actually went into the blend (the LSTM is zeroed below its confidence bar)
                        "agi_counted": float(decision.get("agi_bias", 0.0) or 0.0),
                        "agi_confidence": float(decision.get("agi_confidence", 0.0) or 0.0),
                        "weights_used": {"lstm": decision.get("agi_weight_used"), "ppo": decision.get("ppo_weight_used"),
                                         "dreamer": decision.get("dreamer_weight_used"),
                                         "council": decision.get("council_weight_used")},
                        "council": float(decision.get("council_target", 0.0) or 0.0),
                        "ppo_exposure": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                        "dreamer_exposure": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                        "raw_target": float(decision["raw_target"]),
                        "exposure": float(exposure),
                        "decision_profile": dict(decision["profile"]),
                        "trade_memory_scale": decision.get("trade_memory_scale", 1.0),
                        "trade_memory_why": decision.get("trade_memory_why"),
                        "trade_memory": {
                            "trades": int(trade_memory.get("trades", 0) or 0),
                            "expectancy": float(trade_memory.get("expectancy", 0.0) or 0.0),
                            "profit_factor": float(trade_memory.get("profit_factor", 0.0) or 0.0),
                            "recent_loss_streak": int(trade_memory.get("recent_loss_streak", 0) or 0),
                        },
                    },
                )
                sym_state = last_symbol_state.setdefault(str(symbol), {})
                sym_state["signal"] = regime
                sym_state["regime"] = regime
                sym_state["confidence"] = conf
                sym_state["risk_scalar"] = _risk_scalar(agi_meta)
                sym_state["trend_bias"] = float((agi_meta or {}).get("trend_bias", 0.0) or 0.0)
                sym_state["ppo_exposure"] = float((ppo_meta or {}).get("target", 0.0) or 0.0)
                sym_state["dreamer_exposure"] = float((dreamer_meta or {}).get("target", 0.0) or 0.0)
                sym_state["raw_target"] = float(decision["raw_target"])
                sym_state["blend_exposure"] = float(exposure)

                action_meta = ppo_meta or brain.get_last_action_meta(symbol=symbol)
                stack = _stack_level(symbol, float(exposure), executor, stacking, stack_state)
                sym_state["stack"] = stack
                current_symbol_exposure, total_exposure, symbol_positions, total_positions = _position_exposure_state(
                    symbol, max_lots, executor
                )
                supervisor_decision = supervisor.allow_trade(
                    symbol=symbol,
                    target_exposure=float(exposure),
                    confidence=conf,
                    spread_bps=spread_now,
                    snapshot=guard_snap,
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
                if supervisor_decision.allowed and getattr(supervisor_decision, "target", None) is not None:
                    exposure = float(supervisor_decision.target)
                    sym_state["risk_supervisor"]["capped_target"] = exposure
                # ---- cost gate: open or add only while the bot's own trades in this market have
                # earned more than their spread (scored over the real holding time); closes always pass
                cur_x = float(current_symbol_exposure)
                held_by_gate = False  # a gate said "no more risk here": keep the open position exactly as it is
                if cost_gate["enabled"] and supervisor_decision.allowed and float(exposure) != 0.0 and (
                        cur_x == 0.0 or (float(exposure) > 0) != (cur_x > 0) or abs(float(exposure)) > abs(cur_x) + 1e-9):
                    _, tr_ = _cost_blocked(symbol)
                    proven_ = tr_["n_all"] >= model_memory.min_samples
                    sym_state["cost_gate"] = {"net_edge_bps": tr_["edge_bps"], "n": tr_["n_all"], "spread_bps": spread_now,
                                              "margin_bps": cost_gate["margin_bps"],
                                              "verdict": ("learning" if not proven_ else
                                                          "pass" if tr_["edge_bps"] > cost_gate["margin_bps"] else "block")}
                    if proven_ and tr_["edge_bps"] <= cost_gate["margin_bps"]:
                        exposure = 0.0 if (cur_x != 0.0 and (cur_x > 0) != (float(exposure) > 0)) else cur_x
                        held_by_gate = True
                        if time.time() - _gate_logged.get(f"cost:{symbol}", 0) > 900:
                            _gate_logged[f"cost:{symbol}"] = time.time()
                            logger.info(f"COST BLOCK {symbol} | trades net {tr_['edge_bps']:+.2f} bps after spread "
                                        f"over {tr_['n_all']:.0f} calls (needs > {cost_gate['margin_bps']:+.2f})")
                        if pending_entries.pop(symbol, None):
                            _append_audit("entry_cancelled", {"symbol": symbol, "reason": "cost gate"})
                # ---- setup gate: only proven setups trade normal size; losing ones are blocked
                gate_entry = None
                lot_cap = None
                cur_x = float(current_symbol_exposure)
                if supervisor_decision.allowed and float(exposure) != 0.0 and (
                        cur_x == 0.0 or (float(exposure) > 0) != (cur_x > 0) or abs(float(exposure)) > abs(cur_x) + 1e-9):
                    side_ = 1 if float(exposure) > 0 else -1
                    gkey = setup_gate.key(symbol, side_, conf, float((ppo_meta or {}).get("target", 0.0) or 0.0))
                    verdict, why = setup_gate.verdict(gkey)
                    sym_state["setup"] = {"key": gkey, "verdict": verdict, "why": why}
                    if verdict == "block":
                        exposure = 0.0 if (cur_x != 0.0 and (cur_x > 0) != (side_ > 0)) else cur_x
                        held_by_gate = True
                        if time.time() - _gate_logged.get(gkey, 0) > 900:
                            _gate_logged[gkey] = time.time()
                            logger.info(f"SETUP BLOCK {symbol} | {why}")
                        if pending_entries.pop(symbol, None):
                            _append_audit("entry_cancelled", {"symbol": symbol, "reason": "setup blocked"})
                    else:
                        gate_entry = (gkey, verdict)
                        if verdict in ("probe", "retest"):
                            info_ = executor._symbol_info(symbol)
                            vmin_ = float(getattr(info_, "volume_min", 0.01) or 0.01)
                            same_ = cur_x != 0.0 and (cur_x > 0) == (side_ > 0)
                            lot_cap = (abs(cur_x) * float(max_lots) if same_ else 0.0) + vmin_
                # ---- trend gate: no new risk against BOTH the 4h and 1h trend (closes always pass)
                cur_x = float(current_symbol_exposure)
                if trend_gate.enabled and supervisor_decision.allowed and float(exposure) != 0.0 and (
                        cur_x == 0.0 or (float(exposure) > 0) != (cur_x > 0) or abs(float(exposure)) > abs(cur_x) + 1e-9):
                    side_ = 1 if float(exposure) > 0 else -1
                    tv_, twhy_, ttr_ = trend_gate.check(symbol, side_)
                    sym_state["trend_gate"] = {"verdict": tv_, "why": twhy_, "trends": ttr_}
                    if tv_ == "block":
                        exposure = 0.0 if (cur_x != 0.0 and (cur_x > 0) != (side_ > 0)) else cur_x
                        held_by_gate = True
                        gate_entry = None
                        if time.time() - _gate_logged.get(f"trend:{symbol}:{side_}", 0) > 900:
                            _gate_logged[f"trend:{symbol}:{side_}"] = time.time()
                            logger.info(f"TREND BLOCK {symbol} | {twhy_}")
                        if pending_entries.pop(symbol, None):
                            _append_audit("entry_cancelled", {"symbol": symbol, "reason": "trend gate"})
                skip_ = entry_skip.get(symbol)
                if skip_ and skip_[1] > time.time() and float(exposure) != 0.0 and (float(exposure) > 0) == (skip_[0] > 0) \
                        and gate_entry is not None and pending_entries.get(symbol) is None:
                    exposure = cur_x if (cur_x == 0.0 or (cur_x > 0) == (skip_[0] > 0)) else 0.0
                    gate_entry = None
                # Holding: sending cur_x to the executor would re-size it (add with stacking, trim as equity falls)
                hold_only = held_by_gate and float(exposure) != 0.0 and abs(float(exposure) - cur_x) < 1e-12
                if hold_only and pending_entries.pop(symbol, None):
                    _append_audit("entry_cancelled", {"symbol": symbol, "reason": "gate: hold only"})
                pending_now = False
                if supervisor_decision.allowed and smart_entry["enabled"] and not hold_only:
                    pend = pending_entries.get(symbol)
                    cur_lots = abs(float(current_symbol_exposure)) * float(max_lots)
                    want_lots = executor.normalize_volume(
                        symbol, executor.risk_capped_lots(symbol, abs(float(exposure)) * float(max_lots)) * stack) if exposure else 0.0
                    same_side = float(current_symbol_exposure) == 0.0 or (float(exposure) > 0) == (float(current_symbol_exposure) > 0)
                    opening = float(exposure) != 0.0 and same_side and want_lots > cur_lots + 1e-9
                    if pend and (not opening or (float(exposure) > 0) != (pend["side"] > 0)):
                        pending_entries.pop(symbol, None)
                        _append_audit("entry_cancelled", {"symbol": symbol, "reason": "signal changed"})
                        pend = None
                    if opening:
                        if pend is None:
                            pend = _plan_entry(symbol, float(exposure), executor, smart_entry)
                            if pend is not None:
                                pending_entries[symbol] = pend
                                _append_audit("entry_planned", {"symbol": symbol, "side": "BUY" if pend["side"] > 0 else "SELL",
                                                                "ref": pend["ref"], "limit": pend["limit"], "k_atr": pend["k"],
                                                                "expires": pend["expires"]})
                                logger.info(f"ENTRY PLAN {symbol} | waiting up to {smart_entry['wait_min']:.0f} min for "
                                            f"{pend['limit']:.5g} (now {pend['ref']:.5g}, {pend['k']:.2f} ATR pullback)")
                        if pend is not None:
                            pend.update(exposure=float(exposure), max_lots=max_lots, action_meta=action_meta, ctx={
                                "regime": regime, "confidence": conf, "target_exposure": float(exposure),
                                "raw_target": float(decision["raw_target"]),
                                "ppo_target": float((ppo_meta or {}).get("target", 0.0) or 0.0),
                                "dreamer_target": float((dreamer_meta or {}).get("target", 0.0) or 0.0),
                                "agi_bias": float((agi_meta or {}).get("trend_bias", 0.0) or 0.0),
                                "agi_risk_scalar": _risk_scalar(agi_meta), "stack": stack, "lot_cap": lot_cap})
                            pend["gate"] = gate_entry
                            pending_now = True
                news_block = news_trader.blocks_regular(symbol)
                if news_block:
                    if pending_entries.pop(symbol, None):
                        _append_audit("entry_cancelled", {"symbol": symbol, "reason": news_block})
                    sym_state["news_block"] = news_block
                    order_meta = None
                    bar_settled = True
                elif hold_only:
                    order_meta = None
                    bar_settled = True
                elif pending_now:
                    order_meta = None
                    bar_settled = True
                elif supervisor_decision.allowed:
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
                            "agi_risk_scalar": _risk_scalar(agi_meta),
                            "stack": stack,
                            "lot_cap": lot_cap,
                        },
                    )
                    if order_meta and order_meta.get("executed") and gate_entry \
                            and str(order_meta.get("request_action", "")) not in ("close", "reduce"):
                        setup_gate.record_open(symbol, 1 if float(exposure) > 0 else -1, *gate_entry)
                    req_action = str((order_meta or {}).get("request_action", "") or "")
                    bar_settled = bool(order_meta and (order_meta.get("executed") or req_action in ("noop", "blocked")))
                    if order_meta and order_meta.get("executed"):
                        supervisor.mark_trade(symbol)
                    elif order_meta:
                        # Nothing was sent (noop/blocked): no cooldown, no trade alert.
                        order_meta = None
                else:
                    order_meta = None
                    bar_settled = not str(supervisor_decision.reason).startswith(("spread_bps", "cooldown"))
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
                if bar_settled and bar_time is not None:
                    last_decided_bar[symbol] = bar_time
                executor.manage_open_positions(symbol)
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

        news_active = any(ev["phase"] in ("prep", "armed", "in_trade") for ev in news_trader.events.values())
        time.sleep(5 if (pending_entries or news_active) else max(5, loop_sleep_sec))


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)
