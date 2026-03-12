import atexit
import datetime
import json
import os
import subprocess
import time

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from Python.agi_brain import SmartAGI
from Python.hybrid_brain import HybridBrain
from Python.mt5_executor import MT5Executor
from Python.risk_engine import RiskEngine
from Python.risk_supervisor import RiskSupervisor
from Python.config_utils import load_project_config
from Python.event_intel import EventIntel
from Python.trade_learning import build_trade_learning
from alerts.telegram_alerts import TelegramAlerter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "server_agi.lock")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
AUDIT_LOG = os.path.join(LOG_DIR, "audit_events.jsonl")
TRADE_EVENTS_LOG = os.path.join(LOG_DIR, "trade_events.jsonl")
ACTIVE_MODELS_PATH = os.path.join(BASE_DIR, "models", "registry", "active.json")

os.makedirs(LOG_DIR, exist_ok=True)
logger.add(SERVER_LOG, rotation="10 MB", level="INFO")


def _utc_now() -> datetime.datetime:
    return datetime.datetime.now(datetime.timezone.utc)


def _json_default(v):
    if isinstance(v, (datetime.datetime, datetime.date)):
        return v.isoformat()
    return str(v)


def _append_jsonl(path: str, row: dict):
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
        return False

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


def _low_volatility_memory_base(trade_memory: dict | None) -> float:
    memory = trade_memory if isinstance(trade_memory, dict) else {}
    trades = int(memory.get("trades", 0) or 0)
    expectancy = float(memory.get("expectancy", 0.0) or 0.0)
    profit_factor = float(memory.get("profit_factor", 0.0) or 0.0)
    recent_loss_streak = int(memory.get("recent_loss_streak", 0) or 0)

    min_trades = int(os.environ.get("AGI_LOW_VOL_MIN_TRADES", "20"))
    min_profit_factor = float(os.environ.get("AGI_LOW_VOL_MIN_PROFIT_FACTOR", "1.15"))
    min_expectancy = float(os.environ.get("AGI_LOW_VOL_MIN_EXPECTANCY", "0.0"))
    max_recent_loss_streak = int(os.environ.get("AGI_LOW_VOL_MAX_RECENT_LOSS_STREAK", "3"))
    base_floor = float(os.environ.get("AGI_LOW_VOL_BASE_EXPOSURE", "0.08"))
    base_cap = float(os.environ.get("AGI_LOW_VOL_MAX_EXPOSURE", "0.16"))

    if trades < min_trades:
        return 0.0
    if expectancy <= min_expectancy:
        return 0.0
    if profit_factor < min_profit_factor:
        return 0.0
    if recent_loss_streak > max_recent_loss_streak:
        return 0.0

    expectancy_bonus = min(0.04, expectancy / 250.0)
    profit_factor_bonus = min(0.04, max(0.0, profit_factor - min_profit_factor) / 5.0)
    return float(min(base_cap, base_floor + expectancy_bonus + profit_factor_bonus))


def _signal_to_exposure(
    signal: str,
    confidence: float,
    closes,
    trade_memory: dict | None = None,
    low_vol_base: float | None = None,
):
    if len(closes) < 8:
        return 0.0

    momentum = (closes[-1] - closes[-8]) / (abs(closes[-8]) + 1e-12)
    direction = 1.0 if momentum >= 0 else -1.0

    if signal == "LOW_VOLATILITY":
        base = _low_volatility_memory_base(trade_memory) if low_vol_base is None else float(low_vol_base)
    elif signal == "MED_VOLATILITY":
        base = 0.35
    else:
        base = 0.65

    conf_scale = min(1.0, max(0.0, float(confidence)))
    return direction * base * conf_scale


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


def main(live=False):
    if not _acquire_single_instance_lock():
        raise RuntimeError("Server_AGI is already running (lock file exists)")

    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    cfg = _load_cfg(live=live)
    ok = _init_mt5(cfg)
    if not ok:
        raise RuntimeError(f"MT5 init failed: {mt5.last_error()}")

    risk = RiskEngine()
    supervisor = RiskSupervisor(cfg)
    executor = MT5Executor(risk)
    brain = HybridBrain(risk, executor)
    agi = SmartAGI()

    trading_cfg = cfg.get("trading", {})
    symbols = trading_cfg.get("symbols", ["EURUSDm", "GBPUSDm"])
    timeframe = _to_mt5_timeframe(trading_cfg.get("timeframe", "M5"))
    confidence_threshold = float(trading_cfg.get("confidence_threshold", 0.85))
    max_lots = float(cfg.get("risk", {}).get("max_lots", 1.0))

    token, chat_id = _load_telegram_cfg(cfg)
    alerter = TelegramAlerter(token, chat_id)
    event_intel = EventIntel(cfg, LOG_DIR)

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

    while True:
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
                learn = build_trade_learning(
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

        for symbol in symbols:
            try:
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue

                pred = agi.predict(df, production=True)
                conf = float(pred.get("confidence", 0.0))
                sig = str(pred.get("signal", "LOW_VOLATILITY"))
                trade_memory = trade_learning_by_symbol.get(str(symbol), {})
                low_vol_base = _low_volatility_memory_base(trade_memory) if sig == "LOW_VOLATILITY" else 0.0

                agi_exposure = 0.0
                if conf >= confidence_threshold:
                    agi_exposure = _signal_to_exposure(
                        sig,
                        conf,
                        df["close"].values,
                        trade_memory=trade_memory,
                        low_vol_base=low_vol_base,
                    )

                ppo_exposure = brain.predict_ppo_exposure(symbol, df)
                dreamer_exposure = brain.predict_dreamer_exposure(symbol, df)
                exposure = brain.blend_exposure(agi_exposure, ppo_exposure, conf, dreamer_exposure=dreamer_exposure)
                if sig == "LOW_VOLATILITY":
                    if low_vol_base <= 0.0:
                        exposure = 0.0
                    else:
                        exposure = max(-low_vol_base, min(low_vol_base, float(exposure)))

                logger.info(
                    f"DECISION {symbol} | signal={sig} conf={conf:.4f} agi={agi_exposure:.4f} "
                    f"ppo={(0.0 if ppo_exposure is None else float(ppo_exposure)):.4f} "
                    f"dreamer={(0.0 if dreamer_exposure is None else float(dreamer_exposure)):.4f} "
                    f"blend={exposure:.4f}"
                )
                _append_audit(
                    "signal",
                    {
                        "symbol": symbol,
                        "signal": sig,
                        "confidence": conf,
                        "agi_exposure": float(agi_exposure),
                        "ppo_exposure": None if ppo_exposure is None else float(ppo_exposure),
                        "dreamer_exposure": None if dreamer_exposure is None else float(dreamer_exposure),
                        "exposure": float(exposure),
                        "threshold": float(confidence_threshold),
                        "trade_memory": {
                            "trades": int(trade_memory.get("trades", 0) or 0),
                            "expectancy": float(trade_memory.get("expectancy", 0.0) or 0.0),
                            "profit_factor": float(trade_memory.get("profit_factor", 0.0) or 0.0),
                            "recent_loss_streak": int(trade_memory.get("recent_loss_streak", 0) or 0),
                            "low_vol_base": float(low_vol_base),
                        },
                    },
                )
                sym_state = last_symbol_state.setdefault(str(symbol), {})
                sym_state["signal"] = sig
                sym_state["confidence"] = conf
                sym_state["agi_exposure"] = float(agi_exposure)
                sym_state["ppo_exposure"] = None if ppo_exposure is None else float(ppo_exposure)
                sym_state["dreamer_exposure"] = None if dreamer_exposure is None else float(dreamer_exposure)
                sym_state["blend_exposure"] = float(exposure)

                action_meta = brain.get_last_action_meta()
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
                if supervisor_decision.allowed:
                    order_meta = brain.live_trade(symbol, exposure, max_lots, action_meta=action_meta)
                    if order_meta:
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
                            "signal": sig,
                        },
                    )
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
                    logger.info(
                        f"ACTION {symbol} | mode={order_meta.get('entry_mode')} "
                        f"volume={order_meta.get('volume_lots')} "
                        f"TP={order_meta.get('tp_price')} SL={order_meta.get('sl_price')}"
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


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)

