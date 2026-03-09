import atexit
import datetime
import json
import os
import time

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

from Python.agi_brain import SmartAGI
from Python.hybrid_brain import HybridBrain
from Python.mt5_executor import MT5Executor
from Python.risk_engine import RiskEngine
from Python.config_utils import load_project_config
from alerts.telegram_alerts import TelegramAlerter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "server_agi.lock")
LOG_DIR = os.path.join(BASE_DIR, "logs")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
AUDIT_LOG = os.path.join(LOG_DIR, "audit_events.jsonl")
TRADE_EVENTS_LOG = os.path.join(LOG_DIR, "trade_events.jsonl")

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


def _signal_to_exposure(signal: str, confidence: float, closes):
    if len(closes) < 8:
        return 0.0

    momentum = (closes[-1] - closes[-8]) / (abs(closes[-8]) + 1e-12)
    direction = 1.0 if momentum >= 0 else -1.0

    if signal == "LOW_VOLATILITY":
        base = 0.0
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


def _scan_trade_events(alerter, known_open_tickets, seen_closed_deals, last_deal_check):
    now_utc = _utc_now()

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
            alerter.trade_closed(
                symbol=payload["symbol"],
                ticket=payload["ticket"],
                pnl=pnl,
                volume=payload["volume"],
                price=payload["price"],
            )
        except Exception:
            continue

    if len(seen_closed_deals) > 20000:
        seen_closed_deals = set(sorted(seen_closed_deals)[-10000:])

    return current_tickets, seen_closed_deals, now_utc - datetime.timedelta(seconds=3)


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
    heartbeat_sec = int(os.environ.get("AGI_HEARTBEAT_SEC", "60"))
    loop_sleep_sec = int(os.environ.get("AGI_LOOP_SEC", "20"))
    last_heartbeat = 0.0

    while True:
        now = time.time()

        if now - last_heartbeat >= max(15, heartbeat_sec):
            uptime = int(now - start_time)
            acc = mt5.account_info()
            if acc:
                risk.update_equity(float(acc.equity))

            alerter.heartbeat(
                uptime=str(uptime) + " sec",
                mt5_connected=mt5.initialize(),
                trading_enabled=not risk.halt,
            )

            snap = _account_snapshot()
            alerter.snapshot(
                balance=0.0 if snap["balance"] is None else snap["balance"],
                equity=0.0 if snap["equity"] is None else snap["equity"],
                pnl_today=snap["pnl_today"],
                floating=snap["floating"],
                open_positions=snap["open_positions"],
            )
            _append_audit("snapshot", snap)
            last_heartbeat = now

        for symbol in symbols:
            try:
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue

                pred = agi.predict(df, production=True)
                conf = float(pred.get("confidence", 0.0))
                sig = str(pred.get("signal", "LOW_VOLATILITY"))

                agi_exposure = 0.0
                if conf >= confidence_threshold:
                    agi_exposure = _signal_to_exposure(sig, conf, df["close"].values)

                ppo_exposure = brain.predict_ppo_exposure(symbol, df)
                exposure = brain.blend_exposure(agi_exposure, ppo_exposure, conf)

                logger.info(
                    f"DECISION {symbol} | signal={sig} conf={conf:.4f} agi={agi_exposure:.4f} "
                    f"ppo={(0.0 if ppo_exposure is None else float(ppo_exposure)):.4f} blend={exposure:.4f}"
                )
                _append_audit(
                    "signal",
                    {
                        "symbol": symbol,
                        "signal": sig,
                        "confidence": conf,
                        "agi_exposure": float(agi_exposure),
                        "ppo_exposure": None if ppo_exposure is None else float(ppo_exposure),
                        "exposure": float(exposure),
                        "threshold": float(confidence_threshold),
                    },
                )

                action_meta = brain.get_last_action_meta()
                order_meta = brain.live_trade(symbol, exposure, max_lots, action_meta=action_meta)
                executor.manage_open_positions(symbol)
                if order_meta:
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

        known_open_tickets, seen_closed_deals, last_deal_check = _scan_trade_events(
            alerter,
            known_open_tickets,
            seen_closed_deals,
            last_deal_check,
        )

        time.sleep(max(5, loop_sleep_sec))


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)


