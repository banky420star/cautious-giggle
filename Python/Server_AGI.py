import atexit
import os
import time

import MetaTrader5 as mt5
import pandas as pd
import yaml

from Python.agi_brain import SmartAGI
from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.hybrid_brain import HybridBrain
from alerts.telegram_alerts import TelegramAlerter

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOCK_DIR = os.path.join(BASE_DIR, ".tmp")
LOCK_PATH = os.path.join(LOCK_DIR, "server_agi.lock")


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


def _load_cfg():
    cfg_path = os.path.join(BASE_DIR, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _load_telegram_cfg(cfg):
    token = os.environ.get("TELEGRAM_TOKEN") or cfg.get("telegram", {}).get("token")
    chat_id = os.environ.get("TELEGRAM_CHAT_ID") or cfg.get("telegram", {}).get("chat_id")

    if token in ("", "YOUR_BOT_TOKEN_HERE"):
        token = None
    if chat_id in ("", "YOUR_CHAT_ID_HERE"):
        chat_id = None

    return token, chat_id


def _init_mt5(cfg):
    mt5_cfg = cfg.get("mt5", {})
    login = int(os.environ.get("MT5_LOGIN", mt5_cfg.get("login", 0)) or 0)
    password = os.environ.get("MT5_PASSWORD", mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER", mt5_cfg.get("server", ""))

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


def main(live=False):
    if not _acquire_single_instance_lock():
        raise RuntimeError("Server_AGI is already running (lock file exists)")

    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    cfg = _load_cfg()
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
            last_heartbeat = now

        for symbol in symbols:
            try:
                df = _fetch_symbol_df(symbol, timeframe)
                if df is None or df.empty:
                    continue

                pred = agi.predict(df, production=True)
                conf = float(pred.get("confidence", 0.0))
                sig = str(pred.get("signal", "LOW_VOLATILITY"))

                exposure = 0.0
                if conf >= confidence_threshold:
                    exposure = _signal_to_exposure(sig, conf, df["close"].values)

                brain.live_trade(symbol, exposure, max_lots)
                executor.manage_open_positions(symbol)

                acc = mt5.account_info()
                if acc:
                    risk.update_equity(float(acc.equity))
                    if abs(exposure) > 0.01:
                        action = "BUY" if exposure > 0 else "SELL"
                        alerter.trade(
                            symbol=symbol,
                            action=action,
                            exposure=abs(exposure),
                            confidence=conf,
                            balance=float(acc.balance),
                            equity=float(acc.equity),
                            free_margin=float(acc.margin_free),
                        )
            except Exception as exc:
                risk.record_error()
                alerter.alert(f"Execution loop error on {symbol}: {exc}")

        time.sleep(max(5, loop_sleep_sec))


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)
