import json
import os
import socket
import threading
import time

import MetaTrader5 as mt5
import pandas as pd

from Python.data_feed import fetch_training_data
from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.hybrid_brain import HybridBrain
from alerts.telegram_alerts import TelegramAlerter


def _ok(payload: dict) -> bytes:
    return (json.dumps(payload) + "\n").encode("utf-8")


def _parse_request(raw: bytes) -> dict:
    try:
        return json.loads(raw.decode("utf-8", errors="replace").strip())
    except Exception:
        return {}


def _heartbeat_loop(risk: RiskEngine, alerter: TelegramAlerter):
    start_time = time.time()
    while True:
        uptime = int(time.time() - start_time)
        alerter.heartbeat(
            uptime=f"{uptime} sec",
            mt5_connected=mt5.initialize(),
            trading_enabled=not risk.halt,
        )
        time.sleep(120)


def _build_df(symbol: str, timeframe: str = "M5") -> pd.DataFrame:
    df = fetch_training_data(symbol, period="10d", timeframe=timeframe)
    if df is None:
        return pd.DataFrame()
    if "symbol" not in df.columns and not df.empty:
        df = df.copy()
        df["symbol"] = symbol
    return df


def main(live=False):
    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    host = os.environ.get("AGI_HOST", "0.0.0.0")
    port = int(os.environ.get("AGI_PORT", "9090"))
    token = os.environ.get("AGI_TOKEN", "").strip()

    mt5.initialize()

    risk = RiskEngine()
    executor = MT5Executor(risk)
    brain = HybridBrain(risk, executor)

    alerter = TelegramAlerter(
        os.environ.get("TELEGRAM_TOKEN"),
        os.environ.get("TELEGRAM_CHAT_ID"),
    )
    threading.Thread(target=_heartbeat_loop, args=(risk, alerter), daemon=True).start()

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((host, port))
    server.listen(32)

    while True:
        conn, _addr = server.accept()
        with conn:
            raw = b""
            while b"\n" not in raw:
                part = conn.recv(4096)
                if not part:
                    break
                raw += part

            req = _parse_request(raw)
            if not req:
                conn.sendall(_ok({"action": "ERROR", "error": "Invalid JSON request"}))
                continue

            if token and req.get("token", "").strip() != token:
                conn.sendall(_ok({"action": "ERROR", "error": "Unauthorized"}))
                continue

            action = str(req.get("action", "health")).lower()
            symbol = str(req.get("symbol", "EURUSDm"))

            if action == "health":
                conn.sendall(_ok({"action": "HEALTH", "mt5_connected": bool(mt5.initialize()), "halt": risk.halt}))
                continue

            if action == "risk_status":
                conn.sendall(_ok({
                    "action": "RISK_STATUS",
                    "halt": risk.halt,
                    "daily_trades": risk.daily_trades,
                    "max_daily_trades": risk.max_daily_trades,
                    "realized_pnl_today": risk.realized_pnl_today,
                    "max_daily_loss": risk.max_daily_loss,
                }))
                continue

            df = _build_df(symbol)
            if df.empty:
                conn.sendall(_ok({"action": "ERROR", "symbol": symbol, "error": "No market data"}))
                continue

            if action == "predict":
                decision = brain.predict(symbol, df)
                conn.sendall(_ok(decision))
                continue

            if action == "trade":
                decision = brain.decide_and_trade(symbol, df)
                conn.sendall(_ok(decision))
                continue

            conn.sendall(_ok({"action": "ERROR", "error": f"Unsupported action: {action}"}))


if __name__ == "__main__":
    import sys

    live_flag = "--live" in sys.argv
    main(live=live_flag)
