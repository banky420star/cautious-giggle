import os
import time
import json
import socket
import threading
import MetaTrader5 as mt5
from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.hybrid_brain import HybridBrain
from alerts.telegram_alerts import TelegramAlerter


def _json_response(conn, payload: dict):
    conn.sendall((json.dumps(payload) + "\n").encode("utf-8"))


def _handle_request(req: dict, brain: HybridBrain, risk: RiskEngine):
    action = str(req.get("action", "")).lower()
    symbol = req.get("symbol", "EURUSDm")

    if action == "health":
        return {
            "action": "HEALTH",
            "mt5_connected": bool(mt5.initialize()),
            "trading_enabled": bool(risk.can_trade()),
            "daily_trades": int(risk.daily_trades),
            "halt": bool(risk.halt),
        }

    if action == "risk_status":
        return {
            "action": "RISK_STATUS",
            "max_daily_loss": risk.max_daily_loss,
            "max_daily_trades": risk.max_daily_trades,
            "max_lots": risk.max_lots,
            "realized_pnl_today": risk.realized_pnl_today,
            "daily_trades": risk.daily_trades,
            "halt": risk.halt,
        }

    if action == "predict":
        result = brain.trade_cycle(symbol, execute=False)
        result["request_action"] = action
        return result

    if action == "trade":
        result = brain.trade_cycle(symbol, execute=True)
        result["request_action"] = action
        return result

    return {"action": "ERROR", "error": f"Unknown action '{action}'", "symbol": symbol}


def _client_loop(conn, addr, brain: HybridBrain, risk: RiskEngine, token: str):
    with conn:
        try:
            buf = b""
            while b"\n" not in buf and len(buf) < 65536:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                buf += chunk

            raw = buf.decode("utf-8", errors="replace").strip()
            if not raw:
                _json_response(conn, {"action": "ERROR", "error": "Empty request"})
                return

            req = json.loads(raw.splitlines()[0])
            if token:
                provided = str(req.get("token", "")).strip()
                if provided != token:
                    _json_response(conn, {"action": "ERROR", "error": "Invalid token"})
                    return

            resp = _handle_request(req, brain, risk)
            _json_response(conn, resp)
        except Exception as e:
            _json_response(conn, {"action": "ERROR", "error": str(e), "peer": str(addr)})


def _start_socket_server(host: str, port: int, brain: HybridBrain, risk: RiskEngine, token: str):
    srv = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    srv.bind((host, port))
    srv.listen(64)

    while True:
        conn, addr = srv.accept()
        th = threading.Thread(target=_client_loop, args=(conn, addr, brain, risk, token), daemon=True)
        th.start()


def main(live=False):
    if live:
        os.environ["AGI_IS_LIVE"] = "1"

    mt5.initialize()

    risk = RiskEngine()
    executor = MT5Executor(risk)
    brain = HybridBrain(risk, executor)

    alerter = TelegramAlerter(
        os.environ.get("TELEGRAM_TOKEN"),
        os.environ.get("TELEGRAM_CHAT_ID")
    )

    host = os.environ.get("AGI_HOST", "0.0.0.0")
    port = int(os.environ.get("AGI_PORT", "9090"))
    token = os.environ.get("AGI_TOKEN", "").strip()

    socket_thread = threading.Thread(
        target=_start_socket_server,
        args=(host, port, brain, risk, token),
        daemon=True,
    )
    socket_thread.start()

    start_time = time.time()

    while True:
        uptime = int(time.time() - start_time)

        alerter.heartbeat(
            uptime=str(uptime) + " sec",
            mt5_connected=mt5.initialize(),
            trading_enabled=not risk.halt
        )

        time.sleep(120)


if __name__ == "__main__":
    import sys
    live_flag = "--live" in sys.argv
    main(live=live_flag)
