"""
Server AGI — Main engine with socket server, risk polling, and autonomy loop.
Works on both Windows (MT5 live) and Mac (dry-run dev mode).

Usage:
  python -m Python.Server_AGI          # dry-run dev mode
  python -m Python.Server_AGI --live   # live trading (Windows + MT5 only)
"""
import os
import sys
import time
import json
import socket
import asyncio
import threading
from datetime import datetime
from loguru import logger

# ── Conditional MT5 import ──────────────────────────────────────────
_mt5 = None
if sys.platform == "win32":
    try:
        import MetaTrader5 as mt5
        _mt5 = mt5
    except ImportError:
        pass

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from Python.risk_engine import RiskEngine
from Python.mt5_executor import MT5Executor
from Python.hybrid_brain import HybridBrain
from Python.data_feed import get_latest_data, fetch_training_data

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "server_agi.log"), rotation="10 MB", level="INFO")


class AGIServer:
    """Main server engine that coordinates brain, risk, and execution."""

    def __init__(self, live: bool = False):
        self.live = live
        if live:
            os.environ["AGI_IS_LIVE"] = "1"

        # Initialize MT5 if available
        if _mt5 is not None and live:
            if not _mt5.initialize():
                logger.error("MT5 failed to initialize — running in dry-run mode")
                self.live = False
            else:
                logger.success("MT5 connected successfully")

        # Core components
        self.risk = RiskEngine()
        self.executor = MT5Executor(self.risk)
        self.brain = HybridBrain(self.risk, self.executor)
        self.risk_engine = self.risk  # Alias for AutonomyLoop compatibility

        # Socket server config
        self.host = os.environ.get("AGI_HOST", "127.0.0.1")
        self.port = int(os.environ.get("AGI_PORT", "9090"))
        self.token = os.environ.get("AGI_TOKEN", "").strip()

        # Trading symbols from config
        import yaml
        config_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "config.yaml")
        try:
            with open(config_path) as f:
                cfg = yaml.safe_load(f)
            self.symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
        except Exception:
            self.symbols = ["EURUSD"]

        self.start_time = time.time()
        logger.success(f"AGIServer initialized | live={self.live} | symbols={self.symbols}")

    def handle_command(self, request: dict) -> dict:
        """Process a command from the socket server or n8n bridge."""
        # Token auth
        if self.token and request.get("token") != self.token:
            return {"error": "Invalid token", "action": "ERROR"}

        command = request.get("action", "").lower()
        symbol = request.get("symbol", self.symbols[0])

        if command == "predict":
            return self._handle_predict(symbol)
        elif command == "trade":
            return self._handle_trade(symbol)
        elif command == "health":
            return self._handle_health()
        elif command == "risk_status":
            return self._handle_risk_status()
        else:
            return {"error": f"Unknown command: {command}", "action": "ERROR"}

    def _handle_predict(self, symbol: str) -> dict:
        """Get prediction without executing."""
        try:
            df = fetch_training_data(symbol, period="5d", interval="5m")
            if df is None or df.empty or len(df) < 100:
                return {"error": f"Insufficient data for {symbol}", "action": "ERROR"}

            decision = self.brain.decide(symbol, df)
            return decision
        except Exception as e:
            return {"error": str(e), "action": "ERROR"}

    def _handle_trade(self, symbol: str) -> dict:
        """Get prediction and execute."""
        try:
            df = fetch_training_data(symbol, period="5d", interval="5m")
            if df is None or df.empty or len(df) < 100:
                return {"error": f"Insufficient data for {symbol}", "action": "ERROR"}

            decision = self.brain.live_trade(symbol, df)
            return decision if decision else {"action": "HOLD", "reason": "risk_blocked"}
        except Exception as e:
            return {"error": str(e), "action": "ERROR"}

    def _handle_health(self) -> dict:
        uptime = int(time.time() - self.start_time)
        mt5_ok = _mt5 is not None and _mt5.initialize() if _mt5 else False
        return {
            "status": "OK",
            "action": "HEALTH",
            "uptime_sec": uptime,
            "mt5_connected": mt5_ok,
            "trading_enabled": not self.risk.halt,
            "daily_trades": self.risk.daily_trades,
            "realized_pnl": self.risk.realized_pnl_today,
            "mode": "LIVE" if self.live else "DRY-RUN",
        }

    def _handle_risk_status(self) -> dict:
        return {
            "action": "RISK_STATUS",
            "halt": self.risk.halt,
            "daily_trades": self.risk.daily_trades,
            "max_daily_trades": self.risk.max_daily_trades,
            "realized_pnl": self.risk.realized_pnl_today,
            "max_daily_loss": self.risk.max_daily_loss,
            "current_dd": self.risk.current_dd,
            "can_trade": self.risk.can_trade(),
        }

    def run_socket_server(self):
        """Run the TCP socket server for n8n bridge communication."""
        server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        server.bind((self.host, self.port))
        server.listen(5)
        server.settimeout(1.0)  # Allow periodic checking
        logger.success(f"Socket server listening on {self.host}:{self.port}")

        while True:
            try:
                conn, addr = server.accept()
                threading.Thread(target=self._handle_connection, args=(conn, addr), daemon=True).start()
            except socket.timeout:
                continue
            except Exception as e:
                logger.error(f"Socket server error: {e}")
                time.sleep(1)

    def _handle_connection(self, conn, addr):
        try:
            data = b""
            while b"\n" not in data:
                chunk = conn.recv(4096)
                if not chunk:
                    break
                data += chunk

            raw = data.decode("utf-8", errors="replace").strip()
            if not raw:
                return

            request = json.loads(raw.split("\n")[0])
            response = self.handle_command(request)
            conn.sendall((json.dumps(response) + "\n").encode("utf-8"))

        except Exception as e:
            error_resp = json.dumps({"error": str(e), "action": "ERROR"}) + "\n"
            try:
                conn.sendall(error_resp.encode("utf-8"))
            except Exception:
                pass
        finally:
            conn.close()


def main(live=False):
    server = AGIServer(live=live)

    # Start socket server in background
    socket_thread = threading.Thread(target=server.run_socket_server, daemon=True)
    socket_thread.start()

    # Optionally start autonomy loop
    try:
        from Python.autonomy_loop import AutonomyLoop
        autonomy = AutonomyLoop(server)

        async def run_autonomy():
            await autonomy.start()

        logger.info("Starting AutonomyLoop...")
        asyncio.run(run_autonomy())

    except KeyboardInterrupt:
        logger.info("Server shutting down...")
    except Exception as e:
        logger.error(f"Autonomy loop error: {e}")
        # Fall back to simple heartbeat loop
        logger.info("Running in simple heartbeat mode...")
        while True:
            uptime = int(time.time() - server.start_time)
            logger.debug(f"Heartbeat: uptime={uptime}s trades={server.risk.daily_trades}")
            time.sleep(120)


if __name__ == "__main__":
    live_flag = "--live" in sys.argv or "--production" in sys.argv
    main(live=live_flag)
