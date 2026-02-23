import socket, select, os, sys, pandas as pd, numpy as np
from datetime import datetime
from loguru import logger

# Add parent directory to path so imports work
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# ── Full file logging ───────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
LOG_FILE = os.path.join(LOG_DIR, f"agi_server_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logger.add(LOG_FILE, rotation="10 MB", retention="7 days", level="DEBUG",
           format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {name}:{function}:{line} - {message}")
logger.info(f"Full log file: {LOG_FILE}")
# ────────────────────────────────────────────────────────────────────

from Python.agi_brain import SmartAGI
from Python.risk_engine import RiskEngine
from Python.data_feed import fetch_realtime

class AGIServer:
    def __init__(self, host='127.0.0.1', port=9090):
        self.agi = SmartAGI()
        self.risk = RiskEngine()
        self.excel_path = os.path.expanduser("~/Documents/cautious-giggle/control.xlsx")
        self.trade_count = 0
        self.hold_count = 0
        self.data_cache = {}       # cache real data per symbol
        self.cache_time = {}       # track cache freshness
        self.server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.server.bind((host, port))
        self.server.listen(5)
        logger.success(f"AGI Server LIVE on {host}:{port}")
        logger.info(f"Excel path: {self.excel_path}")
        logger.info(f"Logging to: {LOG_FILE}")
        logger.info("Using REAL market data via Yahoo Finance")

    def get_market_data(self, symbol: str) -> pd.DataFrame:
        """Fetch real market data with 5-min caching."""
        now = datetime.now()
        if symbol in self.cache_time and (now - self.cache_time[symbol]).seconds < 300:
            logger.debug(f"Using cached data for {symbol} (age: {(now - self.cache_time[symbol]).seconds}s)")
            return self.data_cache[symbol]

        logger.info(f"Fetching fresh real data for {symbol}...")
        df = fetch_realtime(symbol, period="5d", interval="5m")
        self.data_cache[symbol] = df
        self.cache_time[symbol] = now
        return df

    def write_to_excel(self, row, col, value):
        try:
            import openpyxl
            wb = openpyxl.load_workbook(self.excel_path)
            ws = wb.active
            ws.cell(row=row, column=col).value = value
            wb.save(self.excel_path)
            logger.debug(f"Excel write: row={row}, col={col}, value={value}")
        except Exception as e:
            logger.error(f"Excel error: {e}")

    def send_telegram(self, signal, confidence, symbol, price=0.0):
        """Send Telegram alert via HTTP API (non-blocking)."""
        try:
            from alerts.telegram_alerts import send_trade_alert
            send_trade_alert(signal, symbol, price, confidence)
        except Exception as e:
            logger.warning(f"Telegram skipped: {e}")

    def run(self):
        logger.info("Server loop started — waiting for connections (REAL DATA MODE)...")
        while True:
            readable, _, _ = select.select([self.server], [], [], 0.1)
            if readable:
                client, addr = self.server.accept()
                logger.info(f"Client connected from {addr}")
                data = client.recv(1024).decode().strip()
                logger.info(f"Received command: '{data}'")

                if data == "/shutdown":
                    logger.warning("Shutdown command received — stopping server")
                    break

                # Parse symbol from command (e.g., "ANALYZE EURUSD" or just anything)
                parts = data.split()
                symbol = parts[1] if len(parts) > 1 else "EURUSD"

                # Fetch REAL market data
                df = self.get_market_data(symbol)
                logger.info(f"Data: {len(df)} candles | Latest close: {df['close'].iloc[-1]:.5f} | {symbol}")

                # AGI prediction on real data
                result = self.agi.predict(df)
                logger.info(f"Prediction => Signal: {result['signal']} | Confidence: {result['confidence']:.4f} | Symbol: {result['symbol']}")

                # Risk check + trade execution
                if result["signal"] != "HOLD" and self.risk.can_trade(10000, result["signal"], df['close'].iloc[-1]):
                    self.trade_count += 1
                    self.write_to_excel(10, 2, result["signal"])
                    self.write_to_excel(10, 3, result["confidence"])
                    self.write_to_excel(10, 4, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    self.write_to_excel(10, 5, float(df['close'].iloc[-1]))

                    response = f"TRADE: {result['signal']} {symbol} @ {df['close'].iloc[-1]:.5f} | conf={result['confidence']:.4f}"
                    client.send(response.encode())
                    logger.success(f"TRADE #{self.trade_count} EXECUTED: {response}")

                    # Telegram alert (non-blocking)
                    self.send_telegram(result["signal"], result["confidence"], symbol, float(df['close'].iloc[-1]))
                else:
                    self.hold_count += 1
                    response = f"HOLD {symbol} @ {df['close'].iloc[-1]:.5f} | conf={result['confidence']:.4f}"
                    client.send(response.encode())
                    logger.info(f"HOLD #{self.hold_count}: {response}")

                client.close()
                logger.info(f"Stats: {self.trade_count} trades | {self.hold_count} holds | Total: {self.trade_count + self.hold_count}")

        self.server.close()
        logger.success(f"Server shutdown complete. Total: {self.trade_count} trades, {self.hold_count} holds")

if __name__ == "__main__":
    AGIServer().run()
