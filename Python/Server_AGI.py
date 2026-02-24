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

        # Send Telegram startup alert
        try:
            from alerts.telegram_alerts import send_startup_alert
            send_startup_alert()
        except Exception:
            pass
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

    def send_telegram(self, signal, confidence, symbol, price=0.0,
                      lots=0.01, sl_pct=2.0, tp_pct=4.0):
        """Send Telegram alert in MT5 execution format (non-blocking)."""
        try:
            from alerts.telegram_alerts import send_trade_alert
            send_trade_alert(signal, symbol, price, confidence,
                             lots=lots, sl_pct=sl_pct, tp_pct=tp_pct)
        except Exception as e:
            logger.warning(f"Telegram skipped: {e}")

    def send_telegram_rejected(self, symbol, reason="Risk blocked"):
        """Send Telegram rejection alert."""
        try:
            from alerts.telegram_alerts import send_trade_rejected
            send_trade_rejected(symbol, reason="Invalid Params", details=reason)
        except Exception as e:
            logger.warning(f"Telegram rejection skipped: {e}")

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
                price = float(df['close'].iloc[-1])
                if result["signal"] != "HOLD" and self.risk.can_trade(
                    10000, result["signal"], price, confidence=result["confidence"]
                ):
                    self.trade_count += 1
                    lots = self.risk.lot_size(10000, price)
                    sl_tp = self.risk.compute_sl_tp(result["signal"], price)

                    self.write_to_excel(10, 2, result["signal"])
                    self.write_to_excel(10, 3, result["confidence"])
                    self.write_to_excel(10, 4, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
                    self.write_to_excel(10, 5, price)
                    self.write_to_excel(10, 6, lots)
                    self.write_to_excel(10, 7, sl_tp["sl"])
                    self.write_to_excel(10, 8, sl_tp["tp"])

                    response = (f"TRADE: {result['signal']} {symbol} @ {price:.5f} | "
                                f"conf={result['confidence']:.4f} | lots={lots} | "
                                f"sl={sl_tp['sl']:.5f} | tp={sl_tp['tp']:.5f}")
                    client.send(response.encode())
                    logger.success(f"TRADE #{self.trade_count} EXECUTED: {response}")

                    # Telegram: "Executed! (One Way)" + "buy EURUSD Q=0.1 SL=2% TP=4% M=505"
                    sl_pct = round(abs(price - sl_tp['sl']) / price * 100, 1)
                    tp_pct = round(abs(sl_tp['tp'] - price) / price * 100, 1)
                    self.send_telegram(
                        result["signal"], result["confidence"], symbol, price,
                        lots=lots, sl_pct=sl_pct, tp_pct=tp_pct
                    )
                else:
                    self.hold_count += 1
                    response = f"HOLD {symbol} @ {df['close'].iloc[-1]:.5f} | conf={result['confidence']:.4f}"
                    client.send(response.encode())
                    logger.info(f"HOLD #{self.hold_count}: {response}")

                    # Telegram: "Invalid Params" when risk blocks the trade
                    if result["signal"] != "HOLD":
                        self.send_telegram_rejected(
                            symbol,
                            f"Confidence {result['confidence']:.2%} or risk limit"
                        )

                client.close()
                logger.info(f"Stats: {self.trade_count} trades | {self.hold_count} holds | Total: {self.trade_count + self.hold_count}")

        self.server.close()
        logger.success(f"Server shutdown complete. Total: {self.trade_count} trades, {self.hold_count} holds")

if __name__ == "__main__":
    AGIServer().run()
