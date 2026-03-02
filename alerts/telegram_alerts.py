import requests
import time
from datetime import datetime

class TelegramAlerter:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id
        self.last_sent = {}

    def _send(self, text):
        if not self.token or not self.chat_id:
            return
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            requests.post(url, json={
                "chat_id": self.chat_id,
                "text": text
            }, timeout=5)
        except Exception:
            pass

    def heartbeat(self, uptime, mt5_connected, trading_enabled):
        msg = (
            f"💓 HEARTBEAT\n"
            f"Uptime: {uptime}\n"
            f"MT5: {'CONNECTED ✅' if mt5_connected else 'DISCONNECTED ❌'}\n"
            f"Trading: {'ENABLED ✅' if trading_enabled else 'HALTED ❌'}\n"
            f"Time: {datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._send(msg)

    def trade(self, symbol, action, exposure, confidence, balance, equity, free_margin):
        msg = (
            f"📈 TRADE EXECUTED\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Exposure: {round(exposure,4)}\n"
            f"Confidence: {round(confidence,3)}\n"
            f"Balance: {round(balance,2)}\n"
            f"Equity: {round(equity,2)}\n"
            f"Free Margin: {round(free_margin,2)}"
        )
        self._send(msg)

    def snapshot(self, balance, equity, pnl_today, floating, open_positions):
        msg = (
            f"📊 STATUS SNAPSHOT\n"
            f"Balance: {round(balance,2)}\n"
            f"Equity: {round(equity,2)}\n"
            f"PnL Today: {round(pnl_today,2)}\n"
            f"Floating: {round(floating,2)}\n"
            f"Open Positions: {open_positions}"
        )
        self._send(msg)

    def model(self, message):
        self._send(f"🧠 MODEL UPDATE\n{message}")

    def alert(self, message):
        self._send(f"⚠️ ALERT\n{message}")
