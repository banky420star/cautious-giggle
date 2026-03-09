import datetime
import requests


class TelegramAlerter:
    def __init__(self, token, chat_id):
        self.token = token
        self.chat_id = chat_id

    def _send(self, text):
        if not self.token or not self.chat_id:
            return False
        url = f"https://api.telegram.org/bot{self.token}/sendMessage"
        try:
            resp = requests.post(url, json={"chat_id": self.chat_id, "text": text}, timeout=8)
            return bool(resp.ok)
        except Exception:
            return False

    def online(self, message=""):
        body = "🟢 ONLINE\nAGI runtime connected."
        if message:
            body += f"\n{message}"
        self._send(body)

    def offline(self, message=""):
        body = "🔴 OFFLINE\nAGI runtime stopped."
        if message:
            body += f"\n{message}"
        self._send(body)

    def heartbeat(self, uptime, mt5_connected, trading_enabled):
        msg = (
            "💓 HEARTBEAT\n"
            f"Uptime: {uptime}\n"
            f"MT5: {'CONNECTED ✅' if mt5_connected else 'DISCONNECTED ❌'}\n"
            f"Trading: {'ENABLED ✅' if trading_enabled else 'HALTED ❌'}\n"
            f"Time: {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._send(msg)

    def trade(self, symbol, action, exposure, confidence, balance, equity, free_margin):
        msg = (
            "📈 TRADE EXECUTED\n"
            f"Symbol: {symbol}\n"
            f"Action: {action}\n"
            f"Exposure: {round(exposure, 4)}\n"
            f"Confidence: {round(confidence, 3)}\n"
            f"Balance: {round(balance, 2)}\n"
            f"Equity: {round(equity, 2)}\n"
            f"Free Margin: {round(free_margin, 2)}"
        )
        self._send(msg)

    def trade_closed(self, symbol, ticket, pnl, volume, price):
        icon = "✅" if pnl >= 0 else "❌"
        msg = (
            f"📉 TRADE CLOSED {icon}\n"
            f"Symbol: {symbol}\n"
            f"Ticket: {ticket}\n"
            f"Volume: {round(volume, 4)}\n"
            f"Close Price: {round(price, 6)}\n"
            f"Realized PnL: {round(pnl, 2)}"
        )
        self._send(msg)

    def trade_action(self, symbol, order_meta):
        if not order_meta:
            return
        msg = (
            "🧭 ACTION\n"
            f"Symbol: {symbol}\n"
            f"Mode: {order_meta.get('entry_mode')} | Side: {order_meta.get('order_type')}\n"
            f"Volume: {order_meta.get('volume_lots')} | Exposure: {round(order_meta.get('exposure', 0.0), 3)}\n"
            f"Entry: {order_meta.get('entry_price')} | TP: {order_meta.get('tp_price')} | SL: {order_meta.get('sl_price')}"
        )
        self._send(msg)

    def snapshot(self, balance, equity, pnl_today, floating, open_positions):
        msg = (
            "📊 STATUS SNAPSHOT\n"
            f"Balance: {round(balance, 2)}\n"
            f"Equity: {round(equity, 2)}\n"
            f"PnL Today: {round(pnl_today, 2)}\n"
            f"Floating: {round(floating, 2)}\n"
            f"Open Positions: {int(open_positions)}"
        )
        self._send(msg)

    def training(self, stage, message):
        self._send(f"🧠 TRAINING {stage}\n{message}")

    def model(self, message):
        self._send(f"🏆 MODEL UPDATE\n{message}")

    def alert(self, message):
        self._send(f"⚠️ ALERT\n{message}")
