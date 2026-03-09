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

    def heartbeat_full(
        self,
        uptime,
        mt5_connected,
        trading_enabled,
        snapshot=None,
        training=None,
        models=None,
        event_intel=None,
    ):
        snap = snapshot or {}
        tr = training or {}
        md = models or {}
        ei = event_intel or {}
        eis = ei.get("summary", {}) if isinstance(ei, dict) else {}
        msg = (
            "💓 HEARTBEAT (FULL)\n"
            f"Uptime: {uptime}\n"
            f"MT5: {'CONNECTED ✅' if mt5_connected else 'DISCONNECTED ❌'}\n"
            f"Trading: {'ENABLED ✅' if trading_enabled else 'HALTED ❌'}\n"
            f"Balance: {round(float(snap.get('balance', 0.0) or 0.0), 2)}\n"
            f"Equity: {round(float(snap.get('equity', 0.0) or 0.0), 2)}\n"
            f"Free Margin: {round(float(snap.get('free_margin', 0.0) or 0.0), 2)}\n"
            f"PnL Today: {round(float(snap.get('pnl_today', 0.0) or 0.0), 2)}\n"
            f"Floating: {round(float(snap.get('floating', 0.0) or 0.0), 2)}\n"
            f"Open Positions: {int(snap.get('open_positions', 0) or 0)}\n"
            f"LSTM: {'RUNNING' if tr.get('lstm_running') else 'IDLE'}"
            f"{' | ' + str(tr.get('lstm_symbol')) if tr.get('lstm_symbol') else ''}"
            f"{' | epoch ' + str(tr.get('lstm_epoch')) + '/' + str(tr.get('lstm_epochs_total')) if tr.get('lstm_epoch') and tr.get('lstm_epochs_total') else ''}\n"
            f"{'LSTM Score: ' + str(tr.get('lstm_score')) + chr(10) if tr.get('lstm_score') else ''}"
            f"PPO: {'RUNNING' if tr.get('drl_running') else 'IDLE'}"
            f"{' | ' + str(tr.get('drl_symbol')) if tr.get('drl_symbol') else ''}"
            f"{' | score ' + str(tr.get('drl_score')) if tr.get('drl_score') else ''}\n"
            f"Cycle: {'RUNNING' if tr.get('cycle_running') else 'IDLE'}\n"
            f"Champion: {md.get('champion') or 'none'}\n"
            f"Canary: {md.get('canary') or 'none'}\n"
            f"Event Active: {int(eis.get('active_window', 0) or 0)} | High Active: {int(eis.get('high_active', 0) or 0)}\n"
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

    def trade_closed(self, symbol, ticket, pnl, volume, price, reason=None, deal_id=None):
        icon = "🟢⬆️" if pnl >= 0 else "🔴⬇️"
        why = reason if reason else "n/a"
        msg = (
            f"📉 TRADE CLOSED {icon}\n"
            f"Symbol: {symbol}\n"
            f"Ticket: {ticket}\n"
            f"Deal ID: {deal_id if deal_id is not None else 'n/a'}\n"
            f"Volume: {round(volume, 4)}\n"
            f"Close Price: {round(price, 6)}\n"
            f"Reason: {why}\n"
            f"Realized PnL: {round(pnl, 2)}"
        )
        self._send(msg)

    def trade_action(self, symbol, order_meta):
        if not order_meta:
            return
        entry = float(order_meta.get("entry_price", 0.0) or 0.0)
        tp = float(order_meta.get("tp_price", 0.0) or 0.0)
        sl = float(order_meta.get("sl_price", 0.0) or 0.0)
        side = str(order_meta.get("order_type", "BUY")).upper()
        lots = float(order_meta.get("volume_lots", 0.0) or 0.0)
        if side == "BUY":
            tp_dist = max(0.0, tp - entry)
            sl_dist = max(0.0, entry - sl)
        else:
            tp_dist = max(0.0, entry - tp)
            sl_dist = max(0.0, sl - entry)
        rr = (tp_dist / sl_dist) if sl_dist > 1e-12 else 0.0
        exp_profit_usd = order_meta.get("expected_profit_usd")
        exp_loss_usd = order_meta.get("expected_loss_usd")
        if exp_profit_usd is None or exp_loss_usd is None:
            exp_profit_usd = tp_dist
            exp_loss_usd = sl_dist

        msg = (
            "🧭 ACTION\n"
            f"Symbol: {symbol}\n"
            f"Mode: {order_meta.get('entry_mode')} | Side: {order_meta.get('order_type')}\n"
            f"Volume: {order_meta.get('volume_lots')} | Exposure: {round(order_meta.get('exposure', 0.0), 3)}\n"
            f"Entry: {order_meta.get('entry_price')} | TP: {order_meta.get('tp_price')} | SL: {order_meta.get('sl_price')}\n"
            f"Expected Profit(USD): {round(float(exp_profit_usd), 2)} | Expected Loss(USD): {round(float(exp_loss_usd), 2)} | RR: {round(rr, 3)} | Lots: {round(lots, 2)}"
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

    def profitability_daily(self, summary):
        s = summary or {}
        best = (s.get("best_symbols") or [])[:2]
        worst = (s.get("worst_symbols") or [])[:2]
        btxt = ", ".join([f"{x.get('symbol')} {x.get('total_pnl')}" for x in best]) if best else "n/a"
        wtxt = ", ".join([f"{x.get('symbol')} {x.get('total_pnl')}" for x in worst]) if worst else "n/a"
        msg = (
            "📅 DAILY PROFITABILITY\n"
            f"Trades: {int(s.get('trades', 0) or 0)} | Win Rate: {float(s.get('win_rate', 0.0) or 0.0):.2f}%\n"
            f"Total PnL: {float(s.get('total_pnl', 0.0) or 0.0):.2f}\n"
            f"Expectancy: {float(s.get('expectancy', 0.0) or 0.0):.4f} | PF: {float(s.get('profit_factor', 0.0) or 0.0):.3f}\n"
            f"Best: {btxt}\n"
            f"Worst: {wtxt}\n"
            f"Generated: {s.get('generated_at_utc', 'n/a')}"
        )
        self._send(msg)
