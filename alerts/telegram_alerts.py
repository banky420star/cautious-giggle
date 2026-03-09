import datetime
import json
import os

import requests


class TelegramAlerter:
    def __init__(self, token, chat_id, cards_state_path=None):
        self.token = token
        self.chat_id = chat_id
        self.cards_state_path = cards_state_path or self._default_cards_path()
        self.cards = self._load_cards()

    def _default_cards_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "logs", "telegram_cards.json")

    def _load_cards(self):
        try:
            if self.cards_state_path and os.path.exists(self.cards_state_path):
                with open(self.cards_state_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {}
                if isinstance(data, dict):
                    return {str(k): int(v) for k, v in data.items()}
        except Exception:
            pass
        return {}

    def _save_cards(self):
        try:
            if not self.cards_state_path:
                return
            os.makedirs(os.path.dirname(self.cards_state_path), exist_ok=True)
            with open(self.cards_state_path, "w", encoding="utf-8") as f:
                json.dump(self.cards, f, indent=2, ensure_ascii=True)
        except Exception:
            pass

    def _api(self, method, payload):
        if not self.token or not self.chat_id:
            return None
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        try:
            resp = requests.post(url, json=payload, timeout=8)
            if not resp.ok:
                return None
            body = resp.json()
            if not body.get("ok"):
                return None
            return body.get("result")
        except Exception:
            return None

    def _send(self, text):
        out = self._api(
            "sendMessage",
            {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
        )
        return out is not None

    def _upsert_card(self, key, text):
        key = str(key)
        msg_id = self.cards.get(key)
        if msg_id:
            edited = self._api(
                "editMessageText",
                {
                    "chat_id": self.chat_id,
                    "message_id": int(msg_id),
                    "text": text,
                    "disable_web_page_preview": True,
                },
            )
            if edited is not None:
                return True
        sent = self._api(
            "sendMessage",
            {"chat_id": self.chat_id, "text": text, "disable_web_page_preview": True},
        )
        if sent is None:
            return False
        try:
            self.cards[key] = int(sent.get("message_id"))
            self._save_cards()
        except Exception:
            pass
        return True

    def online(self, message=""):
        body = "🟢 ONLINE\nAGI runtime connected."
        if message:
            body += f"\n{message}"
        self._upsert_card("runtime", body)

    def offline(self, message=""):
        body = "🔴 OFFLINE\nAGI runtime stopped."
        if message:
            body += f"\n{message}"
        self._upsert_card("runtime", body)

    def heartbeat(self, uptime, mt5_connected, trading_enabled):
        msg = (
            "💓 HEARTBEAT\n"
            f"Uptime: {uptime}\n"
            f"MT5: {'CONNECTED ✅' if mt5_connected else 'DISCONNECTED ❌'}\n"
            f"Trading: {'ENABLED ✅' if trading_enabled else 'HALTED ❌'}\n"
            f"Time: {datetime.datetime.utcnow().strftime('%H:%M:%S')} UTC"
        )
        self._upsert_card("heartbeat", msg)

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
        self._upsert_card("heartbeat", msg)

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
        self._upsert_card("trade_execution", msg)

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
        self._upsert_card("trade_closed", msg)

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
        exp_profit_usd = order_meta.get("tp_outcome_usd", order_meta.get("expected_profit_usd"))
        exp_loss_usd = order_meta.get("sl_outcome_usd", order_meta.get("expected_loss_usd"))
        if exp_profit_usd is None or exp_loss_usd is None:
            exp_profit_usd = tp_dist
            exp_loss_usd = sl_dist
        tp_icon = "🟢" if float(exp_profit_usd) >= 0 else "🔴"
        sl_icon = "🟢" if float(exp_loss_usd) >= 0 else "🔴"

        msg = (
            "🧭 ACTION\n"
            f"Symbol: {symbol}\n"
            f"Mode: {order_meta.get('entry_mode')} | Side: {order_meta.get('order_type')}\n"
            f"Volume: {order_meta.get('volume_lots')} | Exposure: {round(order_meta.get('exposure', 0.0), 3)}\n"
            f"Entry: {order_meta.get('entry_price')} | TP: {order_meta.get('tp_price')} | SL: {order_meta.get('sl_price')}\n"
            f"{tp_icon} TP Value(USD): {round(float(exp_profit_usd), 2)}\n"
            f"{sl_icon} SL Value(USD): {round(float(exp_loss_usd), 2)}\n"
            f"RR: {round(rr, 3)} | Lots: {round(lots, 2)}"
        )
        self._upsert_card("trade_action", msg)

    def snapshot(self, balance, equity, pnl_today, floating, open_positions):
        msg = (
            "📊 STATUS SNAPSHOT\n"
            f"Balance: {round(balance, 2)}\n"
            f"Equity: {round(equity, 2)}\n"
            f"PnL Today: {round(pnl_today, 2)}\n"
            f"Floating: {round(floating, 2)}\n"
            f"Open Positions: {int(open_positions)}"
        )
        self._upsert_card("snapshot", msg)

    def training(self, stage, message):
        self._upsert_card(f"training_{str(stage).strip().lower()}", f"🧠 TRAINING {stage}\n{message}")

    def model(self, message):
        self._upsert_card("model", f"🏆 MODEL UPDATE\n{message}")

    def alert(self, message):
        self._upsert_card("alerts", f"⚠️ ALERT\n{message}")

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
        self._upsert_card("daily_profitability", msg)
