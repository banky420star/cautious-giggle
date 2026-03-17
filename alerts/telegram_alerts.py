import datetime
import html
import json
import os
import re

import requests


def _utc_now():
    return datetime.datetime.now(datetime.timezone.utc)


def _utc_iso():
    return _utc_now().isoformat(timespec="seconds")


def _utc_clock():
    return _utc_now().strftime("%H:%M:%S")


def _as_float(value, digits=2):
    try:
        return f"{float(value):.{int(digits)}f}"
    except Exception:
        return "-"


def _as_int(value):
    try:
        return str(int(value))
    except Exception:
        return "-"


def _strip_markup(text):
    plain = re.sub(r"<[^>]+>", "", str(text or ""))
    return re.sub(r"\s+", " ", plain).strip()


class TelegramAlerter:
    def __init__(self, token, chat_id, cards_state_path=None):
        self.token = token
        self.chat_id = chat_id
        self.cards_state_path = cards_state_path or self._default_cards_path()
        self.cards = self._load_cards()
        self._last_api_error = None

    def _default_cards_path(self):
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        return os.path.join(root, "logs", "telegram_cards.json")

    def _configured(self):
        return bool(self.token and self.chat_id)

    def _normalize_card(self, key, value):
        base = {
            "key": str(key),
            "message_id": None,
            "category": "ops",
            "delivery_status": "dashboard_only",
            "updated_utc": None,
            "next_retry_utc": None,
            "title": str(key),
            "preview": "",
            "last_text": "",
        }
        if isinstance(value, dict):
            base["message_id"] = value.get("message_id") or value.get("msg_id")
            base["category"] = str(value.get("category") or "ops")
            base["delivery_status"] = str(value.get("delivery_status") or "dashboard_only")
            base["updated_utc"] = value.get("updated_utc")
            base["next_retry_utc"] = value.get("next_retry_utc")
            base["title"] = str(value.get("title") or str(key))
            base["preview"] = str(value.get("preview") or "")
            base["last_text"] = str(value.get("last_text") or "")
        else:
            try:
                base["message_id"] = int(value)
            except Exception:
                base["message_id"] = None
        return base

    def _load_cards(self):
        try:
            if self.cards_state_path and os.path.exists(self.cards_state_path):
                with open(self.cards_state_path, "r", encoding="utf-8") as handle:
                    data = json.load(handle) or {}
                if isinstance(data, dict):
                    return {str(key): self._normalize_card(key, value) for key, value in data.items()}
        except Exception:
            pass
        return {}

    def _save_cards(self):
        try:
            if not self.cards_state_path:
                return
            os.makedirs(os.path.dirname(self.cards_state_path), exist_ok=True)
            with open(self.cards_state_path, "w", encoding="utf-8") as handle:
                json.dump(self.cards, handle, indent=2, ensure_ascii=True, sort_keys=True)
        except Exception:
            pass

    def _api(self, method, payload):
        if not self._configured():
            return None
        self._last_api_error = None
        url = f"https://api.telegram.org/bot{self.token}/{method}"
        body = dict(payload or {})
        body.setdefault("disable_web_page_preview", True)
        body.setdefault("parse_mode", "HTML")
        try:
            resp = requests.post(url, json=body, timeout=8)
            if not resp.ok:
                retry_after = None
                try:
                    parsed = resp.json() or {}
                    retry_after = ((parsed.get("parameters") or {}).get("retry_after"))
                except Exception:
                    parsed = {}
                self._last_api_error = {"status_code": resp.status_code, "retry_after": retry_after, "body": parsed}
                return None
            parsed = resp.json()
            if not parsed.get("ok"):
                retry_after = ((parsed.get("parameters") or {}).get("retry_after")) if isinstance(parsed, dict) else None
                self._last_api_error = {"status_code": resp.status_code, "retry_after": retry_after, "body": parsed}
                return None
            return parsed.get("result")
        except Exception:
            self._last_api_error = {"status_code": None, "retry_after": None, "body": None}
            return None

    def _build_card(self, header, rows=None, detail=None):
        lines = [f"<b>{html.escape(str(header))}</b>"]
        if detail:
            lines.append(html.escape(str(detail)))
        for label, value in rows or []:
            if value is None:
                continue
            txt = str(value).strip()
            if not txt:
                continue
            lines.append(f"<b>{html.escape(str(label))}</b>: {html.escape(txt)}")
        lines.append(f"<b>Updated</b>: {_utc_clock()} UTC")
        return "\n".join(lines)

    def _upsert_card(self, key, text, category="ops", title=None):
        card_key = str(key)
        record = self.cards.get(card_key, self._normalize_card(card_key, {}))
        msg_id = record.get("message_id")
        sent = False
        result = None
        next_retry_utc = record.get("next_retry_utc")
        if next_retry_utc:
            try:
                if datetime.datetime.fromisoformat(str(next_retry_utc)) > _utc_now():
                    return False
            except Exception:
                pass

        if msg_id and self._configured():
            result = self._api(
                "editMessageText",
                {
                    "chat_id": self.chat_id,
                    "message_id": int(msg_id),
                    "text": text,
                },
            )
            sent = result is not None

        if not sent and self._configured():
            result = self._api(
                "sendMessage",
                {
                    "chat_id": self.chat_id,
                    "text": text,
                },
            )
            sent = result is not None

        if sent and result is not None:
            try:
                record["message_id"] = int(result.get("message_id"))
            except Exception:
                pass
            record["next_retry_utc"] = None
        else:
            retry_after = None
            if isinstance(self._last_api_error, dict):
                retry_after = self._last_api_error.get("retry_after")
            try:
                retry_after = int(retry_after) if retry_after is not None else None
            except Exception:
                retry_after = None
            record["next_retry_utc"] = (
                (_utc_now() + datetime.timedelta(seconds=max(1, retry_after))).isoformat(timespec="seconds")
                if retry_after
                else record.get("next_retry_utc")
            )

        plain = _strip_markup(text)
        record.update(
            {
                "key": card_key,
                "category": str(category or "ops"),
                "delivery_status": "both" if sent else "dashboard_only",
                "updated_utc": _utc_iso(),
                "title": str(title or plain.splitlines()[0] or card_key),
                "preview": plain[:240],
                "last_text": str(text or ""),
            }
        )
        self.cards[card_key] = record
        self._save_cards()
        return sent

    def retry_pending_cards(self):
        resent = []
        for key, record in list(self.cards.items()):
            if not isinstance(record, dict):
                continue
            if str(record.get("delivery_status") or "dashboard_only") == "both":
                continue
            text = str(record.get("last_text") or "").strip()
            if not text:
                continue
            if self._upsert_card(
                key,
                text,
                category=str(record.get("category") or "ops"),
                title=str(record.get("title") or key),
            ):
                resent.append(str(key))
        return resent

    def cards_snapshot(self, limit=None):
        rows = [dict(value) for value in self.cards.values() if isinstance(value, dict)]
        rows.sort(key=lambda item: str(item.get("updated_utc") or ""), reverse=True)
        if limit is not None:
            rows = rows[: max(0, int(limit))]
        return rows

    def state_summary(self, limit=12):
        cards = self.cards_snapshot(limit=limit)
        delivery = {"both": 0, "dashboard_only": 0}
        for row in self.cards.values():
            if not isinstance(row, dict):
                continue
            delivery[str(row.get("delivery_status") or "dashboard_only")] = (
                delivery.get(str(row.get("delivery_status") or "dashboard_only"), 0) + 1
            )
        return {
            "configured": self._configured(),
            "card_count": len(self.cards),
            "last_updated_utc": cards[0].get("updated_utc") if cards else None,
            "delivery": delivery,
            "cards": cards,
        }

    def online(self, message=""):
        body = self._build_card(
            "Runtime | ONLINE",
            rows=[("State", "AGI runtime connected"), ("Message", message or "Trading engine initialized")],
        )
        self._upsert_card("runtime", body, category="runtime", title="Runtime online")

    def offline(self, message=""):
        body = self._build_card(
            "Runtime | OFFLINE",
            rows=[("State", "AGI runtime stopped"), ("Message", message or "Runtime exited")],
        )
        self._upsert_card("runtime", body, category="runtime", title="Runtime offline")

    def heartbeat(self, uptime, mt5_connected, trading_enabled):
        body = self._build_card(
            "Runtime | HEARTBEAT",
            rows=[
                ("Uptime", uptime),
                ("MT5", "CONNECTED" if mt5_connected else "DISCONNECTED"),
                ("Trading", "ENABLED" if trading_enabled else "HALTED"),
            ],
        )
        self._upsert_card("heartbeat", body, category="runtime", title="Runtime heartbeat")

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
        summary = ei.get("summary", {}) if isinstance(ei, dict) else {}
        body = self._build_card(
            "Runtime | HEARTBEAT",
            rows=[
                ("Uptime", uptime),
                ("MT5", "CONNECTED" if mt5_connected else "DISCONNECTED"),
                ("Trading", "ENABLED" if trading_enabled else "HALTED"),
                ("Balance", _as_float(snap.get("balance"), 2)),
                ("Equity", _as_float(snap.get("equity"), 2)),
                ("Free margin", _as_float(snap.get("free_margin"), 2)),
                ("PnL today", _as_float(snap.get("pnl_today"), 2)),
                ("Floating", _as_float(snap.get("floating"), 2)),
                ("Open positions", _as_int(snap.get("open_positions", 0))),
                ("LSTM", "RUNNING" if tr.get("lstm_running") else "IDLE"),
                ("PPO", "RUNNING" if tr.get("drl_running") else "IDLE"),
                ("Cycle", "RUNNING" if tr.get("cycle_running") else "IDLE"),
                ("Champion", md.get("champion") or "none"),
                ("Canary", md.get("canary") or "none"),
                ("Event active", _as_int(summary.get("active_window", 0))),
                ("High impact", _as_int(summary.get("high_active", 0))),
            ],
        )
        self._upsert_card("heartbeat", body, category="runtime", title="Runtime heartbeat")

    def trade(self, symbol, action, exposure, confidence, balance, equity, free_margin):
        body = self._build_card(
            "Trading | EXECUTED",
            rows=[
                ("Symbol", symbol),
                ("Action", action),
                ("Exposure", _as_float(exposure, 4)),
                ("Confidence", _as_float(confidence, 3)),
                ("Balance", _as_float(balance, 2)),
                ("Equity", _as_float(equity, 2)),
                ("Free margin", _as_float(free_margin, 2)),
            ],
        )
        self._upsert_card("trade_execution", body, category="trading", title=f"Trade executed {symbol}")

    def trade_closed(self, symbol, ticket, pnl, volume, price, reason=None, deal_id=None):
        body = self._build_card(
            "Trading | CLOSED",
            rows=[
                ("Symbol", symbol),
                ("Ticket", ticket),
                ("Deal ID", deal_id if deal_id is not None else "n/a"),
                ("Volume", _as_float(volume, 4)),
                ("Close price", _as_float(price, 6)),
                ("Reason", reason or "n/a"),
                ("Realized PnL", _as_float(pnl, 2)),
            ],
        )
        self._upsert_card("trade_closed", body, category="trading", title=f"Trade closed {symbol}")

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
        body = self._build_card(
            "Trading | ACTION",
            rows=[
                ("Symbol", symbol),
                ("Lane", order_meta.get("lane")),
                ("Mode", order_meta.get("entry_mode")),
                ("Request action", order_meta.get("request_action")),
                ("Side", side),
                ("Volume lots", _as_float(order_meta.get("volume_lots"), 2)),
                ("Exposure", _as_float(order_meta.get("exposure"), 3)),
                ("Target exposure", _as_float(order_meta.get("target_exposure"), 3)),
                ("PPO", _as_float(order_meta.get("ppo_target"), 3)),
                ("Dreamer", _as_float(order_meta.get("dreamer_target"), 3)),
                ("AGI bias", _as_float(order_meta.get("agi_bias"), 3)),
                ("Entry", _as_float(order_meta.get("entry_price"), 6)),
                ("TP", _as_float(order_meta.get("tp_price"), 6)),
                ("SL", _as_float(order_meta.get("sl_price"), 6)),
                ("TP value USD", _as_float(exp_profit_usd, 2)),
                ("SL value USD", _as_float(exp_loss_usd, 2)),
                ("RR", _as_float(rr, 3)),
                ("Lots", _as_float(lots, 2)),
                ("Magic", order_meta.get("magic")),
                ("Comment", order_meta.get("comment")),
                ("Ticket", order_meta.get("ticket")),
                ("Retcode", order_meta.get("retcode")),
                ("Model version", order_meta.get("model_version")),
            ],
        )
        self._upsert_card("trade_action", body, category="trading", title=f"Trade action {symbol}")

    def snapshot(self, balance, equity, pnl_today, floating, open_positions):
        body = self._build_card(
            "Risk | SNAPSHOT",
            rows=[
                ("Balance", _as_float(balance, 2)),
                ("Equity", _as_float(equity, 2)),
                ("PnL today", _as_float(pnl_today, 2)),
                ("Floating", _as_float(floating, 2)),
                ("Open positions", _as_int(open_positions)),
            ],
        )
        self._upsert_card("snapshot", body, category="risk", title="Risk snapshot")

    def training(self, stage, message):
        body = self._build_card(
            f"Training | {str(stage).upper()}",
            rows=[("Message", message)],
        )
        self._upsert_card(
            f"training_{str(stage).strip().lower()}",
            body,
            category="training",
            title=f"Training {stage}",
        )

    def training_cycle(self, status, symbols, report=None, detail=None):
        payload = report or {}
        rows = payload.get("symbols") or []
        promoted = [str(row.get("symbol") or "n/a") for row in rows if row.get("wins") and row.get("passes_thresholds")]
        blocked = [str(row.get("symbol") or "n/a") for row in rows if not (row.get("wins") and row.get("passes_thresholds"))]
        body = self._build_card(
            f"Training | CYCLE {str(status).upper()}",
            rows=[
                ("Status", str(status).upper()),
                ("Universe", ", ".join(str(x) for x in (symbols or [])) or "n/a"),
                ("Mode", payload.get("mode") or "n/a"),
                ("Promoted", ", ".join(promoted) or "none"),
                ("Blocked", ", ".join(blocked) or "none"),
                ("Skip LSTM", "YES" if payload.get("skip_lstm") else "NO"),
                ("Skip Dreamer", "YES" if payload.get("skip_dreamer") else "NO"),
                ("Detail", detail or payload.get("error") or "n/a"),
            ],
        )
        self._upsert_card("training_cycle", body, category="training", title=f"Training cycle {status}")

    def model(self, message):
        body = self._build_card("Registry | MODEL UPDATE", rows=[("Message", message)])
        self._upsert_card("model", body, category="registry", title="Model update")

    def alert(self, message):
        body = self._build_card("Incident | ALERT", rows=[("Message", message)])
        self._upsert_card("alerts", body, category="incident", title="Alert")

    def profitability_daily(self, summary):
        payload = summary or {}
        best = (payload.get("best_symbols") or [])[:2]
        worst = (payload.get("worst_symbols") or [])[:2]
        body = self._build_card(
            "Trading | DAILY PROFITABILITY",
            rows=[
                ("Trades", _as_int(payload.get("trades", 0))),
                ("Win rate", _as_float(payload.get("win_rate", 0.0), 2) + "%"),
                ("Total PnL", _as_float(payload.get("total_pnl", 0.0), 2)),
                ("Expectancy", _as_float(payload.get("expectancy", 0.0), 4)),
                ("Profit factor", _as_float(payload.get("profit_factor", 0.0), 3)),
                (
                    "Best",
                    ", ".join(f"{row.get('symbol')} {_as_float(row.get('total_pnl'), 2)}" for row in best) or "n/a",
                ),
                (
                    "Worst",
                    ", ".join(f"{row.get('symbol')} {_as_float(row.get('total_pnl'), 2)}" for row in worst) or "n/a",
                ),
                ("Generated", payload.get("generated_at_utc") or "n/a"),
            ],
        )
        self._upsert_card("daily_profitability", body, category="trading", title="Daily profitability")

    def symbol_status(self, symbol, payload=None):
        card = payload or {}
        closed = card.get("last_closed") or {}
        body = self._build_card(
            f"Symbol | {symbol}",
            rows=[
                ("Signal", card.get("signal", "n/a")),
                ("Confidence", _as_float(card.get("confidence"), 4)),
                ("AGI exposure", _as_float(card.get("agi_exposure"), 4)),
                ("PPO exposure", _as_float(card.get("ppo_exposure"), 4)),
                ("Dreamer exposure", _as_float(card.get("dreamer_exposure"), 4)),
                ("Blend exposure", _as_float(card.get("blend_exposure"), 4)),
                ("Open positions", _as_int(card.get("open_positions", 0))),
                ("Floating PnL", _as_float(card.get("floating_pnl"), 2)),
                ("Position side", card.get("position_side") or "n/a"),
                ("Position volume", _as_float(card.get("position_volume"), 2)),
                ("Entry", _as_float(card.get("position_entry"), 6)),
                ("TP", _as_float(card.get("position_tp"), 6)),
                ("SL", _as_float(card.get("position_sl"), 6)),
                ("TP value USD", _as_float(card.get("position_tp_value_usd"), 2)),
                ("SL value USD", _as_float(card.get("position_sl_value_usd"), 2)),
                ("Last close PnL", _as_float(closed.get("profit"), 2)),
                ("Last close reason", closed.get("comment") or "n/a"),
                ("Last close deal", closed.get("deal_id") if closed.get("deal_id") is not None else "n/a"),
            ],
        )
        self._upsert_card(f"symbol_{symbol}", body, category="symbol", title=f"Symbol {symbol}")
