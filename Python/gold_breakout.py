"""Gold Breakout: the Asian-range breakout ("Gold Breakout 15M Strategy", WillyAlgoTrader) as its own strategy.

Rules (tools/gold_breakout_backtest.py tests exactly these on Exness gold: 346 trades, +0.15R/trade, t=2.8):
    range     high/low of gold from 00:00 to 07:00 broker time (GMT+3, as on the user's TradingView page)
    entry     07:00-20:00: first break above the range high -> buy, below the low -> sell; one trade a day
    stop      1% of the entry price; no take-profit
    exit      the stop, or market close at 20:00

Separate from the council: after the order fills, its position is taken off the bot's list of own positions,
so the council (hide_foreign) never resizes or closes it, and this runner closes it itself at 20:00.
Demo accounts only. The main loop checks every ~20 s, so the entry is a market order shortly after the break
(TradingView fills a stop order exactly at the level); a break that is already more than `max_chase_pct`
past the level when first seen is skipped rather than chased.
"""
from __future__ import annotations

import json
import os
import time
from datetime import datetime, timedelta, timezone

from Python.strategy_exec import all_positions, close_ticket, open_hidden, risk_lots

DEFAULTS = {"enabled": False, "symbol": "XAUUSDm", "tz_hours": 3, "range_end_hour": 7, "close_hour": 20,
            "stop_pct": 1.0, "risk_pct": 1.0, "min_lot_max_risk_pct": 2.5, "max_chase_pct": 0.15,
            "magic": 77700}


class GoldBreakout:
    def __init__(self, cfg: dict | None, log_dir: str, logger=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.enabled = bool(self.cfg["enabled"])
        self.log = logger
        from Python.strategy_exec import account_log
        name = account_log("gold_breakout")      # per broker account: demo and live keep separate books
        self.state_path = os.path.join(log_dir, f"{name}_state.json")
        self.events_path = os.path.join(log_dir, f"{name}.jsonl")
        self.state = self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self) -> dict:
        try:
            with open(self.state_path, encoding="utf-8") as f:
                return json.load(f)
        except Exception:
            return {}

    def _save(self) -> None:
        tmp = self.state_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.state_path)

    def _event(self, kind: str, **kw) -> None:
        row = {"t": time.time(), "event": kind, "day": self.state.get("day"), **kw}
        try:
            with open(self.events_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(row, default=str) + "\n")
        except OSError:
            pass
        if self.log:
            self.log.info(f"GOLD BREAKOUT {kind} | " + ", ".join(f"{k}={v}" for k, v in kw.items()))

    # ------------------------------------------------------------------ helpers
    def broker_now(self, now: float | None = None) -> datetime:
        return datetime.fromtimestamp(now if now is not None else time.time(), tz=timezone.utc) + \
            timedelta(hours=int(self.cfg["tz_hours"]))

    def compute_range(self, rates, day: str):
        """High/low of the bars whose broker-clock open time is on `day` before range_end_hour."""
        if rates is None:
            return None
        off = int(self.cfg["tz_hours"]) * 3600
        hi = lo = None
        n = 0
        for r in rates:
            bt = datetime.fromtimestamp(int(r["time"]) + off, tz=timezone.utc)
            if bt.strftime("%Y-%m-%d") != day or bt.hour >= int(self.cfg["range_end_hour"]):
                continue
            h, l_ = float(r["high"]), float(r["low"])
            hi = h if hi is None else max(hi, h)
            lo = l_ if lo is None else min(lo, l_)
            n += 1
        return None if n < 60 else (hi, lo, n)    # 84 M5 bars in a full 7-hour range

    def size(self, mt5, entry: float, stop: float) -> tuple[float, float] | None:
        """Lots risking risk_pct of equity at the stop; the minimum lot is allowed up to min_lot_max_risk_pct."""
        return risk_lots(mt5, self.cfg["symbol"], entry, stop, self.cfg["risk_pct"], self.cfg["min_lot_max_risk_pct"])

    def _hidden_positions(self, mt5):
        return all_positions(mt5, self.cfg["symbol"])

    # ------------------------------------------------------------------ one loop step
    def step(self, mt5, can_open: bool = True, now: float | None = None) -> None:
        if not self.enabled:
            return
        bn = self.broker_now(now)
        day = bn.strftime("%Y-%m-%d")
        if self.state.get("day") != day:
            if self.state.get("ticket") and not self.state.get("closed"):
                self._close(mt5, "carried over from yesterday")   # should not happen: closed at close_hour
            self.state = {"day": day}
            self._save()
        st = self.state
        if bn.weekday() >= 5 or bn.hour < int(self.cfg["range_end_hour"]):
            return
        sym = self.cfg["symbol"]

        # an open trade: watch for the stop (broker side) and close at close_hour
        if st.get("ticket") and not st.get("closed"):
            open_ids = {str(p.ticket) for p in self._hidden_positions(mt5)}
            if str(st["ticket"]) not in open_ids:
                st["closed"] = "stop or closed at the broker"
                self._save()
                self._event("closed", reason=st["closed"], ticket=st["ticket"])
            elif bn.hour >= int(self.cfg["close_hour"]):
                self._close(mt5, f"{self.cfg['close_hour']}:00 session close")
            return
        if st.get("done") or bn.hour >= int(self.cfg["close_hour"]):
            return

        if "hi" not in st:
            rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 400)
            rg = self.compute_range(rates, day)
            if rg is None:
                st["done"] = "no full range today"
                self._save()
                self._event("skip", reason=st["done"])
                return
            st["hi"], st["lo"], st["range_bars"] = rg
            self._save()
            self._event("range", high=st["hi"], low=st["lo"], bars=st["range_bars"])

        tick = mt5.symbol_info_tick(sym)
        if tick is None:
            return
        hi, lo = float(st["hi"]), float(st["lo"])
        side = 1 if float(tick.ask) >= hi else (-1 if float(tick.bid) <= lo else 0)
        if side == 0:
            return
        level = hi if side > 0 else lo
        px = float(tick.ask) if side > 0 else float(tick.bid)
        if abs(px - level) / level * 100.0 > float(self.cfg["max_chase_pct"]):
            st["done"] = f"break seen {abs(px - level) / level * 100:.2f}% past the level: not chased"
            self._save()
            self._event("skip", reason=st["done"], price=px, level=level)
            return
        if not can_open:
            st["done"] = "risk engine halted new trades"
            self._save()
            self._event("skip", reason=st["done"])
            return
        stop = px * (1 - side * float(self.cfg["stop_pct"]) / 100.0)
        sz = self.size(mt5, px, stop)
        if sz is None:
            st["done"] = "minimum lot would risk more than allowed"
            self._save()
            self._event("skip", reason=st["done"], price=px, stop=stop)
            return
        lots, risk_pct = sz
        self._open(mt5, side, px, stop, lots, risk_pct)

    def _open(self, mt5, side, px, stop, lots, risk_pct):
        sym, st = self.cfg["symbol"], self.state
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(lots),
               "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL, "price": px,
               "sl": round(stop, 3), "deviation": 50, "type_time": mt5.ORDER_TIME_GTC,
               "magic": int(self.cfg["magic"]), "comment": "GBO|XAU|O"}
        ok, ticket, res = open_hidden(mt5, req)
        st["done"] = "traded" if ok else f"order failed: {getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}"
        if ok:
            st.update({"ticket": ticket, "side": side, "entry": px, "stop": stop, "lots": lots, "risk_pct": risk_pct})
        self._save()
        self._event("open" if ok else "error", side="BUY" if side > 0 else "SELL", price=px, stop=round(stop, 3),
                    lots=lots, risk_pct=risk_pct, ticket=ticket, detail=None if ok else st["done"])

    def _close(self, mt5, why: str):
        st = self.state
        ok, pos = close_ticket(mt5, self.cfg["symbol"], st.get("ticket"), self.cfg["magic"], "GBO|XAU|C")
        if pos is None:
            st["closed"] = "already closed"
            self._save()
            return
        if ok:
            st["closed"] = why
        self._save()
        self._event("close" if ok else "close_error", reason=why, ticket=st.get("ticket"),
                    profit=round(float(getattr(pos, "profit", 0.0) or 0.0), 2))

    def status(self) -> dict:
        return {"enabled": self.enabled, **self.state}
