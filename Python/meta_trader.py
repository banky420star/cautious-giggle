"""Stand-alone ATR traders: the meta-brain trader (here) and the Gemma trader (Python/gemma_trader.py).

Each one makes its own decisions and keeps its own positions, separate from the council's: positions are
hidden from the council (like Gold Breakout's), at most one per market, stop / target at the live ATR
multiples (Python/meta_brain.STOPS), closed at market after 96 M5 bars (8 h).

Meta-brain trader: the meta-brain (Python/meta_brain.py) is trained on every strategy's signals (the
48-indicator council, bankbot, ICT FVG->iFVG, Sweep+FVG, LuxAlgo SMC, SMC rules, Gold Breakout's range break,
regime) to predict the result in R of a long and a short opened now with those stops. It trades exactly what
its walk-forward test traded: enter when the better side is predicted above the margin. The prediction is
computed by the external-voters worker (voter "meta_brain", every new M5 bar); this class only acts on it.
Config: trading.meta_trader. Runs on demo rules: demo, and live while same_on_live is on (Server_AGI).
"""
from __future__ import annotations

import json
import os
import time

import pandas as pd

from Python.indicator_council import _atr
from Python.meta_brain import HORIZON, STOPS, base
from Python.strategy_exec import account_log, all_positions, close_ticket, open_hidden, risk_lots, crowd_reason


class AtrTrader:
    """Position book + order plumbing shared by the stand-alone ATR traders."""
    NAME = "atr trader"
    TAG = "ATR"
    FILE = "atr_trader"
    DEFAULTS = {"enabled": False, "risk_pct": 5.0, "min_lot_max_risk_pct": 5.0, "max_open": 7, "magic": 77000}

    def __init__(self, cfg: dict | None, log_dir: str, logger=None):
        self.cfg = {**self.DEFAULTS, **(cfg or {})}
        self.enabled = bool(self.cfg.get("enabled"))
        self.log = logger
        name = account_log(self.FILE)         # per broker account: demo and live keep separate books
        self.path = os.path.join(log_dir, f"{name}_state.json")
        self.events = os.path.join(log_dir, f"{name}.jsonl")
        self.state = self._load()        # {"open": {sym: {...}}, "seen": {sym: signal stamp}}

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                st = json.load(f)
        except (OSError, ValueError):
            st = {}
        st.setdefault("open", {})
        st.setdefault("seen", {})
        return st

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def _event(self, kind: str, **kw) -> None:
        row = {"t": time.time(), "event": kind, **kw}
        try:
            with open(self.events, "a", encoding="utf-8") as f:
                f.write(json.dumps(row) + "\n")
        except OSError:
            pass
        if self.log:
            self.log.info(f"{self.NAME} {kind}: " + ", ".join(f"{k}={v}" for k, v in kw.items()))

    def stops_for(self, sym) -> tuple[float, float]:
        """(stop, target) in M5 ATR: this trader's own `stops` ({market or "default": [sl, tp]}) or the shared
        meta_brain.STOPS. Gemma / Jev use 4 / 6.4 (user 2026-11-24): at ~2 ATR normal wiggles stopped 31 of their
        61 trades within ~38 min on a day the direction was right; 4+ ATR also puts them on the stop manager's
        wide rules, which don't trail winners down to +0.6 R."""
        own = self.cfg.get("stops") or {}
        pick = own.get(base(sym)) or own.get(sym) or own.get("default")
        if pick:
            return float(pick[0]), float(pick[1])
        return STOPS.get(base(sym), STOPS["default"])

    # Each trader may hold up to `max_per_symbol` positions per market (user 2026-11-25: 2 per bot per market).
    # The first sits in state["open"] under the market name (what the dashboards read), extra ones under
    # "market#ticket".
    def _keys(self, sym) -> list:
        return [k for k in self.state["open"] if k == sym or k.startswith(f"{sym}#")]

    def per_symbol(self) -> int:
        return max(1, int(self.cfg.get("max_per_symbol", 1) or 1))

    def side_on(self, sym) -> int:
        keys = self._keys(sym)
        return int(self.state["open"][keys[0]]["side"]) if keys else 0

    def manage(self, mt5, sym, now) -> int:
        """Watch this trader's open positions on `sym`: broker stop/target, 8 h limit. How many are still open."""
        keys = self._keys(sym)
        if not keys:
            return 0
        open_ids = {str(p.ticket) for p in all_positions(mt5, sym)}
        n = 0
        for k in keys:
            pos = self.state["open"][k]
            if str(pos["ticket"]) not in open_ids:
                self._event("closed", symbol=sym, ticket=pos["ticket"], reason="stop or target at the broker")
                del self.state["open"][k]
                self._save()
                continue
            if now - float(pos["opened"]) >= HORIZON * 300 and self._close_key(mt5, sym, k, "8 h time limit"):
                continue
            n += 1
        return n

    def _close_key(self, mt5, sym, key, why) -> bool:
        pos = self.state["open"][key]
        ok, p = close_ticket(mt5, sym, pos["ticket"], self.cfg["magic"], f"{self.TAG}|C")
        if ok or p is None:
            self._event("closed", symbol=sym, ticket=pos["ticket"], reason=why,
                        profit=round(float(getattr(p, "profit", 0.0) or 0.0), 2) if p else None)
            del self.state["open"][key]
            self._save()
            return True
        return False

    def close(self, mt5, sym, why) -> bool:
        """Close every position this trader holds on `sym`. True when all of them closed."""
        return all([self._close_key(mt5, sym, k, why) for k in self._keys(sym)])

    def open(self, mt5, sym, side, **why):
        crowd = crowd_reason(mt5, side, symbol=sym)
        if crowd:
            self._event("skip", symbol=sym, reason=crowd)
            self._save()
            return
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 200)
        tick = mt5.symbol_info_tick(sym)
        if rates is None or len(rates) < 50 or tick is None:
            self._event("skip", symbol=sym, reason="no prices")
            return
        df = pd.DataFrame(list(rates) if not hasattr(rates, "dtype") else rates)
        atr = float(_atr(df[["open", "high", "low", "close"]].astype(float)).iloc[-1])
        sl_m, tp_m = self.stops_for(sym)
        px = float(tick.ask if side > 0 else tick.bid)
        sl, tp = px - side * sl_m * atr, px + side * tp_m * atr
        sz = risk_lots(mt5, sym, px, sl, self.cfg["risk_pct"], self.cfg["min_lot_max_risk_pct"])
        if sz is None:
            self._event("skip", symbol=sym, reason="minimum lot would risk more than allowed")
            return
        lots, risk = sz
        digits = int(getattr(mt5.symbol_info(sym), "digits", 5) or 5)
        req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": sym, "volume": float(lots),
               "type": mt5.ORDER_TYPE_BUY if side > 0 else mt5.ORDER_TYPE_SELL, "price": px,
               "sl": round(sl, digits), "tp": round(tp, digits), "deviation": 50, "type_time": mt5.ORDER_TIME_GTC,
               "magic": int(self.cfg["magic"]), "comment": f"{self.TAG}|O"}
        ok, ticket, res = open_hidden(mt5, req)
        if ok:
            key = sym if sym not in self.state["open"] else f"{sym}#{ticket}"
            self.state["open"][key] = {"symbol": sym, "ticket": ticket, "side": side, "entry": px, "sl": req["sl"], "tp": req["tp"],
                                       "lots": lots, "risk_pct": risk, "opened": time.time(), **why}
        self._event("open" if ok else "error", symbol=sym, side="BUY" if side > 0 else "SELL", price=px, sl=req["sl"],
                    tp=req["tp"], lots=lots, risk_pct=risk, ticket=ticket, **why,
                    detail=None if ok else f"{getattr(res, 'retcode', None)} {getattr(res, 'comment', '')}")
        self._save()


class MetaTrader(AtrTrader):
    NAME = "meta trader"
    TAG = "META"
    FILE = "meta_trader"
    DEFAULTS = {"enabled": False, "risk_pct": 5.0, "min_lot_max_risk_pct": 5.0, "max_open": 7,
                "max_signal_age_s": 600,   # act only on a prediction from the last 2 M5 bars
                "magic": 77800}

    def step(self, mt5, ext, symbols, can_open: bool = True, now: float | None = None, block=None) -> None:
        if not self.enabled or ext is None or getattr(ext, "meta", None) is None:
            return
        now = time.time() if now is None else now
        margin = float(ext.meta.margin)
        for sym in symbols:
            held = self.manage(mt5, sym, now)
            if held >= self.per_symbol():
                continue
            val = ext.values.get((sym, "meta_brain"))
            det = ext.details.get((sym, "meta_brain")) or {}
            if not val or now - val[1] > float(self.cfg["max_signal_age_s"]) or "pred_long" not in det:
                continue
            if self.state["seen"].get(sym) == val[1]:
                continue                                  # this prediction was already acted on (or passed)
            self.state["seen"][sym] = val[1]
            pl, ps = float(det["pred_long"]), float(det["pred_short"])
            if max(pl, ps) < margin or (held and (1 if pl >= ps else -1) != self.side_on(sym)):
                self._save()                              # too weak, or against the position it already holds
                continue
            why_not = block(sym) if block else None
            if not can_open or why_not or len(self.state["open"]) >= int(self.cfg["max_open"]):
                self._event("skip", symbol=sym, reason="risk engine halt" if not can_open else why_not or "max open",
                            pred=max(pl, ps))
                self._save()
                continue
            self.open(mt5, sym, 1 if pl >= ps else -1, pred=round(max(pl, ps), 3))
