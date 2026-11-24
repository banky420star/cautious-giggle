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
from Python.strategy_exec import all_positions, close_ticket, open_hidden, risk_lots


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
        self.path = os.path.join(log_dir, f"{self.FILE}_state.json")
        self.events = os.path.join(log_dir, f"{self.FILE}.jsonl")
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

    def manage(self, mt5, sym, now) -> bool:
        """Watch an open position: broker stop/target, 8 h limit. True while it is still open."""
        pos = self.state["open"].get(sym)
        if not pos:
            return False
        open_ids = {str(p.ticket) for p in all_positions(mt5, sym)}
        if str(pos["ticket"]) not in open_ids:
            self._event("closed", symbol=sym, ticket=pos["ticket"], reason="stop or target at the broker")
            del self.state["open"][sym]
            self._save()
            return False
        if now - float(pos["opened"]) >= HORIZON * 300:
            return not self.close(mt5, sym, "8 h time limit")
        return True

    def close(self, mt5, sym, why) -> bool:
        pos = self.state["open"].get(sym)
        ok, p = close_ticket(mt5, sym, pos["ticket"], self.cfg["magic"], f"{self.TAG}|C")
        if ok or p is None:
            self._event("closed", symbol=sym, ticket=pos["ticket"], reason=why,
                        profit=round(float(getattr(p, "profit", 0.0) or 0.0), 2) if p else None)
            del self.state["open"][sym]
            self._save()
            return True
        return False

    def open(self, mt5, sym, side, **why):
        rates = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 200)
        tick = mt5.symbol_info_tick(sym)
        if rates is None or len(rates) < 50 or tick is None:
            self._event("skip", symbol=sym, reason="no prices")
            return
        df = pd.DataFrame(list(rates) if not hasattr(rates, "dtype") else rates)
        atr = float(_atr(df[["open", "high", "low", "close"]].astype(float)).iloc[-1])
        sl_m, tp_m = STOPS.get(base(sym), STOPS["default"])
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
            self.state["open"][sym] = {"ticket": ticket, "side": side, "entry": px, "sl": req["sl"], "tp": req["tp"],
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
            if self.manage(mt5, sym, now):
                continue
            val = ext.values.get((sym, "meta_brain"))
            det = ext.details.get((sym, "meta_brain")) or {}
            if not val or now - val[1] > float(self.cfg["max_signal_age_s"]) or "pred_long" not in det:
                continue
            if self.state["seen"].get(sym) == val[1]:
                continue                                  # this prediction was already acted on (or passed)
            self.state["seen"][sym] = val[1]
            pl, ps = float(det["pred_long"]), float(det["pred_short"])
            if max(pl, ps) < margin:
                self._save()
                continue
            why_not = block(sym) if block else None
            if not can_open or why_not or len(self.state["open"]) >= int(self.cfg["max_open"]):
                self._event("skip", symbol=sym, reason="risk engine halt" if not can_open else why_not or "max open",
                            pred=max(pl, ps))
                self._save()
                continue
            self.open(mt5, sym, 1 if pl >= ps else -1, pred=round(max(pl, ps), 3))
