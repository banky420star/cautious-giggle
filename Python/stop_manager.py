"""Locks in profit on every bot trade by moving its stop, with each market's own rule.

One stop mover for all the bot's traders: the council's positions AND the stand-alone strategies' (Gold Breakout,
meta-brain trader, Gemma trader), which are hidden from the council. Trades opened by hand are never touched.

Rules come from tools/stop_rules_backtest.py (17 months of M5 history, council entries, chosen by the worst of
3 time periods; logs/stop_rules_report.json) and live in config trading.stop_rules, per market, in R
(R = the trade's starting stop distance; trades whose stop is >= 4 ATR use the wide set, rules_wide):
    be_at / be_lock      once the trade has been be_at R in profit, stop to entry + be_lock R
    trail_from / trail_atr   once trail_from R in profit, stop trails trail_atr x ATR behind the best price
    keep_target          false = push the target out to 10 R so the trail takes the exit
The stop only ever moves toward profit. State (each trade's R, ATR and best price) survives restarts.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

from Python.strategy_exec import all_positions, close_ticket, strategy_tickets

# Backtest winners (logs/stop_rules_report.json, 2026-11-24); config trading.stop_rules overrides per market.
DEFAULT_RULES = {
    "XAUUSD": {"be_at": 0.5, "be_lock": 0.0, "trail_from": 1.0, "trail_atr": 1.0, "keep_target": False},
    "BTCUSD": {"be_at": 0.75, "be_lock": 0.25, "keep_target": True},
    "EURUSD": {"trail_from": 1.0, "trail_atr": 0.5, "keep_target": False},
    "USDJPY": {"trail_from": 0.5, "trail_atr": 0.5, "keep_target": False},
    "XAGUSD": {"trail_from": 1.5, "trail_atr": 1.5, "keep_target": False},
    "ETHUSD": {"trail_from": 0.5, "trail_atr": 0.5, "keep_target": True},
    "USOIL": {"trail_from": 0.5, "trail_atr": 0.5, "keep_target": True},
    "default": {"be_at": 0.5, "be_lock": 0.0, "keep_target": True},     # markets added later (tools/market_scout.py)
}
# Trades with a wide stop (>= WIDE_AT x ATR: the council since 2026-11-24 at 8 ATR, Gold Breakout's 1 %) use these:
# re-tuned for 8 / 12.8 ATR, 24 h, with-trend entries (tools/stop_rules_backtest.py --wide,
# logs/stop_rules_report_wide.json). At that width moving the stop barely changes the average (t < 1) but locks
# profit in; a tight trail would cut it (the old executor rule fell from +0.12 R to +0.03 R on gold).
WIDE_AT = 4.0
DEFAULT_WIDE_RULES = {
    "XAUUSD": {"be_at": 0.75, "be_lock": 0.1, "keep_target": True},
    "BTCUSD": {"trail_from": 0.75, "trail_atr": 6.0, "keep_target": False},
    "XAGUSD": {"keep_target": True},                    # best left alone
    "ETHUSD": {"trail_from": 1.0, "trail_atr": 6.0, "keep_target": False},
    "EURUSD": {"be_at": 0.75, "be_lock": 0.1, "keep_target": True},     # not traded by the council now
    "USDJPY": {"be_at": 0.75, "be_lock": 0.1, "keep_target": True},
    "USOIL": {"be_at": 0.75, "be_lock": 0.1, "keep_target": True},
    "default": {"be_at": 0.75, "be_lock": 0.1, "keep_target": True},
}
FAR_TARGET_R = 10.0
MIN_STEP_R = 0.05          # move only when the stop improves by at least this (no modify spam)


def _base(sym: str) -> str:
    s = str(sym)
    return s[:-1] if s.endswith("m") and len(s) > 3 and s[:-1].isupper() else s   # Exness "m" account suffix


def new_stop(side: int, entry: float, r: float, atr: float, best: float, rule: dict) -> float | None:
    """The stop this rule wants now (None = leave it), given the best price reached since entry."""
    if r <= 0:
        return None
    mfe = (best - entry) * side / r
    want = None
    if rule.get("be_at") is not None and mfe >= float(rule["be_at"]):
        want = entry + side * float(rule.get("be_lock", 0.0)) * r
    if rule.get("trail_from") is not None and mfe >= float(rule["trail_from"]):
        t = best - side * float(rule["trail_atr"]) * atr
        want = t if want is None else (max(want, t) if side > 0 else min(want, t))
    return want


class StopManager:
    def __init__(self, cfg: dict | None, log_dir: str, logger=None, council_max_hours: float | None = None):
        cfg = cfg or {}
        # council trades run to stop / target / this many hours (trading.council.hold_to_exit); the strategies
        # keep their own time limits
        self.council_max_hours = council_max_hours
        self.enabled = bool(cfg.get("enabled", True))
        self.rules = {k: dict(v) for k, v in DEFAULT_RULES.items()}
        for k, v in (cfg.get("rules") or {}).items():
            self.rules[_base(k)] = {**self.rules.get(_base(k), {}), **(v or {})}
        self.wide_rules = {k: dict(v) for k, v in DEFAULT_WIDE_RULES.items()}
        for k, v in (cfg.get("rules_wide") or {}).items():
            self.wide_rules[_base(k)] = {**self.wide_rules.get(_base(k), {}), **(v or {})}
        self.log = logger
        self.log_dir = log_dir
        self.path = os.path.join(log_dir, "stop_manager_state.json")
        self.state = self._load()
        self._atr: dict = {}

    def _load(self) -> dict:
        try:
            with open(self.path, encoding="utf-8") as f:
                return json.load(f)
        except (OSError, ValueError):
            return {}

    def _save(self) -> None:
        tmp = self.path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(self.state, f)
        os.replace(tmp, self.path)

    def _atr_now(self, mt5, sym) -> float | None:
        hit = self._atr.get(sym)
        if hit and time.time() - hit[0] < 300:
            return hit[1]
        r = mt5.copy_rates_from_pos(sym, mt5.TIMEFRAME_M5, 1, 200)
        if r is None or len(r) < 30:
            return None
        h, l, c = (np.asarray([x[k] for x in r], dtype=float) for k in ("high", "low", "close"))
        pc = np.r_[c[0], c[:-1]]
        tr = np.maximum(h - l, np.maximum(abs(h - pc), abs(l - pc)))
        a = float(tr[0])
        for x in tr[1:]:
            a += (x - a) / 14.0
        self._atr[sym] = (time.time(), a)
        return a

    def _mine(self, mt5) -> set[str]:
        own = set()
        try:
            own = {str(x) for x in mt5._own_load()} if hasattr(mt5, "_own_load") else set()
        except Exception:
            own = set()
        return own | strategy_tickets(self.log_dir)

    def step(self, mt5, symbols=None) -> list[dict]:
        """Move stops on every bot position that has earned it. Returns the changes made."""
        if not self.enabled:
            return []
        positions = all_positions(mt5)
        mine = self._mine(mt5)
        if self.council_max_hours:
            self._time_exits(mt5, positions)
            positions = all_positions(mt5)
        managed = getattr(mt5, "_own_load", None) is not None
        seen, changes = set(), []
        for p in positions:
            t = str(p.ticket)
            sym = str(p.symbol)
            if (symbols and sym not in symbols) or (managed and t not in mine):
                continue                                   # a trade opened by hand: never touched
            seen.add(t)
            side = 1 if int(p.type) == int(getattr(mt5, "ORDER_TYPE_BUY", 0)) else -1
            entry, sl, tp = float(p.price_open), float(p.sl or 0), float(p.tp or 0)
            tick = mt5.symbol_info_tick(sym)
            if tick is None:
                continue
            px = float(tick.bid if side > 0 else tick.ask)
            st = self.state.get(t)
            if st is None:
                if sl <= 0:
                    continue                               # no stop yet: nothing to measure R from
                atr = self._atr_now(mt5, sym)
                if not atr:
                    continue
                r0 = abs(entry - sl)
                if (sl - entry) * side >= 0 or r0 < 0.5 * atr:
                    # stop already moved (older rule) or unusually tight: use the standard ATR stop as R
                    from Python.meta_brain import STOPS
                    r0 = STOPS.get(_base(sym), STOPS["default"])[0] * atr
                st = self.state[t] = {"r": r0, "atr": atr, "best": entry, "side": side, "sym": sym}
            st["best"] = max(st["best"], px) if side > 0 else min(st["best"], px)
            r = float(st["r"])
            wide = r >= WIDE_AT * float(st["atr"])
            book = self.wide_rules if wide else self.rules
            rule = book.get(_base(sym)) or book.get("default") or {}
            want = new_stop(side, entry, r, float(st["atr"]), float(st["best"]), rule)
            new_sl = sl
            if want is not None and (sl <= 0 or (want - sl) * side >= MIN_STEP_R * r):
                # never place the stop on the wrong side of the current price
                new_sl = min(want, px - 0.1 * float(st["atr"])) if side > 0 else max(want, px + 0.1 * float(st["atr"]))
                if sl > 0 and (new_sl - sl) * side < MIN_STEP_R * r:
                    new_sl = sl
            new_tp = tp
            if not rule.get("keep_target", True) and tp > 0 and abs(tp - entry) < (FAR_TARGET_R - 0.5) * r:
                new_tp = entry + side * FAR_TARGET_R * r
            if new_sl == sl and new_tp == tp:
                continue
            ok, msg = self._send(mt5, p, sym, new_sl, new_tp)
            row = {"ticket": t, "symbol": sym, "side": "BUY" if side > 0 else "SELL", "from_sl": sl, "sl": new_sl,
                   "tp": new_tp, "locked_r": round((new_sl - entry) * side / r, 2) if r else None,
                   "best_r": round((st["best"] - entry) * side / r, 2) if r else None, "ok": ok, "detail": msg}
            changes.append(row)
            if self.log:
                (self.log.info if ok else self.log.warning)(
                    f"STOP MOVE {sym} #{t} {row['side']} sl {sl:g} -> {new_sl:g} (locks {row['locked_r']:+.2f}R, "
                    f"best {row['best_r']:+.2f}R)" + ("" if new_tp == tp else f", target out to {new_tp:g}")
                    + ("" if ok else f" FAILED: {msg}"))
        for t in [k for k in self.state if k not in seen and k not in {str(p.ticket) for p in positions}]:
            del self.state[t]                              # closed: forget it
        self._save()
        return changes

    def _time_exits(self, mt5, positions) -> None:
        """Close council trades older than council_max_hours (the strategies' trades are left to them)."""
        try:
            council = {str(x) for x in mt5._own_load()} if hasattr(mt5, "_own_load") else set()
        except Exception:
            return
        council -= strategy_tickets(self.log_dir)
        now = time.time()
        for p in positions:
            opened = float(getattr(p, "time", 0) or 0)
            if str(p.ticket) not in council or not opened or now - opened < self.council_max_hours * 3600:
                continue
            ok, _ = close_ticket(mt5, str(p.symbol), p.ticket, 0, "AGI|TIME")
            if self.log:
                (self.log.info if ok else self.log.warning)(
                    f"TIME EXIT {p.symbol} #{p.ticket}: council trade open {(now - opened) / 3600:.1f} h "
                    f"(limit {self.council_max_hours:g} h)" + ("" if ok else " FAILED"))

    @staticmethod
    def _send(mt5, p, sym, sl, tp):
        info = mt5.symbol_info(sym)
        digits = int(getattr(info, "digits", 5) or 5) if info is not None else 5
        req = {"action": mt5.TRADE_ACTION_SLTP, "symbol": sym, "position": p.ticket,
               "sl": round(float(sl), digits) if sl else 0.0, "tp": round(float(tp), digits) if tp else 0.0}
        had = getattr(mt5, "hide_foreign", None)
        try:
            if had is not None:
                mt5.hide_foreign = False                   # the strategies' positions are hidden from the council
            res = mt5.order_send(req)
        finally:
            if had is not None:
                mt5.hide_foreign = had
        code = getattr(res, "retcode", None)
        return code in (10008, 10009, 10025), f"{code} {getattr(res, 'comment', '')}"
