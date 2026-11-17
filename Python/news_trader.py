"""News trading around high-impact releases: prep for the liquidity sweep, then trade the play
that has worked before for this market and this kind of news.

Prep (from PREP_MIN before the release)
    * mark the range of the PRE_RANGE_MIN minutes before the release: stop orders pile up just
      beyond its high and low, which is the liquidity a release spike tends to grab
    * stop the normal strategy from opening new trades on that symbol (so it doesn't get swept)

After the release (up to TRIGGER_MIN minutes), two plays are watched:
    fade      price pokes beyond the range by >= SWEEP_ATR x ATR(1m), then a 1-minute bar closes back
              inside -> trade the other way; stop just beyond the sweep extreme; target the far
              side of the range (at least MIN_R x risk)
    breakout  a 1-minute bar closes beyond the range by >= BREAK_ATR x ATR -> trade with it; stop at the
              middle of the range; target TARGET_R x risk
Trades that hit neither stop nor target are closed after HOLD_MIN minutes.

Which play (or none) is chosen per symbol and topic from the playbook (logs/news_playbook.json):
the paper result of BOTH plays is recorded after every event, whichever was traded, so the
evidence keeps growing. A play is only traded when it has >= MIN_SAMPLES events and an average
result >= MIN_AVG_R (after costs). History is seeded by tools/news_backtest.py.
"""
from __future__ import annotations

import json
import os
import time

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PLAYBOOK = os.path.join(ROOT, "logs", "news_playbook.json")

PREP_MIN = 10
PRE_RANGE_MIN = 30
TRIGGER_MIN = 20
HOLD_MIN = 60
SWEEP_ATR = 0.5
BREAK_ATR = 1.0
MIN_R = 1.0
TARGET_R = 1.5
STOP_BUFFER_ATR = 0.3
MIN_SAMPLES = 8
MIN_AVG_R = 0.10
ALWAYS_TRADE = True   # set from config: take part even without a proven edge (at half size)
COST = {"XAUUSDm": 0.30, "BTCUSDm": 18.0, "EURUSDm": 0.00011, "USDJPYm": 0.013, "USOILm": 0.02,
        "XAGUSDm": 0.033, "ETHUSDm": 1.4}   # ~1.3x the live spread (BTC/XAU include slippage)   # round-trip cost in price units (spread + slippage)

# which symbols each news topic is traded on
TOPIC_SYMBOLS = {
    "inflation": ["XAUUSDm", "BTCUSDm"], "fed": ["XAUUSDm", "BTCUSDm"], "jobs": ["XAUUSDm", "BTCUSDm"],
    "growth": ["XAUUSDm"], "geopolitics": ["XAUUSDm"], "currencies": ["XAUUSDm"], "crypto": ["BTCUSDm", "ETHUSDm"],
}
GENERIC = "us_release"


def atr1(h, l, c, n=14):
    tr = np.maximum(h[1:] - l[1:], np.maximum(np.abs(h[1:] - c[:-1]), np.abs(l[1:] - c[:-1])))
    return float(np.mean(tr[-n * 4:])) if len(tr) else 0.0


def pre_range(t, h, l, t0):
    m = (t >= t0 - PRE_RANGE_MIN * 60) & (t < t0)
    if m.sum() < PRE_RANGE_MIN * 0.6:
        return None
    return float(h[m].max()), float(l[m].min())


def find_entry(play, t, o, h, l, c, t0, rng, a):
    """First entry for `play` after t0 using closed 1-minute bars. Returns dict or None."""
    rh, rl = rng
    mid = (rh + rl) / 2
    swept_hi = swept_lo = None
    idx = np.where((t >= t0) & (t < t0 + TRIGGER_MIN * 60))[0]
    for i in idx:
        if play == "fade":
            if h[i] > rh + SWEEP_ATR * a:
                swept_hi = max(swept_hi or h[i], h[i])
            if l[i] < rl - SWEEP_ATR * a:
                swept_lo = min(swept_lo or l[i], l[i])
            if swept_hi is not None and c[i] < rh and (swept_lo is None or swept_hi - rh >= rl - swept_lo):
                e, sl = c[i], swept_hi + STOP_BUFFER_ATR * a
                tp = min(rl, e - MIN_R * (sl - e))
                return {"side": -1, "entry": e, "sl": sl, "tp": tp, "i": int(i)}
            if swept_lo is not None and c[i] > rl:
                e, sl = c[i], swept_lo - STOP_BUFFER_ATR * a
                tp = max(rh, e + MIN_R * (e - sl))
                return {"side": 1, "entry": e, "sl": sl, "tp": tp, "i": int(i)}
        else:
            if c[i] > rh + BREAK_ATR * a:
                e, sl = c[i], mid
                return {"side": 1, "entry": e, "sl": sl, "tp": e + TARGET_R * (e - sl), "i": int(i)}
            if c[i] < rl - BREAK_ATR * a:
                e, sl = c[i], mid
                return {"side": -1, "entry": e, "sl": sl, "tp": e - TARGET_R * (sl - e), "i": int(i)}
    return None


def simulate(play, sym, t, o, h, l, c, t0):
    """Paper-trade one play around one event. Returns {'r': result in risk units, ...} or None."""
    rng = pre_range(t, h, l, t0)
    if rng is None:
        return None
    k = np.where(t < t0)[0]
    if len(k) < 70:
        return None
    a = atr1(h[k[-70]:k[-1] + 1], l[k[-70]:k[-1] + 1], c[k[-70]:k[-1] + 1])
    if a <= 0:
        return None
    en = find_entry(play, t, o, h, l, c, t0, rng, a)
    if en is None:
        return {"r": 0.0, "traded": False}
    side, e, sl, tp = en["side"], en["entry"], en["sl"], en["tp"]
    risk = abs(e - sl)
    if risk <= 0:
        return None
    cost = COST.get(sym, 0.0)
    exit_px = None
    for j in range(en["i"] + 1, len(t)):
        if t[j] > t[en["i"]] + HOLD_MIN * 60:
            exit_px = o[j]
            break
        if (side > 0 and l[j] <= sl) or (side < 0 and h[j] >= sl):
            exit_px = sl
            break
        if (side > 0 and h[j] >= tp) or (side < 0 and l[j] <= tp):
            exit_px = tp
            break
    if exit_px is None:
        return None
    r = (side * (exit_px - e) - cost) / risk
    return {"r": float(r), "traded": True, "side": side, "entry": float(e), "sl": float(sl), "tp": float(tp),
            "exit": float(exit_px), "risk": float(risk), "range": rng}


# ------------------------------------------------------------------ playbook
def load_playbook() -> dict:
    try:
        with open(PLAYBOOK, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def save_playbook(pb: dict):
    tmp = PLAYBOOK + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(pb, f, indent=1)
    os.replace(tmp, PLAYBOOK)


def record(pb: dict, sym: str, topic: str, play: str, r: float, traded: bool, keep: int = 200):
    """Add a paper result. Both the topic bucket and the generic bucket are updated."""
    for key in {topic, GENERIC}:
        rows = pb.setdefault(sym, {}).setdefault(key, {}).setdefault(play, [])
        rows.append(round(float(r), 3) if traded else None)
        del rows[:-keep]


def stats(rows: list) -> dict:
    tr = [x for x in rows if x is not None]
    n = len(tr)
    return {"events": len(rows), "n": n, "avg_r": (sum(tr) / n) if n else 0.0,
            "win": (sum(1 for x in tr if x > 0) / n) if n else 0.0, "total_r": sum(tr)}


def choose(pb: dict, sym: str, topic: str) -> dict:
    """Pick the play for this symbol/topic. Uses the topic's own record when it has enough
    events, otherwise the generic US-release record."""
    out = {"play": None, "basis": None, "why": ""}
    for key in (topic, GENERIC):
        plays = (pb.get(sym) or {}).get(key) or {}
        st = {p: stats(rows) for p, rows in plays.items()}
        ok = {p: s for p, s in st.items() if s["n"] >= MIN_SAMPLES}
        if not ok:
            continue
        best = max(ok, key=lambda p: ok[p]["avg_r"])
        out["proven"] = ok[best]["avg_r"] >= MIN_AVG_R
        s = ok[best]
        label = "this kind of news" if key == topic else "US releases in general"
        others = ", ".join(f"{p} {v['avg_r']:+.2f}R over {v['n']}" for p, v in ok.items() if p != best)
        if s["avg_r"] >= MIN_AVG_R or ALWAYS_TRADE:
            out.update(play=best, basis=key, stats=s,
                       why=("" if out["proven"] else "no proven edge yet, so half size; ") + f"{best} has averaged {s['avg_r']:+.2f}R per trade ({s['win']:.0%} wins, {s['n']} trades) on "
                           f"{label}" + (f"; {others}" if others else ""))
        else:
            out.update(play=None, basis=key, stats=s,
                       why=f"no play has an edge on {label} yet (best: {best} {s['avg_r']:+.2f}R over {s['n']}"
                           + (f"; {others}" if others else "") + ") — standing aside, only protecting")
        return out
    if ALWAYS_TRADE:
        out.update(play="fade", proven=False,
                   why="no history for this market yet: fading the sweep at half size (the most common release pattern)")
    else:
        out["why"] = "not enough past events yet — standing aside, only protecting"
    return out


# ------------------------------------------------------------------ live
class NewsTrader:
    """Live state machine, driven from the bot's main loop."""

    def __init__(self, mt5, executor, cfg: dict, logger, audit):
        nc = ((cfg.get("trading", {}) or {}).get("news_trading", {}) or {})
        self.enabled = bool(nc.get("enabled", False))
        self.mt5, self.executor, self.log, self.audit = mt5, executor, logger, audit
        self.symbols = set(nc.get("symbols", ["XAUUSDm", "BTCUSDm"]))
        self.max_lots = float(nc.get("max_lots", 0.2))
        self.magic = int(nc.get("magic", 770077))
        global ALWAYS_TRADE
        ALWAYS_TRADE = bool(nc.get("always_trade", True))
        self.pb = load_playbook()
        self.events: dict[str, dict] = {}   # id -> event state
        self._cal_t = 0.0
        self.status: dict[str, dict] = {}

    # -- calendar
    def _refresh_calendar(self):
        if time.time() - self._cal_t < 600:
            return
        self._cal_t = time.time()
        from Python.news_markets import get_calendar, topic_of

        for e in get_calendar():
            if e.get("impact") != "High" or e["t"] < time.time() - 2 * 3600 or e["t"] > time.time() + 36 * 3600:
                continue
            topic = topic_of(e["title"], e.get("country"))
            if e.get("country") != "USD" and topic != "crypto":
                continue  # BTC and gold are priced in dollars: US releases are the ones that move them
            # every high-impact US release moves the dollar, so it is traded on every symbol;
            # crypto-only news is traded on bitcoin only
            targets = TOPIC_SYMBOLS["crypto"] if topic == "crypto" else sorted(self.symbols)
            for sym in targets:
                if sym not in self.symbols:
                    continue
                eid = f"{sym}|{int(e['t'])}"   # releases at the same minute are one event
                if eid in self.events and e["title"] not in self.events[eid]["title"]:
                    self.events[eid]["title"] += " + " + e["title"].replace("USD ", "")
                if eid not in self.events:
                    self.events[eid] = {"id": eid, "sym": sym, "t0": e["t"], "title": e["title"], "topic": topic,
                                        "phase": "scheduled"}

    def _bars(self, sym, n=200):
        r = self.mt5.copy_rates_from_pos(sym, self.mt5.TIMEFRAME_M1, 1, n)
        if r is None or not len(r):
            return None
        return (np.asarray(r["time"], float), np.asarray(r["open"], float), np.asarray(r["high"], float),
                np.asarray(r["low"], float), np.asarray(r["close"], float))

    # -- public
    def blocks_regular(self, sym: str) -> str | None:
        """Reason the normal strategy should not open trades on `sym` right now (or None)."""
        if not self.enabled:
            return None
        for ev in self.events.values():
            if ev["sym"] != sym:
                continue
            if ev["phase"] in ("prep", "armed", "in_trade"):
                return f"news: {ev['title']} ({ev['phase']})"
        return None

    def upcoming(self) -> list[dict]:
        rows = []
        for ev in sorted(self.events.values(), key=lambda x: x["t0"]):
            if ev["phase"] in ("done",):
                continue
            ch = choose(self.pb, ev["sym"], ev["topic"])
            rows.append({k: ev.get(k) for k in ("sym", "title", "topic", "t0", "phase", "range", "play", "result")}
                        | {"plan": ch.get("play"), "why": ch.get("why")})
        return rows

    def step(self):
        if not self.enabled:
            return
        try:
            self._refresh_calendar()
        except Exception as exc:
            self.log.warning(f"news calendar refresh failed: {exc}")
        now = time.time()
        for ev in list(self.events.values()):
            try:
                self._step_event(ev, now)
            except Exception as exc:
                self.log.warning(f"news step failed for {ev['id']}: {exc}")
        for eid, ev in list(self.events.items()):
            if ev["phase"] == "done" and now - ev["t0"] > 6 * 3600:
                self.events.pop(eid, None)

    def _step_event(self, ev: dict, now: float):
        sym, t0 = ev["sym"], ev["t0"]
        if ev["phase"] == "scheduled" and now >= t0 - PREP_MIN * 60:
            if now > t0 + TRIGGER_MIN * 60:
                ev["phase"] = "done"
                return
            ch = choose(self.pb, sym, ev["topic"])
            ev.update(phase="prep", play=ch["play"], why=ch["why"], proven=ch.get("proven", True))
            self.log.info(f"NEWS PREP {sym} | {ev['title']} in {max(0, (t0 - now) / 60):.0f} min | "
                          f"pausing new regular trades | plan: {ch['play'] or 'stand aside'} — {ch['why']}")
            self.audit("news_prep", {"symbol": sym, "title": ev["title"], "topic": ev["topic"], "plan": ch["play"],
                                     "why": ch["why"], "t0": t0})
        if ev["phase"] == "prep" and now >= t0:
            b = self._bars(sym)
            rng = pre_range(b[0], b[2], b[3], t0) if b else None
            if rng is None:
                ev["phase"] = "done"
                return
            ev["range"] = rng
            ev["atr"] = atr1(b[2][b[0] < t0][-70:], b[3][b[0] < t0][-70:], b[4][b[0] < t0][-70:])
            ev["phase"] = "armed"
            self.log.info(f"NEWS ARMED {sym} | {ev['title']} released | range {rng[1]:.5g}–{rng[0]:.5g}: "
                          f"stops sit beyond these levels | watching for the sweep")
        if ev["phase"] == "armed":
            b = self._bars(sym)
            if b is None:
                return
            if ev.get("play"):
                en = find_entry(ev["play"], *b, t0, ev["range"], ev["atr"])
                if en is not None and b[0][en["i"]] >= now - 180:   # only act on a fresh trigger
                    self._enter(ev, en)
                    return
            if now > t0 + TRIGGER_MIN * 60:
                ev["phase"] = "review"
        if ev["phase"] == "in_trade":
            if not self._position(sym):
                ev["phase"] = "review"
            elif now > ev["entered_at"] + HOLD_MIN * 60:
                pos = self._position(sym)
                if pos:
                    self.executor.close_positions([pos], order_meta={"reason": "news hold time over"})
                ev["phase"] = "review"
        if ev["phase"] == "review" and now > t0 + (TRIGGER_MIN + HOLD_MIN + 5) * 60:
            self._learn(ev)
            ev["phase"] = "done"

    def _position(self, sym):
        for p in (self.mt5.positions_get(symbol=sym) or []):
            if int(getattr(p, "magic", 0) or 0) == self.magic or int(getattr(p, "ticket", -1)) == ev_ticket(self, sym):
                return p
        return None

    halted = staticmethod(lambda: False)   # set by the bot: True while the risk supervisor halts trading

    def _enter(self, ev, en):
        sym = ev["sym"]
        if self.halted():
            self.log.info(f"NEWS SKIP {sym} | risk halt active (daily loss / drawdown limit)")
            ev["phase"] = "review"
            return
        info = self.mt5.symbol_info(sym)
        point = float(getattr(info, "point", 0.01) or 0.01)
        sl_points = abs(en["entry"] - en["sl"]) / point
        lots = self.executor.risk_capped_lots(sym, self.max_lots, sl_points=int(sl_points))
        if not ev.get("proven", True):
            lots = max(lots / 2, 0.0)
        lots = self.executor.normalize_volume(sym, lots)
        if lots <= 0:
            self.log.info(f"NEWS SKIP {sym} | stop too wide for the risk limit")
            ev["phase"] = "review"
            return
        # flatten anything the normal strategy holds on this symbol so the two don't fight
        mine = [p for p in (self.mt5.positions_get(symbol=sym) or [])]
        if mine:
            self.executor.close_positions(mine, order_meta={"reason": "news play takes over"})
        otype = self.mt5.ORDER_TYPE_BUY if en["side"] > 0 else self.mt5.ORDER_TYPE_SELL
        meta = self.executor.open_position(sym, otype, lots, order_meta={"source": "news", "play": ev["play"]})
        if not meta.get("executed"):
            self.log.warning(f"NEWS ENTRY FAILED {sym} | {meta.get('reason') or meta.get('retcode')}")
            ev["phase"] = "review"
            return
        pos = max(self.mt5.positions_get(symbol=sym) or [], key=lambda p: int(getattr(p, "time", 0) or 0), default=None)
        if pos is not None:
            ev["ticket"] = int(pos.ticket)
            self.mt5.order_send({"action": self.mt5.TRADE_ACTION_SLTP, "symbol": sym, "position": int(pos.ticket),
                                 "sl": round(en["sl"], 5), "tp": round(en["tp"], 5)})
        ev.update(phase="in_trade", entered_at=time.time(), entry=en)
        what = ("sweep of the highs failed, fading it" if ev["play"] == "fade" and en["side"] < 0 else
                "sweep of the lows failed, fading it" if ev["play"] == "fade" else
                "range broken with force, riding it")
        self.log.info(f"NEWS TRADE {sym} | {ev['title']} | {what} | {'buy' if en['side'] > 0 else 'sell'} {lots} "
                      f"@ {en['entry']:.5g} stop {en['sl']:.5g} target {en['tp']:.5g}")
        self.audit("news_trade", {"symbol": sym, "title": ev["title"], "topic": ev["topic"], "play": ev["play"],
                                  "side": en["side"], "lots": lots, "entry": en["entry"], "sl": en["sl"], "tp": en["tp"]})

    def _learn(self, ev):
        b = self._bars(ev["sym"], n=min(1000, int((time.time() - ev["t0"]) / 60) + 120))
        if b is None:
            return
        res = {}
        for play in ("fade", "breakout"):
            r = simulate(play, ev["sym"], *b, ev["t0"])
            if r is not None:
                record(self.pb, ev["sym"], ev["topic"], play, r["r"], r["traded"])
                res[play] = round(r["r"], 2) if r["traded"] else None
        save_playbook(self.pb)
        ev["result"] = res
        self.log.info(f"NEWS LEARN {ev['sym']} | {ev['title']} | paper results: "
                      + ", ".join(f"{p} {'no trigger' if v is None else f'{v:+.2f}R'}" for p, v in res.items()))
        self.audit("news_learn", {"symbol": ev["sym"], "title": ev["title"], "topic": ev["topic"], "results": res})


def ev_ticket(nt: NewsTrader, sym: str) -> int:
    for ev in nt.events.values():
        if ev["sym"] == sym and ev.get("phase") == "in_trade":
            return int(ev.get("ticket", -1))
    return -1
