"""Setup scorecard: only trade setups that have proven themselves.

Every entry is filed under a *setup*: market | side | session | confidence band | what the AI model said.
When the trade closes, its result is added to that setup's score (and to two broader levels:
market|side|session and market|side, so a new combination can lean on its family's record).

Verdict for a new entry:
  full   the exact setup has >= min_trades_full trades and made money on average  -> normal size
  block  the most specific level with >= min_trades_block trades lost on average  -> no trade
         (one smallest-lot "re-test" is still allowed every retest_hours so it can earn its way back)
  probe  not enough history yet                                                   -> smallest lot only

Scores are kept per broker account in logs/setups_<account>.json.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import tempfile
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# UTC hours. South Africa (SAST) = UTC+2: asia 02-09, london 09-15, new york 15-19, late 19-02.
SESSIONS = (("asia", 0, 7), ("london", 7, 13), ("newyork", 13, 17), ("late", 17, 24))
SESSION_NAMES = {"asia": "Asia", "london": "London", "newyork": "New York", "late": "Late US"}


def session_of(hour_utc: int) -> str:
    for name, a, b in SESSIONS:
        if a <= int(hour_utc) % 24 < b:
            return name
    return "late"


def setup_key(symbol: str, side: int, confidence: float, ppo: float, hour_utc: int | None = None,
              conf_hi: float = 0.72) -> str:
    hour_utc = dt.datetime.now(dt.timezone.utc).hour if hour_utc is None else hour_utc
    ai = "none" if not ppo else ("agree" if (ppo > 0) == (side > 0) else "against")
    return "|".join([str(symbol), "BUY" if side > 0 else "SELL", session_of(hour_utc),
                     "hi" if float(confidence) >= conf_hi else "lo", f"ai-{ai}"])


def levels(key: str) -> list[str]:
    p = key.split("|")
    return [key, "|".join(p[:3]), "|".join(p[:2])]


def describe(key: str) -> str:
    p = key.split("|")
    out = [p[0].rstrip("m"), p[1].lower()] if len(p) >= 2 else [key]
    if len(p) >= 3:
        out.append(SESSION_NAMES.get(p[2], p[2]))
    if len(p) >= 4:
        out.append("confidence high" if p[3] == "hi" else "confidence low")
    if len(p) >= 5:
        out.append({"ai-none": "no AI view", "ai-agree": "AI agrees", "ai-against": "AI disagrees"}.get(p[4], p[4]))
    return ", ".join(out)


class SetupGate:
    def __init__(self, cfg: dict | None = None, account_id: str | None = None):
        c = dict(cfg or {})
        self.enabled = bool(c.get("enabled", True))
        self.min_full = int(c.get("min_trades_full", 20))
        self.min_block = int(c.get("min_trades_block", 12))
        self.retest_s = float(c.get("retest_hours", 4)) * 3600.0
        self.conf_hi = float(c.get("confidence_high", 0.72))
        acct = "".join(ch for ch in str(account_id or os.environ.get("EXNESS_ACCOUNT_ID", "")) if ch.isalnum())
        self.path = os.path.join(os.environ.get("AGI_LOG_DIR") or os.path.join(ROOT, "logs"), f"setups_{acct}.json" if acct else "setups.json")
        self.state = {"stats": {}, "open": [], "retest": {}, "era": ""}
        self._last_resolve = 0.0
        self._load()
        # A new trading-rules era (setup_gate.era) starts the scores fresh: results from the old rules say nothing
        # about the new ones. The old scores are kept beside it (setups_<acct>.before-<era>.json).
        era = str(c.get("era") or "")
        if era and self.state.get("era") != era:
            if self.state["stats"]:
                tag = "".join(ch if ch.isalnum() else "-" for ch in era)[:40]
                try:
                    with open(self.path.replace(".json", f".before-{tag}.json"), "w", encoding="utf-8") as f:
                        json.dump(self.state, f, indent=1)
                except OSError:
                    pass
            self.state.update({"stats": {}, "retest": {}, "era": era})
            self.save()

    # ------------------------------------------------------------------ storage
    def _load(self) -> None:
        try:
            with open(self.path, encoding="utf-8") as f:
                s = json.load(f)
            if isinstance(s, dict):
                self.state.update({k: s.get(k, self.state[k]) for k in self.state})
        except (OSError, ValueError):
            pass

    def save(self) -> None:
        try:
            d = os.path.dirname(self.path)
            os.makedirs(d, exist_ok=True)
            with tempfile.NamedTemporaryFile("w", dir=d, delete=False, suffix=".tmp", encoding="utf-8") as f:
                json.dump(self.state, f, indent=1)
            os.replace(f.name, self.path)
        except Exception:
            pass

    # ------------------------------------------------------------------ scoring
    def key(self, symbol: str, side: int, confidence: float, ppo: float, hour_utc: int | None = None) -> str:
        return setup_key(symbol, side, confidence, ppo, hour_utc, self.conf_hi)

    def add_result(self, key: str, pnl: float, t: float | None = None) -> None:
        for k in levels(key):
            s = self.state["stats"].setdefault(k, {"n": 0, "wins": 0, "net": 0.0})
            s["n"] += 1
            s["wins"] += 1 if pnl > 0 else 0
            s["net"] = round(s["net"] + float(pnl), 2)
            s["last"] = t or time.time()

    def stats(self, key: str) -> dict:
        return self.state["stats"].get(key) or {"n": 0, "wins": 0, "net": 0.0}

    def verdict(self, key: str, now: float | None = None) -> tuple[str, str]:
        if not self.enabled:
            return "full", "setup gate off"
        now = now or time.time()
        s = self.stats(key)
        if s["n"] >= self.min_full and s["net"] > 0:
            return "full", f"proven: {s['n']} trades, {100 * s['wins'] / s['n']:.0f}% won, net ${s['net']:+.2f}"
        for k in levels(key):
            ls = self.stats(k)
            if ls["n"] >= self.min_block:
                if ls["net"] < 0:
                    if now - float(self.state["retest"].get(key, 0)) >= self.retest_s:
                        return "retest", (f"losing record ({describe(k)}: {ls['n']} trades, net ${ls['net']:+.2f}); "
                                          f"one smallest-lot re-test")
                    return "block", f"losing record ({describe(k)}: {ls['n']} trades, net ${ls['net']:+.2f})"
                break
        return "probe", f"not proven yet ({s['n']}/{self.min_full} trades): smallest lot"

    # ------------------------------------------------------------------ journal
    def record_open(self, symbol: str, side: int, key: str, verdict: str) -> None:
        self.state["open"].append({"symbol": symbol, "side": 1 if side > 0 else -1, "key": key, "t": time.time()})
        if verdict == "retest":
            self.state["retest"][key] = time.time()
        self.save()

    def resolve(self, mt5, every_s: float = 60.0) -> int:
        """Match journaled entries to the broker's deals and score the ones that have closed."""
        if not self.state["open"] or time.time() - self._last_resolve < every_s:
            return 0
        self._last_resolve = time.time()
        now = dt.datetime.now(dt.timezone.utc)
        t0 = min(float(o["t"]) for o in self.state["open"]) - 600
        deals = mt5.history_deals_get(dt.datetime.fromtimestamp(t0, dt.timezone.utc), now) or []
        ins = [d for d in deals if d.entry != getattr(mt5, "DEAL_ENTRY_OUT", 1) and d.symbol]
        outs: dict[str, list] = {}
        for d in deals:
            if d.entry == getattr(mt5, "DEAL_ENTRY_OUT", 1):
                outs.setdefault(str(d.position_id), []).append(d)
        open_ids = {str(getattr(p, "ticket", "")) for p in (mt5.positions_get() or [])}
        used, keep, done = set(), [], 0
        buy = getattr(mt5, "ORDER_TYPE_BUY", 0)
        used.update(str(o["pid"]) for o in self.state["open"] if o.get("pid"))
        for o in self.state["open"]:
            pid = str(o.get("pid") or "")
            if not pid:
                cands = [d for d in ins if d.symbol == o["symbol"] and str(d.position_id) not in used
                         and ((d.type == buy) == (o["side"] > 0)) and abs(float(d.time) - float(o["t"])) < 180]
                if not cands:
                    if time.time() - float(o["t"]) < 900:
                        keep.append(o)  # the deal may not be listed yet
                    continue
                pid = str(min(cands, key=lambda x: abs(float(x.time) - float(o["t"]))).position_id)
            elif time.time() - float(o["t"]) > 14 * 86400:
                continue
            used.add(pid)
            if pid in open_ids or pid not in outs:
                o["pid"] = pid
                keep.append(o)
                continue
            self.add_result(o["key"], sum(float(x.profit) for x in outs[pid]), float(o["t"]))
            done += 1
        self.state["open"] = keep
        if done:
            self.save()
        return done

    # ------------------------------------------------------------------ dashboard
    def table(self) -> list[dict]:
        rows = []
        for k, s in self.state["stats"].items():
            if k.count("|") != 4:
                continue
            v, why = self.verdict(k)
            rows.append({"key": k, "label": describe(k), "n": s["n"], "win": round(100 * s["wins"] / max(1, s["n"])),
                         "net": s["net"], "verdict": v, "why": why})
        order = {"full": 0, "probe": 1, "retest": 2, "block": 3}
        return sorted(rows, key=lambda r: (order.get(r["verdict"], 9), -r["n"]))
