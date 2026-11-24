"""Gemma trader: Gemma reads the whole council and makes its OWN buy / sell / hold call.

Separate from every other decision: the council, the meta-brain trader and Gold Breakout keep trading as
before; Gemma only decides for its own positions (hidden from the council, one per market, ATR stop and target,
8 h limit, like the meta-brain trader: Python/meta_trader.AtrTrader).

What Gemma sees each M15 bar per market (built in the main loop, sent to a background worker so the
bot never waits on the model):
  * the council: up / down / flat counts, weighted result, strongest voters each way (with track record)
  * your models: gold order blocks, Kronos, LightGBM, news reader, SMC rules
  * the regime model (LSTM) and PPO
  * the meta-brain's predicted result for a long and a short
  * the Smart Money Concepts structure (Python/smc.py describe)
  * the desk picture (desk_context): account equity / today's result, every open position on the account and
    whose it is, its own open risk, the trading session, high-impact news due, recent price action, and its own
    track record on this market and overall
Gemma answers in a fixed JSON schema (reason first, then decision and confidence); it trades when the
confidence is at least `min_confidence`. An opposite call with enough confidence closes its open position.
Config: trading.gemma_trader. Runs on demo rules: demo, and live while same_on_live is on (Server_AGI).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import queue
import threading
import time

from Python.meta_trader import AtrTrader
from Python.strategy_exec import all_positions, strategy_tickets

PROMPT = """You are the head trader. Decide whether to BUY, SELL or HOLD {asset} now, for a trade that lasts up to
8 hours with a stop {sl} ATR away and a target {tp} ATR away (reward/risk {rr:.1f}). The spread costs about {cost:.2f} of
the stop. Only trade when the evidence clearly agrees; otherwise HOLD. Weigh the whole desk: do not pile more
risk onto a move the account already holds, stand aside just before high-impact news, and learn from your own
recent results on this market.

Your desk's readings (votes run from -1 = strong sell to +1 = strong buy):

{brief}

The desk right now:
{desk}

Smart Money Concepts structure:
{smc}

First write your reasoning in two or three sentences, then the decision, then your confidence from 0 to 1."""

SCHEMA = {"type": "object",
          "properties": {"reason": {"type": "string"},
                         "decision": {"type": "string", "enum": ["buy", "sell", "hold"]},
                         "confidence": {"type": "number"}},
          "required": ["reason", "decision", "confidence"]}

LABEL = {"ob_gold": "Gold order-block strategy", "kronos": "Kronos forecaster", "catboost": "LightGBM return model",
         "news_llm": "News reader", "smc_rules": "SMC fixed rules", "smc_gemma": "Gemma SMC read", "meta_brain": "Meta-brain"}


def council_brief(council_meta: dict | None, decision: dict | None, ext=None, symbol: str = "") -> str:
    """The desk's readings as short lines for the model."""
    lines = []
    cm = council_meta or {}
    if cm:
        lines.append(f"Indicator council: {cm.get('up', 0)} say buy, {cm.get('down', 0)} say sell, {cm.get('flat', 0)} neutral; "
                     f"weighted result {float(cm.get('target', 0.0)):+.2f}.")
        votes, assess = cm.get("votes") or {}, cm.get("assess") or {}
        ranked = sorted(((k, v) for k, v in votes.items() if k not in LABEL and abs(v) > 0.05),
                        key=lambda kv: -abs(kv[1]) * float((assess.get(kv[0]) or {}).get("mult", 1.0)))
        for side, name in ((1, "buy"), (-1, "sell")):
            top = [f"{k} {v:+.2f}" + (f" (hit {100 * a['hit']:.0f}%)" if (a := assess.get(k) or {}).get("hit") is not None else "")
                   for k, v in ranked if (v > 0) == (side > 0)][:5]
            if top:
                lines.append(f"Strongest {name} indicators: " + ", ".join(top) + ".")
        models = [f"{LABEL[k]} {v:+.2f}" for k, v in votes.items() if k in LABEL and k != "smc_gemma"]
        if models:
            lines.append("Models: " + ", ".join(models) + ".")
    d = decision or {}
    if d:
        lstm = (f"Regime model (LSTM): {d.get('agi_direction', 'HOLD')} with confidence {float(d.get('agi_confidence', 0)):.2f}"
                if d.get("agi_proven") else "Regime model (LSTM): not proven on this market yet, ignore it")
        lines.append(f"{lstm}; PPO {float(d.get('ppo_target', 0)):+.2f}; the bot's own blended signal {float(d.get('target', 0)):+.2f}.")
    det = (getattr(ext, "details", None) or {}).get((symbol, "meta_brain")) if ext is not None else None
    if det and "pred_long" in det:
        lines.append(f"Meta-brain (trained on every strategy): a long now is predicted to make {det['pred_long']:+.2f}R, "
                     f"a short {det['pred_short']:+.2f}R (it trades only above +0.30R).")
    return "\n".join(lines) or "No readings."


# One shared worker answers for every LLM trader. Each round it takes EVERYTHING queued at that moment, grouped by
# trader (one model load per group), so no model can starve another: with a lock per trader, Gemma's slow calls kept
# new Gemma requests arriving before its batch ended and Qwen never got a turn (2026-11-24).
_JOBS: queue.Queue = queue.Queue()
_WORKER: list = []


def _shared_worker(q: queue.Queue = _JOBS):
    while True:
        jobs = [q.get()]
        try:
            while True:
                jobs.append(q.get_nowait())
        except queue.Empty:
            pass
        _round(jobs)


def _round(jobs) -> None:
    """Answer one round: first come, first served by trader, each trader's jobs together (one model load)."""
    order = []
    for tr, _ in jobs:
        if tr not in order:
            order.append(tr)
    for tr in order:
        for t2, job in jobs:
            if t2 is tr:
                tr._answer(*job)


class GemmaTrader(AtrTrader):
    NAME = "gemma trader"
    TAG = "GEMMA"
    FILE = "gemma_trader"
    DEFAULTS = {"enabled": False, "risk_pct": 5.0, "min_lot_max_risk_pct": 5.0, "max_open": 7, "magic": 77900,
                "model": "gemma4:e4b-it-qat", "url": "http://127.0.0.1:11434", "timeout_s": 180,
                "min_confidence": 0.6, "max_decision_age_s": 1200, "every_min": 15}
    OWN_WORKER = False   # True: answered on its own thread (a fast remote API never waits behind a local model)

    def __init__(self, cfg, log_dir, logger=None, ask=None):
        super().__init__(cfg, log_dir, logger)
        self.decisions: dict = {}      # sym -> {"t", "stamp", "decision", "confidence", "reason"}
        self._stamp: dict = {}
        self._jobs: queue.Queue = queue.Queue() if self.OWN_WORKER else _JOBS
        self._ask = ask                # injectable model call (tests)
        if self.enabled:
            if self.OWN_WORKER:
                threading.Thread(target=_shared_worker, args=(self._jobs,), name=self.FILE, daemon=True).start()
            elif not _WORKER:
                _WORKER.append(threading.Thread(target=_shared_worker, name="llm-traders", daemon=True))
                _WORKER[0].start()

    # ------------------------------------------------------------------ background: ask Gemma
    def slot(self, bar_stamp: str, now: float | None = None) -> str:
        """The decision stamp: the M15 bar, or with `every_min` under 15 the bar plus the current `every_min`
        slice of the clock, so the model is asked again every few minutes inside the bar."""
        every = float(self.cfg.get("every_min") or 15)
        if every >= 15:
            return bar_stamp
        now = time.time() if now is None else now
        sl = int(now // (every * 60) * every * 60)
        return f"{bar_stamp}|{dt.datetime.fromtimestamp(sl, dt.timezone.utc):%H:%M}"

    def wants(self, symbol: str, stamp: str) -> bool:
        """True when this decision slot has not been sent yet (build the brief only then)."""
        return self.enabled and symbol not in (self.cfg.get("skip_symbols") or ()) and self._stamp.get(symbol) != stamp

    def _today(self, now: float) -> dict:
        """{symbol: trades opened} for the current UTC day (state["day"])."""
        day = dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%d")
        if (self.state.get("day") or {}).get("d") != day:
            n: dict = {}                                  # new day, or first run: count today's opens from the log
            t0 = dt.datetime.strptime(day, "%Y-%m-%d").replace(tzinfo=dt.timezone.utc).timestamp()
            try:
                with open(self.events, encoding="utf-8") as f:
                    for ln in f:
                        if '"open"' in ln:
                            e = json.loads(ln)
                            if e.get("event") == "open" and float(e.get("t", 0)) >= t0 and e.get("symbol"):
                                n[e["symbol"]] = n.get(e["symbol"], 0) + 1
            except (OSError, ValueError):
                pass
            self.state["day"] = {"d": day, "n": n}
        return self.state["day"]["n"]

    def submit(self, symbol: str, stamp: str, brief: str, m15, desk: str = "", atr_m5: float | None = None) -> None:
        """Queue one decision per market per M15 bar (never blocks the bot)."""
        if not self.wants(symbol, stamp):
            return
        self._stamp[symbol] = stamp
        self._jobs.put((self, (symbol, stamp, brief, None if m15 is None else m15.tail(1500).copy(), desk, atr_m5)))

    # ------------------------------------------------------------------ the desk picture (main thread: broker calls)
    def track_record(self, trade_log: str) -> dict:
        """{symbol: [profit, ...]} of Gemma's closed trades, oldest first, from the trade log."""
        mine = set()
        try:
            with open(self.events, encoding="utf-8") as f:
                for ln in f:
                    if '"open"' in ln and '"ticket"' in ln:
                        try:
                            mine.add(str(json.loads(ln).get("ticket")))
                        except ValueError:
                            pass
        except OSError:
            pass
        out: dict = {}
        seen = set()
        try:
            with open(trade_log, encoding="utf-8") as f:
                for ln in f:
                    if '"trade_closed"' not in ln:
                        continue
                    try:
                        p = json.loads(ln).get("payload") or {}
                    except ValueError:
                        continue
                    if str(p.get("ticket")) in mine and p.get("deal_id") not in seen:
                        seen.add(p.get("deal_id"))
                        out.setdefault(str(p.get("symbol")), []).append(float(p.get("profit") or 0.0))
        except OSError:
            pass
        return out

    def desk_context(self, mt5, symbol: str, m15=None, atr_m5: float | None = None, now: float | None = None,
                     calendar: list | None = None) -> str:
        """Plain lines on everything around the trade: account, positions, session, news, price, own record."""
        from Python.external_voters import ASSETS
        now = time.time() if now is None else now
        log_dir = os.path.dirname(self.events)
        lines = []
        acc = mt5.account_info()
        positions = all_positions(mt5)
        if acc is not None:
            eq, bal = float(acc.equity), float(acc.balance)
            floating = sum(float(getattr(p, "profit", 0.0) or 0.0) for p in positions)
            lines.append(f"Account: equity ${eq:,.2f}, balance ${bal:,.2f}, open profit/loss ${floating:+,.2f}.")
        # whose positions: Gemma's own, the other stand-alone strategies, else the council bot
        own = {str(v["ticket"]): k for k, v in self.state["open"].items()}
        strat = strategy_tickets(log_dir)
        rows, net = [], {}
        for p in positions:
            side = 1 if int(getattr(p, "type", 0)) == 0 else -1
            who = "yours" if str(p.ticket) in own else ("other strategy" if str(p.ticket) in strat else "council bot")
            net[p.symbol] = net.get(p.symbol, 0) + side
            rows.append(f"{ASSETS.get(p.symbol, p.symbol)} {'BUY' if side > 0 else 'SELL'} {float(p.volume):g} lots "
                        f"${float(getattr(p, 'profit', 0.0) or 0.0):+,.2f} ({who})")
        if rows:
            lines.append(f"Open positions ({len(rows)}): " + "; ".join(rows[:14]) + ".")
            here = [r for r, p in zip(rows, positions) if p.symbol == symbol]
            lines.append(f"On {ASSETS.get(symbol, symbol)}: " + ("; ".join(here) if here else "nothing open") + ".")
        else:
            lines.append("Open positions: none.")
        g_risk = sum(float(v.get("risk_pct", 0.0) or 0.0) for v in self.state["open"].values())
        lines.append(f"Your own open trades: {len(self.state['open'])} of max {self.cfg['max_open']}, "
                     f"risking {g_risk:.1f}% of equity in total; each new trade risks {float(self.cfg['risk_pct']):g}%.")
        # session
        t = dt.datetime.fromtimestamp(now, dt.timezone.utc)
        h = t.hour
        sess = ("Asian session" if h < 7 else "London session" if h < 12 else "London / New York overlap" if h < 16
                else "New York session" if h < 21 else "late New York, thin liquidity")
        lines.append(f"Time: {t:%A %H:%M} UTC, {sess}.")
        # market opens: the first half hour after London / New York opens whipsaws stops
        from Python.entry_window import market_opens, moves
        soon = [(n, t) for n, t in market_opens(now, 6.0) if now - 1800 <= t <= now + 6 * 3600]
        lines.append("Market opens: " + ("; ".join(
            f"{n} {'in ' + str(int((t - now) // 60)) + ' min' if t > now else str(int((now - t) // 60)) + ' min ago'}"
            for n, t in soon) if soon else "none in the next 6 h") + ".")
        # news due in the next 6 h (or just out) that moves this market
        try:
            if calendar is None:
                with open(os.path.join(log_dir, "calendar_cache.json"), encoding="utf-8") as f:
                    calendar = json.load(f).get("events") or []
            due = sorted((e for e in calendar if now - 1800 <= float(e["t"]) <= now + 6 * 3600
                          and str(e.get("impact")) in ("High", "Medium") and moves(e, symbol)), key=lambda e: e["t"])
            lines.append("News in the next 6 h: " + ("; ".join(
                f"{e['title']} ({e['impact']}) " + (f"in {int((float(e['t']) - now) // 60)} min" if float(e["t"]) > now
                                                    else f"{int((now - float(e['t'])) // 60)} min ago")
                for e in due[:5]) if due else "none") + ". New trades are blocked from 30 min before to 30 min after "
                         "High news and around market opens.")
        except (OSError, ValueError, KeyError, TypeError):
            pass
        # price action from the M15 bars
        if m15 is not None and len(m15) >= 100:
            c, hi, lo = m15["close"].astype(float), m15["high"].astype(float), m15["low"].astype(float)
            px = float(c.iloc[-1])
            atr = float(atr_m5) * 3 ** 0.5 if atr_m5 else float((hi - lo).tail(14).mean())
            d_hi, d_lo = float(hi.tail(96).max()), float(lo.tail(96).min())
            where = (px - d_lo) / (d_hi - d_lo) * 100 if d_hi > d_lo else 50.0
            lines.append(f"Price {px:g}: {((px - float(c.iloc[-17])) / atr):+.1f} ATR over 4 h, "
                         f"{((px - float(c.iloc[-97])) / atr):+.1f} ATR over 24 h; 24 h range {d_lo:g} to {d_hi:g}, "
                         f"price at {where:.0f}% of it.")
        # own track record
        rec = self.track_record(os.path.join(log_dir, "trade_events.jsonl"))
        allp = [x for v in rec.values() for x in v]
        if allp:
            mine = rec.get(symbol, [])
            here = (f"on {ASSETS.get(symbol, symbol)} {sum(x > 0 for x in mine)} won / {sum(x <= 0 for x in mine)} lost, "
                    f"net ${sum(mine):+,.2f}" + (f", last {', '.join(f'${x:+.0f}' for x in mine[-3:])}" if mine else "")
                    if mine else f"no closed trades yet on {ASSETS.get(symbol, symbol)}")
            lines.append(f"Your record: {sum(x > 0 for x in allp)} won / {sum(x <= 0 for x in allp)} lost overall, "
                         f"net ${sum(allp):+,.2f}; {here}.")
        else:
            lines.append("Your record: no closed trades yet.")
        return "\n".join(lines)

    def decide(self, symbol, stamp, brief, m15, desk: str = "", atr_m5: float | None = None) -> dict:
        from Python import meta_brain as mb
        from Python import smc
        from Python.external_voters import ASSETS, ollama_json
        a = smc.analyze(m15) if m15 is not None else None
        asset = ASSETS.get(symbol, symbol)
        sl, tp = self.stops_for(symbol)
        # The stops sit at M5 ATR multiples (meta_trader.AtrTrader.open), so the spread is measured against the M5
        # ATR; without it, M15 ATR / sqrt(3) (ATR grows ~ with the square root of the bar length).
        atr = float(atr_m5) if atr_m5 else (float((a or {}).get("atr") or 0.0) / 3 ** 0.5)
        cost = mb.SPREAD_BPS.get(mb.base(symbol), 1.0) / 1e4 * (a["price"] if a else 1.0) / (atr or 1e9) / sl
        prompt = PROMPT.format(asset=asset, sl=sl, tp=tp, rr=tp / sl, cost=cost, brief=brief, desk=desk or "not available",
                               smc=smc.describe(a, asset) if a else "not available")
        ask = self._ask or (lambda p: ollama_json(self.cfg["url"], self.cfg["model"], p, SCHEMA, num_predict=400,
                                                  timeout=int(self.cfg["timeout_s"])))
        js = ask(prompt)
        dec = str(js.get("decision", "hold")).lower()
        conf = max(0.0, min(1.0, float(js.get("confidence", 0.0) or 0.0)))
        return {"t": time.time(), "stamp": stamp, "decision": dec if dec in ("buy", "sell") else "hold",
                "confidence": round(conf, 2), "reason": str(js.get("reason") or "")[:400], "desk": (desk or "")[:1500]}

    def _answer(self, symbol, stamp, brief, m15, desk, atr_m5):
        try:
            self.decisions[symbol] = self.decide(symbol, stamp, brief, m15, desk, atr_m5)
            self.state.setdefault("last", {})[symbol] = self.decisions[symbol]
            self._save()
        except Exception as exc:
            self.decisions[symbol] = {"t": time.time(), "stamp": stamp, "decision": "hold", "confidence": 0.0,
                                      "reason": f"model failed: {type(exc).__name__}: {exc}"[:200], "error": True}
            if self.log:
                self.log.warning(f"{self.NAME} failed on {symbol}: {exc}")

    # ------------------------------------------------------------------ main loop: act on the latest call
    def step(self, mt5, symbols, can_open: bool = True, now: float | None = None, block=None) -> None:
        """`block(symbol) -> reason | None`: hard no-new-trade window (Python/entry_window.py: news, market opens)."""
        if not self.enabled:
            return
        now = time.time() if now is None else now
        for sym in symbols:
            d = self.decisions.get(sym)
            fresh = d is not None and now - d["t"] <= float(self.cfg["max_decision_age_s"])
            side = {"buy": 1, "sell": -1}.get(d["decision"], 0) if fresh else 0
            sure = fresh and d["confidence"] >= float(self.cfg["min_confidence"])
            if self.manage(mt5, sym, now):
                pos = self.state["open"][sym]
                if sure and side == -pos["side"]:
                    self.close(mt5, sym, f"{self.NAME.split()[0].capitalize()} turned {d['decision']} ({d['confidence']:.2f})")
                continue
            if sym in (self.cfg.get("skip_symbols") or ()):
                continue                                  # not traded by this model (trading.<trader>.skip_symbols)
            if not (sure and side) or self.state["seen"].get(sym) == d["stamp"]:
                continue
            self.state["seen"][sym] = d["stamp"]
            why_not = block(sym) if block else None
            per_day = int(self.cfg.get("max_trades_per_market_day") or 0)
            today = self._today(now)
            if not why_not and per_day and today.get(sym, 0) >= per_day:
                why_not = f"already {today[sym]} trades on this market today (max {per_day})"
            if not can_open or why_not or len(self.state["open"]) >= int(self.cfg["max_open"]):
                self._event("skip", symbol=sym, reason="risk engine halt" if not can_open else why_not or "max open")
                self._save()
                continue
            self.open(mt5, sym, side, confidence=d["confidence"], why=d["reason"][:200])
            if sym in self.state["open"]:
                today[sym] = today.get(sym, 0) + 1
                self._save()


class QwenTrader(GemmaTrader):
    """Same desk picture, same rules, a different brain: Qwen 3.5 9B with its own positions and track record
    (logs/qwen_trader.jsonl), so the two models can be compared on real results. Config: trading.qwen_trader."""
    NAME = "qwen trader"
    TAG = "QWEN"
    FILE = "qwen_trader"
    DEFAULTS = {**GemmaTrader.DEFAULTS, "model": "qwen3.5:9b", "magic": 78900}


def _env_value(name: str, files) -> str:
    """Key from the environment, else from the first `NAME=value` line in one of `files` (keys stay out of the repo)."""
    if os.environ.get(name):
        return os.environ[name]
    for path in files or ():
        try:
            with open(os.path.expanduser(path), encoding="utf-8") as f:
                for ln in f:
                    if ln.startswith(name + "="):
                        v = ln.split("=", 1)[1].strip().strip('"').strip("'")
                        if v:
                            return v
        except OSError:
            continue
    return ""


class JevTrader(GemmaTrader):
    """Same desk picture, same rules, but the brain is Jev (TypeSafe's decision model), which answers a choice
    question with calibrated probabilities instead of free text. Replaces the Qwen slot (user 2026-11-24): Qwen 9B
    didn't read the desk well, and on 16 GB it kept pushing Gemma out of GPU memory (both timed out).

    Routes, in order (`routes`): OpenRouter's Decisions API with typesafe/jev-1.13 (OPENROUTER_API_KEY), as Quant-OS and
    jev-trader use it, then jevai.org /api/v1/decisions (JEV_API_KEY). Keys come from the environment or `key_files`. If neither answers the call is a hold (no trade on a guess) and Jev is skipped for
    `cooldown_s` so a dead endpoint never stalls the shared worker Gemma also uses. Config: trading.jev_trader."""
    NAME = "jev trader"
    TAG = "JEV"
    FILE = "jev_trader"
    DEFAULTS = {**GemmaTrader.DEFAULTS, "model": "typesafe-ai/jev", "magic": 79900, "timeout_s": 25,
                "routes": ["openrouter", "jevai"],
                "url": "https://www.jevai.org/api/v1/decisions",
                "openrouter_url": "https://openrouter.ai/api/alpha/decisions", "openrouter_model": "typesafe/jev-1.13",
                "key_files": ["/Volumes/AI_DRIVE/jev-trader/.env", "~/.config/xau-v3/jev.env"], "cooldown_s": 300,
                "every_min": 5}
    OWN_WORKER = True

    QUESTION = {"decision": {
        "type": "choice",
        "instructions": {
            "question": "Open a trade on `asset` now for up to 8 hours, or stand aside?",
            "goal": "Make money after costs on a small account. Each trade risks `risk_per_trade` of equity with a "
                    "broker stop and target `stops` away; `cost` is the spread as a fraction of the stop.",
            "inputs": "`council` holds the indicator votes and model forecasts for this market, `smc` the market "
                      "structure, `desk` the account, every open position, the session, market opens, scheduled "
                      "news and this trader's own record. Positions marked council bot or other strategy belong to "
                      "other traders with their own risk: judge this market on its own merits, and only let your "
                      "own open trades count against a new one. News and market opens are already blocked by the "
                      "system, so you are only asked when trading is allowed.",
        },
        "criteria": {
            "buy": "Price more likely to reach the target above before the stop below, by enough to beat the cost.",
            "sell": "Price more likely to reach the target below before the stop above, by enough to beat the cost.",
            "hold": "No clear edge after costs, the evidence conflicts, or your own open trades already hold "
                    "this same bet. Other traders' positions are not a reason to hold.",
        },
    }}

    def __init__(self, cfg, log_dir, logger=None, ask=None):
        super().__init__(cfg, log_dir, logger, ask)
        self._down_until = 0.0

    def _post(self, url, key, body, extra_headers=None) -> dict:
        import urllib.request
        req = urllib.request.Request(url, data=json.dumps(body, default=str).encode(), method="POST",
                                     headers={"Authorization": "Bearer " + key, "Content-Type": "application/json",
                                              "Accept": "application/json", "User-Agent": "agi-trader/1.0",
                                              **(extra_headers or {})})
        with urllib.request.urlopen(req, timeout=float(self.cfg["timeout_s"])) as r:
            return json.loads(r.read().decode())

    def ask_jev(self, state: dict) -> dict:
        """{"choice", "probabilities"} from the first route that answers. OpenRouter's Decisions API with
        typesafe/jev-1.13 comes first: it is the route Quant-OS and jev-trader run on; jevai.org is the backup."""
        import urllib.error
        errors = []
        files = self.cfg.get("key_files") or ()
        for route in self.cfg.get("routes") or ("openrouter", "jevai"):
            try:
                if route == "openrouter":
                    key = _env_value("OPENROUTER_API_KEY", files)
                    if not key:
                        errors.append("openrouter: no OPENROUTER_API_KEY")
                        continue
                    js = self._post(self.cfg["openrouter_url"], key, {"model": self.cfg["openrouter_model"],
                                                                      "state": state, "questions": self.QUESTION},
                                    {"HTTP-Referer": "https://localhost/cautious-giggle", "X-Title": "cautious-giggle"})
                    a = (js.get("answers") or {}).get("decision") or {}
                    if a.get("choice"):
                        return a
                    errors.append(f"openrouter: no answer {str(js)[:120]}")
                elif route == "jevai":
                    key = _env_value("JEV_API_KEY", files)
                    if not key:
                        errors.append("jevai: no JEV_API_KEY")
                        continue
                    js = self._post(self.cfg["url"], key, {"model": self.cfg["model"], "state": state,
                                                           "questions": self.QUESTION})
                    if int(js.get("code", -1)) != 0:
                        raise RuntimeError(f"code {js.get('code')}: {js.get('message')}")
                    a = (js.get("data") or {}).get("answers", {}).get("decision") or {}
                    if a.get("choice"):
                        return a
                    errors.append(f"jevai: no answer {str(js)[:120]}")
            except urllib.error.HTTPError as exc:
                body = exc.read().decode("utf-8", errors="replace")[:160]
                errors.append(f"{route}: HTTP {exc.code} {body}")
            except (urllib.error.URLError, OSError, ValueError, RuntimeError) as exc:
                errors.append(f"{route}: {exc}"[:160])
        raise RuntimeError("; ".join(errors))

    def decide(self, symbol, stamp, brief, m15, desk: str = "", atr_m5: float | None = None) -> dict:
        from Python import meta_brain as mb
        from Python import smc
        from Python.external_voters import ASSETS
        if time.time() < self._down_until and self._ask is None:
            raise RuntimeError("Jev unreachable, cooling down")
        a = smc.analyze(m15) if m15 is not None else None
        sl, tp = self.stops_for(symbol)
        atr = float(atr_m5) if atr_m5 else (float((a or {}).get("atr") or 0.0) / 3 ** 0.5)
        cost = mb.SPREAD_BPS.get(mb.base(symbol), 1.0) / 1e4 * (a["price"] if a else 1.0) / (atr or 1e9) / sl
        asset = ASSETS.get(symbol, symbol)
        state = {"asset": asset, "stops": f"stop {sl:g} ATR, target {tp:g} ATR (reward/risk {tp / sl:.1f})",
                 "cost": round(cost, 3), "risk_per_trade": f"{float(self.cfg['risk_pct']):g}%",
                 "council": brief, "smc": smc.describe(a, asset) if a else "not available", "desk": desk or "not available"}
        try:
            ans = (self._ask or self.ask_jev)(state)
        except Exception:
            self._down_until = time.time() + float(self.cfg["cooldown_s"])
            raise
        probs = {k: float(v) for k, v in (ans.get("probabilities") or {}).items()}
        dec = str(ans.get("choice", "hold")).lower()
        dec = dec if dec in ("buy", "sell") else "hold"
        conf = probs.get(dec, float(ans.get("confidence", 1.0 if not probs else 0.0) or 0.0))
        why = "Jev " + ", ".join(f"{k} {v:.2f}" for k, v in sorted(probs.items(), key=lambda x: -x[1])) if probs else "Jev"
        return {"t": time.time(), "stamp": stamp, "decision": dec, "confidence": round(max(0.0, min(1.0, conf)), 2),
                "reason": why, "desk": (desk or "")[:1500]}
