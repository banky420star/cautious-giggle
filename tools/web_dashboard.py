"""Local web dashboard server: serves the React app (web/dist) and a JSON snapshot at /api/state.

    python tools/web_dashboard.py [--port 8788]

Binds to 127.0.0.1 only. Read-only: it never places or changes orders. Equity is sampled every
15 s into logs/equity_history.csv so the chart survives restarts.
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import mimetypes
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import monitor_tui as m
import ops_log  # noqa: E402
import account_profiles as ap  # noqa: E402
import market_truth as mtruth  # noqa: E402
import warroom as wr  # noqa: E402
from narrator import Narrator  # noqa: E402
from Python.indicator_council import describe  # noqa: E402

DIST = os.path.join(ROOT, "web", "dist")
EQ_FILE = os.path.join(ROOT, "logs", "equity_history.csv")
NAMES = m.MARKET_NAMES



def money_fmt(v) -> str:
    v = float(v or 0)
    return f"{'+' if v >= 0 else '-'}${abs(v):.2f}"

class State:
    def __init__(self):
        self.cfg = m._cfg()
        self.symbols, news_extra = m.bot_symbols(self.cfg)
        self.broker = m.Broker(self.symbols, extra=news_extra)
        self.logs = m.Logs()
        self.lock = threading.Lock()
        self.log_lock = threading.Lock()
        self.equity: list[tuple[float, float, float]] = []  # (ts, equity, balance)
        self._load_equity()
        self.narrator = Narrator()
        self.news = None
        threading.Thread(target=self._news_loop, daemon=True).start()
        threading.Thread(target=self._sampler, daemon=True).start()
        threading.Thread(target=self._narrate, daemon=True).start()

    # ------------------------------------------------------------------ equity history
    def _load_equity(self):
        try:
            with open(EQ_FILE, "r", encoding="utf-8") as f:
                for line in f:
                    p = line.strip().split(",")
                    if len(p) == 3 and float(p[1]) > 0:  # 0 = broker link was down, not a real value
                        self.equity.append((float(p[0]), float(p[1]), float(p[2])))
        except FileNotFoundError:
            pass
        except Exception:
            self.equity = []
        if not self.equity:  # backfill from the bot's own snapshots
            for ev in self._snapshots():
                self.equity.append(ev)

    def _snapshots(self):
        out = []
        for line in m.tail_lines(os.path.join(ROOT, "logs", "audit_events.jsonl"), 20000, max_bytes=20_000_000):
            if '"snapshot"' not in line:
                continue
            try:
                r = json.loads(line)
                p = r["payload"]
                out.append((dt.datetime.fromisoformat(r["ts"]).timestamp(), float(p["equity"]), float(p["balance"])))
            except Exception:
                continue
        return out

    def _sampler(self):
        while True:
            acc = self.broker.account
            if acc is not None and float(acc.equity or 0) > 0:
                row = (time.time(), float(acc.equity), float(acc.balance))
                with self.lock:
                    self.equity.append(row)
                    self.equity = self.equity[-200_000:]
                try:
                    os.makedirs(os.path.dirname(EQ_FILE), exist_ok=True)
                    with open(EQ_FILE, "a", encoding="utf-8") as f:
                        f.write(f"{row[0]:.0f},{row[1]:.2f},{row[2]:.2f}\n")
                except Exception:
                    pass
            time.sleep(15)

    def _news_loop(self):
        time.sleep(8)
        try:
            from Python.news_markets import NewsMarkets
            if self.broker.mt5 is None:
                return
            self.news = NewsMarkets(self.broker.mt5, poll_sec=300)
            self.news.loop()
        except Exception as exc:
            print("news loop stopped:", exc)

    def _news_snapshot(self) -> dict:
        from Python import news_trader as nt
        try:
            with open(os.path.join(m.LOG_DIR, "news_state.json"), encoding="utf-8") as f:
                live = json.load(f)
        except Exception:
            live = {"enabled": False, "events": []}
        pb = nt.load_playbook()
        book = []
        for sym, topics in pb.items():
            for topic, plays in topics.items():
                for play, rows in plays.items():
                    st = nt.stats(rows)
                    book.append({"symbol": sym, "topic": topic, "play": play, **{k: round(v, 3) for k, v in st.items()}})
        nm = self.news
        return {"enabled": live.get("enabled"), "plan": live.get("events", []),
                "recent": nm.recent(30) if nm else [], "upcoming": nm.upcoming() if nm else [],
                "playbook": book, "errors": (nm.status.get("errors") if nm else [])}

    def _narrate(self):
        time.sleep(6)  # let the broker poller fill in first
        while True:
            try:
                with self.log_lock:
                    self.logs.refresh()
                self.narrator.tick(self.logs, m.read_progress(), self.broker, m.tail_lines)
            except Exception:
                pass
            time.sleep(10)

    # ------------------------------------------------------------------ snapshot
    def snapshot(self) -> dict:
        with self.log_lock:
            self.logs.refresh()
        b, logs = self.broker, self.logs
        alive, _ = m.bot_status()
        acc = b.account
        floating = sum(float(getattr(p, "profit", 0) or 0) for p in b.positions)
        today = (b.pnl_today or 0.0) + floating
        eq = float(getattr(acc, "equity", 0) or 0)
        trading = self.cfg.get("trading") or {}
        risk = self.cfg.get("risk") or {}

        out_entry = int(getattr(b.mt5, "DEAL_ENTRY_OUT", 1)) if b.mt5 else 1
        deals = [d for d in b.deals if float(getattr(d, "volume", 0) or 0) > 0 and str(getattr(d, "symbol", ""))]
        closed = [float(d.profit) + float(getattr(d, "commission", 0) or 0) + float(getattr(d, "swap", 0) or 0)
                  for d in deals if int(getattr(d, "entry", -1)) == out_entry]

        infos = {r["symbol"]: r for r in m.market_rows(self.broker)}
        all_syms = list(self.symbols) + list(self.broker.extra)
        truth = mtruth.overview(all_syms, self.cfg, ap.active())
        symbols = []
        for s in all_syms:
            row = self._symbol(s)
            inf = infos.get(s) or {}
            t = truth["markets"][s]
            grad = dict(inf.get("grad") or {}) or None
            if grad is not None:
                grad.update(why=t["training"]["why"], summary=t["training"]["summary"])
            row.update(mode=t["trades"]["mode"], ai=t["ai"], brain=t["ai"]["short"], training=t["training"]["summary"],
                       grad=grad, news=inf.get("news"), chg_pct=inf.get("chg_pct"))
            if row["mode"] == "news only" and not row.get("position"):
                row["why"] = t["trades"]["text"]
            symbols.append(row)

        with self.lock:
            eqh = list(self.equity)
        # thin the series to <= 1500 points for the browser
        step = max(1, len(eqh) // 1500)
        eq_series = [{"t": int(t * 1000), "equity": round(e, 2), "balance": round(bl, 2)} for t, e, bl in eqh[::step]]
        if eqh and eqh[-1] is not eqh[::step][-1]:
            t, e, bl = eqh[-1]
            eq_series.append({"t": int(t * 1000), "equity": round(e, 2), "balance": round(bl, 2)})

        return {
            "ts": time.time(),
            "bot": {"running": alive, "learning_mode": bool((trading.get("demo_learning") or {}).get("enabled")),
                    "next_check_s": int(300 - time.time() % 300), "broker_ok": bool(b.ok and not b.err), "broker_err": b.err},
            "account": {"equity": eq, "balance": float(getattr(acc, "balance", 0) or 0),
                        "free_margin": float(getattr(acc, "margin_free", 0) or 0), "floating": floating,
                        "today": today, "today_pct": (today / (eq - today) * 100) if eq and eq != today else 0.0,
                        "daily_loss_limit": self._loss_limit(risk)},
            "stats": {"closed": len(closed), "wins": sum(1 for x in closed if x > 0),
                      "losses": sum(1 for x in closed if x <= 0), "pnl_closed": sum(closed)},
            "equity": eq_series,
            "symbols": symbols,
            "trades": self._trades(deals, out_entry),
            "training": self._training(),
            "feed": self.narrator.feed(60),
            "news": self._news_snapshot(),
            "markets": m.market_rows(self.broker),
            "mission": self._mission(),
            "scorecard": self._scorecard(),
            "changes": __import__("Python.auto_changes", fromlist=["recent"]).recent(40),
            "blotter": self._blotter(),
            "setups": self._setups(),
            "models": self._models(),
            "account_mode": ap.active(),
            "ai_line": truth["line"],
            "ops": ops_log.events(os.path.join(m.LOG_DIR, "server.log"), NAMES, 50),
            "tip": dict(zip(("term", "text"), wr.GLOSSARY[int(time.time() // 12) % len(wr.GLOSSARY)])),
            "glossary": [{"term": t, "text": x} for t, x in wr.GLOSSARY],
        }

    def _loss_limit(self, risk: dict) -> float:
        """The daily loss stop the bot actually uses (demo learning overrides the default)."""
        dl = ((self.cfg.get("trading") or {}).get("demo_learning") or {})
        if dl.get("enabled") and "max_daily_loss" in (dl.get("risk") or {}) and self._is_demo():
            return float(dl["risk"]["max_daily_loss"])
        return float(risk.get("max_daily_loss", 0) or 0)

    def _is_demo(self) -> bool:
        try:
            from Python import demo_exploration
            return bool(demo_exploration.demo_rules_apply(self.broker.mt5, self.cfg))
        except Exception:
            return False

    def _symbol(self, s: str) -> dict:
        b, logs = self.broker, self.logs
        tick = b.ticks.get(s)
        sg = (logs.last_signal.get(s) or {}).get("payload") or {}
        cv = (logs.last_council.get(s) or {}).get("payload") or {}
        said = self._models_said(s, sg) or {"lstm": {"counted": 0.0, "raw": 0.0}, "ppo": {"counted": 0.0}}
        mine = [p for p in b.positions if str(getattr(p, "symbol", "")) in (s, s.rstrip("m"))]
        prof = sg.get("decision_profile") if isinstance(sg.get("decision_profile"), dict) else {}
        thr = float(prof.get("min_trade_threshold", 0.15) or 0.15)
        raw = float(sg.get("raw_target", 0) or 0) * float(sg.get("risk_scalar", 1) or 1)
        up, down = int(cv.get("up", 0) or 0), int(cv.get("down", 0) or 0)
        pos = None
        if mine:
            p0 = mine[0]
            pos = {"side": "BUY" if int(p0.type) == 0 else "SELL", "volume": sum(float(p.volume) for p in mine),
                   "entry": float(p0.price_open), "sl": float(p0.sl or 0), "tp": float(p0.tp or 0),
                   "pnl": sum(float(p.profit) for p in mine)}
        lean = "BUY" if raw > 0 else "SELL"
        if not sg:
            why = "Waiting for its first check."
        elif pos:
            why = "Managing the trade: stop-loss and take-profit are set; it exits if the signals turn."
        elif abs(raw) >= thr and s in self._blocks():
            why = f"Signals agree on {lean}, but the account is too small for this market's minimum lot (needs ~${self._blocks()[s]['need']:,})."
        elif abs(raw) >= thr:
            why = f"Signals agree on {lean}. Placing a trade at the next check."
        elif (raw > 0 and down > up) or (raw < 0 and up > down):
            why = (f"Leaning {lean}: more signals say {'down' if raw > 0 else 'up'}, but the ones with the better "
                   f"track record say {'up' if raw > 0 else 'down'}.")
        elif abs(raw) >= thr * 0.6:
            why = f"Leaning {lean}. Almost strong enough to trade."
        else:
            why = "Signals are mixed. Not worth paying the spread yet."
        assess = cv.get("assess") or {}
        ranked = sorted(assess.items(), key=lambda kv: -float((kv[1] or {}).get("mult", 1)))
        trusts = [describe(k) for k, v in ranked[:3] if (v or {}).get("verdict") == "works"]
        ignores = [describe(k) for k, v in ranked[::-1][:3] if (v or {}).get("verdict") == "doesn't work"]
        hist = [{"t": h[0], "v": round(h[1], 3)} for h in (logs.sig_hist.get(s) or [])[-120:]]
        pe = logs.pending_entry.get(s)
        pending = None
        if pe and float(pe.get("expires", 0)) > time.time() - 30 and not pos:
            pending = {"side": pe.get("side"), "limit": float(pe["limit"]), "ref": float(pe["ref"]),
                       "expires_in": max(0, int(float(pe["expires"]) - time.time()))}
            why = (f"Signal says {pe.get('side')}. Waiting for a better price (${pending['limit']:,.2f}) instead of "
                   f"chasing; enters at market if it doesn't come within the time limit.")
        return {
            "symbol": s, "name": NAMES.get(s, s),
            "bid": float(tick.bid) if tick else None, "ask": float(tick.ask) if tick else None,
            "prev": b.prev_bid.get(s), "bars": list(b.bars.get(s, []))[-150:],
            "position": pos, "signal": raw, "threshold": thr, "why": why,
            "up": up, "down": down, "voters": len(cv.get("votes") or {}), "ignored": len(cv.get("ignored") or []),
            "council": float(sg.get("council", 0) or 0), "lstm_conf": float(sg.get("confidence", 0) or 0),
            "trusts": trusts, "ignores": ignores, "signal_history": hist, "pending": pending,
            "trades": [{"side": "BUY" if int(p.type) == 0 else "SELL", "volume": float(p.volume),
                        "entry": float(p.price_open), "pnl": round(float(p.profit), 2)} for p in mine],
            "candles": (b.ohlc.get(s) or [])[-84:],
            "stack_max": self._stack_max(),
            "pipeline": {
                "lstm": said["lstm"]["counted"],
                "lstm_raw": said["lstm"]["raw"],
                "ppo": said["ppo"]["counted"],
                "dreamer": float(sg.get("dreamer_exposure", 0) or 0),
                "council": float(sg.get("council", 0) or 0),
                "blend": float(sg.get("raw_target", 0) or 0),
                "final": float(sg.get("exposure", 0) or 0),
            } if sg else None,
            "briefing": self._briefing(s, sg, cv, thr, pos, pending),
            "ballot": self._ballot(s, cv, said),
            "kronos": ((self._xv().get("details") or {}).get(f"{s}:kronos")),
        }

    _xv_cache: tuple = (0.0, {})

    def _xv(self) -> dict:
        t, v = self._xv_cache
        if time.time() - t < 5:
            return v
        try:
            with open(os.path.join(m.LOG_DIR, "external_voters.json"), encoding="utf-8") as f:
                v = json.load(f)
        except (OSError, ValueError):
            v = {}
        self._xv_cache = (time.time(), v)
        return v

    _rd_cache: tuple = (0.0, {})

    def _readings(self) -> dict:
        t, v = self._rd_cache
        if time.time() - t < 5:
            return v
        try:
            with open(os.path.join(m.LOG_DIR, "council_readings.json"), encoding="utf-8") as f:
                v = json.load(f)
        except (OSError, ValueError):
            v = {}
        self._rd_cache = (time.time(), v)
        return v

    def _models_said(self, s: str, sg: dict) -> dict:
        """The AI models' part of the last decision, exactly as it went into the blend. Used by the gauge,
        the step-by-step briefing and the voter list, so they cannot disagree."""
        if not sg:
            return {}
        tr = self.cfg.get("trading") or {}
        bar = float(tr.get("confidence_threshold", 0) or 0)
        w = sg.get("weights_used") or {}
        conf = float(sg.get("agi_confidence", sg.get("confidence", 0)) or 0)
        raw = float(sg.get("agi_bias", 0) or 0)
        counted = float(sg["agi_counted"]) if "agi_counted" in sg else (raw if conf >= bar else 0.0)
        ai = mtruth.ai(s, self.cfg, ap.active())
        side = lambda v: "BUY" if v > 0 else "SELL" if v < 0 else "no view"  # noqa: E731
        if not raw:
            lt = "No view this bar (or no pattern model for this market yet)."
        elif not counted:
            lt = (f"Leans {side(raw)} {raw:+.2f} but is only {conf:.0%} sure, below the {bar:.0%} it needs, "
                  f"so it is NOT counted.")
        elif w.get("lstm") == 0:
            lt = f"Leans {side(raw)} {raw:+.2f}, but its record here is losing, so it is ignored."
        else:
            lt = f"Leans {side(raw)} {raw:+.2f}, {conf:.0%} sure (bar {bar:.0%}): counted."
        lt += " The pattern model has no exam of its own; it reads the last 60 bars."
        ppo = float(sg.get("ppo_exposure", 0) or 0)
        tag = {"passed": "passed its exam", "trial": "passed its exam, on trial",
               "unproven": "FAILED its exam; exploration model", "none": "no model loaded"}.get(ai["state"], ai["state"])
        if ppo:
            pt = f"Trader model ({tag}) wants {side(ppo)} {ppo:+.2f}."
        elif ai["state"] in ("none", "unknown"):
            pt = "No trader model for this market, so it has no vote."
        else:
            pt = f"Trader model ({tag}) has no view this bar."
        return {"lstm": {"raw": raw, "counted": counted, "conf": conf, "bar": bar, "weight": w.get("lstm"), "text": lt},
                "ppo": {"v": ppo, "counted": ppo if w.get("ppo", 1) else 0.0, "weight": w.get("ppo"), "text": pt,
                        "state": ai["state"]}}

    def _ballot(self, s: str, cv: dict, said: dict | None = None) -> list[dict]:
        """Every voter in this market's last council: what it said, how much it counts, its record, and why."""
        votes, assess = cv.get("votes") or {}, cv.get("assess") or {}
        rd = ((self._readings().get(s) or {}).get("readings") or {})
        xv = self._xv()
        weights, details = xv.get("weights") or {}, xv.get("details") or {}
        ignored = set(cv.get("ignored") or [])
        out = []
        for k, v in votes.items():
            if v is None:
                continue
            a = assess.get(k) or {}
            ext = k in weights
            base = float(weights.get(k, 1) if ext else 1)
            mult = float(a.get("mult", 1) or 1)
            side = "BUY" if v > 0.05 else "SELL" if v < -0.05 else "no view"
            det = details.get(f"{s}:{k}") if ext else None
            meter = rd.get(k)
            if det and det.get("reason"):
                reason = det["reason"]
            elif meter and meter.get("text"):
                reason = meter["text"]
            else:
                rule = describe(k)
                reason = (f"Rule: {rule}. It fired on the {side} side." if side != "no view"
                          else f"Rule: {rule}. Its condition isn't met right now, so it abstains.")
            n = float(a.get("n_all", 0) or 0)
            out.append({"key": k, "name": describe(k), "model": ext, "vote": round(float(v), 3), "side": side,
                        "weight": round(base * mult, 2), "base": base, "mult": round(mult, 2),
                        "pull": 0.0 if k in ignored else round(float(v) * base * mult, 3), "ignored": k in ignored,
                        "record": {"n": round(n), "net_bps": a.get("edge_bps"), "hit": a.get("hit_rate"),
                                   "verdict": a.get("verdict") if n >= 60 else "learning"},
                        "reason": reason, "levels": (det or {}).get("levels"),
                        "tf": (k.split("@")[1] if "@" in k else "M5") if not ext else None,
                        "strategy": k.split("@")[0] if not ext else k,
                        "meter": meter or {"v": round(float(v), 3), "min": -1, "max": 1, "buy": [0.05, 1], "sell": [-1, -0.05],
                                           "fmt": 2, "short": f"{float(v):+.2f}",
                                           "text": "vote from −1 (strong sell) to +1 (strong buy)"},
                        "headlines": (det or {}).get("headlines")})
        for k, name in (("lstm", "LSTM pattern model"), ("ppo", "PPO trader model")):
            m_ = (said or {}).get(k)
            if not m_:
                continue
            v = m_["raw"] if k == "lstm" else m_["v"]
            if not v:
                continue
            counted = m_["counted"]
            a = assess.get(k) or {}
            n = float(a.get("n_all", 0) or 0)
            out.append({"key": k, "name": name, "model": True, "vote": round(v, 3),
                        "side": "BUY" if v > 0.05 else "SELL" if v < -0.05 else "no view",
                        "weight": m_.get("weight") or 0, "base": m_.get("weight") or 0, "mult": 1.0,
                        "stage": "blend", "pull": round(counted * float(m_.get("weight") or 0), 3),
                        "ignored": not counted or m_.get("weight") == 0,
                        "record": {"n": round(n), "net_bps": a.get("edge_bps"), "hit": a.get("hit_rate"),
                                   "verdict": a.get("verdict") if n >= 60 else "learning"},
                        "reason": m_["text"], "tf": None, "strategy": k,
                        "meter": {"v": round(v, 3), "min": -1, "max": 1, "buy": [0.05, 1], "sell": [-1, -0.05], "fmt": 2,
                                  "short": f"{v:+.2f}", "text": ""}})
        out.sort(key=lambda r: -abs(r["pull"]))
        return out

    def _blocks(self) -> dict:
        """Markets the bot skipped in the last 20 min because the minimum lot is too risky for the balance."""
        now = time.time()
        c = getattr(self, "_blk_cache", None)
        if c and now - c[0] < 5:
            return c[1]
        out = {}
        for e in ops_log.events(os.path.join(m.LOG_DIR, "server.log"), NAMES, 400, 300_000):
            if e["kind"] == "blocked" and e.get("t") and now - e["t"] < 1200 and e["sym"] not in out:
                out[e["sym"]] = e
        self._blk_cache = (now, out)
        return out

    def _briefing(self, s, sg, cv, thr, pos, pending) -> list[dict]:
        """Step-by-step: how this market's decision was made, in plain English."""
        if not sg:
            return [{"k": "status", "text": "Waiting for the first 5-minute check on this market."}]
        side = lambda v: "BUY" if v > 0 else "SELL" if v < 0 else "no view"  # noqa: E731
        council, raw = float(sg.get("council", 0) or 0), float(sg.get("raw_target", 0) or 0)
        final, conf = float(sg.get("exposure", 0) or 0), float(sg.get("confidence", 0) or 0)
        rs = float(sg.get("risk_scalar", 1) or 1)
        up, down = int(cv.get("up", 0) or 0), int(cv.get("down", 0) or 0)
        out = []
        said = self._models_said(s, sg)
        out.append({"k": "LSTM", "v": said["lstm"]["counted"], "text": said["lstm"]["text"]})
        out.append({"k": "PPO", "v": said["ppo"]["counted"], "text": said["ppo"]["text"]})
        out.append({"k": "Council", "v": council, "text": (
            f"{up} indicator views say up, {down} say down (5m to 4h). Weighted by each one's recent record, the council "
            f"says {side(council)} {council:+.2f}.")})
        out.append({"k": "Blend", "v": raw, "text": (
            f"Votes combined into {raw:+.2f}, then scaled by market risk (×{rs:.2f}) and each model's track record "
            f"in this kind of market → {final:+.2f}.")})
        if pos:
            act = f"In a {pos['side'].lower()} trade ({money_fmt(pos['pnl'])}); the stop and target do the work, and it exits if the order line goes back inside ±{thr:.2f}."
        elif pending:
            act = f"Past the ±{thr:.2f} trade line, so a {pending['side'].lower()} is planned: waiting for a better price first."
        elif abs(final) >= thr and s in self._blocks():
            bk = self._blocks()[s]
            act = (f"Past the ±{thr:.2f} trade line, but BLOCKED: the smallest lot would risk {bk['risk']:.1f}% of the "
                   f"account at the stop (cap 4%). Trades here once the account is about ${bk['need']:,}.")
        elif abs(final) >= thr:
            act = f"Past the ±{thr:.2f} trade line: places a {side(final).lower()} at the next check."
        else:
            act = f"Inside the ±{thr:.2f} trade line ({abs(final) / thr:.0%} of the way there): not enough edge to pay the spread, so it waits."
        out.append({"k": "Action", "v": final, "text": act})
        return out

    _models_cache: tuple = (0.0, None)

    def _models(self) -> dict:
        """The user's own models in the council: live vote per market, live track record (model memory),
        and the history test (tools/voter_backtest.py)."""
        t, v = self._models_cache
        if v is not None and time.time() - t < 10:
            return v
        def _read(p):
            try:
                with open(p, encoding="utf-8") as f:
                    return json.load(f)
            except (OSError, ValueError):
                return {}
        try:
            from Python.model_memory import ModelMemory
            from Python.external_voters import NAMES as XN, DEFAULTS as XD
            st = _read(os.path.join(m.LOG_DIR, "external_voters.json"))
            bt = _read(os.path.join(m.LOG_DIR, "voter_backtest.json"))
            mem = ModelMemory(os.path.join(m.LOG_DIR, "memory", "model_memory.json"))
            weights = st.get("weights") or XD["weights"]
            syms = list(NAMES)
            cover = {"ob_gold": XD["ob_gold"]["symbols"], "catboost": list(XD["catboost"]["pairs"]),
                     "kronos": syms, "news_llm": syms}
            errs = st.get("errors") or {}
            voters = []
            for key in ("ob_gold", "kronos", "catboost", "news_llm"):
                rows = []
                for s in cover[key]:
                    a = mem.assess(s, f"ind:{key}", "*")
                    tb = (((bt.get("markets") or {}).get(s) or {}).get("voters") or {}).get(key)
                    rows.append({"symbol": s, "name": NAMES.get(s, s), "vote": (st.get("values") or {}).get(f"{s}:{key}"),
                                 "live": {"n": a["n_all"], "hit": a["hit_rate"], "edge_bps": a["edge_bps"],
                                          "verdict": a["verdict"] if a["n_all"] >= 60 else "learning", "mult": a["mult"]},
                                 "test": tb, "error": errs.get(f"{s}:{key}")})
                voters.append({"key": key, "name": XN.get(key, key), "weight": weights.get(key), "rows": rows})
            compare = {s: {"name": NAMES.get(s, s), "spread_bps": d.get("spread_bps"),
                           "base": (d.get("voters") or {}).get("indicators_only"),
                           "with": (d.get("voters") or {}).get("indicators_plus_models")}
                       for s, d in (bt.get("markets") or {}).items()}
            v = {"live": bool(st) and time.time() - float(st.get("t", 0)) < 180, "updated": st.get("t"),
                 "news_model": st.get("news_model"), "voters": voters,
                 "test": {"t": bt.get("t"), "done": bool(bt.get("done")), "days": bt.get("days"),
                          "horizon_min": bt.get("horizon_min"), "compare": compare,
                          "running": bool(bt) and not bt.get("done")}}
        except Exception as exc:
            v = {"live": False, "voters": [], "error": f"{type(exc).__name__}: {exc}"}
        self._models_cache = (time.time(), v)
        return v

    _setups_cache: tuple = (0.0, None)

    def _setups(self) -> dict:
        """The setup scorecard of the active account: which setups trade normal size, smallest lot, or are blocked."""
        t, v = self._setups_cache
        if v is not None and time.time() - t < 10:
            return v
        try:
            from Python.setup_gate import SetupGate
            c = m._cfg() or {}
            g = SetupGate(((c.get("trading") or {}).get("setup_gate") or {}),
                          account_id=ap.env_for(ap.active()).get("EXNESS_ACCOUNT_ID", ""))
            rows = g.table()
            v = {"enabled": g.enabled, "rows": rows, "open": len(g.state.get("open") or []),
                 "min_full": g.min_full, "min_block": g.min_block,
                 "counts": {k: sum(1 for r in rows if r["verdict"] == k) for k in ("full", "probe", "retest", "block")}}
        except Exception as exc:
            v = {"enabled": False, "rows": [], "error": type(exc).__name__}
        self._setups_cache = (time.time(), v)
        return v

    def _blotter(self) -> list[dict]:
        """Every open trade: where price sits between its stop and its target, in R and in $."""
        b = self.broker
        rows = []
        for p in b.positions or []:
            sym = str(getattr(p, "symbol", ""))
            s = sym if sym in NAMES or sym.endswith("m") else sym + "m"
            buy = int(p.type) == 0
            tick = b.ticks.get(s)
            now = float(getattr(p, "price_current", 0) or 0) or (float(tick.bid if buy else tick.ask) if tick else None)
            entry, sl, tp = float(p.price_open), float(p.sl or 0), float(p.tp or 0)
            sgn = 1 if buy else -1
            # once the stop has been moved to breakeven or past it the trade can no longer lose, and R measured
            # against that tiny distance is meaningless: report it as locked
            locked = bool(sl and tp and (sl - entry) * sgn >= -0.15 * abs(tp - entry))
            r = ((now - entry) * sgn / abs(entry - sl)) if (now and sl and entry != sl and not locked) else None
            prog = None
            if now and sl and tp and tp != sl:
                prog = max(0.0, min(1.0, (now - sl) / (tp - sl)))
            rows.append({"ticket": int(getattr(p, "ticket", 0) or 0), "symbol": s, "name": NAMES.get(s, s),
                         "side": "BUY" if buy else "SELL", "volume": float(p.volume), "entry": entry, "now": now,
                         "sl": sl or None, "tp": tp or None, "pnl": round(float(p.profit), 2), "r": None if r is None else round(r, 2), "locked": locked,
                         "progress": prog, "entry_pos": (max(0.0, min(1.0, (entry - sl) / (tp - sl))) if sl and tp and tp != sl else None),
                         "opened": int(getattr(p, "time", 0) or 0),
                         "dist_sl_pct": (abs(now - sl) / now * 100) if (now and sl) else None,
                         "dist_tp_pct": (abs(tp - now) / now * 100) if (now and tp) else None})
        rows.sort(key=lambda x: -x["opened"])
        return rows

    def _scorecard(self) -> dict:
        """Who called it: recent closed trades with every voter's call, and the running tally."""
        from Python import trade_scorecard as tc
        summ = (tc._json(tc.SUMMARY).get("voters") or {})
        board = sorted(({"key": k, **v} for k, v in summ.items() if v.get("calls")),
                       key=lambda r: (-(r.get("hit") or 0), -r["calls"]))
        recent = []
        for r in tc.recent(25):
            calls = [c for c in r.get("calls", []) if c.get("call") != "no view"]
            calls.sort(key=lambda c: (c["call"] != "right", -abs(c.get("vote", 0))))
            recent.append({**r, "name": NAMES.get(r.get("symbol"), r.get("symbol")), "calls": calls,
                           "n_right": sum(c["call"] == "right" for c in calls), "n_wrong": sum(c["call"] == "wrong" for c in calls)})
        return {"recent": recent, "board": board[:60]}

    def _stack_max(self) -> int:
        st = (((self.cfg.get("trading") or {}).get("demo_learning") or {}).get("stacking") or {})
        return int(st.get("max_per_symbol", 3)) if st.get("enabled") else 1

    def _mission(self) -> list[dict]:
        """Training mission board: every market's training status, reward scenario and exam record."""
        try:
            with open(os.path.join(m.LOG_DIR, "graduation.json"), encoding="utf-8") as f:
                st = json.load(f).get("symbols") or {}
        except Exception:
            st = {}
        gcfg = ((self.cfg.get("trading") or {}).get("graduation") or {})
        order = [str(x) for x in ((self.cfg.get("drl") or {}).get("scenario_order") or [])]
        about = {k: str((v or {}).get("about", "")) for k, v in (((self.cfg.get("drl") or {}).get("reward_scenarios") or {}).items())}
        from Python import train_progress as tp
        try:
            running = bool(m.training_state().get("running"))
        except Exception:
            running = False
        out = []
        for s in list(self.symbols) + list(self.broker.extra):
            v = st.get(s) or {}
            hist = []
            for h in (v.get("history") or [])[-8:]:
                g = (((h.get("exam") or {}).get("gates") or [{}]) or [{}])[0]
                hist.append({"scenario": h.get("scenario", "balanced"), "passed": bool(h.get("passed")),
                             "crashed": mtruth.crashed(h),
                             "ret": g.get("total_return"), "sharpe": g.get("sharpe"), "dd": g.get("max_drawdown")})
            tr = mtruth.training(s, st)
            row = {"symbol": s, "name": NAMES.get(s, s), "status": v.get("status", "live"),
                   "scenario": v.get("scenario"), "about": about.get(v.get("scenario") or "", ""),
                   "attempt": v.get("attempts", 0), "max_attempts": int(gcfg.get("max_attempts", 5) or 5),
                   "best": v.get("best_scenario"), "why": tr["why"], "summary": tr["summary"], "history": hist, "order": order,
                   "ai": mtruth.ai(s, self.cfg, ap.active()),
                   "progress": None}
            pr = {}
            if running and v.get("status") == "training":
                pr = tp.read_symbol(s)
                shared = tp.read()
                if str(shared.get("symbol")) == s and float(shared.get("ts", 0)) > float(pr.get("ts", 0) or 0):
                    pr = shared
            if pr and pr.get("kind") == "ppo" and not pr.get("done") and time.time() - float(pr.get("ts", 0)) < 300:
                row["progress"] = {"step": pr.get("step"), "total": pr.get("total"), "eta_s": pr.get("eta_s"),
                                   "fps": pr.get("fps"), "reward": pr.get("ep_reward_mean"),
                                   "lane": pr.get("lane", "GPU")}
            out.append(row)
        return out

    def _trades(self, deals, out_entry) -> list[dict]:
        rows = []
        for d in sorted(deals, key=lambda d: getattr(d, "time", 0))[-30:]:
            sym = str(getattr(d, "symbol", ""))
            sym = sym if sym.endswith("m") else sym + "m"
            closing = int(getattr(d, "entry", 0)) == out_entry
            rows.append({"t": int(getattr(d, "time", 0) or 0) * 1000, "symbol": NAMES.get(sym, sym),
                         "side": "BUY" if int(getattr(d, "type", 0)) == 0 else "SELL",
                         "volume": float(getattr(d, "volume", 0) or 0), "price": float(getattr(d, "price", 0) or 0),
                         "closing": closing, "pnl": float(getattr(d, "profit", 0) or 0) if closing else None})
        return rows[::-1]

    def _training_note(self) -> str:
        try:
            with open(os.path.join(m.LOG_DIR, "graduation.json"), encoding="utf-8") as f:
                st = json.load(f).get("symbols") or {}
        except Exception:
            st = {}
        q = [NAMES.get(s, s) for s, v in st.items() if v.get("status") in ("retrain_queued", "queued", "exam_failed")]
        if q:
            return "Not training right now. Queued next: " + ", ".join(q) + "."
        return "Not training right now. Markets are retrained automatically when their live results slip."

    def _training(self) -> dict:
        running = bool(m.training_state().get("running"))
        prog = m.read_progress() if running else {}
        stages = wr.cycle_stages(wr.stage_lines(m.CYCLE_LOG), wr.cycle_symbols(self.symbols))
        if not running:
            for r in stages:
                if r["status"] == "running":
                    r["status"] = "interrupted"
        cur = next((r for r in stages if r["status"] == "running"), None) if running else None
        summ = wr.training_summary(prog, stages)
        frac = summ["overall"]
        mission = wr.MISSIONS.get(cur["kind"]) if cur else None
        nxt = next((r for r in stages if r["status"] == "pending"), None)
        return {
            "running": running,
            "note": None if running else m.training_note(),
            "progress": frac,
            "stage": summ,
            "stages": [{"label": r["label"], "status": r["status"]} for r in stages],
            "current": ({"name": mission["name"], "symbol": NAMES.get(cur["symbol"], cur["symbol"]),
                         "learn": mission["learn"], "goal": mission["goal"], "test": mission["test"]} if cur else None),
            "next": (f"{wr.MISSIONS[nxt['kind']]['name']} for {NAMES.get(nxt['symbol'], nxt['symbol'])}" if nxt else None),
        }


STATE: State | None = None


def _live_rules(cfg: dict) -> list[str]:
    """What the bot does on a real-money account."""
    tr = (cfg or {}).get("trading") or {}
    dl = tr.get("demo_learning") or {}
    if dl.get("same_on_live"):
        r = dl.get("risk") or {}
        mk = (tr.get("news_trading") or {}).get("symbols") or tr.get("symbols") or []
        stk = dl.get("stacking") or {}
        return [
            "EXACTLY the same as demo: nothing is switched off on real money",
            f"Risk {float(r.get('risk_percent', 2.5)):g}% of the account per trade, up to {float(r.get('max_lots', 0.2)):g} lots",
            f"Daily loss stop ${float(r.get('max_daily_loss', 150)):g} (a fixed dollar amount, whatever the live balance is); "
            f"halts at {float(r.get('max_drawdown_pct', 40)):g}% drawdown",
            f"All {len(mk)} markets all day, up to {int(stk.get('max_per_symbol', 3))} trades each, max {int(r.get('max_open_positions', 14))} open",
            "Learning mode (half trade threshold), indicator council, exploration models and news trading all on",
        ]
    risk = (cfg or {}).get("risk") or {}
    sup = risk.get("supervisor") or {}
    return [
        f"Risk per trade {float(tr.get('risk_percent', 1.0)):g}% (demo: 2.5%), max {float(risk.get('max_lots', 0.05)):g} lots",
        f"Daily loss stop ${float(risk.get('max_daily_loss', 25)):g}; halts at {float(sup.get('max_drawdown_pct', 8)):g}% drawdown",
        f"Only {', '.join(tr.get('symbols') or ['BTCUSDm', 'XAUUSDm'])}: the other markets stay demo-only until their AI passes",
        "No trade stacking, no learning mode, no exploration models",
        "Indicator council off (opt-in with trading.council.allow_live)",
        "Only exam-approved AI models trade",
    ]


def _reexec_with(profile: str):
    """Restart this dashboard on the newly active account so it shows the right balance."""
    time.sleep(1.5)
    env = {k: v for k, v in os.environ.items() if not k.startswith("EXNESS_")}
    env.update(ap.env_for(profile))
    env["CG_ACCOUNT_PROFILE"] = profile
    os.execve(sys.executable, [sys.executable] + sys.argv, env)


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _send(self, code, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json(self, code, obj):
        return self._send(code, json.dumps(obj).encode(), "application/json")

    def _same_origin(self) -> bool:
        # A page on another site can fire requests at localhost; it cannot add custom headers without
        # a CORS preflight (which this server never approves), so require one, plus a local Origin.
        origin = self.headers.get("Origin") or ""
        host = self.headers.get("Host") or ""
        local = ("127.0.0.1", "localhost")
        return (self.headers.get("X-CG-Client") == "dashboard"
                and host.split(":")[0] in local
                and (not origin or origin.split("://", 1)[-1].split(":")[0] in local))

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        if not self._same_origin():
            return self._json(403, {"ok": False, "error": "forbidden"})
        try:
            n = min(int(self.headers.get("Content-Length") or 0), 16384)
            body = json.loads(self.rfile.read(n) or b"{}")
        except Exception:
            return self._json(400, {"ok": False, "error": "bad request"})
        if path == "/api/account/live_setup":
            if ap.active() == "live":
                return self._json(409, {"ok": False, "error": "switch to demo before changing the live keys"})
            return self._json(200, ap.save_live(*(str(body.get(k, "") or "")[:4000]
                                                  for k in ("api_key", "account_id", "gateway", "private_key"))))
        prof = str(body.get("profile", ""))
        if path == "/api/account/check" and prof in ap.FILES:
            return self._json(200, ap.check(prof))
        if path == "/api/account/switch" and prof in ap.FILES:
            res = ap.switch(prof, str(body.get("confirm", "")))
            self._json(200, res)
            if res.get("ok"):
                threading.Thread(target=_reexec_with, args=(prof,), daemon=True).start()
            return None
        return self._json(404, {"ok": False, "error": "not found"})

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/api/account":
            ap.ensure_template()
            return self._json(200, {"active": ap.active(), "demo": ap.status("demo"), "live": ap.status("live"),
                                    "form": ap.live_form(),
                                    "confirm": ap.CONFIRM, "live_rules": _live_rules(STATE.cfg if STATE else {})})
        if path == "/api/state":
            try:
                body = json.dumps(STATE.snapshot()).encode()
                return self._send(200, body, "application/json")
            except Exception as exc:
                return self._send(500, json.dumps({"error": str(exc)}).encode(), "application/json")
        rel = "index.html" if path in ("/", "") else path.lstrip("/")
        full = os.path.realpath(os.path.join(DIST, rel))
        if not full.startswith(os.path.realpath(DIST)) or not os.path.isfile(full):
            full = os.path.join(DIST, "index.html")
        if not os.path.isfile(full):
            return self._send(404, b"dashboard not built: cd web && npm run build", "text/plain")
        with open(full, "rb") as f:
            self._send(200, f.read(), mimetypes.guess_type(full)[0] or "application/octet-stream")


def main():
    global STATE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8788)
    args = ap.parse_args()
    STATE = State()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
