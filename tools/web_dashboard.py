"""Local web dashboard server: serves the React app (web/dist) and a JSON snapshot at /api/state.

    python tools/web_dashboard.py [--port 8787]

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

import monitor_tui as m  # noqa: E402
import warroom as wr  # noqa: E402
from narrator import Narrator  # noqa: E402
from Python.indicator_council import describe  # noqa: E402

DIST = os.path.join(ROOT, "web", "dist")
EQ_FILE = os.path.join(ROOT, "logs", "equity_history.csv")
NAMES = {"BTCUSDm": "Bitcoin", "XAUUSDm": "Gold"}


class State:
    def __init__(self):
        self.cfg = m._cfg()
        self.symbols = [str(s) for s in ((self.cfg.get("trading") or {}).get("symbols") or ["BTCUSDm", "XAUUSDm"])]
        self.broker = m.Broker(self.symbols)
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
                    if len(p) == 3:
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
            if acc is not None:
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

        symbols = []
        for s in self.symbols:
            symbols.append(self._symbol(s))

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
                        "daily_loss_limit": float(risk.get("max_daily_loss", 0) or 0)},
            "stats": {"closed": len(closed), "wins": sum(1 for x in closed if x > 0),
                      "losses": sum(1 for x in closed if x <= 0), "pnl_closed": sum(closed)},
            "equity": eq_series,
            "symbols": symbols,
            "trades": self._trades(deals, out_entry),
            "training": self._training(),
            "feed": self.narrator.feed(60),
            "news": self._news_snapshot(),
            "tip": dict(zip(("term", "text"), wr.GLOSSARY[int(time.time() // 12) % len(wr.GLOSSARY)])),
        }

    def _symbol(self, s: str) -> dict:
        b, logs = self.broker, self.logs
        tick = b.ticks.get(s)
        sg = (logs.last_signal.get(s) or {}).get("payload") or {}
        cv = (logs.last_council.get(s) or {}).get("payload") or {}
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
        }

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

    def _training(self) -> dict:
        prog = m.read_progress()
        stages = wr.cycle_stages(m.tail_lines(m.CYCLE_LOG, 400), self.symbols)
        cur = next((r for r in stages if r["status"] == "running"), None)
        summ = wr.training_summary(prog, stages)
        frac = summ["overall"]
        mission = wr.MISSIONS.get(cur["kind"]) if cur else None
        nxt = next((r for r in stages if r["status"] == "pending"), None)
        return {
            "running": bool(m.training_state().get("running")),
            "progress": frac,
            "stage": summ,
            "stages": [{"label": r["label"], "status": r["status"]} for r in stages],
            "current": ({"name": mission["name"], "symbol": NAMES.get(cur["symbol"], cur["symbol"]),
                         "learn": mission["learn"], "goal": mission["goal"], "test": mission["test"]} if cur else None),
            "next": (f"{wr.MISSIONS[nxt['kind']]['name']} for {NAMES.get(nxt['symbol'], nxt['symbol'])}" if nxt else None),
        }


STATE: State | None = None


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

    def do_GET(self):
        path = self.path.split("?", 1)[0]
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
    ap.add_argument("--port", type=int, default=8787)
    args = ap.parse_args()
    STATE = State()
    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    print(f"dashboard on http://127.0.0.1:{args.port}", flush=True)
    srv.serve_forever()


if __name__ == "__main__":
    main()
