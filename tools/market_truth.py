"""One source of truth for what each market's AI is, what the market trades, and how its training went.

Every screen (war room cards, boot screen, training panel, terminal monitors) reads these functions
instead of re-deriving the story from its own clues, so they cannot contradict each other. Each
statement is built from the records that decide it: the model registry (what is actually loaded), the
exploration marker (failed the exam), the config (what an account trades) and graduation.json
(what training really did, with crashed runs kept apart from real exam results).
"""
from __future__ import annotations

import json
import os

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
GRAD = os.path.join(ROOT, "logs", "graduation.json")
SCEN = ["balanced", "patient", "hunter", "steady", "trend_rider"]


def _grad() -> dict:
    try:
        with open(GRAD, encoding="utf-8") as f:
            return json.load(f).get("symbols") or {}
    except (OSError, ValueError):
        return {}


def crashed(h: dict) -> bool:
    """A history entry whose training never reached a real exam (crash, no data, killed)."""
    if h.get("crashed"):
        return True
    e = h.get("exam") or {}
    return not h.get("passed") and (bool(e.get("error")) or not e.get("gates"))


def exam_result(h: dict) -> dict | None:
    """Headline numbers of one real exam: return and Sharpe on unseen history."""
    if crashed(h):
        return None
    g = ((h.get("exam") or {}).get("gates") or [{}])[0]
    return {"scenario": h.get("scenario"), "passed": bool(h.get("passed")),
            "return_pct": round(float(g.get("total_return", 0) or 0) * 100, 2),
            "sharpe": round(float(g.get("sharpe", 0) or 0), 1),
            "max_dd_pct": round(float(g.get("max_drawdown", 0) or 0) * 100, 2),
            "bars": int(g.get("steps", 0) or 0)}


def training(symbol: str, st: dict | None = None) -> dict:
    v = (st if st is not None else _grad()).get(symbol) or {}
    hist = v.get("history") or []
    exams = [r for r in (exam_result(h) for h in hist) if r]
    n_crash = sum(1 for h in hist if crashed(h))
    best = max(exams, key=lambda r: r["return_pct"]) if exams else None
    status = v.get("status") or "not started"
    if exams:
        res = (f"{len(exams)} real exam{'s' if len(exams) != 1 else ''}, none passed; best {best['return_pct']:+.2f}% "
               f"on unseen history ({best['scenario']} reward)") if not any(r["passed"] for r in exams) else \
              f"passed with the {next(r for r in exams if r['passed'])['scenario']} reward"
    else:
        res = "no exam has run yet"
    if n_crash:
        res += f"; {n_crash} run{'s' if n_crash != 1 else ''} crashed before the exam (not counted)"
    return {"status": status, "why": v.get("why") or "", "summary": res, "exams": exams, "crashes": n_crash,
            "best_return_pct": best["return_pct"] if best else None}


def ai(symbol: str, cfg: dict | None, account: str | None) -> dict:
    """What the AI part of this market really is right now."""
    try:
        from Python import demo_exploration
        from Python.model_registry import ModelRegistry
        act = ModelRegistry().get_symbol_active(symbol) or {}
    except Exception:
        return {"state": "unknown", "short": "AI status unknown", "text": "Could not read the model registry."}
    live_too = demo_exploration.same_on_live(cfg)
    if act.get("champion"):
        return {"state": "passed", "short": "AI (passed) + indicators",
                "text": "Its AI model passed the exam on unseen history and votes alongside the indicators."}
    can = act.get("canary")
    if can and not demo_exploration.is_exploration(can):
        return {"state": "trial", "short": "AI on trial + indicators",
                "text": "Its AI model passed the exam and is on a live trial before it is fully trusted."}
    explore = demo_exploration.settings(cfg or {})
    if can and not explore["enabled"]:
        return {"state": "none", "short": "indicators only",
                "text": "Its AI model failed the exam and exploration is off, so the indicator council decides alone."}
    if can:
        where = "on this account and on real money too (same_on_live is on)" if live_too else \
                ("on this demo account only" if account == "demo" else "on demo accounts only, not here")
        return {"state": "unproven", "short": "indicators + unproven AI",
                "text": (f"No AI model has passed the exam here. A model that failed still votes, "
                         f"{where}, at {explore['size_scale']:.0%} size, to collect real evidence.")}
    return {"state": "none", "short": "indicators only", "text": "No AI model yet: the indicator council decides alone."}


def trades(symbol: str, cfg: dict | None, account: str | None) -> dict:
    """Which kinds of trade this market takes on the current account."""
    tr = (cfg or {}).get("trading") or {}
    demo_rules = account == "demo" or bool((tr.get("demo_learning") or {}).get("same_on_live"))
    if demo_rules and (tr.get("demo_learning") or {}).get("trade_all_markets", False):
        return {"mode": "all signals + news", "text": "Trades every signal and news release (learning settings)."}
    if symbol in (tr.get("symbols") or []):
        return {"mode": "signals + news", "text": "Approved market: trades signals and news releases."}
    return {"mode": "news only", "text": "Not approved for signal trading here: trades news releases only."}


def market(symbol: str, cfg: dict | None, account: str | None, st: dict | None = None) -> dict:
    return {"ai": ai(symbol, cfg, account), "trades": trades(symbol, cfg, account), "training": training(symbol, st)}


def overview(symbols: list[str], cfg: dict | None, account: str | None) -> dict:
    st = _grad()
    rows = {s: market(s, cfg, account, st) for s in symbols}
    n = {k: sum(1 for r in rows.values() if r["ai"]["state"] == k) for k in ("passed", "trial", "unproven", "none")}
    return {"markets": rows, "counts": n,
            "line": (f"{n['passed'] + n['trial']} with a proven AI · {n['unproven']} with an unproven AI · "
                     f"{n['none']} indicators only")}
