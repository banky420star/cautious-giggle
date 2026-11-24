"""Learning view for the war-room dashboard: what each layer has learned, where it is relayed, and the audit.

learning_snapshot() -> {
  audit:   latest tools/pipeline_audit.py results grouped by section (rerun every 15 min by start_audit_thread)
  layers:  each learner, what it learns from, and the decision layer it feeds (with its live effect)
  markets: per market: real trade record + the size it allows, move filter now, what model memory has learned
           per model (all hours / GO / STOP), PPO trial status
}
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
import time
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
LOG = os.path.join(ROOT, "logs")
MARKETS = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
NAMES = {"XAUUSDm": "Gold", "BTCUSDm": "Bitcoin", "EURUSDm": "EUR/USD", "USDJPYm": "USD/JPY", "XAGUSDm": "Silver",
         "ETHUSDm": "Ether", "USOILm": "Oil"}
MODELS = ["lstm", "ppo", "council", "blend", "trade"]
_cache = {"t": 0.0, "v": None}


def _jload(path, default=None):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def _tail(path, n_bytes=6_000_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n_bytes))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def _audit():
    a = _jload(os.path.join(LOG, "pipeline_audit.json"), {}) or {}
    sec = defaultdict(list)
    for r in a.get("results", []):
        sec[r["area"]].append(r)
    order = ["runtime", "ai", "feed", "live", "council", "kronos", "ob_gold", "catboost", "news", "signal", "filter", "learning", "lstm", "ppo"]
    out = []
    for k in sorted(sec, key=lambda x: order.index(x) if x in order else 99):
        rows = sec[k]
        out.append({"section": k, "fail": sum(r["status"] == "FAIL" for r in rows), "warn": sum(r["status"] == "WARN" for r in rows),
                    "pass": sum(r["status"] == "PASS" for r in rows),
                    "rows": sorted(rows, key=lambda r: {"FAIL": 0, "WARN": 1, "PASS": 2}[r["status"]])})
    return {"t": a.get("t"), "sections": out}


def _snapshot():
    mem = _jload(os.path.join(LOG, "memory", "model_memory.json"), {}) or {}
    stats = mem.get("stats", {})
    rec = {}
    try:
        import csv
        with open(os.path.join(LOG, "learning", "trade_learning_by_symbol.csv"), encoding="utf-8") as f:
            rec = {r["symbol"]: r for r in csv.DictReader(f)}
    except OSError:
        pass
    ev = _tail(os.path.join(LOG, "audit_events.jsonl"))
    last_sig, last_mf = {}, {}
    for e in ev:
        p = e.get("payload") or {}
        if e.get("event") == "signal":
            last_sig[p.get("symbol")] = (e.get("ts"), p)
        elif e.get("event") == "move_filter":
            last_mf[p.get("symbol")] = (e.get("ts"), p)
    reg = (_jload(os.path.join(ROOT, "models", "registry", "active.json"), {}) or {}).get("symbols", {})
    sys.path.insert(0, ROOT)
    try:
        from Python.trade_memory_scale import trade_memory_scale as _trade_memory_scale
    except Exception:  # noqa: BLE001
        _trade_memory_scale = None

    def grade(s, m, ctx):
        st = stats.get(f"{s}|{m}|{ctx}")
        if not st or st.get("w", 0) < 5:
            return None
        return {"bps": round(st["m"], 2), "n": round(st["w"], 0),
                "hit": round(st["hits"] / st["calls"], 3) if st.get("calls") else None}

    markets = []
    for s in MARKETS:
        r = rec.get(s) or {}
        tm = {"trades": int(r.get("trades") or 0), "profit_factor": float(r.get("profit_factor") or 0),
              "recent_loss_streak": int(r.get("recent_loss_streak") or 0)}
        scale, why = _trade_memory_scale(tm) if _trade_memory_scale else (1.0, "?")
        sig = last_sig.get(s, (None, {}))[1]
        mf_ts, mf = last_mf.get(s, (None, None))
        cs = (reg.get(s) or {}).get("canary_state") or {}
        markets.append({
            "symbol": s, "name": NAMES[s],
            "record": {"trades": tm["trades"], "win_rate": float(r.get("win_rate") or 0), "pnl": float(r.get("total_pnl") or 0),
                       "profit_factor": tm["profit_factor"], "loss_streak": tm["recent_loss_streak"]},
            "size": {"scale": sig.get("trade_memory_scale", scale), "why": sig.get("trade_memory_why") or why},
            "filter": None if not mf else {"p": mf.get("p"), "go": mf.get("go"), "ts": mf_ts},
            "grades": {m: {"all": grade(s, m, "*"), "go": grade(s, m, "mf:go"), "stop": grade(s, m, "mf:stop")} for m in MODELS},
            "trial": {"canary": os.path.basename(str((reg.get(s) or {}).get("canary") or "")) or None,
                      "champion": os.path.basename(str((reg.get(s) or {}).get("champion") or "")) or None,
                      "trades": cs.get("trades"), "pnl": cs.get("realized_pnl"), "reason": cs.get("reason"),
                      "min_trades": ((reg.get(s) or {}).get("canary_policy") or {}).get("min_trades"),
                      "checked": cs.get("checked_at")},
        })
    layers = [
        {"learner": "Model memory", "learns_from": "every model's call, scored 10 minutes later after costs",
         "feeds": "each model's weight in the vote (×0.5 to ×1.5 per market and market mood)", "live": True},
        {"learner": "GO / STOP grades", "learns_from": "the same calls, split by the big-move filter",
         "feeds": "recorded per model now; will feed weights once there are enough", "live": False},
        {"learner": "Trade record", "learns_from": "real closed trades per market (last 30 days)",
         "feeds": "position size: half size while a market's record loses", "live": True},
        {"learner": "Setup gate", "learns_from": "real closed trades per kind of setup",
         "feeds": "smallest lot until a setup proves itself; blocks losing setups", "live": True},
        {"learner": "Voter scorecard", "learns_from": "which voters called each closed trade right",
         "feeds": "shown in 'Who called it'; mutes come from model memory", "live": True},
        {"learner": "Big-move filter", "learns_from": "a year of 5-minute bars per market (trained offline)",
         "feeds": "watch-only: logged with every decision, blocks nothing yet", "live": False},
        {"learner": "PPO trial", "learns_from": "this model's own closed trades since its trial began",
         "feeds": "promotes a new model to champion only after enough trades without a net loss", "live": True},
    ]
    return {"audit": _audit(), "layers": layers, "markets": markets, "memory_updated": mem.get("updated")}


def learning_snapshot(max_age_s: float = 30.0):
    if _cache["v"] is None or time.time() - _cache["t"] > max_age_s:
        try:
            _cache["v"] = _snapshot()
        except Exception as exc:  # noqa: BLE001
            _cache["v"] = {"error": str(exc)}
        _cache["t"] = time.time()
    return _cache["v"]


def start_audit_thread(every_s: float = 900.0):
    """Rerun tools/pipeline_audit.py in the background so the dashboard's audit stays current."""
    def loop():
        while True:
            try:
                subprocess.run([sys.executable, os.path.join(ROOT, "tools", "pipeline_audit.py")], cwd=ROOT,
                               env={**os.environ, "PYTHONPATH": ROOT}, stdout=subprocess.DEVNULL,
                               stderr=subprocess.DEVNULL, timeout=600)
            except Exception:  # noqa: BLE001
                pass
            time.sleep(every_s)
    threading.Thread(target=loop, name="audit", daemon=True).start()
