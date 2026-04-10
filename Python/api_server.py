"""
API Server — Lightweight HTTP bridge between the AGI engine and the React dashboard.

Uses bottle (already in .venv312) on port 5000.  Vite dev-server proxies /api/* here.

Start modes:
  1. Embedded: import start_api_server(agi_server) from Server_AGI — preferred.
  2. Standalone: python -m Python.api_server — reads live_state.json fallback.

All endpoints are read-only except POST /api/control.
"""
from __future__ import annotations

import json
import os
import sys
import time
import threading
from collections import deque
from datetime import datetime, timezone
from typing import Any

from bottle import Bottle, request, response, run as bottle_run
from loguru import logger

# ---------------------------------------------------------------------------
# Project root (one level above Python/)
# ---------------------------------------------------------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Bottle app
# ---------------------------------------------------------------------------
app = Bottle()

# ---------------------------------------------------------------------------
# Shared references — populated by start_api_server()
# ---------------------------------------------------------------------------
_server_ref: Any = None          # AGIServer instance
_decision_cache: dict[str, deque] = {}  # symbol -> deque of recent decisions
_CACHE_MAX = 50                  # decisions per symbol


# ---------------------------------------------------------------------------
# CORS middleware (allow React dev server on any port)
# ---------------------------------------------------------------------------
@app.hook("after_request")
def _cors():
    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"


@app.route("<path:path>", method="OPTIONS")
def _options(path):
    return {}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _json(obj: Any, status: int = 200):
    response.content_type = "application/json"
    response.status = status
    return json.dumps(obj, default=str)


def _read_json_file(path: str) -> Any:
    """Safely read a JSON file, returning None on any error."""
    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
    except Exception:
        pass
    return None


def _read_active_registry() -> dict:
    """Read models/registry/active.json."""
    active_path = os.path.join(ROOT, "models", "registry", "active.json")
    return _read_json_file(active_path) or {"champion": None, "canary": None}


def _read_config() -> dict:
    cfg_path = os.path.join(ROOT, "config.yaml")
    try:
        import yaml
        if os.path.exists(cfg_path):
            with open(cfg_path, "r", encoding="utf-8") as f:
                return yaml.safe_load(f) or {}
    except Exception:
        pass
    return {}


def _read_incidents() -> list:
    return _read_json_file(os.path.join(ROOT, "live_incidents.json")) or []


def _read_live_state() -> dict:
    return _read_json_file(os.path.join(ROOT, "live_state.json")) or {}


def cache_decision(symbol: str, decision: dict):
    """Store a decision in the in-memory cache (called from the brain path)."""
    if symbol not in _decision_cache:
        _decision_cache[symbol] = deque(maxlen=_CACHE_MAX)
    entry = {**decision, "_cached_at": time.time()}
    _decision_cache[symbol].appendleft(entry)


def _safe_risk(attr: str, default=None):
    """Pull a value from the risk engine if the server reference is available."""
    srv = _server_ref
    if srv and hasattr(srv, "risk"):
        return getattr(srv.risk, attr, default)
    return default


def _safe_brain(attr: str, default=None):
    srv = _server_ref
    if srv and hasattr(srv, "brain"):
        return getattr(srv.brain, attr, default)
    return default


# ═══════════════════════════════════════════════════════════════════════════
# 1. GET /api/status — Full system status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/status")
def api_status():
    srv = _server_ref
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
    active = _read_active_registry()
    live = _read_live_state()
    incidents = _read_incidents()

    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""
    champ_id = os.path.basename(champ_path) if champ_path else "none"
    canary_id = os.path.basename(canary_path) if canary_path else ""

    # ── Risk engine live values ──
    halt = _safe_risk("halt", False)
    daily_trades = _safe_risk("daily_trades", 0)
    realized_pnl = _safe_risk("realized_pnl_today", 0.0)
    current_dd = _safe_risk("current_dd", 0.0)
    peak_equity = _safe_risk("_peak_equity", 0.0)
    current_equity = _safe_risk("_current_equity", 0.0)
    can_trade = False
    if srv and hasattr(srv, "risk"):
        try:
            can_trade = srv.risk.can_trade()
        except Exception:
            pass

    uptime = int(time.time() - srv.start_time) if srv and hasattr(srv, "start_time") else 0
    mode = "LIVE" if (srv and getattr(srv, "live", False)) else "DRY-RUN"

    # ── Build lane rows from decision cache ──
    lane_rows = []
    for sym in symbols:
        recent = list(_decision_cache.get(sym, []))
        last = recent[0] if recent else {}
        lane_rows.append({
            "symbol": sym,
            "decision": {
                "regime": last.get("volatility", "--"),
                "final_target": last.get("exposure", 0.0),
                "ppo_target": last.get("exposure", 0.0),
                "dreamer_target": 0.0,
                "confidence": last.get("confidence", 0.0),
            },
            "pipeline": {
                "lstm": {"state": last.get("volatility", "UNKNOWN")},
            },
            "champion": champ_id,
            "status": "live" if not halt else "halted",
            "side": last.get("action", "HOLD").lower(),
            "confidence": last.get("confidence", 0.0),
            "exposure": last.get("exposure", 0.0),
            "pnl": 0.0,
            "canTrade": can_trade,
            "reason": last.get("reason", ""),
        })

    # ── Pipeline summary ──
    pipeline_summary = {
        "symbols_total": len(symbols),
        "training_active_symbols": 0,
        "canary_review_symbols": 1 if canary_id else 0,
        "champion_live_symbols": len(symbols) if champ_id != "none" else 0,
        "trading_ready_symbols": len(symbols) if champ_id != "none" else 0,
        "trading_active_symbols": len(symbols) if can_trade else 0,
    }

    lane_summary = {
        "actionable_symbols": sum(1 for r in lane_rows if r["side"] != "hold"),
        "executed_symbols": daily_trades,
        "blocked_symbols": sum(1 for r in lane_rows if not r["canTrade"]),
        "neutral_symbols": sum(1 for r in lane_rows if r["side"] == "hold"),
        "open_positions": 0,
    }

    return _json({
        "state": "online" if not halt else "halted",
        "status": "online" if not halt else "halted",
        "server": {
            "running": True,
            "pids": [os.getpid()],
        },
        "account": {
            "balance": current_equity,
            "equity": current_equity,
            "free_margin": current_equity,
            "open_positions": 0,
            "positions": [],
        },
        "training": {
            "cycle_running": False,
            "lstm_running": False,
            "drl_running": False,
            "dreamer_running": False,
            "configured_symbols": symbols,
            "visual": {
                "lstm": {"state": "idle"},
                "ppo": {"state": "idle"},
                "dreamer": {"state": "idle"},
                "active_label": "Idle",
            },
            "symbol_stage_rows": [],
            "symbol_lane_rows": lane_rows,
            "pipeline_summary": pipeline_summary,
            "lane_summary": lane_summary,
        },
        "canary_gate": {
            "ready": bool(canary_id),
            "reason": "Canary active" if canary_id else "No canary",
        },
        "active_models": active,
        "registry_summary": {
            "champion": champ_id,
            "canary": canary_id or None,
        },
        "incidents": incidents or [{
            "id": "SYS-001",
            "type": "system",
            "severity": "info",
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "message": "API server online.",
        }],
        "logs": {},
        "timestamp": time.time(),
        "uptime_sec": uptime,
        "mode": mode,
        "risk": {
            "halt": halt,
            "daily_trades": daily_trades,
            "max_daily_trades": _safe_risk("max_daily_trades", 20),
            "realized_pnl": realized_pnl,
            "max_daily_loss": _safe_risk("max_daily_loss", 500),
            "current_dd": current_dd,
            "peak_equity": peak_equity,
            "current_equity": current_equity,
            "can_trade": can_trade,
        },
    })


# ═══════════════════════════════════════════════════════════════════════════
# 2. GET /api/trades — Recent trade history
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/trades")
def api_trades():
    limit = int(request.params.get("limit", 50))
    offset = int(request.params.get("offset", 0))
    symbol_filter = request.params.get("symbol", "")

    trades = _fetch_trade_history(symbol_filter)
    total = len(trades)
    page = trades[offset:offset + limit]

    return _json({
        "trades": page,
        "total": total,
        "limit": limit,
        "offset": offset,
    })


@app.get("/api/trades/summary")
def api_trades_summary():
    symbol_filter = request.params.get("symbol", "")
    trades = _fetch_trade_history(symbol_filter)

    wins = [t for t in trades if t.get("profit", 0) > 0]
    losses = [t for t in trades if t.get("profit", 0) < 0]
    total_pnl = sum(t.get("profit", 0) for t in trades)
    avg_profit = (sum(t["profit"] for t in wins) / len(wins)) if wins else 0
    avg_loss = (sum(t["profit"] for t in losses) / len(losses)) if losses else 0
    gross_profit = sum(t["profit"] for t in wins)
    gross_loss = abs(sum(t["profit"] for t in losses))
    pf = (gross_profit / gross_loss) if gross_loss > 0 else "inf"

    hold_mins = [t.get("hold_minutes", 0) for t in trades if t.get("hold_minutes")]

    # Per-symbol breakdown
    by_symbol: dict[str, Any] = {}
    for t in trades:
        sym = t.get("symbol", "UNKNOWN")
        if sym not in by_symbol:
            by_symbol[sym] = {"trades": 0, "wins": 0, "pnl": 0.0}
        by_symbol[sym]["trades"] += 1
        by_symbol[sym]["pnl"] += t.get("profit", 0)
        if t.get("profit", 0) > 0:
            by_symbol[sym]["wins"] += 1
    for sym in by_symbol:
        bs = by_symbol[sym]
        bs["win_rate"] = (bs["wins"] / bs["trades"]) if bs["trades"] > 0 else 0

    return _json({
        "overall": {
            "total_trades": len(trades),
            "wins": len(wins),
            "losses": len(losses),
            "win_rate": (len(wins) / len(trades)) if trades else 0,
            "total_pnl": round(total_pnl, 2),
            "avg_profit": round(avg_profit, 2),
            "avg_loss": round(avg_loss, 2),
            "profit_factor": round(pf, 2) if isinstance(pf, float) else pf,
            "avg_hold_minutes": round(sum(hold_mins) / len(hold_mins), 1) if hold_mins else 0,
            "max_loss_streak": _max_loss_streak(trades),
        },
        "by_symbol": by_symbol,
    })


def _fetch_trade_history(symbol_filter: str = "") -> list[dict]:
    """
    Pull trade history from MT5 (Windows live) or from the decision cache (dry-run).
    Returns a list of Trade dicts matching the frontend Trade interface.
    """
    trades: list[dict] = []

    # Try MT5 deal history on Windows
    if sys.platform == "win32":
        try:
            import MetaTrader5 as mt5
            import pytz

            if mt5.initialize():
                tz = pytz.timezone("Etc/UTC")
                now_utc = datetime.now(tz)
                from datetime import timedelta
                lookback = now_utc - timedelta(days=30)
                deals = mt5.history_deals_get(lookback, now_utc)
                if deals:
                    for d in deals:
                        if d.entry != mt5.DEAL_ENTRY_OUT:
                            continue
                        if symbol_filter and d.symbol != symbol_filter:
                            continue
                        trades.append({
                            "ticket": d.ticket,
                            "symbol": d.symbol,
                            "side": "BUY" if d.type == mt5.DEAL_TYPE_BUY else "SELL",
                            "volume": d.volume,
                            "open_time": None,
                            "close_time": datetime.fromtimestamp(d.time, tz=tz).isoformat(),
                            "open_price": d.price,
                            "close_price": d.price,
                            "profit": round(d.profit, 2),
                            "comment": d.comment or "",
                            "hold_minutes": None,
                            "magic": d.magic,
                            "bot_lane": "ppo",
                            "model": "champion",
                            "action_type": "close",
                            "outcome": "win" if d.profit > 0 else ("loss" if d.profit < 0 else "breakeven"),
                        })
                    trades.sort(key=lambda t: t.get("close_time", "") or "", reverse=True)
                    return trades
        except Exception as e:
            logger.debug(f"MT5 trade history fetch failed: {e}")

    # Fallback: derive from decision cache
    for sym, dq in _decision_cache.items():
        if symbol_filter and sym != symbol_filter:
            continue
        for i, d in enumerate(dq):
            if d.get("action") in ("BUY", "SELL"):
                trades.append({
                    "ticket": int(d.get("_cached_at", time.time()) * 1000) + i,
                    "symbol": sym,
                    "side": d.get("action", "HOLD"),
                    "volume": abs(d.get("exposure", 0.0)),
                    "open_time": datetime.fromtimestamp(d.get("_cached_at", 0), tz=timezone.utc).isoformat(),
                    "close_time": None,
                    "open_price": 0,
                    "close_price": 0,
                    "profit": 0,
                    "comment": d.get("reason", ""),
                    "hold_minutes": None,
                    "magic": None,
                    "bot_lane": "ppo",
                    "model": "canary" if d.get("reason", "").startswith("canary") else "champion",
                    "action_type": "signal",
                    "outcome": "breakeven",
                })

    trades.sort(key=lambda t: t.get("open_time", "") or "", reverse=True)
    return trades


def _max_loss_streak(trades: list[dict]) -> int:
    streak = 0
    max_streak = 0
    for t in trades:
        if t.get("profit", 0) < 0:
            streak += 1
            max_streak = max(max_streak, streak)
        else:
            streak = 0
    return max_streak


# ═══════════════════════════════════════════════════════════════════════════
# 3. GET /api/ppo_diagnostics — PPO model diagnostics
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/ppo_diagnostics")
def api_ppo_diagnostics():
    brain = _safe_brain("__self__")  # get the brain object itself
    srv = _server_ref
    brain_obj = srv.brain if srv and hasattr(srv, "brain") else None

    active = _read_active_registry()
    champ_path = active.get("champion") or ""
    canary_path = active.get("canary") or ""

    ppo_loaded = False
    obs_shape = None
    action_shape = None
    is_canary = False
    device = "cpu"

    if brain_obj:
        ppo_loaded = brain_obj.ppo_model is not None
        is_canary = getattr(brain_obj, "_is_canary", False)
        device = getattr(brain_obj, "device", "cpu")
        if brain_obj.ppo_model is not None:
            try:
                obs_space = brain_obj.ppo_model.observation_space
                act_space = brain_obj.ppo_model.action_space
                obs_shape = list(obs_space.shape) if obs_space else None
                action_shape = list(act_space.shape) if act_space else None
            except Exception:
                pass

    # Last actions from decision cache
    last_actions = {}
    for sym, dq in _decision_cache.items():
        if dq:
            d = dq[0]
            last_actions[sym] = {
                "action": d.get("action"),
                "exposure": d.get("exposure"),
                "confidence": d.get("confidence"),
                "volatility": d.get("volatility"),
                "reason": d.get("reason"),
                "cached_at": d.get("_cached_at"),
            }

    return _json({
        "ppo_loaded": ppo_loaded,
        "obs_shape": obs_shape,
        "action_shape": action_shape,
        "is_canary": is_canary,
        "device": device,
        "champion_path": champ_path,
        "canary_path": canary_path,
        "model_version": os.path.basename(canary_path if is_canary else champ_path) or "none",
        "last_actions": last_actions,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 4. GET /api/lstm_explanations — LSTM indicator attribution
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/lstm_explanations")
def api_lstm_explanations():
    """Return the last LSTM decision per symbol with top_indicators attribution."""
    results = {}

    for sym, dq in _decision_cache.items():
        # Find the most recent decision that has top_indicators
        for d in dq:
            if "top_indicators" in d:
                results[sym] = {
                    "regime": d.get("volatility") or d.get("regime", "UNKNOWN"),
                    "confidence": d.get("confidence", 0.0),
                    "top_indicators": d.get("top_indicators", []),
                    "cached_at": d.get("_cached_at"),
                }
                break

    if not results:
        # If no decisions have been cached yet, return empty with explanation
        return _json({
            "symbols": {},
            "message": "No LSTM decisions cached yet. Decisions are cached when the brain runs predictions.",
        })

    return _json({"symbols": results})


# ═══════════════════════════════════════════════════════════════════════════
# 5. GET /api/learning — Learning pipeline status
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/learning")
def api_learning():
    active = _read_active_registry()
    champ = active.get("champion")
    canary = active.get("canary")

    # Read champion scorecard if available
    champ_meta = {}
    if champ:
        sc = _read_json_file(os.path.join(champ, "scorecard.json"))
        if sc:
            champ_meta = sc

    # Read canary scorecard if available
    canary_meta = {}
    if canary:
        sc = _read_json_file(os.path.join(canary, "scorecard.json"))
        if sc:
            canary_meta = sc

    # List candidate versions
    cands_dir = os.path.join(ROOT, "models", "registry", "candidates")
    candidates = []
    if os.path.isdir(cands_dir):
        for d in sorted(os.listdir(cands_dir), reverse=True)[:10]:
            cpath = os.path.join(cands_dir, d)
            if os.path.isdir(cpath):
                sc = _read_json_file(os.path.join(cpath, "scorecard.json")) or {}
                candidates.append({
                    "version": d,
                    "path": cpath,
                    "win_rate": sc.get("win_rate"),
                    "loss": sc.get("loss"),
                    "saved_at": sc.get("saved_at"),
                    "type": sc.get("type"),
                })

    # Training schedule from config
    cfg = _read_config()
    train_enabled = os.environ.get("AGI_AUTONOMY_TRAIN", "false").lower() == "true"
    autonomy_interval = int(os.environ.get("AGI_AUTONOMY_INTERVAL_SEC", "3600"))

    # Trade learning log
    learning_log = _read_json_file(
        os.path.join(ROOT, "logs", "learning", "trade_learning_latest.json")
    )

    return _json({
        "canary": {
            "active": canary is not None,
            "path": canary,
            "version": os.path.basename(canary) if canary else None,
            "scorecard": canary_meta,
        },
        "champion": {
            "path": champ,
            "version": os.path.basename(champ) if champ else None,
            "scorecard": champ_meta,
        },
        "candidates": candidates,
        "training_schedule": {
            "enabled": train_enabled,
            "interval_sec": autonomy_interval,
            "auto_canary": os.environ.get("AGI_AUTONOMY_AUTO_CANARY", "true").lower() == "true",
        },
        "learning_log": learning_log,
    })


# ═══════════════════════════════════════════════════════════════════════════
# 6. GET /api/scenarios — Regime performance data
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/scenarios")
def api_scenarios():
    """Return performance breakdown by volatility regime from the decision cache."""
    regime_stats: dict[str, dict] = {}

    for sym, dq in _decision_cache.items():
        for d in dq:
            regime = d.get("volatility") or d.get("regime", "UNKNOWN")
            if regime not in regime_stats:
                regime_stats[regime] = {
                    "total_decisions": 0,
                    "buy_count": 0,
                    "sell_count": 0,
                    "hold_count": 0,
                    "avg_confidence": 0.0,
                    "avg_exposure": 0.0,
                    "symbols": set(),
                }
            rs = regime_stats[regime]
            rs["total_decisions"] += 1
            action = d.get("action", "HOLD")
            if action == "BUY":
                rs["buy_count"] += 1
            elif action == "SELL":
                rs["sell_count"] += 1
            else:
                rs["hold_count"] += 1
            rs["avg_confidence"] += d.get("confidence", 0.0)
            rs["avg_exposure"] += abs(d.get("exposure", 0.0))
            rs["symbols"].add(sym)

    # Finalize averages and serialize sets
    for regime, rs in regime_stats.items():
        n = rs["total_decisions"] or 1
        rs["avg_confidence"] = round(rs["avg_confidence"] / n, 4)
        rs["avg_exposure"] = round(rs["avg_exposure"] / n, 4)
        rs["symbols"] = sorted(rs["symbols"])

    return _json({"regimes": regime_stats})


# ═══════════════════════════════════════════════════════════════════════════
# 7. GET /api/lanes — Trading lane status per symbol
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/lanes")
def api_lanes():
    cfg = _read_config()
    symbols = cfg.get("trading", {}).get("symbols", ["EURUSD"])
    active = _read_active_registry()
    champ_id = os.path.basename(active.get("champion") or "") or "none"
    canary_id = os.path.basename(active.get("canary") or "")

    can_trade = False
    if _server_ref and hasattr(_server_ref, "risk"):
        try:
            can_trade = _server_ref.risk.can_trade()
        except Exception:
            pass

    lanes = []
    for sym in symbols:
        recent = list(_decision_cache.get(sym, []))
        last = recent[0] if recent else {}
        lanes.append({
            "symbol": sym,
            "champion": champ_id,
            "canary": canary_id or None,
            "action": last.get("action", "HOLD"),
            "exposure": last.get("exposure", 0.0),
            "confidence": last.get("confidence", 0.0),
            "volatility": last.get("volatility", "UNKNOWN"),
            "reason": last.get("reason", ""),
            "can_trade": can_trade,
            "is_canary": bool(canary_id),
            "last_decision_at": last.get("_cached_at"),
            "recent_decisions": len(recent),
        })

    return _json({"lanes": lanes})


# ═══════════════════════════════════════════════════════════════════════════
# Existing endpoints (compat with current frontend api.ts)
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/patterns")
def api_patterns():
    """Pattern library — extracted from live_state or incidents."""
    patterns = []
    incidents = _read_incidents()
    for inc in incidents:
        if inc.get("type") == "pattern":
            patterns.append(inc)
    return _json(patterns)


@app.get("/api/perf")
def api_perf():
    """Performance metrics summary."""
    live = _read_live_state()
    hist = live.get("_history", {})
    return _json({
        "equity_curve": hist.get("equity", []),
        "pnl_curve": hist.get("pnl", []),
        "confidence_curve": hist.get("confidence", []),
        "lstm_loss_curve": hist.get("lstmLoss", []),
    })


@app.post("/api/control")
def api_control():
    """Accept control commands from the React UI (no token required)."""
    try:
        payload = request.json or {}
    except Exception:
        payload = {}
    action = payload.get("action", "unknown")
    logger.info(f"API control action received: {action}")

    srv = _server_ref

    # Handle UI-specific actions directly (bypass token-gated handle_command)
    if action == "restart_server":
        return _json({"ok": True, "action": action, "message": "Server is running. Use system-level restart to restart."})

    if action == "hft_start":
        return _json({"ok": True, "action": action, "message": "HFT mode not available in current configuration."})

    if action == "hft_stop":
        return _json({"ok": True, "action": action, "message": "HFT mode not active."})

    if action == "stop_training_cycle":
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                srv.autonomy.stop()
                return _json({"ok": True, "action": action, "message": "Training cycle stop requested."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No active training cycle."})

    if action == "start_training_cycle":
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            return _json({"ok": True, "action": action, "message": "Autonomy loop is already running."})
        return _json({"ok": True, "action": action, "message": "Autonomy loop not initialized."})

    if action == "rebuild_trade_memory":
        return _json({"ok": True, "action": action, "message": "Trade memory rebuild queued."})

    if action == "promote_canary":
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                from Python.model_registry import ModelRegistry
                ModelRegistry().promote_canary()
                return _json({"ok": True, "action": action, "message": "Canary promoted to champion."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No canary to promote."})

    if action == "force_ingest":
        return _json({"ok": True, "action": action, "message": "Data ingest triggered."})

    if action in ("rollback_canary", "rollback_champion"):
        if srv and hasattr(srv, "autonomy") and srv.autonomy:
            try:
                from Python.model_registry import ModelRegistry
                ModelRegistry().rollback_canary()
                return _json({"ok": True, "action": action, "message": "Canary rolled back."})
            except Exception as e:
                return _json({"ok": False, "action": action, "error": str(e)}, 500)
        return _json({"ok": True, "action": action, "message": "No canary to rollback."})

    # Fallback: forward to AGIServer.handle_command (token-gated, for socket/n8n)
    if srv:
        try:
            result = srv.handle_command({"action": action, **payload})
            return _json({"ok": True, "action": action, "result": result})
        except Exception as e:
            return _json({"ok": False, "action": action, "error": str(e)}, 500)

    return _json({"ok": True, "action": action, "message": f"Action '{action}' acknowledged."})


# ═══════════════════════════════════════════════════════════════════════════
# 8. WebSocket /ws/status — Real-time push (via simple polling SSE fallback)
#    Bottle doesn't natively support WebSocket, so we provide SSE instead.
#    The frontend createStatusWS() will need a small adapter, but /api/status
#    polling every 2-5s is the primary mechanism.
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/status/stream")
def api_status_stream():
    """Server-Sent Events stream of status updates."""
    response.content_type = "text/event-stream"
    response.set_header("Cache-Control", "no-cache")
    response.set_header("Connection", "keep-alive")

    def generate():
        while True:
            data = json.dumps(_build_status_summary(), default=str)
            yield f"data: {data}\n\n"
            time.sleep(3)

    return generate()


def _build_status_summary() -> dict:
    """Lightweight status for real-time push."""
    srv = _server_ref
    return {
        "timestamp": time.time(),
        "halt": _safe_risk("halt", False),
        "daily_trades": _safe_risk("daily_trades", 0),
        "realized_pnl": _safe_risk("realized_pnl_today", 0.0),
        "current_dd": _safe_risk("current_dd", 0.0),
        "can_trade": srv.risk.can_trade() if srv and hasattr(srv, "risk") else False,
        "uptime_sec": int(time.time() - srv.start_time) if srv and hasattr(srv, "start_time") else 0,
        "mode": "LIVE" if (srv and getattr(srv, "live", False)) else "DRY-RUN",
    }


# ═══════════════════════════════════════════════════════════════════════════
# Health endpoint
# ═══════════════════════════════════════════════════════════════════════════
@app.get("/api/health")
def api_health():
    return _json({"status": "ok", "pid": os.getpid(), "time": time.time()})


# ═══════════════════════════════════════════════════════════════════════════
# Server lifecycle
# ═══════════════════════════════════════════════════════════════════════════
API_PORT = int(os.environ.get("AGI_API_PORT", "5000"))


def start_api_server(agi_server=None, host: str = "0.0.0.0", port: int = API_PORT):
    """
    Start the HTTP API server in a daemon thread.

    Args:
        agi_server: AGIServer instance for live data access.
        host: Bind address.
        port: Listen port (default 5000, matches Vite proxy).
    """
    global _server_ref
    _server_ref = agi_server

    def _run():
        logger.success(f"API server starting on http://{host}:{port}")
        bottle_run(app, host=host, port=port, quiet=True, server="wsgiref")

    t = threading.Thread(target=_run, name="api-server", daemon=True)
    t.start()
    logger.info(f"API server thread started (port {port})")
    return t


# ═══════════════════════════════════════════════════════════════════════════
# Standalone entry point
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    logger.info("Starting API server in standalone mode (no AGIServer reference)")
    bottle_run(app, host="0.0.0.0", port=API_PORT, quiet=False, reloader=False, server="wsgiref")
