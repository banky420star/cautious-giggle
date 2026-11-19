"""Trend gate: no new trades against BOTH the 4-hour and the 1-hour trend.

Trend on a timeframe = last COMPLETED bar's close above (+1) or below (-1) its 50-bar EMA.
A new trade or an add is blocked when its side is against both timeframes. Mixed or agreeing
trends pass. Closing or reducing a position always passes (the gate only stops new risk).

Why: the 2026-11-19 post-mortem of 208 real demo trades found the 42 buys made against both the
4h and 1h trend lost $94.50 (24% won), more than half of the total loss. A year-long backtest showed
following the trend is not an edge by itself, so this is a filter against the worst trades, not a signal.

Config (trading.trend_gate): enabled (true), ema (50), timeframes (["4h", "1h"]), cache_s (300).
"""
import time

from loguru import logger

_TF = {"4h": "TIMEFRAME_H4", "1h": "TIMEFRAME_H1", "1d": "TIMEFRAME_D1", "15m": "TIMEFRAME_M15"}


def ema_trend(closes, span: int = 50) -> int:
    """+1 / -1 / 0: last close vs its EMA(span). Needs at least 2*span closes, else 0 (no view)."""
    closes = [float(c) for c in closes]
    if len(closes) < 2 * span:
        return 0
    a = 2.0 / (span + 1.0)
    e = closes[0]
    for c in closes[1:]:
        e = a * c + (1.0 - a) * e
    last = closes[-1]
    return 1 if last > e else -1 if last < e else 0


def verdict(side: int, trends: dict) -> tuple[str, str]:
    """side +1 buy / -1 sell; trends {tf: +1/-1/0}. 'block' only if every timeframe points the other way."""
    if not trends or any(v == 0 for v in trends.values()):
        return "pass", "trend unknown on some timeframe: no gate"
    against = [tf for tf, v in trends.items() if v == -side]
    word = "buy" if side > 0 else "sell"
    if len(against) == len(trends):
        dirn = "down" if side > 0 else "up"
        return "block", f"{word} against the {' and '.join(trends)} trend (all {dirn})"
    return "pass", f"{word} ok: " + ", ".join(f"{tf} {'up' if v > 0 else 'down'}" for tf, v in trends.items())


class TrendGate:
    def __init__(self, cfg: dict | None, mt5_mod):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.span = int(cfg.get("ema", 50))
        self.tfs = [str(t) for t in (cfg.get("timeframes") or ["4h", "1h"])]
        self.cache_s = float(cfg.get("cache_s", 300))
        self.mt5 = mt5_mod
        self._cache = {}

    def trends(self, symbol: str) -> dict:
        now = time.time()
        hit = self._cache.get(symbol)
        if hit and now - hit[0] < self.cache_s:
            return hit[1]
        out = {}
        for tf in self.tfs:
            try:
                const = getattr(self.mt5, _TF[tf])
                rates = self.mt5.copy_rates_from_pos(symbol, const, 1, self.span * 4)   # from 1 = completed bars only
                out[tf] = ema_trend([r["close"] for r in rates], self.span) if rates is not None and len(rates) else 0
            except Exception as exc:
                logger.warning(f"trend gate: {symbol} {tf} bars unavailable ({exc})")
                out[tf] = 0
        self._cache[symbol] = (now, out)
        return out

    def check(self, symbol: str, side: int) -> tuple[str, str, dict]:
        if not self.enabled:
            return "pass", "trend gate off", {}
        tr = self.trends(symbol)
        v, why = verdict(side, tr)
        return v, why, tr
