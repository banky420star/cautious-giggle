"""Position-size multiplier from a market's real closed-trade record (shared by the bot and the dashboard)."""
from __future__ import annotations


def trade_memory_scale(trade_memory: dict | None, cfg: dict | None = None) -> tuple[float, str]:
    """Size multiplier from this market's real closed-trade record (logs/learning, rebuilt every few minutes).
    1.0 = normal; `losing_scale` (default 0.5) when the record is losing or on a losing streak. Never above 1.0.
    trading.trade_memory_scale: {enabled, min_trades, losing_scale, max_loss_streak}."""
    c = ((((cfg or {}).get("trading") or {}).get("trade_memory_scale")) or {}) if isinstance(cfg, dict) else {}
    if not bool(c.get("enabled", True)):
        return 1.0, "off"
    m = trade_memory or {}
    n = int(m.get("trades", 0) or 0)
    pf = float(m.get("profit_factor", 0.0) or 0.0)
    streak = int(m.get("recent_loss_streak", 0) or 0)
    low = max(0.1, min(1.0, float(c.get("losing_scale", 0.5))))
    if streak > int(c.get("max_loss_streak", 3)):
        return low, f"{streak} losses in a row"
    if n < int(c.get("min_trades", 20)):
        return 1.0, f"only {n} trades so far"
    if pf < 1.0:
        return low, f"losing record: profit factor {pf:.2f} over {n} trades"
    return 1.0, f"profit factor {pf:.2f} over {n} trades"
