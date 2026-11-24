"""Order plumbing shared by the stand-alone strategies (Gold Breakout, meta-brain trader).

Their positions are kept off the council's list (Exness adapter "own" ids), so the council never closes,
resizes or stacks on them; each strategy tracks its own tickets."""
from __future__ import annotations

import json
import os

DONE = (10008, 10009)

# Event logs (logs/<name>.jsonl) of the stand-alone strategies; their "open" rows carry the ticket.
STRATEGY_LOGS = ("gold_breakout", "meta_trader", "gemma_trader", "qwen_trader", "jev_trader")


def strategy_tickets(log_dir: str) -> set[str]:
    """Tickets of every position a stand-alone strategy opened. The broker API has no magic numbers, so this
    is how the council's own trade record (e.g. a PPO model's trial) leaves the strategies' trades out."""
    out: set[str] = set()
    for name in STRATEGY_LOGS:
        try:
            with open(os.path.join(log_dir, f"{name}.jsonl"), encoding="utf-8") as f:
                for ln in f:
                    if '"ticket"' not in ln:
                        continue
                    try:
                        t = json.loads(ln).get("ticket")
                    except ValueError:
                        continue
                    if t not in (None, "", 0):
                        out.add(str(t))
        except OSError:
            pass
    return out


def risk_lots(mt5, symbol: str, entry: float, stop: float, risk_pct: float, min_lot_max_risk_pct: float):
    """(lots, actual risk % of equity) risking `risk_pct` at the stop, or None if even the minimum lot would
    risk more than `min_lot_max_risk_pct`."""
    acc, info = mt5.account_info(), mt5.symbol_info(symbol)
    if acc is None or info is None:
        return None
    equity = float(acc.equity)
    tick_size = float(getattr(info, "trade_tick_size", 0) or info.point)
    tick_value = float(getattr(info, "trade_tick_value_loss", 0) or getattr(info, "trade_tick_value", 0) or 0)
    if equity <= 0 or tick_size <= 0 or tick_value <= 0 or entry == stop:
        return None
    loss_per_lot = abs(entry - stop) / tick_size * tick_value
    step = float(getattr(info, "volume_step", 0.01) or 0.01)
    vmin = float(getattr(info, "volume_min", 0.01) or 0.01)
    lots = int(equity * float(risk_pct) / 100.0 / loss_per_lot / step) * step
    if lots < vmin:
        if vmin * loss_per_lot <= equity * float(min_lot_max_risk_pct) / 100.0:
            lots = vmin
        else:
            return None
    return round(lots, 8), round(lots * loss_per_lot / equity * 100.0, 2)


def all_positions(mt5, symbol: str | None = None) -> list:
    """Positions including the ones hidden from the council."""
    had = getattr(mt5, "hide_foreign", None)
    try:
        if had is not None:
            mt5.hide_foreign = False
        return list((mt5.positions_get(symbol=symbol) if symbol else mt5.positions_get()) or [])
    finally:
        if had is not None:
            mt5.hide_foreign = had


# Account-wide cap on positions in one direction, across every trader (trading.crowd_cap, user 2026-11-24): on
# 2026-11-24 every brain sold the same dollar move at ~5% each and equity swung from $1,928 to $940 in six hours.
# Server_AGI sets this at start; 0 = off.
CROWD = {"max_same_side": 0}


def crowd_reason(mt5, side: int, cap: int | None = None) -> str | None:
    """Why a new `side` (+1 buy / -1 sell) trade may not open: `cap` or more positions in that direction are
    already open across all traders. None when it may."""
    cap = int(CROWD["max_same_side"] if cap is None else cap or 0)
    if cap <= 0 or not side:
        return None
    buy = getattr(mt5, "ORDER_TYPE_BUY", 0)
    n = sum(1 for p in all_positions(mt5) if (1 if int(getattr(p, "type", buy)) == buy else -1) == side)
    if n >= cap:
        return f"{n} {'buy' if side > 0 else 'sell'} positions already open across all traders (cap {cap})"
    return None


def open_hidden(mt5, req: dict):
    """Send an opening order and take the new position off the council's list. Returns (ok, ticket, result)."""
    own_before = set(mt5._own_load()) if hasattr(mt5, "_own_load") else None
    res = mt5.order_send(req)
    ok = res is not None and int(getattr(res, "retcode", 0)) in DONE
    ticket = None
    if ok:
        if own_before is not None:
            own_after = set(mt5._own_load())
            new = own_after - own_before
            mt5._own_save(own_after - new)
            ticket = next(iter(new), None)
        if ticket is None:
            ticket = next((str(p.ticket) for p in all_positions(mt5, req["symbol"])
                           if int(getattr(p, "magic", -2) or -2) == int(req.get("magic", -1))), None)
    return ok, ticket, res


def close_ticket(mt5, symbol: str, ticket, magic: int, comment: str):
    """Close one (possibly hidden) position at market. Returns (ok, position or None)."""
    pos = next((p for p in all_positions(mt5, symbol) if str(p.ticket) == str(ticket)), None)
    if pos is None:
        return False, None
    tick = mt5.symbol_info_tick(symbol)
    buy = pos.type == mt5.ORDER_TYPE_BUY
    req = {"action": mt5.TRADE_ACTION_DEAL, "symbol": symbol, "volume": float(pos.volume),
           "type": mt5.ORDER_TYPE_SELL if buy else mt5.ORDER_TYPE_BUY, "position": pos.ticket,
           "price": float(tick.bid if buy else tick.ask) if tick else 0.0, "deviation": 50,
           "type_time": mt5.ORDER_TIME_GTC, "magic": int(magic), "comment": comment}
    res = mt5.order_send(req)
    return res is not None and int(getattr(res, "retcode", 0)) in DONE, pos
