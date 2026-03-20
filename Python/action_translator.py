from __future__ import annotations

from typing import Optional

from Python.mt5_compat import mt5


def translate_trade_action(
    symbol: str,
    action_meta: dict,
    exposure: float,
    max_lots: float,
    tick: Optional[mt5.Tick] = None,
) -> Optional[dict]:
    if action_meta is None or tick is None:
        return None
    if abs(float(exposure)) < 0.01:
        return None

    direction = 1 if action_meta.get("direction", 0.0) >= 0.0 else -1
    size = float(action_meta.get("size", 0.0))
    if size <= 0.0:
        return None

    entry_mode = action_meta.get("entry_mode", "market")
    entry_offset_pct = float(action_meta.get("entry_offset_pct", 0.0))
    tp_offset_pct = float(action_meta.get("tp_offset_pct", 0.0))
    sl_offset_pct = float(action_meta.get("sl_offset_pct", 0.0))

    mid_price = float((tick.ask + tick.bid) / 2.0)
    entry_price = _compute_entry_price(direction, mid_price, entry_offset_pct)

    if direction >= 0:
        tp_price = float(entry_price * (1.0 + tp_offset_pct))
        sl_price = float(entry_price * (1.0 - sl_offset_pct))
    else:
        tp_price = float(entry_price * (1.0 - tp_offset_pct))
        sl_price = float(entry_price * (1.0 + sl_offset_pct))

    lots = round(abs(size * max_lots), 2)
    order_type = "BUY" if direction > 0 else "SELL"

    return {
        "symbol": symbol,
        "order_type": order_type,
        "entry_mode": entry_mode,
        "volume_lots": lots,
        "entry_price": round(entry_price, 6),
        "tp_price": round(tp_price, 6),
        "sl_price": round(sl_price, 6),
        "exposure": float(exposure),
    }


def _compute_entry_price(direction: float, base_price: float, offset_pct: float) -> float:
    if direction >= 0:
        return base_price * (1.0 + offset_pct)
    return base_price * (1.0 - offset_pct)
