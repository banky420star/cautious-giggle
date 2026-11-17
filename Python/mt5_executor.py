import math
import time

from loguru import logger
from Python.mt5_compat import mt5


MAGIC_BY_SYMBOL = {
    "BTCUSDm": 51000,
    "XAUUSDm": 52000,
}

LANE_MAGIC_OFFSET = {
    "champion": 0,
    "canary": 100,
    "history": 200,
    "unknown": 900,
}

# Every magic this bot emits for a symbol lies in [base, base + MAGIC_SPAN).
MAGIC_SPAN = 1000

# MT5 trade server return codes (numeric so they work with the compat stub).
RET_REQUOTE = 10004
RET_REJECT = 10006
RET_PLACED = 10008
RET_DONE = 10009
RET_DONE_PARTIAL = 10010
RET_MARKET_CLOSED = 10018
RET_NO_MONEY = 10019
RET_PRICE_CHANGED = 10020
RET_PRICE_OFF = 10021
RET_NO_CHANGES = 10025
RET_INVALID_FILL = 10030
RET_CONNECTION = 10031

SUCCESS_RETCODES = {RET_DONE, RET_PLACED, RET_DONE_PARTIAL}
# Retry immediately with a fresh tick.
RETRY_RETCODES = {RET_REQUOTE, RET_PRICE_CHANGED, RET_PRICE_OFF}
# Environmental conditions: never count toward the error halt.
BENIGN_RETCODES = RETRY_RETCODES | {RET_MARKET_CLOSED, RET_NO_CHANGES, RET_CONNECTION}

# symbol_info().filling_mode is a bitmask (SYMBOL_FILLING_FOK=1, SYMBOL_FILLING_IOC=2),
# not an ORDER_FILLING_* enum value.
_SYMBOL_FILLING_FOK = 1
_SYMBOL_FILLING_IOC = 2

_MAX_SEND_ATTEMPTS = 3
_ATR_CACHE_TTL_SEC = 30.0


_SIZE_BLOCK_LOGGED: dict = {}


class MT5Executor:
    def __init__(self, risk):
        self.risk = risk
        self._atr_cache = {}

    # ------------------------------------------------------------------ helpers
    def _symbol_info(self, symbol):
        return mt5.symbol_info(symbol)

    def _symbol_tick(self, symbol):
        tick = mt5.symbol_info_tick(symbol)
        if tick is None:
            return None
        try:
            if float(tick.bid) <= 0.0 or float(tick.ask) <= 0.0 or not math.isfinite(float(tick.bid)):
                return None
        except Exception:
            return None
        return tick

    def get_tick(self, symbol):
        return self._symbol_tick(symbol)

    def _profile(self, symbol) -> dict:
        try:
            return self.risk.get_symbol_profile(symbol) or {}
        except Exception:
            return {}

    def _record_error(self):
        fn = getattr(self.risk, "record_error", None)
        if callable(fn):
            fn()

    def _record_success(self):
        fn = getattr(self.risk, "record_success", None)
        if callable(fn):
            fn()

    def _can_open(self, symbol) -> bool:
        fn = getattr(self.risk, "can_open", None) or getattr(self.risk, "can_trade", None)
        return bool(fn(symbol)) if callable(fn) else True

    def _symbol_magic_base(self, symbol: str) -> int:
        profile = self._profile(symbol)
        for key in ("magic_base", "magic"):
            if key in profile:
                try:
                    return int(profile.get(key))
                except Exception:
                    pass
        return int(MAGIC_BY_SYMBOL.get(str(symbol), 59000))

    def _is_own_position(self, p) -> bool:
        magic = getattr(p, "magic", None)
        if magic is None:
            # Test doubles / brokers that do not report magic: assume ours.
            return True
        base = self._symbol_magic_base(getattr(p, "symbol", ""))
        return 0 <= int(magic) - base < MAGIC_SPAN

    def _lane_for_order(self, order_meta: dict | None) -> str:
        lane = str((order_meta or {}).get("lane", "unknown") or "unknown").strip().lower()
        if lane in LANE_MAGIC_OFFSET:
            return lane
        return "unknown"

    def _symbol_tag(self, symbol: str) -> str:
        symbol_str = str(symbol or "").upper()
        if symbol_str.startswith("BTC"):
            return "BTC"
        if symbol_str.startswith("XAU"):
            return "XAU"
        return symbol_str[:4] or "UNK"

    def _magic_for_order(self, symbol: str, order_meta: dict | None, request_kind: str = "open") -> int:
        base = self._symbol_magic_base(symbol)
        lane = self._lane_for_order(order_meta)
        kind_offset = {"open": 0, "close": 10, "manage": 20}.get(str(request_kind), 90)
        return int(base + int(LANE_MAGIC_OFFSET.get(lane, 900)) + int(kind_offset))

    def _order_comment(self, symbol: str, order_meta: dict | None, request_kind: str = "open") -> str:
        meta = order_meta or {}
        sym = self._symbol_tag(symbol)
        lane = self._lane_for_order(meta)
        lane_tag = {"champion": "CH", "canary": "CA", "history": "HI", "unknown": "UN"}.get(lane, "UN")
        family = str(meta.get("model_family", "P") or "P").upper()[:1]
        version = str(meta.get("model_version", "") or "")
        version_tag = version[-6:] if version else "000000"
        ppo_target = float(meta.get("ppo_target", meta.get("exposure", 0.0)) or 0.0)
        ppo_tag = int(round(ppo_target * 100.0))
        req_tag = {"open": "O", "close": "C", "manage": "M"}.get(str(request_kind), "U")
        comment = f"AGI|{sym}|{lane_tag}|{req_tag}|{family}{version_tag}|P{ppo_tag:+03d}"
        return comment[:31]

    def _result_ticket(self, result):
        if result is None:
            return None
        for attr in ("order", "deal"):
            value = getattr(result, attr, None)
            if value not in (None, 0):
                return int(value)
        return None

    def _log_order_send(self, symbol: str, request_action: str, request: dict, result, order_meta: dict | None):
        meta = order_meta or {}
        executed_volume = getattr(result, "volume", None) if result is not None else None
        payload = {
            "action": str(request_action),
            "request_action": str(request_action),
            "side": "buy" if request.get("type") == mt5.ORDER_TYPE_BUY else ("sell" if "type" in request else ""),
            "lots": float(request.get("volume", 0.0) or 0.0),
            "executed_lots": float(executed_volume if executed_volume else 0.0),
            "price": request.get("price"),
            "sl": request.get("sl"),
            "tp": request.get("tp"),
            "target": float(meta.get("exposure", 0.0) or 0.0),
            "ppo": float(meta.get("ppo_target", 0.0) or 0.0),
            "dreamer": float(meta.get("dreamer_target", 0.0) or 0.0),
            "agi": float(meta.get("agi_bias", 0.0) or 0.0),
            "magic": request.get("magic"),
            "comment": request.get("comment"),
            "retcode": getattr(result, "retcode", None) if result is not None else None,
            "ticket": self._result_ticket(result),
        }
        if result is None:
            try:
                payload["last_error"] = str(mt5.last_error())
            except Exception:
                payload["last_error"] = "unknown"
        logger.info(
            "ORDER_SEND {} | action={} side={} lots={:.2f} price={} sl={} tp={} target={:.4f} magic={} retcode={} ticket={}",
            symbol,
            payload["action"],
            payload["side"],
            payload["lots"],
            payload["price"],
            payload["sl"],
            payload["tp"],
            payload["target"],
            payload["magic"],
            payload["retcode"],
            payload["ticket"],
        )
        return payload

    def _select_filling_mode(self, symbol):
        info = self._symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_IOC
        try:
            fm = int(getattr(info, "filling_mode", 0) or 0)
        except Exception:
            fm = 0
        if fm & _SYMBOL_FILLING_FOK:
            return mt5.ORDER_FILLING_FOK
        if fm & _SYMBOL_FILLING_IOC:
            return mt5.ORDER_FILLING_IOC
        # Neither FOK nor IOC allowed => exchange/instant execution with RETURN.
        return mt5.ORDER_FILLING_RETURN

    def normalize_volume(self, symbol, volume: float) -> float:
        """Floor volume to the symbol's volume_step and clamp to [volume_min, volume_max].

        Returns 0.0 when the volume is below volume_min.
        """
        vol = abs(float(volume))
        info = self._symbol_info(symbol)
        step = float(getattr(info, "volume_step", 0.0) or 0.0) if info is not None else 0.0
        vmin = float(getattr(info, "volume_min", 0.0) or 0.0) if info is not None else 0.0
        vmax = float(getattr(info, "volume_max", 0.0) or 0.0) if info is not None else 0.0
        if step <= 0.0:
            step = 0.01
        if vmin <= 0.0:
            vmin = step
        decimals = max(0, int(round(-math.log10(step)))) if step < 1 else 0
        vol = math.floor(vol / step + 1e-9) * step
        if vmax > 0.0:
            vol = min(vol, vmax)
        vol = round(vol, decimals)
        if vol + 1e-12 < vmin:
            return 0.0
        return vol

    def _volume_step(self, symbol) -> float:
        info = self._symbol_info(symbol)
        step = float(getattr(info, "volume_step", 0.0) or 0.0) if info is not None else 0.0
        return step if step > 0.0 else 0.01

    def _min_stop_distance(self, symbol):
        info = self._symbol_info(symbol)
        if info is None:
            return 0.0, 5

        point = float(info.point) if info.point else 0.0001
        stops_level = max(int(getattr(info, "trade_stops_level", 0)), 0)
        freeze_level = max(int(getattr(info, "trade_freeze_level", 0)), 0)
        min_points = max(stops_level, freeze_level) + 2
        return min_points * point, min_points

    def _sanitize_sl_tp(self, symbol, order_type, sl, tp, tick):
        info = self._symbol_info(symbol)
        if info is None or tick is None:
            return sl, tp

        digits = int(info.digits) if info.digits is not None else 5
        min_dist, _ = self._min_stop_distance(symbol)

        bid = float(tick.bid)
        ask = float(tick.ask)

        new_sl = float(sl) if sl else None
        new_tp = float(tp) if tp else None

        if order_type == mt5.ORDER_TYPE_BUY:
            max_sl = bid - min_dist
            min_tp = ask + min_dist
            if new_sl is not None:
                new_sl = round(min(new_sl, max_sl), digits)
                if new_sl <= 0:
                    new_sl = None
            if new_tp is not None:
                new_tp = round(max(new_tp, min_tp), digits)
        else:
            min_sl = ask + min_dist
            max_tp = bid - min_dist
            if new_sl is not None:
                new_sl = round(max(new_sl, min_sl), digits)
            if new_tp is not None:
                new_tp = round(min(new_tp, max_tp), digits)
                if new_tp <= 0:
                    new_tp = None

        return new_sl, new_tp

    # --------------------------------------------------------------- positions
    def get_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        longs = []
        shorts = []
        if positions:
            for p in positions:
                if not self._is_own_position(p):
                    continue
                if p.type == mt5.ORDER_TYPE_BUY:
                    longs.append(p)
                else:
                    shorts.append(p)
        return longs, shorts

    # ---------------------------------------------------------------- sizing
    def risk_capped_lots(self, symbol, requested_lots: float, sl_points: int | None = None) -> float:
        """Cap a lot size so that hitting the stop loses at most risk_percent of equity.

        Uses symbol tick_value/tick_size. Returns requested_lots unchanged if the risk
        engine has no risk_percent or broker data is unavailable.
        """
        risk_pct = float(getattr(self.risk, "risk_percent", 0.0) or 0.0)
        if risk_pct <= 0.0 or requested_lots <= 0.0:
            return requested_lots
        try:
            acc = mt5.account_info()
            info = self._symbol_info(symbol)
            if acc is None or info is None:
                return requested_lots
            equity = float(acc.equity)
            point = float(info.point) if info.point else 0.0
            tick_size = float(getattr(info, "trade_tick_size", 0.0) or point)
            tick_value = float(getattr(info, "trade_tick_value_loss", 0.0) or getattr(info, "trade_tick_value", 0.0) or 0.0)
            if sl_points is None:
                sl_points, _ = self._dynamic_points(symbol)
            sl_dist = float(sl_points) * point
            if equity <= 0 or tick_size <= 0 or tick_value <= 0 or sl_dist <= 0:
                return requested_lots
            loss_per_lot = sl_dist / tick_size * tick_value
            max_lots = equity * risk_pct / 100.0 / loss_per_lot
            capped = min(requested_lots, max_lots)
            vol_min = float(getattr(info, "volume_min", 0.0) or 0.0)
            ceiling = float(getattr(self.risk, "min_lot_max_risk_percent", 0.0) or 0.0)
            if 0 < capped < vol_min and ceiling > 0 and vol_min * loss_per_lot <= equity * ceiling / 100.0:
                logger.info(f"{symbol}: using minimum lot {vol_min} (stop risk "
                            f"{vol_min * loss_per_lot / equity * 100:.2f}% <= ceiling {ceiling:.1f}%)")
                return vol_min
            if 0 < capped < vol_min:
                # the broker's smallest lot is too big for this account: say so instead of skipping silently
                now = time.time()
                if now - _SIZE_BLOCK_LOGGED.get(symbol, 0.0) > 600:
                    _SIZE_BLOCK_LOGGED[symbol] = now
                    risk_min = vol_min * loss_per_lot / equity * 100
                    need = vol_min * loss_per_lot / (max(ceiling, risk_pct) / 100.0)
                    logger.warning(f"SIZE BLOCK {symbol} | minimum lot {vol_min} would risk {risk_min:.1f}% of "
                                   f"${equity:.2f} at the stop (cap {max(ceiling, risk_pct):.1f}%); needs about ${need:.0f}")
            return capped
        except Exception as exc:
            logger.warning(f"risk_capped_lots({symbol}) failed: {exc}")
            return requested_lots

    # ------------------------------------------------------------- reconcile
    def reconcile_exposure(self, symbol, target_exposure, max_lots, order_meta=None, execution_context=None):
        """Move the bot's net position in `symbol` to target_exposure * max_lots.

        Reductions and closes are always allowed (even when the risk engine has halted
        new trading); opening or increasing risk requires risk.can_open/can_trade.
        A trade is recorded only when an order actually executed.
        """
        longs, shorts = self.get_positions(symbol)
        long_lots = sum(float(p.volume) for p in longs)
        short_lots = sum(float(p.volume) for p in shorts)

        raw_target = float(target_exposure) * float(max_lots)
        # demo stacking: each stacked trade is sized like a single trade (its own stop, its own risk cap)
        stack = max(1, int((execution_context or {}).get("stack", 1) or 1))
        target_abs = self.normalize_volume(symbol, self.risk_capped_lots(symbol, abs(raw_target)) * stack)
        target_lots = math.copysign(target_abs, raw_target) if target_abs > 0 else 0.0
        step = self._volume_step(symbol)

        result_meta = {"request_action": "noop", "executed": False, "target_lots": float(target_lots)}
        executed_any = False

        if target_lots > 0:
            same, opposite, have, want, side = longs, shorts, long_lots, target_lots, mt5.ORDER_TYPE_BUY
        elif target_lots < 0:
            same, opposite, have, want, side = shorts, longs, short_lots, -target_lots, mt5.ORDER_TYPE_SELL
        else:
            same, opposite, have, want, side = [], longs + shorts, 0.0, 0.0, None

        # 1) Remove opposite-side (or all, when flattening) exposure first.
        if opposite:
            close_meta = self.close_positions(opposite, order_meta=order_meta, execution_context=execution_context)
            result_meta = close_meta
            if close_meta.get("executed"):
                executed_any = True
            if not close_meta.get("all_closed", close_meta.get("executed", False)):
                # Never open the other side while the old side is still on (hedge risk).
                result_meta["request_action"] = "close_failed"
                result_meta["target_lots"] = float(target_lots)
                if executed_any:
                    self._record_trade(symbol)
                return result_meta

        if side is None:
            if executed_any:
                self._record_trade(symbol)
            result_meta["target_lots"] = float(target_lots)
            return result_meta

        # 2) Same side too large: partial close of the excess.
        excess = have - want
        if excess >= step - 1e-9:
            reduce_meta = self.reduce_positions(symbol, same, excess, order_meta=order_meta, execution_context=execution_context)
            result_meta = reduce_meta
            if reduce_meta.get("executed"):
                executed_any = True
        # 3) Same side too small: add, gated on the risk engine.
        elif want - have >= step - 1e-9:
            add_lots = self.normalize_volume(symbol, want - have)
            if add_lots > 0:
                if not self._can_open(symbol):
                    result_meta = {"request_action": "blocked", "executed": False}
                else:
                    result_meta = self.open_position(
                        symbol, side, add_lots, order_meta=order_meta, execution_context=execution_context
                    )
                    if result_meta.get("executed"):
                        executed_any = True

        if executed_any:
            self._record_trade(symbol)
        result_meta["target_lots"] = float(target_lots)
        return result_meta

    def _record_trade(self, symbol):
        fn = getattr(self.risk, "record_trade", None)
        if callable(fn):
            fn(symbol)

    # ---------------------------------------------------------------- sending
    def _send_with_retry(self, symbol, request: dict, refresh_price):
        """order_send with retry on requote/price-changed. Returns (result, success)."""
        result = None
        for _ in range(_MAX_SEND_ATTEMPTS):
            result = mt5.order_send(request)
            retcode = getattr(result, "retcode", None) if result is not None else None
            if retcode in SUCCESS_RETCODES:
                return result, True
            if retcode in RETRY_RETCODES and refresh_price is not None:
                tick = self._symbol_tick(symbol)
                if tick is None:
                    break
                request["price"] = refresh_price(tick)
                continue
            break
        return result, False

    def _account_failure(self, result):
        retcode = getattr(result, "retcode", None) if result is not None else None
        if retcode in BENIGN_RETCODES:
            return
        self._record_error()

    def _close_one(self, p, volume, order_meta=None):
        close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
        tick = self._symbol_tick(p.symbol)
        if tick is None:
            return {"request_action": "close", "executed": False, "reason": "no_tick"}

        def _price(t):
            return t.bid if close_type == mt5.ORDER_TYPE_SELL else t.ask

        deviation = int(self._profile(p.symbol).get("entry_deviation", 20))
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": p.symbol,
            "volume": float(volume),
            "type": close_type,
            "position": p.ticket,
            "price": _price(tick),
            "deviation": deviation,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(p.symbol),
            "magic": self._magic_for_order(p.symbol, order_meta, request_kind="close"),
            "comment": self._order_comment(p.symbol, order_meta, request_kind="close"),
        }
        result, ok = self._send_with_retry(p.symbol, request, _price)
        meta = self._log_order_send(p.symbol, "close", request, result, order_meta)
        meta["executed"] = ok
        if ok:
            self._record_success()
        else:
            self._account_failure(result)
        return meta

    def close_positions(self, positions, order_meta=None, execution_context=None):
        last_meta = {"request_action": "close", "executed": False, "all_closed": True}
        any_ok = False
        all_ok = True
        for p in positions:
            meta = self._close_one(p, p.volume, order_meta=order_meta)
            any_ok = any_ok or bool(meta.get("executed"))
            all_ok = all_ok and bool(meta.get("executed"))
            last_meta = meta
        last_meta["executed"] = any_ok
        last_meta["all_closed"] = all_ok
        return last_meta

    def reduce_positions(self, symbol, positions, excess_lots, order_meta=None, execution_context=None):
        """Partially close `excess_lots` from `positions`, oldest first."""
        remaining = float(excess_lots)
        last_meta = {"request_action": "reduce", "executed": False}
        any_ok = False
        for p in sorted(positions, key=lambda x: getattr(x, "time", 0)):
            if remaining <= 1e-9:
                break
            vol = self.normalize_volume(symbol, min(float(p.volume), remaining))
            if vol <= 0:
                break
            meta = self._close_one(p, vol, order_meta=order_meta)
            meta["request_action"] = "reduce"
            last_meta = meta
            if meta.get("executed"):
                any_ok = True
                remaining -= vol
            else:
                break
        last_meta["executed"] = any_ok
        return last_meta

    # ------------------------------------------------------------- SL / TP
    def _atr_points(self, symbol, bars=120, period=14):
        cached = self._atr_cache.get(symbol)
        now = time.monotonic()
        if cached is not None and now - cached[0] < _ATR_CACHE_TTL_SEC:
            return cached[1]

        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 1, bars)
        info = self._symbol_info(symbol)
        if rates is None or len(rates) < period + 2 or info is None:
            return None

        point = float(info.point) if info.point else 0.0001
        trs = []
        prev_close = float(rates[0][4])
        for r in rates[1:]:
            high, low, close = float(r[2]), float(r[3]), float(r[4])
            trs.append(max(high - low, abs(high - prev_close), abs(low - prev_close)))
            prev_close = close
        if len(trs) < period:
            return None

        atr = sum(trs[-period:]) / float(period)
        atr_points = int(max(1, round(atr / max(point, 1e-12))))
        self._atr_cache[symbol] = (now, atr_points)
        return atr_points

    def _dynamic_points(self, symbol):
        profile = self._profile(symbol)
        base_sl = int(profile.get("sl_points", 250))
        base_tp = int(profile.get("tp_points", 450))

        atr_points = self._atr_points(symbol)
        if atr_points is None:
            return base_sl, base_tp

        dyn_sl = max(base_sl, int(atr_points * 1.4))
        dyn_tp = max(base_tp, int(atr_points * 2.2))
        return dyn_sl, dyn_tp

    def _get_sl_tp(self, symbol, order_type, entry_price):
        info = self._symbol_info(symbol)
        tick = self._symbol_tick(symbol)
        deviation = int(self._profile(symbol).get("entry_deviation", 20))
        if info is None or tick is None:
            return None, None, deviation

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        _, min_pts = self._min_stop_distance(symbol)

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        sl_dist = max(int(dyn_sl), min_pts) * point
        tp_dist = max(int(dyn_tp), min_pts) * point

        if order_type == mt5.ORDER_TYPE_BUY:
            sl = round(entry_price - sl_dist, digits)
            tp = round(entry_price + tp_dist, digits)
        else:
            sl = round(entry_price + sl_dist, digits)
            tp = round(entry_price - tp_dist, digits)

        sl, tp = self._sanitize_sl_tp(symbol, order_type, sl, tp, tick)
        return sl, tp, deviation

    def open_position(self, symbol, order_type, volume, order_meta=None, execution_context=None):
        tick = self._symbol_tick(symbol)
        if tick is None:
            return {"request_action": "open", "executed": False, "reason": "no_tick"}

        volume = self.normalize_volume(symbol, volume)
        if volume <= 0:
            return {"request_action": "open", "executed": False, "reason": "below_volume_min"}

        def _price(t):
            return t.ask if order_type == mt5.ORDER_TYPE_BUY else t.bid

        price = _price(tick)
        sl, tp, deviation = self._get_sl_tp(symbol, order_type, price)

        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": float(volume),
            "type": order_type,
            "price": price,
            "deviation": deviation,
            "type_time": mt5.ORDER_TIME_GTC,
            "type_filling": self._select_filling_mode(symbol),
            "magic": self._magic_for_order(symbol, order_meta, request_kind="open"),
            "comment": self._order_comment(symbol, order_meta, request_kind="open"),
        }
        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result, ok = self._send_with_retry(symbol, request, _price)
        meta = self._log_order_send(symbol, "open", request, result, order_meta)
        meta["executed"] = ok
        if ok:
            self._record_success()
        else:
            self._account_failure(result)
        return meta

    def manage_open_positions(self, symbol):
        positions = [p for p in (mt5.positions_get(symbol=symbol) or []) if self._is_own_position(p)]
        info = self._symbol_info(symbol)
        tick = self._symbol_tick(symbol)
        if not positions or info is None or tick is None:
            return

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        profile = self._profile(symbol)

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        breakeven_trigger = int(profile.get("breakeven_points", max(25, dyn_sl // 3)))
        trailing_trigger = int(profile.get("trailing_trigger_points", max(40, dyn_sl // 2)))
        trailing_step = int(profile.get("trailing_step_points", max(10, dyn_sl // 8)))
        # Only move SL when it improves by at least this much (avoids modify spam).
        min_move = max(1, trailing_step // 4) * point

        for p in positions:
            is_buy = p.type == mt5.ORDER_TYPE_BUY
            current_price = tick.bid if is_buy else tick.ask
            profit_points = (current_price - p.price_open) / point if is_buy else (p.price_open - current_price) / point

            cur_sl = float(p.sl) if p.sl else None
            cur_tp = float(p.tp) if p.tp else None
            new_sl = cur_sl
            new_tp = cur_tp

            if new_tp is None:
                tp_dist = dyn_tp * point
                new_tp = p.price_open + tp_dist if is_buy else p.price_open - tp_dist

            def _better(candidate, existing):
                if existing is None:
                    return True
                return candidate > existing + min_move if is_buy else candidate < existing - min_move

            if profit_points >= breakeven_trigger:
                be_buffer = 5 * point
                be_sl = p.price_open + be_buffer if is_buy else p.price_open - be_buffer
                if _better(be_sl, new_sl):
                    new_sl = be_sl

            if profit_points >= trailing_trigger:
                trail_dist = trailing_step * point
                trail_sl = current_price - trail_dist if is_buy else current_price + trail_dist
                if _better(trail_sl, new_sl):
                    new_sl = trail_sl

            if new_sl == cur_sl and new_tp == cur_tp:
                continue

            new_sl, new_tp = self._sanitize_sl_tp(symbol, p.type, new_sl, new_tp, tick)
            if new_sl is not None:
                new_sl = round(new_sl, digits)
            if new_tp is not None:
                new_tp = round(new_tp, digits)

            sl_changed = new_sl is not None and (cur_sl is None or abs(new_sl - cur_sl) > 0.5 * point)
            # Only (re)set TP when it is missing; never fight a broker-adjusted TP each cycle.
            tp_changed = new_tp is not None and cur_tp is None
            if not sl_changed and not tp_changed:
                continue

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": p.ticket,
                "sl": new_sl if new_sl is not None else (cur_sl or 0.0),
                "tp": new_tp if new_tp is not None else (cur_tp or 0.0),
                "magic": self._magic_for_order(symbol, None, request_kind="manage"),
                "comment": self._order_comment(symbol, None, request_kind="manage"),
            }
            result = mt5.order_send(req)
            self._log_order_send(symbol, "manage", req, result, {"symbol": symbol})
            retcode = getattr(result, "retcode", None) if result is not None else None
            if retcode in SUCCESS_RETCODES or retcode == RET_NO_CHANGES:
                continue
            self._account_failure(result)
