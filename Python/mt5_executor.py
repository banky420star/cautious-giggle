import os

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


class MT5Executor:
    def __init__(self, risk):
        self.risk = risk

    def _symbol_info(self, symbol):
        return mt5.symbol_info(symbol)

    def _symbol_tick(self, symbol):
        return mt5.symbol_info_tick(symbol)

    def get_tick(self, symbol):
        return self._symbol_tick(symbol)

    def get_mid_price(self, symbol):
        tick = self._symbol_tick(symbol)
        if tick is None:
            return None
        return float((tick.bid + tick.ask) / 2.0)

    def _symbol_magic_base(self, symbol: str) -> int:
        profile = {}
        try:
            profile = self.risk.get_symbol_profile(symbol) or {}
        except Exception:
            profile = {}
        if "magic_base" in profile:
            try:
                return int(profile.get("magic_base"))
            except Exception:
                pass
        if "magic" in profile:
            try:
                return int(profile.get("magic"))
            except Exception:
                pass
        return int(MAGIC_BY_SYMBOL.get(str(symbol), 59000))

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
        payload = {
            "action": str(request_action),
            "request_action": str(request_action),
            "side": str(meta.get("order_type") or request.get("type") or ""),
            "lots": float(request.get("volume", 0.0) or 0.0),
            "executed_lots": float(request.get("volume", 0.0) or 0.0),
            "target": float(meta.get("exposure", 0.0) or 0.0),
            "ppo": float(meta.get("ppo_target", 0.0) or 0.0),
            "dreamer": float(meta.get("dreamer_target", 0.0) or 0.0),
            "agi": float(meta.get("agi_bias", 0.0) or 0.0),
            "magic": request.get("magic"),
            "comment": request.get("comment"),
            "retcode": getattr(result, "retcode", None) if result is not None else None,
            "ticket": self._result_ticket(result),
        }
        logger.info(
            "ORDER_SEND {} | action={} side={} lots={:.2f} target={:.4f} ppo={:.4f} dreamer={:.4f} agi={:.4f} magic={} comment={} retcode={} ticket={}",
            symbol,
            payload["action"],
            payload["side"],
            payload["lots"],
            payload["target"],
            payload["ppo"],
            payload["dreamer"],
            payload["agi"],
            payload["magic"],
            payload["comment"],
            payload["retcode"],
            payload["ticket"],
        )
        return payload

    def _select_filling_mode(self, symbol):
        info = self._symbol_info(symbol)
        if info is None:
            return mt5.ORDER_FILLING_RETURN

        fm = int(getattr(info, "filling_mode", mt5.ORDER_FILLING_RETURN))

        # Some brokers expose bitmask-like values (e.g. 3 => FOK/IOC allowed).
        if fm == 3:
            return mt5.ORDER_FILLING_IOC

        if fm in (mt5.ORDER_FILLING_FOK, mt5.ORDER_FILLING_IOC, mt5.ORDER_FILLING_RETURN):
            return fm

        # Safe fallback for market execution when RETURN is unsupported.
        return mt5.ORDER_FILLING_IOC

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
            # Buy SL must stay below bid, TP must stay above ask.
            max_sl = bid - min_dist
            min_tp = ask + min_dist

            if new_sl is not None:
                if new_sl >= max_sl:
                    new_sl = max_sl
                new_sl = round(new_sl, digits)
                if new_sl <= 0:
                    new_sl = None

            if new_tp is not None:
                if new_tp <= min_tp:
                    new_tp = min_tp
                new_tp = round(new_tp, digits)
        else:
            # Sell SL must stay above ask, TP must stay below bid.
            min_sl = ask + min_dist
            max_tp = bid - min_dist

            if new_sl is not None:
                if new_sl <= min_sl:
                    new_sl = min_sl
                new_sl = round(new_sl, digits)

            if new_tp is not None:
                if new_tp >= max_tp:
                    new_tp = max_tp
                new_tp = round(new_tp, digits)
                if new_tp <= 0:
                    new_tp = None

        return new_sl, new_tp

    def get_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        longs = []
        shorts = []
        if positions:
            for p in positions:
                if p.type == mt5.ORDER_TYPE_BUY:
                    longs.append(p)
                else:
                    shorts.append(p)
        return longs, shorts

    # -- Discipline thresholds --------------------------------------------------
    # Only resize when the delta exceeds this fraction of current lots.
    # Prevents the old behaviour of closing+reopening on every tiny opinion shift.
    REBALANCE_THRESHOLD = 0.25  # 25 % change required to act
    MIN_PARTIAL_CLOSE_LOTS = 0.01

    def _floating_profit(self, positions) -> float:
        """Sum of unrealised profit across given position objects."""
        return sum(float(getattr(p, "profit", 0.0) or 0.0) for p in positions)

    def _partial_close(self, positions, reduce_lots, order_meta=None, execution_context=None):
        """Close the smallest positions first until *reduce_lots* are removed."""
        remaining = round(float(reduce_lots), 2)
        last_meta = {"request_action": "partial_close", "executed": False}
        # Close smallest first to preserve the larger (usually better) entries.
        for p in sorted(positions, key=lambda p: p.volume):
            if remaining < self.MIN_PARTIAL_CLOSE_LOTS:
                break
            close_vol = min(p.volume, remaining)
            tick = self._symbol_tick(p.symbol)
            if tick is None:
                self.risk.record_error()
                continue
            close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": round(close_vol, 2),
                "type": close_type,
                "position": p.ticket,
                "price": close_price,
                "deviation": 20,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(p.symbol),
            }
            request["magic"] = self._magic_for_order(p.symbol, order_meta, request_kind="close")
            request["comment"] = self._order_comment(p.symbol, order_meta, request_kind="close")
            result = mt5.order_send(request)
            last_meta = self._log_order_send(p.symbol, "partial_close", request, result, order_meta)
            last_meta["executed"] = bool(result is not None and result.retcode == mt5.TRADE_RETCODE_DONE)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()
            else:
                remaining = round(remaining - close_vol, 2)
                self.risk.record_trade(p.symbol)
        return last_meta

    def reconcile_exposure(self, symbol, target_exposure, max_lots, order_meta=None, execution_context=None):
        if not self.risk.can_trade(symbol):
            return {"request_action": "blocked", "executed": False}

        longs, shorts = self.get_positions(symbol)

        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)
        current_lots = long_lots - short_lots  # positive = net long
        current_side_is_long = current_lots >= 0

        target_lots = round(float(target_exposure) * float(max_lots), 2)
        result_meta = {
            "request_action": "noop",
            "executed": False,
            "target_lots": float(target_lots),
        }

        # ---- Flat target: close everything ---------------------------------
        if abs(target_lots) < 0.01:
            if long_lots > 0:
                result_meta = self.close_positions(longs, order_meta=order_meta, execution_context=execution_context)
                if result_meta.get("executed"):
                    self.risk.record_trade(symbol)
            if short_lots > 0:
                result_meta = self.close_positions(shorts, order_meta=order_meta, execution_context=execution_context)
                if result_meta.get("executed"):
                    self.risk.record_trade(symbol)
            return result_meta

        target_is_long = target_lots > 0

        # ---- Side reversal: close opposite then open new -------------------
        if target_is_long and short_lots > 0:
            result_meta = self.close_positions(shorts, order_meta=order_meta, execution_context=execution_context)
            if result_meta.get("executed"):
                self.risk.record_trade(symbol)
            short_lots = 0.0
        elif not target_is_long and long_lots > 0:
            result_meta = self.close_positions(longs, order_meta=order_meta, execution_context=execution_context)
            if result_meta.get("executed"):
                self.risk.record_trade(symbol)
            long_lots = 0.0

        # ---- Same-side logic -----------------------------------------------
        if target_is_long:
            same_lots = long_lots
        else:
            same_lots = short_lots

        abs_target = abs(target_lots)
        delta = abs_target - same_lots

        # Rule 1: same side, profitable, target only smaller -> hold
        if delta < 0 and same_lots > 0:
            same_positions = longs if target_is_long else shorts
            floating = self._floating_profit(same_positions)
            if floating >= 0 and abs(delta) / max(same_lots, 0.01) < self.REBALANCE_THRESHOLD:
                result_meta["request_action"] = "hold"
                return result_meta

        # Rule 2: same side, need to reduce -> partial close (not close-all + reopen)
        if delta < -self.MIN_PARTIAL_CLOSE_LOTS:
            reduce = round(abs(delta), 2)
            # Only act if the reduction is significant enough
            if same_lots > 0 and reduce / same_lots >= self.REBALANCE_THRESHOLD:
                same_positions = longs if target_is_long else shorts
                result_meta = self._partial_close(
                    same_positions, reduce, order_meta=order_meta, execution_context=execution_context
                )
            else:
                result_meta["request_action"] = "hold"
            return result_meta

        # Rule 3: same side, need to add -> open additional
        if delta >= 0.01:
            order_type = mt5.ORDER_TYPE_BUY if target_is_long else mt5.ORDER_TYPE_SELL
            result_meta = self.open_position(
                symbol,
                order_type,
                round(delta, 2),
                order_meta=order_meta,
                execution_context=execution_context,
            )
            if result_meta.get("executed"):
                self.risk.record_trade(symbol)
        else:
            result_meta["request_action"] = "hold"

        return result_meta

    def close_positions(self, positions, order_meta=None, execution_context=None):
        last_meta = {"request_action": "close", "executed": False}
        for p in positions:
            tick = self._symbol_tick(p.symbol)
            if tick is None:
                self.risk.record_error()
                continue

            close_type = mt5.ORDER_TYPE_SELL if p.type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_BUY
            close_price = tick.bid if close_type == mt5.ORDER_TYPE_SELL else tick.ask

            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": close_type,
                "position": p.ticket,
                "price": close_price,
                "deviation": 20,
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": self._select_filling_mode(p.symbol),
            }
            request["magic"] = self._magic_for_order(p.symbol, order_meta, request_kind="close")
            request["comment"] = self._order_comment(p.symbol, order_meta, request_kind="close")
            result = mt5.order_send(request)
            last_meta = self._log_order_send(p.symbol, "close", request, result, order_meta)
            last_meta["executed"] = bool(result is not None and result.retcode == mt5.TRADE_RETCODE_DONE)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()
        return last_meta

    def _atr_points(self, symbol, bars=120, period=14):
        rates = mt5.copy_rates_from_pos(symbol, mt5.TIMEFRAME_M5, 0, bars)
        info = self._symbol_info(symbol)
        if rates is None or len(rates) < period + 2 or info is None:
            return None

        point = float(info.point) if info.point else 0.0001
        highs = [float(r[2]) for r in rates]
        lows = [float(r[3]) for r in rates]
        closes = [float(r[4]) for r in rates]

        trs = []
        for i in range(1, len(rates)):
            tr = max(
                highs[i] - lows[i],
                abs(highs[i] - closes[i - 1]),
                abs(lows[i] - closes[i - 1]),
            )
            trs.append(tr)

        if len(trs) < period:
            return None

        atr = sum(trs[-period:]) / float(period)
        atr_points = int(max(1, round(atr / max(point, 1e-12))))
        return atr_points

    def _dynamic_points(self, symbol):
        profile = self.risk.get_symbol_profile(symbol)
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
        if info is None or tick is None:
            return None, None, 20

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        _, min_pts = self._min_stop_distance(symbol)
        deviation = int(self.risk.get_symbol_profile(symbol).get("entry_deviation", 20))

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        sl_points = max(int(dyn_sl), min_pts)
        tp_points = max(int(dyn_tp), min_pts)

        sl_dist = sl_points * point
        tp_dist = tp_points * point

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
            self.risk.record_error()
            return {"request_action": "open", "executed": False}

        price = tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid
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
        }
        request["magic"] = self._magic_for_order(symbol, order_meta, request_kind="open")
        request["comment"] = self._order_comment(symbol, order_meta, request_kind="open")

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)
        meta = self._log_order_send(symbol, "open", request, result, order_meta)
        meta["executed"] = bool(result is not None and result.retcode == mt5.TRADE_RETCODE_DONE)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.risk.record_error()
        return meta

    def manage_open_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        info = self._symbol_info(symbol)
        tick = self._symbol_tick(symbol)
        if not positions or info is None or tick is None:
            return

        point = float(info.point) if info.point else 0.0001
        digits = int(info.digits) if info.digits is not None else 5
        profile = self.risk.get_symbol_profile(symbol)

        dyn_sl, dyn_tp = self._dynamic_points(symbol)
        breakeven_trigger = int(profile.get("breakeven_points", max(25, dyn_sl // 3)))
        trailing_trigger = int(profile.get("trailing_trigger_points", max(40, dyn_sl // 2)))
        trailing_step = int(profile.get("trailing_step_points", max(10, dyn_sl // 8)))

        for p in positions:
            current_price = tick.bid if p.type == mt5.ORDER_TYPE_BUY else tick.ask
            profit_points = (
                (current_price - p.price_open) / point
                if p.type == mt5.ORDER_TYPE_BUY
                else (p.price_open - current_price) / point
            )

            new_sl = float(p.sl) if p.sl else None
            new_tp = float(p.tp) if p.tp else None

            # Add TP if missing
            if new_tp is None:
                tp_dist = dyn_tp * point
                new_tp = (
                    p.price_open + tp_dist
                    if p.type == mt5.ORDER_TYPE_BUY
                    else p.price_open - tp_dist
                )

            # Break-even promotion
            if profit_points >= breakeven_trigger:
                be_buffer = 5 * point
                be_sl = (
                    p.price_open + be_buffer
                    if p.type == mt5.ORDER_TYPE_BUY
                    else p.price_open - be_buffer
                )
                if (p.type == mt5.ORDER_TYPE_BUY and (new_sl is None or be_sl > new_sl)) or (
                    p.type == mt5.ORDER_TYPE_SELL and (new_sl is None or be_sl < new_sl)
                ):
                    new_sl = be_sl

            # Trailing after trigger
            if profit_points >= trailing_trigger:
                trail_dist = trailing_step * point
                trail_sl = (
                    current_price - trail_dist
                    if p.type == mt5.ORDER_TYPE_BUY
                    else current_price + trail_dist
                )
                if (p.type == mt5.ORDER_TYPE_BUY and (new_sl is None or trail_sl > new_sl)) or (
                    p.type == mt5.ORDER_TYPE_SELL and (new_sl is None or trail_sl < new_sl)
                ):
                    new_sl = trail_sl

            new_sl, new_tp = self._sanitize_sl_tp(symbol, p.type, new_sl, new_tp, tick)

            if new_sl is not None:
                new_sl = round(new_sl, digits)
            if new_tp is not None:
                new_tp = round(new_tp, digits)

            sl_changed = (new_sl is not None) and (p.sl is None or abs(float(new_sl) - float(p.sl)) > (0.5 * point))
            tp_changed = (new_tp is not None) and (p.tp is None or abs(float(new_tp) - float(p.tp)) > (0.5 * point))

            if not sl_changed and not tp_changed:
                continue

            req = {
                "action": mt5.TRADE_ACTION_SLTP,
                "symbol": symbol,
                "position": p.ticket,
            }
            if new_sl is not None:
                req["sl"] = new_sl
            if new_tp is not None:
                req["tp"] = new_tp
            req["magic"] = self._magic_for_order(symbol, None, request_kind="manage")
            req["comment"] = self._order_comment(symbol, None, request_kind="manage")

            result = mt5.order_send(req)
            self._log_order_send(symbol, "manage", req, result, {"symbol": symbol})
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

