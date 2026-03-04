import MetaTrader5 as mt5


class MT5Executor:
    def __init__(self, risk):
        self.risk = risk

    def _symbol_info(self, symbol):
        return mt5.symbol_info(symbol)

    def _symbol_tick(self, symbol):
        return mt5.symbol_info_tick(symbol)

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

    def reconcile_exposure(self, symbol, target_exposure, max_lots):
        if not self.risk.can_trade():
            return

        longs, shorts = self.get_positions(symbol)

        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)

        net_lots = long_lots - short_lots
        target_lots = round(float(target_exposure) * float(max_lots), 2)
        delta = target_lots - net_lots

        if abs(delta) < 0.01:
            return

        if delta > 0:
            if short_lots > 0:
                self.close_positions(shorts)
            self.open_position(symbol, mt5.ORDER_TYPE_BUY, abs(delta))
        else:
            if long_lots > 0:
                self.close_positions(longs)
            self.open_position(symbol, mt5.ORDER_TYPE_SELL, abs(delta))

        self.risk.record_trade()

    def close_positions(self, positions):
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
            result = mt5.order_send(request)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

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

    def open_position(self, symbol, order_type, volume):
        tick = self._symbol_tick(symbol)
        if tick is None:
            self.risk.record_error()
            return

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

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        result = mt5.order_send(request)
        if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
            self.risk.record_error()

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

            result = mt5.order_send(req)
            if result is None or result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

