import MetaTrader5 as mt5
from loguru import logger


class MT5Executor:
    def __init__(self, risk):
        self.risk = risk

    def get_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        longs = []
        shorts = []
        if positions:
            for p in positions:
                if p.type == mt5.POSITION_TYPE_BUY:
                    longs.append(p)
                else:
                    shorts.append(p)
        return longs, shorts

    def reconcile_exposure(
        self,
        symbol,
        target_exposure,
        max_lots,
        sl_points: float | None = None,
        tp_points: float | None = None,
        use_limit_orders: bool = False,
        limit_offset_points: float = 0.0,
    ):
        if not self.risk.can_trade():
            return

        longs, shorts = self.get_positions(symbol)

        long_lots = sum(p.volume for p in longs)
        short_lots = sum(p.volume for p in shorts)

        net_lots = long_lots - short_lots
        target_lots = round(float(target_exposure) * float(max_lots), 2)
        delta = round(target_lots - net_lots, 2)

        if abs(delta) < 0.01:
            return

        if delta > 0:
            if short_lots > 0:
                self.close_positions(shorts)
            self.open_position(
                symbol,
                mt5.ORDER_TYPE_BUY,
                abs(delta),
                sl_points=sl_points,
                tp_points=tp_points,
                use_limit_order=use_limit_orders,
                limit_offset_points=limit_offset_points,
            )
        else:
            if long_lots > 0:
                self.close_positions(longs)
            self.open_position(
                symbol,
                mt5.ORDER_TYPE_SELL,
                abs(delta),
                sl_points=sl_points,
                tp_points=tp_points,
                use_limit_order=use_limit_orders,
                limit_offset_points=limit_offset_points,
            )

        self.risk.record_trade()

    def close_positions(self, positions):
        for p in positions:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == mt5.POSITION_TYPE_BUY else mt5.ORDER_TYPE_BUY,
                "position": p.ticket,
            }
            self._send(request, context=f"close ticket={p.ticket}")

    def _symbol_point(self, symbol: str) -> float:
        info = mt5.symbol_info(symbol)
        return float(info.point) if info and info.point else 0.0001

    def _entry_price(self, symbol: str, order_type: int) -> float | None:
        tick = mt5.symbol_info_tick(symbol)
        if not tick:
            return None
        return float(tick.ask if order_type == mt5.ORDER_TYPE_BUY else tick.bid)

    def _build_sl_tp(self, symbol: str, order_type: int, price: float, sl_points, tp_points):
        point = self._symbol_point(symbol)
        sl = None
        tp = None
        if sl_points and sl_points > 0:
            sl = price - (sl_points * point) if order_type == mt5.ORDER_TYPE_BUY else price + (sl_points * point)
        if tp_points and tp_points > 0:
            tp = price + (tp_points * point) if order_type == mt5.ORDER_TYPE_BUY else price - (tp_points * point)
        return sl, tp

    def open_position(
        self,
        symbol,
        order_type,
        volume,
        sl_points: float | None = None,
        tp_points: float | None = None,
        use_limit_order: bool = False,
        limit_offset_points: float = 0.0,
    ):
        price = self._entry_price(symbol, order_type)
        if price is None:
            self.risk.record_error()
            logger.error(f"No tick data available for {symbol}")
            return

        sl, tp = self._build_sl_tp(symbol, order_type, price, sl_points, tp_points)

        if use_limit_order and limit_offset_points > 0:
            point = self._symbol_point(symbol)
            request = {
                "action": mt5.TRADE_ACTION_PENDING,
                "symbol": symbol,
                "volume": volume,
                "type": mt5.ORDER_TYPE_BUY_LIMIT if order_type == mt5.ORDER_TYPE_BUY else mt5.ORDER_TYPE_SELL_LIMIT,
                "price": price - (limit_offset_points * point) if order_type == mt5.ORDER_TYPE_BUY else price + (limit_offset_points * point),
                "type_time": mt5.ORDER_TIME_GTC,
                "type_filling": mt5.ORDER_FILLING_RETURN,
            }
        else:
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": symbol,
                "volume": volume,
                "type": order_type,
                "price": price,
                "type_filling": mt5.ORDER_FILLING_IOC,
            }

        if sl is not None:
            request["sl"] = sl
        if tp is not None:
            request["tp"] = tp

        self._send(request, context=f"open {symbol}")

    def _send(self, request: dict, context: str = "order_send"):
        result = mt5.order_send(request)
        ok_codes = {mt5.TRADE_RETCODE_DONE, mt5.TRADE_RETCODE_PLACED}
        if result is None or getattr(result, "retcode", None) not in ok_codes:
            self.risk.record_error()
            code = getattr(result, "retcode", None)
            logger.error(f"{context} failed. retcode={code} request={request}")
        else:
            logger.info(f"{context} success: ticket={getattr(result, 'order', None)} retcode={getattr(result, 'retcode', None)}")
