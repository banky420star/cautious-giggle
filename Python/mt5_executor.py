import MetaTrader5 as mt5
import time

class MT5Executor:
    def __init__(self, risk):
        self.risk = risk

    def get_positions(self, symbol):
        positions = mt5.positions_get(symbol=symbol)
        longs = []
        shorts = []
        if positions:
            for p in positions:
                if p.type == 0:
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
        target_lots = round(target_exposure * max_lots, 2)
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
            request = {
                "action": mt5.TRADE_ACTION_DEAL,
                "symbol": p.symbol,
                "volume": p.volume,
                "type": mt5.ORDER_TYPE_SELL if p.type == 0 else mt5.ORDER_TYPE_BUY,
                "position": p.ticket
            }
            result = mt5.order_send(request)
            if result.retcode != mt5.TRADE_RETCODE_DONE:
                self.risk.record_error()

    def open_position(self, symbol, order_type, volume):
        request = {
            "action": mt5.TRADE_ACTION_DEAL,
            "symbol": symbol,
            "volume": volume,
            "type": order_type,
            "price": mt5.symbol_info_tick(symbol).ask if order_type == mt5.ORDER_TYPE_BUY else mt5.symbol_info_tick(symbol).bid,
        }
        result = mt5.order_send(request)
        if result.retcode != mt5.TRADE_RETCODE_DONE:
            self.risk.record_error()
