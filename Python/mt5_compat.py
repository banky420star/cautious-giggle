from __future__ import annotations

try:
    import MetaTrader5 as mt5
    MT5_AVAILABLE = True
    MT5_IMPORT_ERROR = None
except Exception as exc:
    MT5_AVAILABLE = False
    MT5_IMPORT_ERROR = exc

    class _MissingMetaTrader5:
        ORDER_TYPE_BUY = 0
        ORDER_TYPE_SELL = 1
        ORDER_FILLING_FOK = 0
        ORDER_FILLING_IOC = 1
        ORDER_FILLING_RETURN = 2
        ORDER_TIME_GTC = 0
        TRADE_ACTION_DEAL = 1
        TRADE_ACTION_SLTP = 6
        TRADE_RETCODE_DONE = 10009
        TIMEFRAME_M1 = 1
        TIMEFRAME_M5 = 5
        TIMEFRAME_M15 = 15
        TIMEFRAME_M30 = 30
        TIMEFRAME_H1 = 60
        TIMEFRAME_H4 = 240
        TIMEFRAME_D1 = 1440

        class Tick:
            pass

        def initialize(self, *args, **kwargs):
            return False

        def last_error(self):
            return str(MT5_IMPORT_ERROR)

        def symbol_info(self, *args, **kwargs):
            return None

        def symbol_info_tick(self, *args, **kwargs):
            return None

        def symbol_select(self, *args, **kwargs):
            return False

        def copy_rates_from_pos(self, *args, **kwargs):
            return None

        def copy_rates_from(self, *args, **kwargs):
            return None

        def copy_rates_range(self, *args, **kwargs):
            return None

        def positions_get(self, *args, **kwargs):
            return []

        def order_send(self, *args, **kwargs):
            raise RuntimeError(
                "MetaTrader5 is required for live trading operations and is unavailable in this environment."
            ) from MT5_IMPORT_ERROR

        def __getattr__(self, name):
            raise RuntimeError(
                "MetaTrader5 is required for live trading operations and is unavailable in this environment."
            ) from MT5_IMPORT_ERROR

    mt5 = _MissingMetaTrader5()
