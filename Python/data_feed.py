import MetaTrader5 as mt5
import os

def get_latest_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        if os.environ.get("AGI_IS_LIVE") == "1":
            raise Exception("Live data feed failure")
        return None
    return rates
