import os
from datetime import datetime

import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
from loguru import logger


_YF_INTERVAL_BY_TIMEFRAME = {
    "M1": "1m",
    "M5": "5m",
    "M15": "15m",
    "M30": "30m",
    "H1": "60m",
    "H4": "60m",
    "D1": "1d",
}


def _normalize_rates_df(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame()

    out = df.copy()
    rename_map = {
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
        "tick_volume": "volume",
    }
    out = out.rename(columns=rename_map)

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], unit="s", errors="coerce")
    elif isinstance(out.index, pd.DatetimeIndex):
        out = out.reset_index().rename(columns={out.index.name or "index": "time"})
        out["time"] = pd.to_datetime(out["time"], errors="coerce")
    else:
        out["time"] = pd.to_datetime(datetime.utcnow())

    keep = [c for c in ["time", "open", "high", "low", "close", "volume"] if c in out.columns]
    out = out[keep]

    for col in ["open", "high", "low", "close", "volume"]:
        if col not in out.columns:
            out[col] = 0.0

    out["symbol"] = symbol
    out = out.dropna(subset=["open", "high", "low", "close"]).sort_values("time")
    out["volume"] = out["volume"].fillna(0.0)
    return out.reset_index(drop=True)


def _fetch_from_mt5(symbol: str, timeframe: str = "M5", bars: int = 5000) -> pd.DataFrame:
    if not mt5.initialize():
        return pd.DataFrame()

    tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}", mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
    if rates is None:
        return pd.DataFrame()

    return _normalize_rates_df(pd.DataFrame(rates), symbol)


def _fetch_from_yfinance(symbol: str, period: str = "60d", timeframe: str = "M5") -> pd.DataFrame:
    ticker = symbol.replace("m", "")
    interval = _YF_INTERVAL_BY_TIMEFRAME.get(timeframe, "5m")

    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    return _normalize_rates_df(df, symbol)


def fetch_training_data(symbol: str, period: str = "60d", timeframe: str = "M5") -> pd.DataFrame:
    mt5_df = _fetch_from_mt5(symbol, timeframe=timeframe, bars=6000)
    if not mt5_df.empty:
        return mt5_df

    logger.warning(f"MT5 data unavailable for {symbol}. Falling back to yfinance.")
    yf_df = _fetch_from_yfinance(symbol, period=period, timeframe=timeframe)
    if yf_df.empty:
        logger.error(f"No training data available for {symbol} from MT5 or yfinance.")
    return yf_df


def get_combined_training_df(symbols: list[str], period: str = "60d", timeframe: str = "M5") -> pd.DataFrame:
    frames = []
    for sym in symbols:
        df = fetch_training_data(sym, period=period, timeframe=timeframe)
        if df is None or df.empty:
            continue
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, ignore_index=True)
    combined = combined.sort_values("time").reset_index(drop=True)
    return combined


def get_latest_data(symbol, timeframe, bars):
    tf_const = getattr(mt5, f"TIMEFRAME_{timeframe}", mt5.TIMEFRAME_M5)
    rates = mt5.copy_rates_from_pos(symbol, tf_const, 0, bars)
    if rates is None:
        if os.environ.get("AGI_IS_LIVE") == "1":
            raise Exception("Live data feed failure")
        return None
    return rates
