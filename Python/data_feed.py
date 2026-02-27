import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

# Optional MT5 support
try:
    import MetaTrader5 as mt5
except ImportError:
    mt5 = None

# Yahoo Finance ticker mapping for forex/commodities
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",       # futures proxy
    "BTCUSD": "BTC-USD",

    # Common broker suffix variants -> map to yfinance
    "EURUSDm": "EURUSD=X",
    "GBPUSDm": "GBPUSD=X",
    "USDJPYm": "USDJPY=X",
    "XAUUSDm": "GC=F",
    "EURUSDM": "EURUSD=X",
    "GBPUSDM": "GBPUSD=X",
    "USDJPYM": "USDJPY=X",
    "XAUUSDM": "GC=F",
}

REQUIRED_COLS = ["open", "high", "low", "close", "volume"]

def _flatten_cols(df: pd.DataFrame) -> pd.DataFrame:
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df

def _standardize(df: pd.DataFrame, symbol: str) -> pd.DataFrame:
    df = _flatten_cols(df).copy()

    df = df.rename(columns={
        "Open": "open", "High": "high", "Low": "low",
        "Close": "close", "Adj Close": "close",
        "Volume": "volume"
    })

    # Ensure required cols exist
    for c in REQUIRED_COLS:
        if c not in df.columns:
            df[c] = np.nan

    # Clean index
    df = df[REQUIRED_COLS].copy()
    df = df[~df.index.duplicated(keep="last")]
    df = df.sort_index()
    df = df.dropna(subset=["open", "high", "low", "close"])

    # Volume fix: yfinance FX volume is often 0/NaN.
    # Create a proxy volume based on true range if missing/zero.
    vol = df["volume"].copy()
    vol_missing = vol.isna() | (vol <= 0)

    if vol_missing.all():
        # Proxy: scaled True Range (not perfect, but at least informative)
        tr = (df["high"] - df["low"]).abs()
        proxy = (tr / (df["close"].abs() + 1e-12)) * 1e6
        df["volume"] = proxy.clip(lower=0).fillna(0.0)
        logger.warning(f"{symbol}: yfinance volume unusable -> using TR-based proxy volume.")
    else:
        # Partial missing: fill missing with median of non-zero
        med = float(vol[~vol_missing].median()) if (~vol_missing).any() else 0.0
        df.loc[vol_missing, "volume"] = med
        df["volume"] = df["volume"].fillna(0.0)

    # Final safety
    df["symbol"] = symbol
    df = df[["open", "high", "low", "close", "volume", "symbol"]]

    return df

def _download(ticker: str, period: str, interval: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)
    if df is None or df.empty:
        return pd.DataFrame()
    # Immediate clean-up of yfinance artifacts
    try:
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.loc[:, ~df.columns.duplicated(keep="last")]
        df = df[~df.index.duplicated(keep="last")].sort_index()
    except Exception as e:
        logger.warning(f"Quick clean failed: {e}")
    return df

def fetch_realtime(symbol: str = "EURUSD", period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    ticker = SYMBOL_MAP.get(symbol, f"{symbol}=X")
    logger.info(f"Fetching real-time data: {ticker} | period={period} | interval={interval}")

    try:
        df = _download(ticker, period=period, interval=interval)
        if df is None or df.empty:
            logger.warning(f"No data returned for {ticker}, falling back to daily")
            df = _download(ticker, period="60d", interval="1d")
        if df is None or df.empty:
            raise RuntimeError("No data after fallbacks")

        df = _standardize(df, symbol)

        logger.success(
            f"Fetched {len(df)} candles for {symbol} | "
            f"Range: {df.index[0]} → {df.index[-1]} | "
            f"Latest close: {df['close'].iloc[-1]:.5f}"
        )
        return df

    except Exception as e:
        logger.error(f"Realtime data fetch failed for {symbol}: {e}")
        logger.warning("Using synthetic fallback data (do NOT trust for trading).")
        idx = pd.date_range(end=datetime.now(), periods=500, freq=interval)
        data = np.random.rand(len(idx), 5).astype(np.float64)
        df = pd.DataFrame(data, columns=["open", "high", "low", "close", "volume"], index=idx)
        df["symbol"] = symbol
        return df

def fetch_training_data(symbol: str = "EURUSD", period: str = "60d") -> pd.DataFrame:
    """
    Training data fetch:
    - Prefer 1h bars (more samples for RL/LSTM)
    - Fall back to 1d if needed
    """
    ticker = SYMBOL_MAP.get(symbol, f"{symbol}=X")
    logger.info(f"Fetching training data: {ticker} | period={period}")

    try:
        df = _download(ticker, period=period, interval="1h")
        if df is None or df.empty:
            df = _download(ticker, period=period, interval="1d")
        if df is None or df.empty:
            return pd.DataFrame()

        df = _standardize(df, symbol)

        # Minimum length guardrails (helps avoid training on junk)
        if len(df) < 250:
            logger.warning(f"{symbol}: training dataset small ({len(df)} rows). Consider longer period.")
        logger.success(f"Training data: {len(df)} candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Training data fetch failed for {symbol}: {e}")
        return pd.DataFrame()

def fetch_ohlc_history(symbol: str, timeframe: str = "1h", count: int = 500) -> pd.DataFrame:
    """
    Fetches OHLC history from MT5 if available and initialized, otherwise falls back to yfinance.
    Priority: MT5 (live broker truth) > yfinance (public fallback).
    """
    if mt5 is not None and mt5.initialize():
        try:
            mt5_tf_map = {
                "1m": mt5.TIMEFRAME_M1, "5m": mt5.TIMEFRAME_M5, "15m": mt5.TIMEFRAME_M15,
                "30m": mt5.TIMEFRAME_M30, "1h": mt5.TIMEFRAME_H1, "4h": mt5.TIMEFRAME_H4,
                "1d": mt5.TIMEFRAME_D1
            }
            tf = mt5_tf_map.get(timeframe, mt5.TIMEFRAME_H1)
            
            # Fetch from MT5
            rates = mt5.copy_rates_from_pos(symbol, tf, 0, count)
            if rates is not None and len(rates) > 0:
                df = pd.DataFrame(rates)
                df['time'] = pd.to_datetime(df['time'], unit='s')
                df.set_index('time', inplace=True)
                df = df.rename(columns={
                    "real_volume": "volume", "tick_volume": "volume_tick"
                })
                # Use tick volume as proxy if real volume is 0
                if 'volume' in df.columns and (df['volume'] == 0).all():
                    df['volume'] = df['volume_tick']
                
                logger.info(f"{symbol}: Fetched {len(df)} candles from MT5 history.")
                return _standardize(df, symbol)
        except Exception as e:
            logger.warning(f"MT5 history fetch failed for {symbol}: {e}. Falling back to yfinance.")

    # Fallback to yfinance
    pd_period_map = {"1h": "60d", "1d": "2y", "5m": "5d"}
    period = pd_period_map.get(timeframe, "60d")
    return fetch_training_data(symbol, period=period)

def get_combined_training_df(symbols: list[str], period: str = "60d") -> pd.DataFrame:
    """
    Fetches training data for multiple symbols and concatenates them.
    Useful for training one policy on multiple market regimes.
    """
    all_dfs = []
    for sym in symbols:
        df_pd = fetch_training_data(sym, period=period)
        if not df_pd.empty and len(df_pd) > 200:
            all_dfs.append(df_pd)
    
    if not all_dfs:
        return pd.DataFrame()
    
    # Simple stack
    return pd.concat(all_dfs, axis=0).sort_index()

if __name__ == "__main__":
    for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
        df = fetch_realtime(sym)
        print(f"\n{sym}: {len(df)} candles")
        if not df.empty:
            print(df.tail(3))
        print("---")
