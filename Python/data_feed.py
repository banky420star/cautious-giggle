import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime

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
    return yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=False)

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

if __name__ == "__main__":
    for sym in ["EURUSDm", "GBPUSDm", "XAUUSDm"]:
        df = fetch_realtime(sym)
        print(f"\n{sym}: {len(df)} candles")
        print(df.tail(3))
        print("---")
