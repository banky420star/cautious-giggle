import yfinance as yf
import pandas as pd
import numpy as np
from loguru import logger
from datetime import datetime, timedelta

# Yahoo Finance ticker mapping for forex/commodities
SYMBOL_MAP = {
    "EURUSD": "EURUSD=X",
    "GBPUSD": "GBPUSD=X",
    "USDJPY": "USDJPY=X",
    "XAUUSD": "GC=F",       # Gold futures (closest proxy)
    "BTCUSD": "BTC-USD",
    
    # MetaTrader 5 Broker Suffix Support
    "EURUSDm": "EURUSD=X",
    "GBPUSDm": "GBPUSD=X",
    "USDJPYm": "USDJPY=X",
    "XAUUSDm": "GC=F",
    "EURUSDM": "EURUSD=X",
    "GBPUSDM": "GBPUSD=X",
    "USDJPYM": "USDJPY=X",
    "XAUUSDM": "GC=F",
}

def fetch_realtime(symbol: str = "EURUSD", period: str = "5d", interval: str = "5m") -> pd.DataFrame:
    """Fetch real market data from Yahoo Finance."""
    ticker = SYMBOL_MAP.get(symbol, f"{symbol}=X")
    logger.info(f"Fetching real data: {ticker} | period={period} | interval={interval}")

    try:
        df = yf.download(ticker, period=period, interval=interval, progress=False)
        if df.empty:
            logger.warning(f"No data returned for {ticker}, falling back to daily")
            df = yf.download(ticker, period="60d", interval="1d", progress=False)

        # Flatten multi-level columns if present
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })

        # Ensure we have the right columns
        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0

        df['symbol'] = symbol
        df = df[['open', 'high', 'low', 'close', 'volume', 'symbol']].dropna()

        logger.success(f"Fetched {len(df)} candles for {symbol} | "
                      f"Range: {df.index[0]} → {df.index[-1]} | "
                      f"Latest close: {df['close'].iloc[-1]:.5f}")
        return df

    except Exception as e:
        logger.error(f"Data fetch failed for {symbol}: {e}")
        # Fallback to synthetic data (never crash)
        logger.warning("Using synthetic fallback data")
        idx = pd.date_range(end=datetime.now(), periods=200, freq="5min")
        df = pd.DataFrame(np.random.rand(200, 5) * 100, columns=['open','high','low','close','volume'], index=idx)
        df['symbol'] = symbol
        return df

def fetch_training_data(symbol: str = "EURUSD", period: str = "60d") -> pd.DataFrame:
    """Fetch longer-term data for LSTM training."""
    ticker = SYMBOL_MAP.get(symbol, f"{symbol}=X")
    logger.info(f"Fetching training data: {ticker} | period={period}")

    try:
        # For training, get daily data over a longer period
        df = yf.download(ticker, period=period, interval="1h", progress=False)
        if df.empty:
            df = yf.download(ticker, period=period, interval="1d", progress=False)

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)

        df = df.rename(columns={
            "Open": "open", "High": "high", "Low": "low",
            "Close": "close", "Volume": "volume"
        })

        for col in ['open', 'high', 'low', 'close', 'volume']:
            if col not in df.columns:
                df[col] = 0.0

        df['symbol'] = symbol
        df = df[['open', 'high', 'low', 'close', 'volume', 'symbol']].dropna()

        logger.success(f"Training data: {len(df)} candles for {symbol}")
        return df

    except Exception as e:
        logger.error(f"Training data fetch failed: {e}")
        return pd.DataFrame()


if __name__ == "__main__":
    for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
        df = fetch_realtime(sym)
        print(f"\n{sym}: {len(df)} candles")
        print(df.tail(3))
        print("---")
