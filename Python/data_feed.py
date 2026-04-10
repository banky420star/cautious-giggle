"""
Data Feed — High-fidelity market data with MT5 (Windows) and Yahoo Finance (Mac/Linux) fallback.

Provides:
  - fetch_training_data(symbol, period)  → pd.DataFrame with [open, high, low, close, volume]
  - get_combined_training_df(symbols, period) → single concatenated pd.DataFrame
  - get_latest_data(symbol, timeframe, bars)  → np.ndarray or None
"""
import os
import sys
import numpy as np
import pandas as pd
from loguru import logger

# ── MT5 conditional import (only available on Windows) ──────────────
_mt5 = None
if sys.platform == "win32":
    try:
        import MetaTrader5 as mt5
        _mt5 = mt5
    except ImportError:
        logger.warning("MetaTrader5 package not installed — using Yahoo Finance fallback.")

# ── Yahoo Finance import ────────────────────────────────────────────
try:
    import yfinance as yf
except ImportError:
    yf = None
    logger.warning("yfinance not installed — pip install yfinance")

# ── Symbol mapping: broker names → Yahoo Finance tickers ────────────
_YF_MAP = {
    # FX pairs
    "EURUSD":   "EURUSD=X",
    "EURUSDm":  "EURUSD=X",
    "GBPUSD":   "GBPUSD=X",
    "GBPUSDm":  "GBPUSD=X",
    "USDJPY":   "USDJPY=X",
    "USDJPYm":  "USDJPY=X",
    "AUDUSD":   "AUDUSD=X",
    "AUDUSDm":  "AUDUSD=X",
    "USDCAD":   "USDCAD=X",
    "USDCADm":  "USDCAD=X",
    "USDCHF":   "USDCHF=X",
    "USDCHFm":  "USDCHF=X",
    "NZDUSD":   "NZDUSD=X",
    "NZDUSDm":  "NZDUSD=X",
    "GBPJPY":   "GBPJPY=X",
    "GBPJPYm":  "GBPJPY=X",
    "EURJPY":   "EURJPY=X",
    "EURJPYm":  "EURJPY=X",
    # Commodities / Metals
    "XAUUSD":   "GC=F",
    "XAUUSDm":  "GC=F",
    "XAGUSD":   "SI=F",
    "XAGUSDm":  "SI=F",
    # Indices
    "US30":     "YM=F",
    "US500":    "ES=F",
    "NAS100":   "NQ=F",
    # Crypto
    "BTCUSD":   "BTC-USD",
    "ETHUSD":   "ETH-USD",
}

# ── MT5 timeframe mapping ──────────────────────────────────────────
_MT5_TIMEFRAMES = {}
if _mt5:
    _MT5_TIMEFRAMES = {
        "M1":  _mt5.TIMEFRAME_M1,
        "M5":  _mt5.TIMEFRAME_M5,
        "M15": _mt5.TIMEFRAME_M15,
        "M30": _mt5.TIMEFRAME_M30,
        "H1":  _mt5.TIMEFRAME_H1,
        "H4":  _mt5.TIMEFRAME_H4,
        "D1":  _mt5.TIMEFRAME_D1,
    }


def _yf_ticker(symbol: str) -> str:
    """Resolve a broker symbol to a Yahoo Finance ticker."""
    clean = symbol.strip()
    if clean in _YF_MAP:
        return _YF_MAP[clean]
    # If it already looks like a YF ticker, pass through
    if "=" in clean or "-" in clean or "." in clean:
        return clean
    # Last resort: try appending =X for FX
    return clean + "=X"


def _synthesize_fx_volume(df: pd.DataFrame) -> pd.DataFrame:
    """
    Yahoo Finance returns 0 volume for FX pairs.
    Synthesize a volume proxy from price volatility (High-Low range * 1e6).
    """
    if "volume" in df.columns and (df["volume"] == 0).all():
        df["volume"] = ((df["high"] - df["low"]) * 1e6).clip(lower=100).astype(float)
        logger.debug("Synthesized FX volume proxy from H-L range.")
    return df


def _fetch_via_mt5(symbol: str, timeframe: str = "M5", bars: int = 5000) -> pd.DataFrame | None:
    """Try fetching data via MT5 (Windows only)."""
    if _mt5 is None:
        return None
    try:
        if not _mt5.initialize():
            logger.warning("MT5 initialize() failed.")
            return None
        
        tf = _MT5_TIMEFRAMES.get(timeframe, _mt5.TIMEFRAME_M5)
        rates = _mt5.copy_rates_from_pos(symbol, tf, 0, bars)
        
        if rates is None or len(rates) == 0:
            logger.warning(f"MT5 returned no data for {symbol}")
            return None
        
        df = pd.DataFrame(rates)
        df["time"] = pd.to_datetime(df["time"], unit="s")
        df = df.rename(columns={"tick_volume": "volume"})
        
        # Ensure standard column names
        for col in ["open", "high", "low", "close", "volume"]:
            if col not in df.columns:
                logger.error(f"MT5 data missing column: {col}")
                return None
        
        df = df[["time", "open", "high", "low", "close", "volume"]].copy()
        df = df.sort_values("time").reset_index(drop=True)
        logger.success(f"MT5: {symbol} → {len(df)} bars loaded")
        return df
        
    except Exception as e:
        logger.warning(f"MT5 fetch error for {symbol}: {e}")
        return None


def _fetch_via_yfinance(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame | None:
    """Fetch data via Yahoo Finance."""
    if yf is None:
        logger.error("yfinance not installed. Cannot fetch data.")
        return None
    
    ticker = _yf_ticker(symbol)
    
    try:
        logger.info(f"YFinance: Fetching {ticker} (period={period}, interval={interval})...")
        data = yf.download(ticker, period=period, interval=interval, progress=False, auto_adjust=True)
        
        if data is None or data.empty:
            logger.error(f"YFinance returned no data for {ticker}")
            return None
        
        # yfinance sometimes returns MultiIndex columns — flatten
        if isinstance(data.columns, pd.MultiIndex):
            data.columns = [col[0].lower() if isinstance(col, tuple) else col.lower() for col in data.columns]
        else:
            data.columns = [c.lower() for c in data.columns]
        
        # Ensure required columns exist
        required = ["open", "high", "low", "close", "volume"]
        missing = [c for c in required if c not in data.columns]
        if missing:
            # Try alternate names
            rename_map = {}
            for col in data.columns:
                cl = col.lower()
                if "open" in cl and "open" not in data.columns:
                    rename_map[col] = "open"
                elif "high" in cl and "high" not in data.columns:
                    rename_map[col] = "high"
                elif "low" in cl and "low" not in data.columns:
                    rename_map[col] = "low"
                elif "close" in cl and "close" not in data.columns:
                    rename_map[col] = "close"
                elif "vol" in cl and "volume" not in data.columns:
                    rename_map[col] = "volume"
            if rename_map:
                data = data.rename(columns=rename_map)
        
        # Verify we have what we need
        for col in required:
            if col not in data.columns:
                logger.error(f"YFinance data missing column '{col}' for {ticker}. Columns: {list(data.columns)}")
                return None
        
        df = data[required].copy()
        df = df.dropna()
        
        # Synthesize volume for FX
        df = _synthesize_fx_volume(df)
        
        df = df.reset_index(drop=True)
        logger.success(f"YFinance: {ticker} → {len(df)} bars loaded")
        return df
        
    except Exception as e:
        logger.error(f"YFinance error for {ticker}: {e}")
        return None


def fetch_training_data(symbol: str, period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch historical training data for a symbol.
    Tries MT5 first (Windows), falls back to Yahoo Finance.
    
    Returns pd.DataFrame with columns: [open, high, low, close, volume]
    """
    # Try MT5 first on Windows
    df = _fetch_via_mt5(symbol)
    if df is not None and not df.empty:
        # Drop the time column for training compatibility
        if "time" in df.columns:
            df = df.drop(columns=["time"])
        return df
    
    # Fallback to Yahoo Finance
    df = _fetch_via_yfinance(symbol, period=period, interval=interval)
    if df is not None and not df.empty:
        return df
    
    logger.error(f"All data sources failed for {symbol}")
    return pd.DataFrame()


def get_combined_training_df(symbols: list[str], period: str = "60d", interval: str = "1h") -> pd.DataFrame:
    """
    Fetch and concatenate training data for multiple symbols.
    Each symbol's data is independently normalized via the TradingEnv, 
    so concatenation is safe for joint training.
    
    Returns pd.DataFrame with columns: [open, high, low, close, volume, symbol]
    """
    frames = []
    for sym in symbols:
        df = fetch_training_data(sym, period=period, interval=interval)
        if df is not None and not df.empty and len(df) > 100:
            df = df.copy()
            df["symbol"] = sym
            frames.append(df)
            logger.info(f"  ✓ {sym}: {len(df)} bars added to combined dataset")
        else:
            logger.warning(f"  ✗ {sym}: skipped (insufficient data)")
    
    if not frames:
        logger.error("No valid data from any symbol!")
        return pd.DataFrame()
    
    combined = pd.concat(frames, ignore_index=True)
    logger.success(f"Combined training dataset: {len(combined)} total bars from {len(frames)} symbols")
    return combined


def get_latest_data(symbol: str, timeframe: str = "M5", bars: int = 200):
    """
    Get latest market data for live inference.
    Returns np.ndarray on MT5, pd.DataFrame on YF, or None on failure.
    """
    # Try MT5
    if _mt5 is not None:
        try:
            if _mt5.initialize():
                tf = _MT5_TIMEFRAMES.get(timeframe, _mt5.TIMEFRAME_M5)
                rates = _mt5.copy_rates_from_pos(symbol, tf, 0, bars)
                if rates is not None and len(rates) > 0:
                    return rates
        except Exception as e:
            logger.warning(f"MT5 live data error: {e}")
    
    # Fallback for dev: use yfinance with short period
    df = _fetch_via_yfinance(symbol, period="5d", interval="5m")
    if df is not None and not df.empty:
        if len(df) > bars:
            df = df.tail(bars).reset_index(drop=True)
        return df
    
    if os.environ.get("AGI_IS_LIVE") == "1":
        raise Exception(f"Live data feed failure for {symbol}")
    return None
