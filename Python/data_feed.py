import os
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd
import yfinance as yf
from loguru import logger


def get_latest_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        if os.environ.get("AGI_IS_LIVE") == "1":
            raise Exception("Live data feed failure")
        return None
    return rates


def _to_yfinance_symbol(symbol: str) -> str:
    base = symbol.replace("m", "").upper()
    if base == "BTCUSD":
        return "BTC-USD"
    if base == "XAUUSD":
        return "GC=F"
    if len(base) == 6 and base.isalpha():
        return f"{base}=X"
    return base


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [str(c[0]).lower() for c in out.columns]
    else:
        out.columns = [str(c).lower() for c in out.columns]

    out = out.rename(columns={"adj close": "close", "adj_close": "close"})

    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"missing required column: {col}")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[["open", "high", "low", "close", "volume"]].copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna()
    out = out.ffill().bfill()
    return out


def fetch_training_data(symbol: str, period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    ticker = _to_yfinance_symbol(symbol)
    try:
        raw = yf.download(
            ticker,
            period=period,
            interval=interval,
            auto_adjust=False,
            progress=False,
            threads=False,
        )
    except Exception as exc:
        logger.error(f"data download failed for {symbol} ({ticker}): {exc}")
        return pd.DataFrame()

    if raw is None or raw.empty:
        logger.warning(f"no data for {symbol} ({ticker})")
        return pd.DataFrame()

    try:
        df = _normalize_ohlcv(raw)
    except Exception as exc:
        logger.error(f"normalization failed for {symbol}: {exc}")
        return pd.DataFrame()

    df["symbol"] = symbol
    df.index = pd.to_datetime(df.index)
    return df


def get_combined_training_df(symbols: Iterable[str], period: str = "60d", interval: str = "5m") -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        df = fetch_training_data(symbol, period=period, interval=interval)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined.replace([float("inf"), float("-inf")], pd.NA).dropna().ffill().bfill()
    return combined
