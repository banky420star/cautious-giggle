import math
from datetime import datetime, timezone
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger


def _to_mt5_timeframe(interval: str):
    m = (interval or "5m").lower().strip()
    mapping = {
        "1m": mt5.TIMEFRAME_M1,
        "5m": mt5.TIMEFRAME_M5,
        "15m": mt5.TIMEFRAME_M15,
        "30m": mt5.TIMEFRAME_M30,
        "1h": mt5.TIMEFRAME_H1,
        "4h": mt5.TIMEFRAME_H4,
        "1d": mt5.TIMEFRAME_D1,
    }
    return mapping.get(m, mt5.TIMEFRAME_M5)


def _interval_minutes(interval: str) -> int:
    m = (interval or "5m").lower().strip()
    if m.endswith("m"):
        return max(1, int(m[:-1]))
    if m.endswith("h"):
        return max(1, int(m[:-1])) * 60
    if m.endswith("d"):
        return max(1, int(m[:-1])) * 24 * 60
    return 5


def _period_days(period: str) -> int:
    p = (period or "60d").lower().strip()
    try:
        if p.endswith("d"):
            return max(1, int(p[:-1]))
        if p.endswith("w"):
            return max(1, int(p[:-1])) * 7
        if p.endswith("mo"):
            return max(1, int(p[:-2])) * 30
    except Exception:
        pass
    return 60


def _bars_for(period: str, interval: str) -> int:
    days = _period_days(period)
    mins = _interval_minutes(interval)
    bars = int(math.ceil((days * 24 * 60) / max(1, mins)))
    return max(300, min(150_000, bars + 50))


def get_latest_data(symbol, timeframe, bars):
    rates = mt5.copy_rates_from_pos(symbol, timeframe, 0, bars)
    if rates is None:
        raise RuntimeError(f"MT5 data feed failure for {symbol}")
    return rates


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]

    if "tick_volume" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"tick_volume": "volume"})

    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"missing required column: {col}")

    if "volume" not in out.columns:
        out["volume"] = 0.0

    out = out[["open", "high", "low", "close", "volume"]].copy()
    out = out.replace([float("inf"), float("-inf")], pd.NA).dropna().ffill().bfill()
    return out


def _assert_recent_bars(df: pd.DataFrame, interval: str, stale_bars: int = 3):
    if df.empty or not isinstance(df.index, pd.DatetimeIndex):
        raise RuntimeError("cannot validate freshness: missing datetime index")
    last_ts = pd.to_datetime(df.index.max(), utc=True)
    now_ts = datetime.now(timezone.utc)
    max_age_min = max(1, _interval_minutes(interval)) * max(1, int(stale_bars))
    age_min = (now_ts - last_ts.to_pydatetime()).total_seconds() / 60.0
    if age_min > max_age_min:
        raise RuntimeError(
            f"stale MT5 data: last={last_ts.isoformat()} age_min={age_min:.1f} > allowed={max_age_min}"
        )


def fetch_training_data(
    symbol: str,
    period: str = "60d",
    interval: str = "5m",
    strict: bool = False,
    require_fresh: bool = False,
) -> pd.DataFrame:
    tf = _to_mt5_timeframe(interval)
    bars = _bars_for(period, interval)

    if not mt5.initialize():
        msg = f"MT5 initialize failed for {symbol}: {mt5.last_error()}"
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars)
    if rates is None or len(rates) < 100:
        msg = f"no MT5 data for {symbol} | tf={interval} bars={bars}"
        logger.warning(msg)
        if strict:
            raise RuntimeError(msg)
        return pd.DataFrame()

    raw = pd.DataFrame(rates)
    if raw.empty:
        if strict:
            raise RuntimeError(f"empty MT5 frame for {symbol}")
        return pd.DataFrame()

    raw["time"] = pd.to_datetime(raw["time"], unit="s", utc=True)
    raw = raw.set_index("time")

    try:
        df = _normalize_ohlcv(raw)
    except Exception as exc:
        msg = f"normalization failed for {symbol}: {exc}"
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
        return pd.DataFrame()

    if require_fresh:
        _assert_recent_bars(df, interval=interval, stale_bars=3)

    df["symbol"] = symbol
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
