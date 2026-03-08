import math
import os
from datetime import datetime, timezone
from typing import Iterable

import MetaTrader5 as mt5
import pandas as pd
from loguru import logger

try:
    import yaml
except Exception:
    yaml = None


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
    return max(300, min(600_000, bars + 50))


def _resolve_cfg_value(v):
    if isinstance(v, str) and v.startswith("ENV:"):
        return os.environ.get(v.split(":", 1)[1], "")
    return v


def _load_mt5_cfg() -> dict:
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    cfg_path = os.path.join(root, "config.yaml")
    if not os.path.exists(cfg_path) or yaml is None:
        return {}
    try:
        with open(cfg_path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f) or {}
        return cfg.get("mt5", {}) if isinstance(cfg, dict) else {}
    except Exception:
        return {}


def _initialize_mt5() -> bool:
    mt5_cfg = _load_mt5_cfg()
    login_raw = os.environ.get("MT5_LOGIN") or _resolve_cfg_value(mt5_cfg.get("login", ""))
    password = os.environ.get("MT5_PASSWORD") or _resolve_cfg_value(mt5_cfg.get("password", ""))
    server = os.environ.get("MT5_SERVER") or _resolve_cfg_value(mt5_cfg.get("server", ""))
    try:
        login = int(str(login_raw).strip()) if str(login_raw).strip() else 0
    except Exception:
        login = 0
    if login and password and server:
        return bool(mt5.initialize(login=login, password=str(password), server=str(server)))
    return bool(mt5.initialize())


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
    bars: int | None = None,
    min_bars: int | None = None,
) -> pd.DataFrame:
    tf = _to_mt5_timeframe(interval)
    bars_req = int(bars) if bars is not None else _bars_for(period, interval)
    min_required = int(min_bars) if min_bars is not None else 100

    if not _initialize_mt5():
        msg = f"MT5 initialize failed for {symbol}: {mt5.last_error()}"
        logger.error(msg)
        if strict:
            raise RuntimeError(msg)
        return pd.DataFrame()

    rates = mt5.copy_rates_from_pos(symbol, tf, 0, bars_req)
    got = 0 if rates is None else len(rates)
    if rates is None or got < max(100, min_required):
        msg = f"insufficient MT5 data for {symbol} | tf={interval} requested={bars_req} got={got} required={max(100, min_required)}"
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


def get_combined_training_df(
    symbols: Iterable[str],
    period: str = "60d",
    interval: str = "5m",
    bars: int | None = None,
    min_bars: int | None = None,
) -> pd.DataFrame:
    frames = []
    for symbol in symbols:
        df = fetch_training_data(symbol, period=period, interval=interval, bars=bars, min_bars=min_bars)
        if not df.empty:
            frames.append(df)

    if not frames:
        return pd.DataFrame()

    combined = pd.concat(frames, axis=0).sort_index()
    combined = combined.replace([float("inf"), float("-inf")], pd.NA).dropna().ffill().bfill()
    return combined
