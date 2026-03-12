import numpy as np
import pandas as pd


ENGINEERED_V2 = "engineered_v2"
ULTIMATE_150 = "ultimate_150"
FEATURE_VERSIONS = {ENGINEERED_V2, ULTIMATE_150}

ENGINEERED_LSTM_COLUMNS = [
    "open",
    "high",
    "low",
    "close",
    "volume",
    "ret_1",
    "ret_5",
    "ret_10",
    "rsi_14",
    "atr_14",
    "ema_12",
    "ema_26",
    "macd_line",
    "macd_signal",
    "bb_width_20",
    "stoch_k_14",
    "vol_z_20",
]


def _as_series(df: pd.DataFrame, col: str) -> pd.Series:
    obj = df[col]
    if isinstance(obj, pd.DataFrame):
        return obj.iloc[:, 0].astype(float)
    return obj.astype(float)


def normalize_feature_version(feature_version: str | None, default: str = ENGINEERED_V2) -> str:
    version = str(feature_version or default).strip().lower()
    return version if version in FEATURE_VERSIONS else str(default)


def _normalize_ohlcv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = [str(c).lower() for c in out.columns]
    if "tick_volume" in out.columns and "volume" not in out.columns:
        out = out.rename(columns={"tick_volume": "volume"})
    if "volume" not in out.columns:
        out["volume"] = 0.0
    for col in ["open", "high", "low", "close"]:
        if col not in out.columns:
            raise ValueError(f"missing required column: {col}")

    if "time" in out.columns:
        out["time"] = pd.to_datetime(out["time"], utc=True, errors="coerce")
        out = out.dropna(subset=["time"]).sort_values("time").set_index("time")
    elif not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.RangeIndex(len(out))
    elif out.index.tz is None:
        out.index = pd.to_datetime(out.index, utc=True, errors="coerce")

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return out


def build_lstm_feature_frame(df: pd.DataFrame, feature_version: str = ENGINEERED_V2) -> tuple[pd.DataFrame, list[str]]:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        features = _build_ultimate_feature_frame(df)
        return features, list(features.columns)
    features = _build_engineered_lstm_frame(df)
    return features, list(ENGINEERED_LSTM_COLUMNS)


def build_env_feature_matrix(df: pd.DataFrame, feature_version: str = ENGINEERED_V2) -> np.ndarray:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        features, _ = build_lstm_feature_frame(df, feature_version=ULTIMATE_150)
        return features.to_numpy(dtype=np.float32)
    return _build_engineered_env_matrix(df)


def feature_count_for_version(feature_version: str) -> int:
    version = normalize_feature_version(feature_version, default=ENGINEERED_V2)
    if version == ULTIMATE_150:
        sample = pd.DataFrame(
            {
                "time": pd.date_range("2026-01-01", periods=300, freq="5min", tz="UTC"),
                "open": np.linspace(1.0, 1.2, 300),
                "high": np.linspace(1.01, 1.21, 300),
                "low": np.linspace(0.99, 1.19, 300),
                "close": np.linspace(1.0, 1.2, 300),
                "volume": np.linspace(100, 400, 300),
            }
        )
        return int(build_env_feature_matrix(sample, feature_version=ULTIMATE_150).shape[1])
    return 21


def _build_engineered_lstm_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    close = _as_series(out, "close")
    high = _as_series(out, "high")
    low = _as_series(out, "low")
    volume = _as_series(out, "volume")

    out["ret_1"] = close.pct_change().fillna(0.0)
    out["ret_5"] = close.pct_change(5).fillna(0.0)
    out["ret_10"] = close.pct_change(10).fillna(0.0)

    delta = close.diff().fillna(0.0)
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    out["rsi_14"] = (100 - (100 / (1 + rs))).fillna(50.0)

    tr1 = (high - low).abs()
    tr2 = (high - close.shift(1)).abs()
    tr3 = (low - close.shift(1)).abs()
    out["atr_14"] = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1).rolling(14).mean().bfill()

    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()
    out["macd_line"] = out["ema_12"] - out["ema_26"]
    out["macd_signal"] = out["macd_line"].ewm(span=9, adjust=False).mean()

    bb_mid = close.rolling(20).mean()
    bb_std = close.rolling(20).std().fillna(0.0)
    out["bb_width_20"] = ((bb_std * 4.0) / (bb_mid.abs() + 1e-12)).fillna(0.0)

    low_14 = low.rolling(14).min()
    high_14 = high.rolling(14).max()
    out["stoch_k_14"] = (((close - low_14) / ((high_14 - low_14) + 1e-12)) * 100.0).fillna(50.0)

    vol_mean_20 = volume.rolling(20).mean()
    vol_std_20 = volume.rolling(20).std().fillna(0.0)
    out["vol_z_20"] = ((volume - vol_mean_20) / (vol_std_20 + 1e-12)).fillna(0.0)

    out = out.replace([np.inf, -np.inf], np.nan).ffill().bfill().dropna()
    return out[ENGINEERED_LSTM_COLUMNS].copy()


def _build_engineered_env_matrix(df: pd.DataFrame) -> np.ndarray:
    out = _normalize_ohlcv(df)
    o = out["open"].to_numpy(dtype=np.float64)
    h = out["high"].to_numpy(dtype=np.float64)
    l = out["low"].to_numpy(dtype=np.float64)
    c = out["close"].to_numpy(dtype=np.float64)
    v = out["volume"].to_numpy(dtype=np.float64)
    dates = out.index if isinstance(out.index, pd.DatetimeIndex) else None
    eps = 1e-12

    def shift(arr: np.ndarray, n: int) -> np.ndarray:
        if n <= 0:
            return arr.copy()
        shifted = np.empty_like(arr)
        shifted[:n] = arr[0]
        shifted[n:] = arr[:-n]
        return shifted

    range_ = np.maximum(h - l, eps)
    close_shift1 = shift(c, 1)
    close_shift5 = shift(c, 5)
    close_shift20 = shift(c, 20)

    log_ret1 = np.log(np.maximum(c, eps) / np.maximum(close_shift1, eps))
    log_ret5 = np.log(np.maximum(c, eps) / np.maximum(close_shift5, eps))
    log_ret20 = np.log(np.maximum(c, eps) / np.maximum(close_shift20, eps))

    body_ratio = (c - o) / range_
    upper_wick = (h - np.maximum(o, c)) / range_
    lower_wick = (np.minimum(o, c) - l) / range_
    range_ratio = (h - l) / (c + eps)

    rv_20 = pd.Series(log_ret1).rolling(20, min_periods=1).std().fillna(0.0).to_numpy(dtype=np.float64)
    vol_ma20 = pd.Series(np.maximum(v, 0.0)).rolling(20, min_periods=1).mean().to_numpy(dtype=np.float64)
    rel_volume = np.maximum(v, 0.0) / (vol_ma20 + eps)
    spread_est_bps = ((h - l) / (c + eps)) * 10000.0

    ma50 = pd.Series(c).rolling(50, min_periods=1).mean().to_numpy(dtype=np.float64)
    htf_trend = (c / (ma50 + eps)) - 1.0

    hour_sin = np.zeros_like(c)
    hour_cos = np.zeros_like(c)
    dow_sin = np.zeros_like(c)
    dow_cos = np.zeros_like(c)
    if dates is not None:
        hour = dates.hour.astype(np.float64)
        dow = dates.dayofweek.astype(np.float64)
        hour_sin = np.sin(2.0 * np.pi * hour / 24.0)
        hour_cos = np.cos(2.0 * np.pi * hour / 24.0)
        dow_sin = np.sin(2.0 * np.pi * dow / 7.0)
        dow_cos = np.cos(2.0 * np.pi * dow / 7.0)

    valid_rv = rv_20[np.isfinite(rv_20)]
    if len(valid_rv) > 10:
        q1 = np.quantile(valid_rv, 0.33)
        q2 = np.quantile(valid_rv, 0.66)
        vol_bucket = np.where(rv_20 <= q1, 0.0, np.where(rv_20 <= q2, 0.5, 1.0))
    else:
        vol_bucket = np.zeros_like(c)

    close_rel = (c / (close_shift1 + eps)) - 1.0
    open_rel = (o / (c + eps)) - 1.0
    high_rel = (h / (c + eps)) - 1.0
    low_rel = (l / (c + eps)) - 1.0
    log_vol = np.log1p(np.maximum(v, 0.0))

    matrix = np.column_stack(
        [
            open_rel,
            high_rel,
            low_rel,
            close_rel,
            log_vol,
            log_ret1,
            log_ret5,
            log_ret20,
            body_ratio,
            upper_wick,
            lower_wick,
            range_ratio,
            rv_20,
            rel_volume,
            spread_est_bps,
            hour_sin,
            hour_cos,
            dow_sin,
            dow_cos,
            htf_trend,
            vol_bucket,
        ]
    )
    return np.nan_to_num(matrix, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def _build_ultimate_feature_frame(df: pd.DataFrame) -> pd.DataFrame:
    out = _normalize_ohlcv(df)
    close = out["close"].astype(float)
    high = out["high"].astype(float)
    low = out["low"].astype(float)
    open_ = out["open"].astype(float)
    volume = out["volume"].astype(float)
    eps = 1e-12

    feats: dict[str, pd.Series] = {}
    feats["open_rel"] = open_ / (close + eps) - 1.0
    feats["high_rel"] = high / (close + eps) - 1.0
    feats["low_rel"] = low / (close + eps) - 1.0
    feats["close_ret_1"] = close.pct_change().fillna(0.0)
    feats["body_ratio"] = (close - open_) / ((high - low).abs() + eps)
    feats["upper_wick_ratio"] = (high - np.maximum(open_, close)) / ((high - low).abs() + eps)
    feats["lower_wick_ratio"] = (np.minimum(open_, close) - low) / ((high - low).abs() + eps)
    feats["range_ratio"] = (high - low) / (close.abs() + eps)
    feats["log_volume"] = np.log1p(np.maximum(volume, 0.0))
    feats["gap_ratio"] = open_ / (close.shift(1).fillna(close.iloc[0]) + eps) - 1.0

    windows = [3, 5, 8, 13, 21, 34, 55]
    for win in windows:
        ret = close.pct_change(win).fillna(0.0)
        logret = np.log(close / (close.shift(win).fillna(close.iloc[0]) + eps)).fillna(0.0)
        range_mean = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).mean()
        range_std = ((high - low) / (close.abs() + eps)).rolling(win, min_periods=1).std().fillna(0.0)
        ma = close.rolling(win, min_periods=1).mean()
        ema = close.ewm(span=max(2, win), adjust=False).mean()
        vol_mean = volume.rolling(win, min_periods=1).mean()
        vol_std = volume.rolling(win, min_periods=1).std().fillna(0.0)
        price_std = close.rolling(win, min_periods=1).std().fillna(0.0)
        atr = pd.concat(
            [(high - low).abs(), (high - close.shift(1)).abs(), (low - close.shift(1)).abs()],
            axis=1,
        ).max(axis=1).rolling(win, min_periods=1).mean()
        delta = close.diff().fillna(0.0)
        gain = delta.clip(lower=0).rolling(win, min_periods=1).mean()
        loss = (-delta.clip(upper=0)).rolling(win, min_periods=1).mean()
        rs = gain / (loss + eps)
        rsi = (100.0 - (100.0 / (1.0 + rs))).fillna(50.0)
        bb_width = ((close.rolling(win, min_periods=1).std().fillna(0.0) * 4.0) / (ma.abs() + eps)).fillna(0.0)
        highest = high.rolling(win, min_periods=1).max()
        lowest = low.rolling(win, min_periods=1).min()
        slope = ma.diff(win).fillna(0.0) / (ma.shift(win).abs() + eps)

        feats[f"ret_{win}"] = ret
        feats[f"logret_{win}"] = logret
        feats[f"range_mean_{win}"] = range_mean
        feats[f"range_std_{win}"] = range_std
        feats[f"close_ma_rel_{win}"] = close / (ma + eps) - 1.0
        feats[f"volume_rel_{win}"] = volume / (vol_mean + eps) - 1.0
        feats[f"realized_vol_{win}"] = close.pct_change().rolling(win, min_periods=1).std().fillna(0.0)
        feats[f"close_z_{win}"] = (close - ma) / (price_std + eps)
        feats[f"momentum_{win}"] = close.diff(win).fillna(0.0) / (close.shift(win).abs() + eps)
        feats[f"ema_rel_{win}"] = close / (ema + eps) - 1.0
        feats[f"rsi_{win}"] = (rsi / 100.0) * 2.0 - 1.0
        feats[f"atr_rel_{win}"] = atr / (close.abs() + eps)
        feats[f"bb_width_{win}"] = bb_width
        feats[f"breakout_high_{win}"] = close / (highest + eps) - 1.0
        feats[f"breakout_low_{win}"] = close / (lowest + eps) - 1.0
        feats[f"slope_{win}"] = slope.fillna(0.0)

    if isinstance(out.index, pd.DatetimeIndex):
        idx = out.index
        hour = pd.Series(idx.hour.astype(np.float32), index=out.index)
        dow = pd.Series(idx.dayofweek.astype(np.float32), index=out.index)
        month = pd.Series(idx.month.astype(np.float32), index=out.index)
        feats["hour_sin"] = np.sin(2.0 * np.pi * hour / 24.0)
        feats["hour_cos"] = np.cos(2.0 * np.pi * hour / 24.0)
        feats["dow_sin"] = np.sin(2.0 * np.pi * dow / 7.0)
        feats["dow_cos"] = np.cos(2.0 * np.pi * dow / 7.0)
        feats["month_sin"] = np.sin(2.0 * np.pi * month / 12.0)
        feats["month_cos"] = np.cos(2.0 * np.pi * month / 12.0)
    else:
        for name in ["hour_sin", "hour_cos", "dow_sin", "dow_cos", "month_sin", "month_cos"]:
            feats[name] = pd.Series(0.0, index=out.index)

    if isinstance(out.index, pd.DatetimeIndex):
        resamples = [
            ("15min", "m15"),
            ("1h", "h1"),
            ("4h", "h4"),
            ("1d", "d1"),
        ]
        for rule, label in resamples:
            htf = (
                out[["open", "high", "low", "close", "volume"]]
                .resample(rule)
                .agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
                .ffill()
            )
            htf = htf.reindex(out.index, method="ffill")
            htf_ma = htf["close"].rolling(8, min_periods=1).mean()
            htf_std = htf["close"].rolling(8, min_periods=1).std().fillna(0.0)
            htf_delta = htf["close"].diff().fillna(0.0)
            htf_gain = htf_delta.clip(lower=0).rolling(8, min_periods=1).mean()
            htf_loss = (-htf_delta.clip(upper=0)).rolling(8, min_periods=1).mean()
            htf_rs = htf_gain / (htf_loss + eps)
            htf_rsi = (100.0 - (100.0 / (1.0 + htf_rs))).fillna(50.0)
            feats[f"{label}_close_rel"] = htf["close"] / (close + eps) - 1.0
            feats[f"{label}_range_rel"] = (htf["high"] - htf["low"]) / (close.abs() + eps)
            feats[f"{label}_volume_rel"] = htf["volume"] / (volume.rolling(20, min_periods=1).mean() + eps) - 1.0
            feats[f"{label}_trend"] = htf["close"] / (htf_ma + eps) - 1.0
            feats[f"{label}_rsi"] = (htf_rsi / 100.0) * 2.0 - 1.0
            feats[f"{label}_bb_width"] = ((htf_std * 4.0) / (htf_ma.abs() + eps)).fillna(0.0)
    else:
        for label in ["m15", "h1", "h4", "d1"]:
            for suffix in ["close_rel", "range_rel", "volume_rel", "trend", "rsi", "bb_width"]:
                feats[f"{label}_{suffix}"] = pd.Series(0.0, index=out.index)

    feats["cross_trend_h1_h4"] = feats["h1_trend"] - feats["h4_trend"]
    feats["cross_trend_m15_h1"] = feats["m15_trend"] - feats["h1_trend"]
    feats["cross_rsi_h1_d1"] = feats["h1_rsi"] - feats["d1_rsi"]
    feats["cross_rsi_m15_h4"] = feats["m15_rsi"] - feats["h4_rsi"]
    feats["cross_volume_h1_d1"] = feats["h1_volume_rel"] - feats["d1_volume_rel"]
    feats["cross_range_h1_h4"] = feats["h1_range_rel"] - feats["h4_range_rel"]
    feats["cross_close_h1_d1"] = feats["h1_close_rel"] - feats["d1_close_rel"]
    feats["cross_bb_h1_d1"] = feats["h1_bb_width"] - feats["d1_bb_width"]
    feats["cross_bb_m15_h1"] = feats["m15_bb_width"] - feats["h1_bb_width"]
    feats["cross_ret_5_21"] = feats["ret_5"] - feats["ret_21"]
    feats["cross_ret_13_55"] = feats["ret_13"] - feats["ret_55"]
    feats["cross_vol_8_34"] = feats["realized_vol_8"] - feats["realized_vol_34"]

    feature_df = pd.DataFrame(feats, index=out.index)
    feature_df = feature_df.replace([np.inf, -np.inf], np.nan).ffill().bfill().fillna(0.0)
    return feature_df.astype(np.float32)
