"""compact_v3: the smaller PPO input (proposal 2026-11-19, approved by the user).

Per decision PPO sees ~205 numbers instead of 16,400 (100 bars x 164 indicators):
  * the 164 ultimate_150 indicators of the LAST closed bar
  * a 30-minute history of 6 core indicators (lags 1..6 bars): 36
  * the big-move filter's chance of a large move in the next hour, and that chance minus the GO threshold: 2
  * the trend the live trend gate uses: 4h and 1h (completed bars, close vs EMA50) and whether they agree: 3
plus, from the environment / live executor, 5 trade-state numbers (see drl.trading_env):
  position, open profit in stop units, time in trade, distance to stop, distance to target.

Big-move readings come from data/move_filter_oof/<SYM>.npz (tools/move_filter_oof.py): honest readings for training
bars, the live filter's own readings after its training period. Live, the bot passes the current reading for the
last bar (mf_last). Missing readings sit at the threshold (no view).
"""
import os

import numpy as np
import pandas as pd

COMPACT_V3 = "compact_v3"
CORE = ["close_ret_1", "range_ratio", "body_ratio", "volume_rel_5", "rsi_5", "close_z_13"]
LAGS = 6
TREND_EMA = 50
TRADE_STATE_COUNT = 5
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_OOF = {}


def _oof(symbol: str | None):
    if not symbol:
        return None
    b = str(symbol).upper()
    b = b[:-1] if b[-1:] in ("M", "C") and not os.path.exists(os.path.join(_ROOT, "data", "move_filter_oof", f"{b}.npz")) else b
    if b not in _OOF:
        p = os.path.join(_ROOT, "data", "move_filter_oof", f"{b}.npz")
        if os.path.exists(p):
            z = np.load(p)
            _OOF[b] = (pd.Series(z["p"].astype(np.float64), index=z["t"].astype(np.int64)), float(z["thr"]))
        else:
            _OOF[b] = None
    return _OOF[b]


def htf_trend(close: pd.Series, rule: str, span: int = TREND_EMA) -> pd.Series:
    """+1/-1/0 per 5-minute bar: last COMPLETED `rule` bar's close vs its EMA(span). Same as Python.trend_gate.

    close is indexed by bar OPEN time; a 5-minute bar is known at open+5min, so a higher bar counts as completed
    once its right edge <= open+5min."""
    if not isinstance(close.index, pd.DatetimeIndex) or len(close) == 0:
        return pd.Series(0.0, index=close.index)
    known = close.copy()
    known.index = close.index + pd.Timedelta(minutes=5)
    b = known.resample(rule, label="right", closed="right").last().dropna()
    if len(b) < 2 * span:
        sign = pd.Series(0.0, index=b.index)
    else:
        e = b.ewm(span=span, adjust=False).mean()
        sign = np.sign(b - e)
        sign.iloc[: 2 * span] = 0.0                     # same warm-up rule as the live gate (needs 2*span bars)
    out = sign.reindex(known.index, method="ffill").fillna(0.0)
    out.index = close.index
    return out


def build_compact_matrix(df: pd.DataFrame, symbol: str | None = None, mf_last: float | None = None) -> np.ndarray:
    from Python.feature_pipeline import ULTIMATE_150, build_lstm_feature_frame, _normalize_ohlcv

    base, _ = build_lstm_feature_frame(df, feature_version=ULTIMATE_150)
    n = len(base)
    cols = [base.to_numpy(dtype=np.float64)]
    core = base.reindex(columns=CORE).fillna(0.0)
    for lag in range(1, LAGS + 1):
        cols.append(core.shift(lag).fillna(0.0).to_numpy(dtype=np.float64))
    # big-move filter
    o = _normalize_ohlcv(df)
    idx = o.index[-n:] if len(o) >= n else o.index
    oof = _oof(symbol)
    thr = oof[1] if oof else 0.5
    p = np.full(n, thr)
    if oof is not None and isinstance(idx, pd.DatetimeIndex):
        secs = (pd.DatetimeIndex(idx).as_unit("s").asi8).astype(np.int64)
        hit = oof[0].reindex(secs).to_numpy()
        p = np.where(np.isfinite(hit), hit, p)
    if mf_last is not None and np.isfinite(mf_last):
        p[-1] = float(mf_last)
    cols.append(np.stack([p, p - thr], axis=1))
    # trend (what the live gate sees)
    close = o["close"].iloc[-n:] if len(o) >= n else o["close"]
    if isinstance(close.index, pd.DatetimeIndex):
        t4, t1 = htf_trend(close, "4h").to_numpy(), htf_trend(close, "1h").to_numpy()
    else:
        t4 = t1 = np.zeros(n)
    agree = np.where((t4 == t1) & (t4 != 0), t4, 0.0)
    cols.append(np.stack([t4, t1, agree], axis=1))
    out = np.concatenate(cols, axis=1)
    return np.nan_to_num(out, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)


def trend_columns(n_features: int) -> tuple[int, int]:
    """Column positions of the 4h and 1h trend inside a compact_v3 row (used by the env's trend gate)."""
    return n_features - 3, n_features - 2
