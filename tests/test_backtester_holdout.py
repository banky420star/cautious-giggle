import numpy as np
import pandas as pd

from Python.backtester import _holdout_slice


def test_holdout_slice_works_on_time_indexed_frames():
    idx = pd.date_range("2026-01-01", periods=100, freq="5min", tz="UTC")
    df = pd.DataFrame({"close": np.arange(100.0)}, index=idx)
    out = _holdout_slice(df, str(idx[80]))
    assert len(out) == 20 and out["close"].iloc[0] == 80.0


def test_holdout_slice_works_with_time_column():
    t = pd.date_range("2026-01-01", periods=10, freq="5min", tz="UTC")
    df = pd.DataFrame({"time": t, "close": np.arange(10.0)})
    assert len(_holdout_slice(df, str(t[7]))) == 3
