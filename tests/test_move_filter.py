import os

import numpy as np
import pandas as pd
import pytest

from Python import move_filter

HAS_MODEL = os.path.exists(os.path.join(move_filter._DIR, "XAUUSD.npz.pkl"))


def test_numpy_trees_match_hand_built_tree():
    # one split on feature 0 at 1.0: left leaf -1, right leaf +2; NaN goes left (missing_type NaN, default_left)
    t = {"feat": np.array([0, -1, -1]), "thr": np.array([1.0, 0, 0]), "left": np.array([1, -1, -1]),
         "right": np.array([2, 0, 0]), "dleft": np.array([1, 0, 0], dtype=np.int8),
         "miss": np.array([2, 0, 0], dtype=np.int8), "leaf": np.array([0, -1.0, 2.0])}
    p = move_filter.predict_trees([t], np.array([[0.5], [3.0], [np.nan]]))
    assert np.allclose(p, 1 / (1 + np.exp(-np.array([-1.0, 2.0, -1.0]))))


def test_symbol_suffixes_resolve_to_trained_market():
    if not HAS_MODEL:
        pytest.skip("move filter not trained")
    assert move_filter.base_symbol("XAUUSDm") == "XAUUSD"
    assert move_filter.base_symbol("USOILm") == "USOIL"
    assert move_filter.base_symbol("BTCUSD") == "BTCUSD"


def test_prob_needs_history_and_model():
    assert move_filter.prob("NOSUCHm", None) is None
    if not HAS_MODEL:
        return
    t = pd.date_range("2026-01-01", periods=100, freq="5min")
    short = pd.DataFrame({"time": t, "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0, "volume": 1.0})
    assert move_filter.prob("XAUUSDm", short) is None
