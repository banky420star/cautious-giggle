"""
Tests for the _normalize_obs_safe fix in Python/hybrid_brain.py.

HybridBrain has heavy dependencies (stable-baselines3, torch, etc.) that may
not be present in the test environment.  Instead of importing HybridBrain
directly, we exercise the method's logic through a minimal stub that replicates
the exact implementation from lines 436-450 of hybrid_brain.py.
"""

from typing import Optional

import numpy as np
import pytest


# ---------------------------------------------------------------------------
# Fakes used across tests
# ---------------------------------------------------------------------------

class _FakeVecNorm:
    """A VecNorm stand-in whose normalize_obs doubles each element."""

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        return obs * 2.0


class _BrokenVecNorm:
    """A VecNorm stand-in that always raises on normalize_obs."""

    def normalize_obs(self, obs: np.ndarray) -> np.ndarray:
        raise ValueError("shape mismatch")


# ---------------------------------------------------------------------------
# Minimal stub carrying the exact implementation under test
# ---------------------------------------------------------------------------

class _BrainStub:
    """
    Minimal stub that carries only the attributes and method needed to test
    _normalize_obs_safe in isolation.

    The body of _normalize_obs_safe is an exact copy of the production code in
    Python/hybrid_brain.py (lines 436-450).
    """

    def __init__(self):
        self._vecnorm_disabled = False

    def _normalize_obs_safe(self, bundle: dict, obs: np.ndarray) -> Optional[np.ndarray]:
        if bundle.get("_vecnorm_failed"):
            return None
        vec_norm = bundle.get("vec_norm")
        if vec_norm is None:
            return obs
        try:
            return vec_norm.normalize_obs(obs.reshape(1, -1)).reshape(-1)
        except Exception as exc:
            bundle["_vecnorm_failed"] = True
            if not self._vecnorm_disabled:
                self._vecnorm_disabled = True
            return None


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def brain():
    return _BrainStub()


@pytest.fixture
def obs():
    return np.array([1.0, 2.0, 3.0], dtype=np.float32)


# ---------------------------------------------------------------------------
# 1. Normal path: vec_norm doubles each value
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_normal(brain, obs):
    """When a working vec_norm is present, the returned array equals obs * 2."""
    bundle = {"vec_norm": _FakeVecNorm()}

    result = brain._normalize_obs_safe(bundle, obs)

    assert result is not None
    np.testing.assert_array_almost_equal(result, obs * 2.0)


# ---------------------------------------------------------------------------
# 2. No vec_norm in bundle → obs returned unchanged
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_no_vecnorm(brain, obs):
    """When bundle has no 'vec_norm' key, obs is returned as-is."""
    bundle = {}

    result = brain._normalize_obs_safe(bundle, obs)

    assert result is not None
    np.testing.assert_array_equal(result, obs)


# ---------------------------------------------------------------------------
# 3. Broken vec_norm → returns None
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_exception_returns_none(brain, obs):
    """When vec_norm.normalize_obs raises, the method should return None."""
    bundle = {"vec_norm": _BrokenVecNorm()}

    result = brain._normalize_obs_safe(bundle, obs)

    assert result is None


# ---------------------------------------------------------------------------
# 4. Broken vec_norm → sets bundle["_vecnorm_failed"] to True
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_sets_failed_flag(brain, obs):
    """After an exception in normalize_obs, bundle['_vecnorm_failed'] must be True."""
    bundle = {"vec_norm": _BrokenVecNorm()}

    brain._normalize_obs_safe(bundle, obs)

    assert bundle.get("_vecnorm_failed") is True


# ---------------------------------------------------------------------------
# 5. Once failed flag is set, normalization is skipped (returns None)
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_skips_on_failed_flag(brain, obs):
    """When bundle['_vecnorm_failed'] is already True, the method returns None
    immediately without invoking vec_norm.normalize_obs."""
    bundle = {
        "vec_norm": _FakeVecNorm(),   # would succeed if called
        "_vecnorm_failed": True,
    }

    result = brain._normalize_obs_safe(bundle, obs)

    assert result is None


# ---------------------------------------------------------------------------
# 6. After exception, _vecnorm_disabled is set to True on the brain instance
# ---------------------------------------------------------------------------

def test_normalize_obs_safe_does_not_suppress_normal_errors_silently(brain, obs):
    """After a normalize_obs exception, brain._vecnorm_disabled must be True,
    indicating the error was recorded on the instance (not silently swallowed)."""
    bundle = {"vec_norm": _BrokenVecNorm()}

    assert brain._vecnorm_disabled is False

    brain._normalize_obs_safe(bundle, obs)

    assert brain._vecnorm_disabled is True
