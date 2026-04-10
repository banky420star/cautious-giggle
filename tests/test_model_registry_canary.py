import shutil
import uuid
from pathlib import Path

import pytest

from Python.model_registry import ModelRegistry


def _tmp_registry_root() -> Path:
    root = Path(".tmp") / f"registry_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_canary_promotion_requires_survival_metrics():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root), registry_config={})
        reg.set_canary("candidate_v1")

        with pytest.raises(RuntimeError):
            reg.promote_canary_to_champion()

        reg.update_canary_metrics(trades=12, realized_pnl=15.0, drawdown=0.03, runtime_minutes=45.0)
        reg.promote_canary_to_champion()

        active = reg._read_active()
        assert active["champion"] == "candidate_v1"
        assert active["canary"] is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_rollback_clears_canary_without_touching_champion():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root), registry_config={})
        active = reg._read_active()
        active["champion"] = "champion_v1"
        reg._write_active(active)
        reg.set_canary("candidate_v2")
        reg.rollback_to_champion()

        out = reg._read_active()
        assert out["champion"] == "champion_v1"
        assert out["canary"] is None
    finally:
        shutil.rmtree(root, ignore_errors=True)
