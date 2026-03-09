import shutil
import uuid
from pathlib import Path

import pytest

from Python.model_registry import ModelRegistry


def _tmp_registry_root() -> Path:
    root = Path(".tmp") / f"registry_symbol_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def test_symbol_canary_requires_symbol_metrics_to_promote():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root))
        reg.set_canary("candidate_eur_v1", symbol="EURUSDm")

        with pytest.raises(RuntimeError):
            reg.promote_canary_to_champion(symbol="EURUSDm")

        reg.update_canary_metrics(
            trades=15,
            realized_pnl=12.5,
            drawdown=0.03,
            runtime_minutes=50.0,
            symbol="EURUSDm",
        )
        reg.promote_canary_to_champion(symbol="EURUSDm")

        active = reg._read_active()
        eur = active.get("symbols", {}).get("EURUSDm", {})
        assert eur.get("champion") == "candidate_eur_v1"
        assert eur.get("canary") is None
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_symbol_rollback_only_clears_target_symbol_canary():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root))
        reg.set_canary("candidate_eur_v2", symbol="EURUSDm")
        reg.set_canary("candidate_xau_v2", symbol="XAUUSDm")

        reg.rollback_to_champion(symbol="EURUSDm")
        active = reg._read_active()
        eur = active.get("symbols", {}).get("EURUSDm", {})
        xau = active.get("symbols", {}).get("XAUUSDm", {})

        assert eur.get("canary") is None
        assert xau.get("canary") == "candidate_xau_v2"
    finally:
        shutil.rmtree(root, ignore_errors=True)
