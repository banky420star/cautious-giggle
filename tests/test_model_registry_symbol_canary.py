import shutil
import uuid
from pathlib import Path

import pytest

from Python.model_registry import ModelRegistry


def _tmp_registry_root() -> Path:
    root = Path(".tmp") / f"registry_symbol_test_{uuid.uuid4().hex}"
    root.mkdir(parents=True, exist_ok=True)
    return root


def _candidate(root: Path, name: str, symbol: str) -> str:
    path = root / "candidates" / name
    path.mkdir(parents=True, exist_ok=True)
    (path / "metadata.json").write_text(f'{{"symbol": "{symbol}", "symbols": ["{symbol}"]}}', encoding="utf-8")
    (path / "scorecard.json").write_text(f'{{"symbol": "{symbol}", "symbols": ["{symbol}"]}}', encoding="utf-8")
    (path / "ppo_trading.zip").write_bytes(b"model")
    (path / "vec_normalize.pkl").write_bytes(b"vec")
    return str(path)


def test_symbol_canary_requires_symbol_metrics_to_promote():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root), registry_config={})
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
        reg = ModelRegistry(root=str(root), registry_config={})
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


def test_symbol_load_active_model_ignores_mismatched_symbol_entries():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root), registry_config={})
        btc = _candidate(root, "btc_ok", "BTCUSDm")
        xau = _candidate(root, "xau_ok", "XAUUSDm")
        reg._write_active(
            {
                "champion": xau,
                "canary": None,
                "symbols": {
                    "BTCUSDm": {"champion": xau, "canary": None, "canary_policy": {}, "canary_state": {}},
                    "XAUUSDm": {"champion": xau, "canary": None, "canary_policy": {}, "canary_state": {}},
                },
            }
        )

        assert reg.load_active_model(symbol="BTCUSDm") is None
        assert reg.load_active_model(symbol="XAUUSDm") == xau

        active = reg._read_active()
        assert active["symbols"]["BTCUSDm"]["champion"] is None

        reg.set_canary(btc, symbol="BTCUSDm")
        assert reg.load_active_model(symbol="BTCUSDm") == btc
    finally:
        shutil.rmtree(root, ignore_errors=True)


def test_set_canary_rejects_mismatched_symbol_artifact():
    root = _tmp_registry_root()
    try:
        reg = ModelRegistry(root=str(root), registry_config={})
        xau = _candidate(root, "xau_ok", "XAUUSDm")

        with pytest.raises(RuntimeError):
            reg.set_canary(xau, symbol="BTCUSDm")
    finally:
        shutil.rmtree(root, ignore_errors=True)
