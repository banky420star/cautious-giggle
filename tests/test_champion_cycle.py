from tools import champion_cycle
import json
from pathlib import Path


def test_resolve_cycle_symbols_prefers_env_list(monkeypatch):
    monkeypatch.setenv("AGI_CYCLE_SYMBOLS", "BTCUSDm,ETHUSDm")
    monkeypatch.delenv("AGI_CYCLE_SYMBOL", raising=False)
    cfg = {"trading": {"symbols": ["EURUSDm", "GBPUSDm"]}}

    assert champion_cycle._resolve_cycle_symbols(cfg) == ["BTCUSDm", "ETHUSDm"]


def test_resolve_cycle_symbols_supports_single_symbol_env(monkeypatch):
    monkeypatch.delenv("AGI_CYCLE_SYMBOLS", raising=False)
    monkeypatch.setenv("AGI_CYCLE_SYMBOL", "BTCUSDm")
    cfg = {"trading": {"symbols": ["EURUSDm", "GBPUSDm"]}}

    assert champion_cycle._resolve_cycle_symbols(cfg) == ["BTCUSDm"]


def test_resolve_cycle_symbols_falls_back_to_trading_config(monkeypatch):
    monkeypatch.delenv("AGI_CYCLE_SYMBOLS", raising=False)
    monkeypatch.delenv("AGI_CYCLE_SYMBOL", raising=False)
    cfg = {"trading": {"symbols": ["BTCUSDm", "XAUUSDm"]}}

    assert champion_cycle._resolve_cycle_symbols(cfg) == ["BTCUSDm", "XAUUSDm"]


def test_resolve_cycle_symbols_defaults_to_btc_and_gold(monkeypatch):
    monkeypatch.delenv("AGI_CYCLE_SYMBOLS", raising=False)
    monkeypatch.delenv("AGI_CYCLE_SYMBOL", raising=False)

    assert champion_cycle._resolve_cycle_symbols({}) == ["BTCUSDm", "XAUUSDm"]


def test_latest_candidate_filters_by_symbol(tmp_path):
    root = tmp_path / "candidates"
    root.mkdir()

    eur = root / "20260101_000000"
    eur.mkdir()
    (eur / "scorecard.json").write_text(json.dumps({"symbol": "EURUSDm"}), encoding="utf-8")

    btc = root / "20260101_000100"
    btc.mkdir()
    (btc / "scorecard.json").write_text(json.dumps({"symbol": "BTCUSDm"}), encoding="utf-8")

    class _Reg:
        candidates_dir = str(root)

    out = champion_cycle._latest_candidate(_Reg(), symbol="BTCUSDm")

    assert Path(out) == btc
