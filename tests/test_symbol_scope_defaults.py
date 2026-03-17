from types import SimpleNamespace

from Python import config_utils
from Python.autonomy_loop import AutonomyLoop


def test_default_symbols_are_btc_and_xau():
    assert config_utils.DEFAULT_TRADING_SYMBOLS == ["BTCUSDm", "XAUUSDm"]


def test_resolve_trading_symbols_defaults_to_btc_and_gold(monkeypatch):
    monkeypatch.delenv("AGI_RUNTIME_SYMBOLS", raising=False)
    assert config_utils.resolve_trading_symbols({}, fallback=None) == ["BTCUSDm", "XAUUSDm"]


def test_resolve_trading_symbols_prefers_env(monkeypatch):
    monkeypatch.setenv("AGI_RUNTIME_SYMBOLS", "BTCUSDm,XAUUSDm")
    cfg = {"trading": {"symbols": ["EURUSDm", "GBPUSDm"]}}

    assert config_utils.resolve_trading_symbols(cfg, env_keys=("AGI_RUNTIME_SYMBOLS",)) == ["BTCUSDm", "XAUUSDm"]


def test_autonomy_loop_load_symbols_prefers_autonomy_env(monkeypatch):
    monkeypatch.setenv("AGI_AUTONOMY_SYMBOLS", "BTCUSDm,XAUUSDm")
    monkeypatch.setattr(AutonomyLoop, "_ensure_symbol_registry_seeded", lambda self: None)
    monkeypatch.setattr(AutonomyLoop, "_init_alerter", lambda self: SimpleNamespace(alert=lambda *args, **kwargs: None))
    monkeypatch.setattr(AutonomyLoop, "_load_evaluation_config", lambda self: {})
    monkeypatch.setattr(
        "Python.autonomy_loop.load_project_config",
        lambda *args, **kwargs: {"trading": {"symbols": ["EURUSDm", "GBPUSDm"]}},
    )

    loop = AutonomyLoop(SimpleNamespace())

    assert loop._load_symbols_cfg() == ["BTCUSDm", "XAUUSDm"]


def test_autonomy_loop_defaults_to_btc_and_gold_on_config_failure(monkeypatch):
    monkeypatch.delenv("AGI_AUTONOMY_SYMBOLS", raising=False)
    monkeypatch.setattr(AutonomyLoop, "_ensure_symbol_registry_seeded", lambda self: None)
    monkeypatch.setattr(AutonomyLoop, "_init_alerter", lambda self: SimpleNamespace(alert=lambda *args, **kwargs: None))
    monkeypatch.setattr(AutonomyLoop, "_load_evaluation_config", lambda self: {})
    monkeypatch.setattr("Python.autonomy_loop.load_project_config", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("boom")))

    loop = AutonomyLoop(SimpleNamespace())

    assert loop._load_symbols_cfg() == ["BTCUSDm", "XAUUSDm"]
