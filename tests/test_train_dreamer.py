from types import SimpleNamespace

from training import train_dreamer


def test_resolve_symbols_prefers_configured_dreamer_subset(monkeypatch):
    monkeypatch.delenv("AGI_DREAMER_SYMBOLS", raising=False)
    monkeypatch.delenv("AGI_DREAMER_SYMBOL", raising=False)
    cfg = {
        "trading": {"symbols": ["EURUSDm", "XAUUSDm", "BTCUSDm"]},
        "drl": {"dreamer": {"symbols": ["EURUSDm", "BTCUSDm"]}},
    }
    args = SimpleNamespace(symbol=None, symbols=None)

    assert train_dreamer._resolve_symbols(args, cfg) == ["EURUSDm", "BTCUSDm"]


def test_resolve_symbols_falls_back_to_all_trading_symbols(monkeypatch):
    monkeypatch.delenv("AGI_DREAMER_SYMBOLS", raising=False)
    monkeypatch.delenv("AGI_DREAMER_SYMBOL", raising=False)
    cfg = {"trading": {"symbols": ["EURUSDm", "XAUUSDm"]}, "drl": {"dreamer": {}}}
    args = SimpleNamespace(symbol=None, symbols=None)

    assert train_dreamer._resolve_symbols(args, cfg) == ["EURUSDm", "XAUUSDm"]


def test_resolve_symbols_honors_environment_override(monkeypatch):
    monkeypatch.setenv("AGI_DREAMER_SYMBOLS", "AUDUSDm,NZDUSDm")
    cfg = {"trading": {"symbols": ["EURUSDm"]}, "drl": {"dreamer": {"symbols": ["BTCUSDm"]}}}
    args = SimpleNamespace(symbol=None, symbols=None)

    assert train_dreamer._resolve_symbols(args, cfg) == ["AUDUSDm", "NZDUSDm"]
