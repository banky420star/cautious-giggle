from Python.hybrid_brain import HybridBrain


class _Dummy:
    pass


def test_hybrid_brain_reads_dreamer_config(monkeypatch):
    monkeypatch.delenv("AGI_DREAMER_ENABLED", raising=False)
    monkeypatch.delenv("AGI_DREAMER_BLEND", raising=False)
    monkeypatch.setattr(
        HybridBrain,
        "_load_cfg",
        lambda self: {
            "trading": {"symbols": ["EURUSDm", "XAUUSDm"]},
            "drl": {"dreamer": {"enabled": True, "blend": 0.22, "symbols": ["XAUUSDm"]}},
        },
    )
    monkeypatch.setattr(HybridBrain, "_load_ppo_from_registry", lambda self: None)
    monkeypatch.setattr(HybridBrain, "_load_dreamer_policies", lambda self: None)
    monkeypatch.setattr(HybridBrain, "_start_autonomy_if_enabled", lambda self: None)

    brain = HybridBrain(_Dummy(), _Dummy())

    assert brain.dreamer_enabled is True
    assert brain.dreamer_blend == 0.22
    assert brain.dreamer_symbols == ["XAUUSDm"]
