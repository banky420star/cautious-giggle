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
    assert brain.active_symbols == ["EURUSDm", "XAUUSDm"]
    assert brain.dreamer_symbols == ["XAUUSDm"]


def test_hybrid_brain_prefers_runtime_symbols_env(monkeypatch):
    monkeypatch.setenv("AGI_RUNTIME_SYMBOLS", "BTCUSDm,XAUUSDm")
    monkeypatch.delenv("AGI_DREAMER_ENABLED", raising=False)
    monkeypatch.delenv("AGI_DREAMER_BLEND", raising=False)
    monkeypatch.setattr(
        HybridBrain,
        "_load_cfg",
        lambda self: {
            "trading": {"symbols": ["EURUSDm", "GBPUSDm"]},
            "drl": {"dreamer": {"enabled": True, "blend": 0.22}},
        },
    )
    monkeypatch.setattr(HybridBrain, "_load_ppo_from_registry", lambda self: None)
    monkeypatch.setattr(HybridBrain, "_load_dreamer_policies", lambda self: None)
    monkeypatch.setattr(HybridBrain, "_start_autonomy_if_enabled", lambda self: None)

    brain = HybridBrain(_Dummy(), _Dummy())

    assert brain.active_symbols == ["BTCUSDm", "XAUUSDm"]
    assert brain.dreamer_symbols == ["BTCUSDm", "XAUUSDm"]


def test_bundle_targets_symbol():
    assert HybridBrain._bundle_targets_symbol({"symbol": "BTCUSDm"}, "BTCUSDm") is True
    assert HybridBrain._bundle_targets_symbol({"symbol": "XAUUSDm"}, "BTCUSDm") is False
    assert HybridBrain._bundle_targets_symbol({"symbols": ["BTCUSDm", "XAUUSDm"]}, "XAUUSDm") is True
    assert HybridBrain._bundle_targets_symbol({"symbols": ["XAUUSDm"]}, "BTCUSDm") is False


def test_hybrid_brain_loads_symbol_scoped_ppo(monkeypatch):
    monkeypatch.delenv("AGI_RUNTIME_SYMBOLS", raising=False)
    monkeypatch.setattr(
        HybridBrain,
        "_load_cfg",
        lambda self: {"trading": {"symbols": ["BTCUSDm", "XAUUSDm"]}, "drl": {"dreamer": {"enabled": False}}},
    )
    monkeypatch.setattr(
        HybridBrain,
        "_load_ppo_bundles_for_symbol",
        lambda self, symbol=None: [{"model": f"model:{symbol or 'global'}", "vec_norm": None, "meta": {"symbol": symbol}}],
    )
    monkeypatch.setattr(HybridBrain, "_load_dreamer_policies", lambda self: None)
    monkeypatch.setattr(HybridBrain, "_start_autonomy_if_enabled", lambda self: None)

    brain = HybridBrain(_Dummy(), _Dummy())

    assert "BTCUSDm" in brain.ppo_bundles_by_symbol
    assert "XAUUSDm" in brain.ppo_bundles_by_symbol


def test_get_last_action_meta_is_symbol_scoped():
    brain = HybridBrain.__new__(HybridBrain)
    brain._last_action_meta = None
    brain._last_action_meta_by_symbol = {
        "BTCUSDm": {"target": 0.7},
        "XAUUSDm": {"target": -0.2},
    }

    assert brain.get_last_action_meta("BTCUSDm")["target"] == 0.7
    assert brain.get_last_action_meta("XAUUSDm")["target"] == -0.2


def test_predict_ppo_action_does_not_fallback_to_other_symbol_bundle():
    brain = HybridBrain.__new__(HybridBrain)
    brain.active_symbols = ["BTCUSDm", "XAUUSDm"]
    brain.ppo_bundles_by_symbol = {"XAUUSDm": [{"target": 0.4}]}
    brain.ppo_bundles = [{"target": 0.9}]
    brain._last_action_meta = None
    brain._last_action_meta_by_symbol = {}

    assert brain.predict_ppo_action("BTCUSDm", object()) is None


def test_predict_dreamer_action_does_not_fallback_to_other_symbol_policy():
    brain = HybridBrain.__new__(HybridBrain)
    brain.dreamer_enabled = True
    brain.active_symbols = ["BTCUSDm", "XAUUSDm"]
    brain.dreamer_policies_by_symbol = {"XAUUSDm": object()}
    brain.dreamer_metadata_by_symbol = {}

    assert brain.predict_dreamer_action("BTCUSDm", object()) is None


def test_candidate_targets_symbol_uses_metadata_or_scorecard(tmp_path):
    cand = tmp_path / "candidate"
    cand.mkdir()
    (cand / "metadata.json").write_text('{"symbol":"BTCUSDm"}', encoding="utf-8")

    brain = HybridBrain.__new__(HybridBrain)

    assert brain._candidate_targets_symbol(str(cand), "BTCUSDm") is True
    assert brain._candidate_targets_symbol(str(cand), "XAUUSDm") is False


def test_symbol_ppo_min_abs_prefers_symbol_override():
    brain = HybridBrain.__new__(HybridBrain)
    brain.ppo_min_abs = 0.03
    brain.drl_symbol_overrides = {"BTCUSDm": {"action": {"min_target_abs": 0.015}}}

    assert brain._symbol_ppo_min_abs("BTCUSDm") == 0.015
    assert brain._symbol_ppo_min_abs("XAUUSDm") == 0.03
