from Python.Server_AGI import _blend_symbol_decision


def test_low_volatility_does_not_force_flat():
    agi = {"regime": "LOW_VOLATILITY", "confidence": 0.9, "risk_scalar": 0.95, "trend_bias": 0.0}
    ppo = {"target": 0.8}
    dreamer = {"target": 0.4}

    btc = _blend_symbol_decision("BTCUSDm", agi, ppo, dreamer)
    xau = _blend_symbol_decision("XAUUSDm", agi, ppo, dreamer)

    assert btc["target"] > 0.0
    assert xau["target"] > 0.0


def test_xau_and_btc_blend_differ():
    agi = {"regime": "MED_VOLATILITY", "confidence": 0.7, "risk_scalar": 0.8, "trend_bias": 0.1}
    ppo = {"target": 0.5}
    dreamer = {"target": 0.5}

    btc = _blend_symbol_decision("BTCUSDm", agi, ppo, dreamer)
    xau = _blend_symbol_decision("XAUUSDm", agi, ppo, dreamer)

    assert btc["target"] != xau["target"]


def test_normalize_active_weights_lets_council_trade_alone():
    from Python.Server_AGI import _blend_symbol_decision
    cfg = {"trading": {"confidence_threshold": 0.85, "normalize_active_weights": True}}
    weak_lstm = {"confidence": 0.6, "trend_bias": 0.6, "risk_scalar": 1.0}
    council = {"target": -0.4}
    on = _blend_symbol_decision("BTCUSDm", weak_lstm, None, None, cfg=cfg, council_meta=council)
    assert abs(on["raw_target"] + 0.4) < 1e-9 and on["weights_normalized"]
    off = _blend_symbol_decision("BTCUSDm", weak_lstm, None, None,
                                 cfg={"trading": {"confidence_threshold": 0.85}}, council_meta=council)
    assert abs(off["raw_target"]) < abs(on["raw_target"])
    nothing = _blend_symbol_decision("BTCUSDm", weak_lstm, None, None, cfg=cfg, council_meta=None)
    assert nothing["target"] == 0.0


def test_plan_entry_sets_pullback_limit(monkeypatch):
    import Python.Server_AGI as srv

    class T:  # tick
        bid, ask = 100.0, 100.2

    class I:
        point = 0.01

    class Ex:
        def _symbol_info(self, s):
            return I()

        def _atr_points(self, s):
            return 100  # 1.00 price units

    monkeypatch.setattr(srv.mt5, "symbol_info_tick", lambda s: T(), raising=False)
    st = srv._smart_entry_settings({"trading": {"smart_entry": {"enabled": True, "wait_minutes": 15,
                                                                "pullback_atr": {"BTCUSDm": 0.5}}}})
    buy = srv._plan_entry("BTCUSDm", 0.3, Ex(), st)
    assert abs(buy["limit"] - 99.7) < 1e-9 and buy["side"] == 1          # ask 100.2 - 0.5 ATR
    sell = srv._plan_entry("BTCUSDm", -0.3, Ex(), st)
    assert abs(sell["limit"] - 100.5) < 1e-9 and sell["side"] == -1      # bid 100.0 + 0.5 ATR
    assert 890 < buy["expires"] - buy["created"] < 910
