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
