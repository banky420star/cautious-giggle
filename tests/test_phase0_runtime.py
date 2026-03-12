from Python.Server_AGI import _low_volatility_memory_base


def test_low_volatility_memory_gate_requires_positive_history(monkeypatch):
    monkeypatch.setenv("AGI_LOW_VOL_MIN_TRADES", "20")
    monkeypatch.setenv("AGI_LOW_VOL_MIN_PROFIT_FACTOR", "1.15")
    monkeypatch.setenv("AGI_LOW_VOL_MIN_EXPECTANCY", "0.0")
    monkeypatch.setenv("AGI_LOW_VOL_MAX_RECENT_LOSS_STREAK", "3")

    blocked = _low_volatility_memory_base(
        {"trades": 50, "expectancy": -1.0, "profit_factor": 1.6, "recent_loss_streak": 0}
    )
    allowed = _low_volatility_memory_base(
        {"trades": 50, "expectancy": 10.0, "profit_factor": 1.6, "recent_loss_streak": 1}
    )

    assert blocked == 0.0
    assert allowed > 0.0
