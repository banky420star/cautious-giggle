from Python.risk_supervisor import RiskSupervisor


def test_risk_supervisor_blocks_spread_and_drawdown():
    supervisor = RiskSupervisor(
        {
            "risk": {
                "max_daily_loss": 100.0,
                "max_symbol_exposure": 0.35,
                "max_total_exposure": 1.2,
                "supervisor": {
                    "enabled": True,
                    "max_drawdown_pct": 8.0,
                    "max_spread_bps": 20.0,
                },
            }
        }
    )
    supervisor.halt_until = None  # clear any halt loaded from persisted state

    decision = supervisor.allow_trade(
        symbol="EURUSDm",
        target_exposure=0.2,
        confidence=0.8,
        spread_bps=25.0,
        snapshot={"pnl_today": 0.0},
        symbol_positions=0,
        total_positions=0,
        current_symbol_exposure=0.0,
        total_exposure=0.0,
        drawdown_pct=2.0,
    )
    assert not decision.allowed
    assert "spread_bps" in decision.reason

    decision = supervisor.allow_trade(
        symbol="EURUSDm",
        target_exposure=0.2,
        confidence=0.8,
        spread_bps=5.0,
        snapshot={"pnl_today": 0.0},
        symbol_positions=0,
        total_positions=0,
        current_symbol_exposure=0.0,
        total_exposure=0.0,
        drawdown_pct=9.0,
    )
    assert not decision.allowed
    assert "drawdown_pct" in decision.reason


def test_risk_supervisor_allows_risk_reduction_even_when_halted():
    supervisor = RiskSupervisor({"risk": {"supervisor": {"enabled": True}}})
    supervisor.halt_until = supervisor._now().replace(year=supervisor._now().year + 1)

    decision = supervisor.allow_trade(
        symbol="BTCUSDm",
        target_exposure=0.05,
        confidence=0.7,
        spread_bps=5.0,
        snapshot={"pnl_today": -1000.0},
        symbol_positions=1,
        total_positions=1,
        current_symbol_exposure=0.2,
        total_exposure=0.2,
        drawdown_pct=20.0,
    )
    assert decision.allowed
    assert decision.reason == "risk_reduction"
