from Python.model_evaluator import _default_gates


def test_default_gates_are_strict():
    gates = _default_gates()
    assert gates["max_drawdown"] <= 0.10
    assert gates["min_sharpe"] >= 0.30
    assert gates["min_steps_per_symbol"] >= 600
    assert gates["sharpe_margin"] >= 0.05
