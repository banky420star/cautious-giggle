from Python.model_evaluator import _default_gates


def test_default_gates_are_reasonable():
    gates = _default_gates()
    assert gates["max_drawdown"] <= 0.25
    assert gates["min_sharpe"] >= 0.01
    assert gates["min_steps_per_symbol"] >= 600
    assert gates["sharpe_margin"] >= 0.0
    assert gates["min_pass_rate"] <= 1.0
    assert gates["min_forward_win_rate"] <= 1.0
