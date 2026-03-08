from Python.model_evaluator import evaluate_candidate_vs_champion


def test_candidate_must_win_multiple_metrics(monkeypatch):
    def fake_run_multi(symbols, model_dir, period="120d", interval="5m", reward_weights=None):
        if model_dir == "candidate":
            return {
                "avg_score": 10.5,
                "avg_return": 0.02,
                "worst_drawdown": 0.09,
                "avg_sharpe": 0.34,
                "per_symbol": [
                    {
                        "symbol": symbols[0],
                        "score": 10.5,
                        "max_drawdown": 0.09,
                        "sharpe": 0.34,
                        "total_return": 0.02,
                        "steps": 700,
                    }
                ],
            }
        return {
            "avg_score": 10.0,
            "avg_return": 0.019,
            "worst_drawdown": 0.095,
            "avg_sharpe": 0.33,
            "per_symbol": [
                {
                    "symbol": symbols[0],
                    "score": 10.0,
                    "max_drawdown": 0.095,
                    "sharpe": 0.33,
                    "total_return": 0.019,
                    "steps": 700,
                }
            ],
        }

    monkeypatch.setattr("Python.model_evaluator.run_multi", fake_run_multi)
    monkeypatch.setattr("Python.model_evaluator.os.path.isdir", lambda _: True)

    gates = {
        "max_drawdown": 0.10,
        "min_sharpe": 0.30,
        "min_return": 0.015,
        "score_margin": 0.30,
        "min_steps_per_symbol": 600,
        "min_pass_rate": 0.80,
        "return_margin": 0.0,
        "sharpe_margin": 0.05,
        "drawdown_margin": 0.0,
        "forward_windows": [],
        "min_forward_win_rate": 0.67,
    }

    report = evaluate_candidate_vs_champion(
        candidate_dir="candidate",
        champion_dir="champion",
        symbols=["EURUSDm"],
        gates=gates,
    )

    assert report["passes_thresholds"] is True
    assert report["wins"] is False
    assert report["win_checks"]["score"] is True
    assert report["win_checks"]["sharpe"] is False
