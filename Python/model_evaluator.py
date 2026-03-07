import os
import time

from Python.backtester import run_multi


def _default_gates():
    return {
        "max_drawdown": 0.20,
        "min_sharpe": -0.10,
        "min_return": -0.02,
        "score_margin": 0.25,
    }


def evaluate_candidate_vs_champion(
    candidate_dir: str,
    champion_dir: str | None,
    symbols: list[str],
    period="120d",
    gates: dict | None = None,
) -> dict:
    g = _default_gates()
    if isinstance(gates, dict):
        g.update({k: gates[k] for k in g.keys() if k in gates})

    cand = run_multi(symbols, candidate_dir, period=period)
    if cand.get("error"):
        return {"wins": False, "passes_thresholds": False, "error": cand["error"], "gates": g}

    champ = None
    if champion_dir and os.path.isdir(champion_dir):
        champ = run_multi(symbols, champion_dir, period=period)
        if champ.get("error"):
            champ = None

    passes = (
        float(cand.get("worst_drawdown", 1.0)) <= float(g["max_drawdown"])
        and float(cand.get("avg_sharpe", -999.0)) >= float(g["min_sharpe"])
        and float(cand.get("avg_return", -999.0)) >= float(g["min_return"])
    )

    wins = True
    margin = float(g["score_margin"])
    if champ:
        wins = float(cand.get("avg_score", 0.0)) > (float(champ.get("avg_score", 0.0)) + margin)

    per_symbol = []
    for row in cand.get("per_symbol", []):
        dd_ok = float(row.get("max_drawdown", 1.0)) <= float(g["max_drawdown"])
        sh_ok = float(row.get("sharpe", -999.0)) >= float(g["min_sharpe"])
        rt_ok = float(row.get("total_return", -999.0)) >= float(g["min_return"])
        per_symbol.append(
            {
                "symbol": row.get("symbol"),
                "score": float(row.get("score", 0.0)),
                "max_drawdown": float(row.get("max_drawdown", 1.0)),
                "sharpe": float(row.get("sharpe", 0.0)),
                "total_return": float(row.get("total_return", 0.0)),
                "passes": bool(dd_ok and sh_ok and rt_ok),
                "checks": {"dd_ok": bool(dd_ok), "sharpe_ok": bool(sh_ok), "return_ok": bool(rt_ok)},
            }
        )

    return {
        "candidate": cand,
        "champion": champ,
        "wins": bool(wins),
        "passes_thresholds": bool(passes),
        "gates": g,
        "per_symbol_gates": per_symbol,
        "ts": time.time(),
    }
