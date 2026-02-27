import os
import time
from loguru import logger
from Python.backtester import run_multi

def evaluate_candidate_vs_champion(candidate_dir: str, champion_dir: str | None, symbols: list[str], period="120d") -> dict:
    cand = run_multi(symbols, candidate_dir, period=period)
    if cand.get("error"):
        return {"wins": False, "passes_thresholds": False, "error": cand["error"]}

    champ = None
    if champion_dir and os.path.isdir(champion_dir):
        champ = run_multi(symbols, champion_dir, period=period)
        if champ.get("error"):
            champ = None

    # thresholds (tune)
    passes = (
        cand["worst_drawdown"] <= 0.20 and
        cand["avg_sharpe"] >= -0.10 and
        cand["avg_return"] >= -0.02
    )

    wins = True
    if champ:
        wins = cand["avg_score"] > (champ["avg_score"] + 0.25)

    return {
        "candidate": cand,
        "champion": champ,
        "wins": bool(wins),
        "passes_thresholds": bool(passes),
        "ts": time.time(),
    }
