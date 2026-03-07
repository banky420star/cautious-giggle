import os
import subprocess
import sys

import yaml
from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry


def _latest_candidate(reg: ModelRegistry):
    root = reg.candidates_dir
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    return dirs[0]


def main():
    cfg_path = os.path.join(PROJECT_ROOT, "config.yaml")
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f) or {}

    symbols = cfg.get("trading", {}).get("symbols", ["EURUSDm", "GBPUSDm"])
    eval_period = cfg.get("drl", {}).get("eval_period", "120d")

    logger.info("Cycle start: train LSTM per symbol")
    subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT)

    logger.info("Cycle step: train PPO candidate")
    subprocess.check_call([sys.executable, "training/train_drl.py"], cwd=PROJECT_ROOT)

    reg = ModelRegistry()
    candidate = _latest_candidate(reg)
    if not candidate:
        raise RuntimeError("No PPO candidate found after training")

    champion = reg._read_active().get("champion")
    logger.info(f"Cycle step: evaluate candidate={os.path.basename(candidate)} vs champion={champion}")
    report = evaluate_candidate_vs_champion(candidate, champion, symbols=symbols, period=eval_period)

    if report.get("error"):
        raise RuntimeError(f"Evaluator error: {report['error']}")

    wins = bool(report.get("wins"))
    passes = bool(report.get("passes_thresholds"))
    logger.info(f"Evaluation result | wins={wins} passes_thresholds={passes}")

    if wins and passes:
        reg.set_canary(candidate)
        logger.success(f"Canary set to {candidate}")
    else:
        logger.warning("Candidate blocked by gates; champion unchanged")


if __name__ == "__main__":
    main()

