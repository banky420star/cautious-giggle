import json
import os
import subprocess
import sys
from pathlib import Path

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.model_evaluator import evaluate_candidate_vs_champion
from Python.model_registry import ModelRegistry
from Python.config_utils import load_project_config


def _latest_candidate(reg: ModelRegistry, symbol: str | None = None):
    root = reg.candidates_dir
    dirs = [os.path.join(root, d) for d in os.listdir(root) if os.path.isdir(os.path.join(root, d))]
    if not dirs:
        return None
    dirs.sort(key=lambda p: os.path.getmtime(p), reverse=True)
    if not symbol:
        return dirs[0]

    safe = symbol.replace("/", "_")
    for d in dirs:
        scorecard = os.path.join(d, "scorecard.json")
        if not os.path.exists(scorecard):
            continue
        try:
            with open(scorecard, "r", encoding="utf-8") as f:
                meta = json.load(f) or {}
            if str(meta.get("symbol", "")).replace("/", "_") == safe:
                return d
        except Exception:
            continue
    return None


def _gates_from_cfg(cfg: dict) -> dict:
    ev = cfg.get("evaluation", {}) if isinstance(cfg, dict) else {}
    return {
        "max_drawdown": float(ev.get("max_drawdown", 0.10)),
        "min_sharpe": float(ev.get("min_sharpe", 0.30)),
        "min_return": float(ev.get("min_return", ev.get("min_expected_payoff", 0.015))),
        "score_margin": float(ev.get("score_margin", 0.30)),
        "min_steps_per_symbol": int(ev.get("min_steps_per_symbol", 600)),
        "min_pass_rate": float(ev.get("min_pass_rate", 0.80)),
        "return_margin": float(ev.get("return_margin", 0.0)),
        "sharpe_margin": float(ev.get("sharpe_margin", 0.05)),
        "drawdown_margin": float(ev.get("drawdown_margin", 0.0)),
        "forward_windows": ev.get("forward_windows", []),
        "min_forward_win_rate": float(ev.get("min_forward_win_rate", 0.67)),
    }


def _write_cycle_report(report: dict, name: str = "champion_cycle_last_report.json"):
    path = Path(PROJECT_ROOT) / "logs" / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")


def _run_train_drl(symbol: str | None = None):
    env = os.environ.copy()
    if symbol:
        env["AGI_DRL_SYMBOL"] = symbol
    subprocess.check_call([sys.executable, "training/train_drl.py"], cwd=PROJECT_ROOT, env=env)


def main():
    cfg = load_project_config(PROJECT_ROOT, live_mode=False)

    trading_cfg = cfg.get("trading", {})
    drl_cfg = cfg.get("drl", {})

    symbols = trading_cfg.get("symbols", ["EURUSDm", "GBPUSDm"])
    eval_period = str(drl_cfg.get("eval_period", "120d"))
    eval_interval = str(drl_cfg.get("interval", trading_cfg.get("timeframe", "M5")))
    reward_cfg = drl_cfg.get("reward", {}) if isinstance(drl_cfg.get("reward", {}), dict) else {}
    reward_weights = reward_cfg.get("weights", {}) if isinstance(reward_cfg.get("weights", {}), dict) else {}

    per_symbol = bool(drl_cfg.get("per_symbol", True))
    gates = _gates_from_cfg(cfg)

    logger.info("Cycle start: train LSTM per symbol")
    subprocess.check_call([sys.executable, "training/train_lstm.py"], cwd=PROJECT_ROOT)

    reg = ModelRegistry()
    cycle_report = {
        "mode": "per_symbol" if per_symbol else "global",
        "symbols": [],
        "eval_period": eval_period,
        "eval_interval": eval_interval,
        "gates": gates,
    }

    if per_symbol:
        for symbol in symbols:
            logger.info(f"Cycle step: train PPO candidate for {symbol}")
            _run_train_drl(symbol=symbol)

            candidate = _latest_candidate(reg, symbol=symbol)
            if not candidate:
                raise RuntimeError(f"No PPO candidate found after training for {symbol}")

            champion = reg.load_active_model(prefer_canary=False, symbol=symbol)
            logger.info(f"Evaluate {symbol} candidate={os.path.basename(candidate)} vs champion={champion}")
            report = evaluate_candidate_vs_champion(
                candidate,
                champion,
                symbols=[symbol],
                period=eval_period,
                interval=eval_interval,
                reward_weights=reward_weights,
                gates=gates,
            )

            if report.get("error"):
                raise RuntimeError(f"Evaluator error on {symbol}: {report['error']}")

            wins = bool(report.get("wins"))
            passes = bool(report.get("passes_thresholds"))
            logger.info(f"{symbol} | wins={wins} passes_thresholds={passes} pass_rate={report.get('pass_rate', 0):.2f}")

            if wins and passes:
                reg.set_canary(candidate, symbol=symbol)
                logger.success(f"Canary set for {symbol}: {candidate}")
            else:
                logger.warning(f"Candidate blocked for {symbol}; champion unchanged")

            cycle_report["symbols"].append(
                {
                    "symbol": symbol,
                    "candidate": candidate,
                    "champion": champion,
                    "wins": wins,
                    "passes_thresholds": passes,
                    "pass_rate": report.get("pass_rate"),
                    "per_symbol_gates": report.get("per_symbol_gates", []),
                    "forward_windows": report.get("forward_windows", []),
                }
            )
    else:
        logger.info("Cycle step: train PPO candidate")
        _run_train_drl(symbol=None)

        candidate = _latest_candidate(reg)
        if not candidate:
            raise RuntimeError("No PPO candidate found after training")

        champion = reg.load_active_model(prefer_canary=False)
        logger.info(f"Cycle step: evaluate candidate={os.path.basename(candidate)} vs champion={champion}")
        report = evaluate_candidate_vs_champion(
            candidate,
            champion,
            symbols=symbols,
            period=eval_period,
            interval=eval_interval,
            reward_weights=reward_weights,
            gates=gates,
        )

        if report.get("error"):
            raise RuntimeError(f"Evaluator error: {report['error']}")

        wins = bool(report.get("wins"))
        passes = bool(report.get("passes_thresholds"))
        logger.info(f"Evaluation result | wins={wins} passes_thresholds={passes} pass_rate={report.get('pass_rate', 0):.2f}")

        if wins and passes:
            reg.set_canary(candidate)
            logger.success(f"Canary set to {candidate}")
        else:
            logger.warning("Candidate blocked by gates; champion unchanged")

        cycle_report["symbols"].append(
            {
                "symbol": "GLOBAL",
                "candidate": candidate,
                "champion": champion,
                "wins": wins,
                "passes_thresholds": passes,
                "pass_rate": report.get("pass_rate"),
                "per_symbol_gates": report.get("per_symbol_gates", []),
                "forward_windows": report.get("forward_windows", []),
            }
        )

    _write_cycle_report(cycle_report)


if __name__ == "__main__":
    main()
