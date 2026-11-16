import os
import time

def run_multi(*args, **kwargs):
    from Python.backtester import run_multi as _run_multi

    return _run_multi(*args, **kwargs)


def _default_gates():
    # Sharpe is annualised; returns are over the out-of-sample holdout window, after costs.
    return {
        "max_drawdown": 0.10,
        "min_sharpe": 0.50,
        "min_return": 0.0,
        "baseline_margin": 0.0,     # candidate avg_return must exceed MA-cross baseline by this
        "require_beat_baseline": True,
        "require_out_of_sample": True,
        "score_margin": 0.30,
        "min_steps_per_symbol": 600,
        "min_pass_rate": 0.80,
        "return_margin": 0.0,
        "sharpe_margin": 0.05,
        "drawdown_margin": 0.0,
        "forward_windows": [],
        "min_forward_win_rate": 0.67,
    }


def _merge_gates(gates: dict | None):
    g = _default_gates()
    if isinstance(gates, dict):
        for k in g.keys():
            if k in gates:
                g[k] = gates[k]
    return g


def _candidate_holdout_start(candidate_dir: str) -> str | None:
    try:
        from Python.backtester import _load_model_meta

        return _load_model_meta(candidate_dir).get("holdout_start")
    except Exception:
        return None


def _call_run_multi(symbols, model_dir, period, interval, reward_weights, holdout_start):
    kwargs = {"period": period, "interval": interval, "reward_weights": reward_weights}
    if holdout_start:
        kwargs["holdout_start"] = holdout_start
    return run_multi(symbols, model_dir, **kwargs)


def _evaluate_once(
    candidate_dir: str,
    champion_dir: str | None,
    symbols: list[str],
    period: str,
    interval: str,
    reward_weights: dict | None,
    holdout_start: str | None = None,
):
    # Both models are scored on the *candidate's* unseen holdout window, so the
    # candidate never gets credit for data it trained on.
    cand = _call_run_multi(symbols, candidate_dir, period, interval, reward_weights, holdout_start)
    if cand.get("error"):
        return {"error": cand["error"], "candidate": cand, "champion": None}

    champ = None
    if champion_dir and os.path.isdir(champion_dir):
        champ = _call_run_multi(symbols, champion_dir, period, interval, reward_weights, holdout_start)
        if champ.get("error"):
            champ = None

    return {"candidate": cand, "champion": champ}


def evaluate_candidate_vs_champion(
    candidate_dir: str,
    champion_dir: str | None,
    symbols: list[str],
    period: str = "120d",
    interval: str = "5m",
    reward_weights: dict | None = None,
    gates: dict | None = None,
) -> dict:
    g = _merge_gates(gates)
    holdout_start = _candidate_holdout_start(candidate_dir)
    if g.get("require_out_of_sample") and not holdout_start:
        return {
            "wins": False,
            "passes_thresholds": False,
            "error": "candidate has no holdout_start (trained without a chronological split); retrain",
            "gates": g,
        }

    main = _evaluate_once(candidate_dir, champion_dir, symbols, period, interval, reward_weights, holdout_start)
    if main.get("error"):
        return {"wins": False, "passes_thresholds": False, "error": main["error"], "gates": g}

    cand = main["candidate"]
    champ = main["champion"]

    per_symbol = []
    pass_count = 0
    for row in cand.get("per_symbol", []):
        dd_ok = float(row.get("max_drawdown", 1.0)) <= float(g["max_drawdown"])
        sh_ok = float(row.get("sharpe", -999.0)) >= float(g["min_sharpe"])
        rt_ok = float(row.get("total_return", -999.0)) >= float(g["min_return"])
        st_ok = int(row.get("steps", 0)) >= int(g["min_steps_per_symbol"])
        passed = bool(dd_ok and sh_ok and rt_ok and st_ok)
        if passed:
            pass_count += 1

        per_symbol.append(
            {
                "symbol": row.get("symbol"),
                "score": float(row.get("score", 0.0)),
                "max_drawdown": float(row.get("max_drawdown", 1.0)),
                "sharpe": float(row.get("sharpe", 0.0)),
                "total_return": float(row.get("total_return", 0.0)),
                "steps": int(row.get("steps", 0)),
                "passes": passed,
                "checks": {
                    "dd_ok": bool(dd_ok),
                    "sharpe_ok": bool(sh_ok),
                    "return_ok": bool(rt_ok),
                    "steps_ok": bool(st_ok),
                },
            }
        )

    pass_rate = pass_count / max(1, len(per_symbol))
    base_passes = (
        float(cand.get("worst_drawdown", 1.0)) <= float(g["max_drawdown"])
        and float(cand.get("avg_sharpe", -999.0)) >= float(g["min_sharpe"])
        and float(cand.get("avg_return", -999.0)) >= float(g["min_return"])
        and pass_rate >= float(g["min_pass_rate"])
    )
    # Every symbol must beat the better of the MA-cross and buy&hold baselines.
    baseline_ret = cand.get("baseline_avg_return")
    beats_baseline = True
    baseline_checks = []
    if g.get("require_beat_baseline"):
        margin_b = float(g.get("baseline_margin", 0.0))
        rows = cand.get("per_symbol", []) or []
        for row in rows:
            bl = row.get("baselines") or {}
            refs = [float(v.get("total_return")) for v in bl.values() if isinstance(v, dict) and v.get("total_return") is not None]
            if not refs and baseline_ret is not None:
                refs = [float(baseline_ret)]
            ok = bool(refs) and float(row.get("total_return", -999.0)) > max(refs) + margin_b
            baseline_checks.append({"symbol": row.get("symbol"), "best_baseline": max(refs) if refs else None, "ok": ok})
        beats_baseline = bool(baseline_checks) and all(c["ok"] for c in baseline_checks)
    base_passes = bool(base_passes and beats_baseline)

    wins = True
    margin = float(g["score_margin"])
    win_checks = {
        "score": True,
        "return": True,
        "sharpe": True,
        "drawdown": True,
    }
    champion_missing = bool(champion_dir and os.path.isdir(champion_dir) and not champ)
    if champion_missing:
        # A champion exists but could not be scored: never promote blind.
        wins = False
        win_checks = {k: False for k in win_checks}
    if champ:
        win_checks["score"] = float(cand.get("avg_score", 0.0)) > (float(champ.get("avg_score", 0.0)) + margin)
        win_checks["return"] = float(cand.get("avg_return", -999.0)) >= (
            float(champ.get("avg_return", -999.0)) + float(g["return_margin"])
        )
        win_checks["sharpe"] = float(cand.get("avg_sharpe", -999.0)) >= (
            float(champ.get("avg_sharpe", -999.0)) + float(g["sharpe_margin"])
        )
        win_checks["drawdown"] = float(cand.get("worst_drawdown", 1.0)) <= (
            float(champ.get("worst_drawdown", 1.0)) + float(g["drawdown_margin"])
        )
        wins = all(win_checks.values())

    forward_windows = [str(x) for x in (g.get("forward_windows") or []) if str(x).strip()]
    if holdout_start:
        # Trailing windows all collapse onto the same holdout slice; they add no information.
        forward_windows = []
    forward_results = []
    if forward_windows:
        wf_wins = 0
        for wf_period in forward_windows:
            fold = _evaluate_once(candidate_dir, champion_dir, symbols, wf_period, interval, reward_weights)
            if fold.get("error"):
                forward_results.append({"period": wf_period, "error": fold["error"]})
                continue

            fc = fold["candidate"]
            fh = fold["champion"]
            fold_win = True
            if fh:
                fold_win = float(fc.get("avg_score", 0.0)) > (float(fh.get("avg_score", 0.0)) + margin)
            wf_wins += 1 if fold_win else 0

            forward_results.append(
                {
                    "period": wf_period,
                    "candidate_score": float(fc.get("avg_score", 0.0)),
                    "champion_score": float(fh.get("avg_score", 0.0)) if fh else None,
                    "wins": bool(fold_win),
                }
            )

        wf_rate = wf_wins / max(1, len(forward_windows))
        base_passes = bool(base_passes and (wf_rate >= float(g["min_forward_win_rate"])))

    return {
        "candidate": cand,
        "champion": champ,
        "wins": bool(wins),
        "passes_thresholds": bool(base_passes),
        "gates": g,
        "win_checks": win_checks,
        "per_symbol_gates": per_symbol,
        "pass_rate": float(pass_rate),
        "beats_baseline": bool(beats_baseline),
        "baseline_checks": baseline_checks,
        "champion_eval_failed": champion_missing,
        "baseline_avg_return": baseline_ret,
        "holdout_start": holdout_start,
        "forward_windows": forward_results,
        "ts": time.time(),
    }
