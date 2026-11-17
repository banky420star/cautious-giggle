"""Graduation: train new markets until they are winners, then let them trade; retrain when needed.

    candidate --train--> exam --pass--> GRADUATED (trades like the other markets)
                           \\--fail--> train again with more PPO steps (up to max_attempts)
                                     --> still failing: 'needs work' (news trading only)
    graduated --live results slip--> retrain (the current model keeps trading meanwhile)

"Winner" means the trained PPO trader passed the same exam as every model: replayed on the last
15% of its history, which it never saw, after costs, it must make money with a Sharpe >= the
configured minimum, a drawdown under the limit, and beat the moving-average baseline.

    python tools/graduation.py run       # loop: waits for any running cycle, trains, updates state
    python tools/graduation.py status

State: logs/graduation.json (read by the bot and the dashboards).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
STATE = os.path.join(ROOT, "logs", "graduation.json")
LOG = os.path.join(ROOT, "logs", "graduation.log")
PY = os.path.join(ROOT, ".venv", "bin", "python")


def log(msg: str):
    line = f"{dt.datetime.now():%Y-%m-%d %H:%M:%S} {msg}"
    print(line, flush=True)
    with open(LOG, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def settings() -> dict:
    from Python.config_utils import load_project_config

    cfg = load_project_config(ROOT, live_mode=False)
    g = ((cfg.get("trading") or {}).get("graduation") or {})
    return {
        "enabled": bool(g.get("enabled", False)),
        "candidates": [str(s) for s in (g.get("candidates") or [])],
        "max_attempts": int(g.get("max_attempts", 3)),
        "base_steps": int((cfg.get("drl") or {}).get("total_timesteps", 500_000)),
        "step_growth": float(g.get("step_growth", 1.5)),
        "retrain_min_trades": int((g.get("retrain_when") or {}).get("min_trades", 20)),
        "retrain_max_win_rate": float((g.get("retrain_when") or {}).get("win_rate_below", 0.40)),
        "retrain_cooldown_h": float((g.get("retrain_when") or {}).get("cooldown_hours", 24)),
        "live": [str(s) for s in ((cfg.get("trading") or {}).get("symbols") or [])],
        "scenarios": [str(x) for x in ((cfg.get("drl") or {}).get("scenario_order")
                                        or list(((cfg.get("drl") or {}).get("reward_scenarios") or {}).keys()))] or ["default"],
        "scenario_about": {k: str((v or {}).get("about", "")) for k, v in
                           (((cfg.get("drl") or {}).get("reward_scenarios") or {}).items())},
    }


def load_state() -> dict:
    try:
        with open(STATE, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {"symbols": {}}


def save_state(st: dict):
    st["updated"] = time.time()
    tmp = STATE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(st, f, indent=1)
    os.replace(tmp, STATE)


def graduated_symbols() -> list[str]:
    """Used by the bot: candidates that passed the exam."""
    return [s for s, v in (load_state().get("symbols") or {}).items()
            if v.get("status") == "graduated" or (v.get("status") in ("training", "retrain_queued") and v.get("was_live"))]


def exam_passed(symbol: str) -> tuple[bool, str]:
    from Python import demo_exploration
    from Python.model_registry import ModelRegistry

    act = ModelRegistry().get_symbol_active(symbol) or {}
    if act.get("champion"):
        return True, "champion"
    can = act.get("canary")
    if can and not demo_exploration.is_exploration(can):
        return True, "canary"
    return False, "exploration only" if can else "no model passed"


def last_exam(symbol: str) -> dict:
    """Headline numbers of the symbol's latest exam, from the cycle report."""
    p = os.path.join(ROOT, "logs", "champion_cycle_last_report.json")
    try:
        with open(p, encoding="utf-8") as f:
            rep = json.load(f)
        for row in rep.get("symbols", []):
            if row.get("symbol") == symbol:
                return {"beat_baseline": bool(row.get("wins")), "passed_gates": bool(row.get("passes_thresholds")),
                        "pass_rate": row.get("pass_rate"), "gates": (row.get("per_symbol_gates") or [])[:3],
                        "error": row.get("error")}
    except Exception:
        pass
    return {}


def cycle_running() -> bool:
    return subprocess.run(["pgrep", "-f", "tools/champion_cycle.py"], capture_output=True).returncode == 0


def run_cycle(symbols: list[str], steps: int, retrain: bool, scenario: str = "default") -> int:
    env = dict(os.environ, AGI_CYCLE_SYMBOLS=",".join(symbols), AGI_DRL_TIMESTEPS=str(steps),
               PYTHONPATH=ROOT, AGI_CYCLE_SKIP_COMPLETED="true")
    if scenario and scenario != "default":
        env["AGI_DRL_SCENARIO"] = scenario
    if retrain:  # pattern models already exist; another PPO attempt is what is needed
        env.update(AGI_CYCLE_SKIP_LSTM="true", AGI_CYCLE_SKIP_DREAMER="true")
    log(f"training {symbols} | reward scenario '{scenario}' | PPO steps {steps:,} | "
        f"{'retrain' if retrain else 'first training'}")
    with open(os.path.join(ROOT, "logs", "champion_cycle_run.log"), "a", encoding="utf-8") as out:
        return subprocess.call(["caffeinate", "-is", PY, "tools/champion_cycle.py"], cwd=ROOT, env=env,
                               stdout=out, stderr=subprocess.STDOUT)


def live_results(symbol: str, days: int = 7) -> dict:
    from Python.mt5_compat import mt5

    mt5.initialize()
    now = dt.datetime.now(dt.timezone.utc)
    deals = mt5.history_deals_get(now - dt.timedelta(days=days), now) or []
    out_entry = int(getattr(mt5, "DEAL_ENTRY_OUT", 1))
    closes = [float(d.profit) for d in deals if int(getattr(d, "entry", -1)) == out_entry
              and str(getattr(d, "symbol", "")).rstrip("m") == symbol.rstrip("m")]
    n = len(closes)
    return {"n": n, "pnl": sum(closes), "win": (sum(1 for x in closes if x > 0) / n) if n else 0.0}


def restart_bot():
    log("restarting the bot so graduated markets start trading")
    subprocess.call(["scripts/mac/stop_bot.sh"], cwd=ROOT, stdout=subprocess.DEVNULL)
    time.sleep(3)
    try:
        os.remove(os.path.join(ROOT, ".tmp", "stop"))
    except FileNotFoundError:
        pass
    subprocess.call(["scripts/mac/start_bot.sh"], cwd=ROOT, stdout=subprocess.DEVNULL)


def scenario_for(v: dict, cfg: dict) -> tuple[str, int]:
    """Reward scenario and PPO step budget for this symbol's next attempt."""
    sc = cfg["scenarios"]
    idx = int(v.get("scenario_idx", 0))
    rounds = idx // len(sc)          # every full pass through the scenarios adds x step_growth practice
    return sc[idx % len(sc)], int(cfg["base_steps"] * cfg["step_growth"] ** rounds)


def plan(st: dict, cfg: dict) -> tuple[list[str], list[str]]:
    """(first-time trainees, retrain trainees) for the next cycle."""
    first, again = [], []
    for s in cfg.get("live", []):
        v = st["symbols"].get(s) or {}
        if v.get("status") == "retrain_queued":
            again.append(s)
    for s in cfg["candidates"]:
        v = st["symbols"].setdefault(s, {"status": "queued", "attempts": 0, "history": []})
        if v["status"] in ("queued",):
            first.append(s)
        elif v["status"] == "exam_failed" and v["attempts"] < cfg["max_attempts"]:
            again.append(s)
        elif v["status"] == "retrain_queued":
            again.append(s)
        elif v["status"] == "needs_work" and time.time() - float(v.get("last_retrain", 0)) > 24 * 3600:
            v["attempts"] = 0          # a new round through the scenarios, now with more PPO steps
            again.append(s)
    return first, again


def check_live(st: dict, cfg: dict):
    """Queue a retrain for any trading market whose live results slipped."""
    for s in cfg["live"] + [k for k, v in st["symbols"].items() if v.get("status") == "graduated"]:
        v = st["symbols"].setdefault(s, {"status": "graduated" if s not in cfg["live"] else "live", "attempts": 0,
                                         "history": []})
        if v.get("status") not in ("graduated", "live"):
            continue
        if time.time() - float(v.get("last_retrain", 0)) < cfg["retrain_cooldown_h"] * 3600:
            continue
        try:
            r = live_results(s)
        except Exception as exc:
            log(f"live check failed for {s}: {exc}")
            continue
        v["live"] = {k: round(x, 3) for k, x in r.items()}
        if r["n"] >= cfg["retrain_min_trades"] and r["pnl"] < 0 and r["win"] < cfg["retrain_max_win_rate"]:
            v["status"] = "retrain_queued"
            v["was_live"] = True
            v["why"] = (f"last 7 days: {r['n']} trades, {r['win']:.0%} wins, ${r['pnl']:+.2f}: slipping, so it trains again "
                        f"(the current model keeps trading meanwhile)")
            log(f"{s}: {v['why']}")


def finish(st: dict, s: str, cfg: dict, rc: int | None, newly: list):
    """Record the result of a finished attempt and decide what comes next."""
    v = st["symbols"][s]
    scen, steps = v.get("scenario", "default"), v.get("steps")
    ok, how = exam_passed(s)
    v["history"].append({"t": time.time(), "attempt": v["attempts"], "scenario": scen, "steps": steps,
                         "passed": ok, "how": how, "rc": rc, "exam": last_exam(s)})
    v["last_retrain"] = time.time()
    v["scenario_idx"] = int(v.get("scenario_idx", 0)) + 1      # next attempt tries the next scenario
    live_sym = s in cfg["live"]
    if ok:
        if not live_sym and v.get("status") != "graduated":
            newly.append(s)
        v["status"] = "live" if live_sym else "graduated"
        v["best_scenario"] = scen
        v["why"] = f"passed the exam on attempt {v['attempts']} with the '{scen}' reward"
    elif v["attempts"] < cfg["max_attempts"]:
        nxt, _ = scenario_for(v, cfg)
        v["status"] = "retrain_queued" if (live_sym or v.get("was_live")) else "exam_failed"
        v["why"] = (f"failed the exam with the '{scen}' reward (attempt {v['attempts']}/{cfg['max_attempts']}); "
                    f"next: '{nxt}' — {cfg['scenario_about'].get(nxt, '')}")
    elif live_sym or v.get("was_live"):
        v["status"] = "live" if live_sym else "graduated"
        v["why"] = f"tried {v['attempts']} reward scenarios without beating the exam; keeps the model it had"
    else:
        v["status"] = "needs_work"
        v["why"] = (f"failed the exam with all {v['attempts']} reward scenarios: trades news only; the next "
                    f"round trains longer")
    if v["status"] in ("graduated", "live"):
        v["attempts"] = 0
        v.pop("was_live", None)
        if v.get("best_scenario") in cfg["scenarios"]:   # retrains start from what worked
            v["scenario_idx"] = cfg["scenarios"].index(v["best_scenario"])
    log(f"{s}: {v['status']} — {v['why']}")


def run():
    log("graduation manager started")
    while True:
        cfg = settings()
        if not cfg["enabled"]:
            time.sleep(600)
            continue
        if cycle_running():
            time.sleep(60)
            continue
        st = load_state()
        st.setdefault("symbols", {})
        # a manager restart can leave attempts marked 'training' after their cycle ended: book them
        stale = [s for s, v in st["symbols"].items() if v.get("status") == "training"]
        newly: list[str] = []
        for s in stale:
            finish(st, s, cfg, None, newly)
        check_live(st, cfg)
        first, again = plan(st, cfg)
        save_state(st)
        if newly:
            restart_bot()
            newly = []
        if not first and not again:
            time.sleep(1800)
            continue
        # markets that already trade go first, then the newcomers; one cycle per reward scenario
        groups: dict[tuple, list[str]] = {}
        for syms, retrain in ((again, True), (first, False)):
            for s in syms:
                scen, steps = scenario_for(st["symbols"][s], cfg)
                groups.setdefault((retrain, scen, steps), []).append(s)
        for (retrain, scen, steps), syms in sorted(groups.items(), key=lambda kv: not kv[0][0]):
            for s in syms:
                v = st["symbols"][s]
                v.update(status="training", attempts=v["attempts"] + 1, steps=steps, scenario=scen, started=time.time(),
                         was_live=bool(v.get("was_live")) or s in cfg["live"])
                v["why"] = f"training with the '{scen}' reward — {cfg['scenario_about'].get(scen, '')}"
            save_state(st)
            rc = run_cycle(syms, steps, retrain, scen)
            for s in syms:
                finish(st, s, cfg, rc, newly)
            save_state(st)
        if newly:
            restart_bot()


def status():
    st = load_state()
    for s, v in (st.get("symbols") or {}).items():
        print(f"{s:9s} {v.get('status', '?'):15s} attempt {v.get('attempts', 0)} · {v.get('scenario', '-'):11s} {v.get('why', '')}")


if __name__ == "__main__":
    {"run": run, "status": status}.get(sys.argv[1] if len(sys.argv) > 1 else "status", status)()
