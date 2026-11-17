import importlib
import sys


def _grad(tmp_path, monkeypatch):
    sys.path.insert(0, "tools")
    g = importlib.import_module("graduation")
    monkeypatch.setattr(g, "STATE", str(tmp_path / "graduation.json"))
    return g


def test_plan_queues_new_and_failed_symbols(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    cfg = {"candidates": ["EURUSDm", "USDJPYm", "USOILm"], "max_attempts": 3}
    st = {"symbols": {"USDJPYm": {"status": "exam_failed", "attempts": 1, "history": []},
                      "USOILm": {"status": "exam_failed", "attempts": 3, "history": []}}}
    first, again = g.plan(st, cfg)
    assert first == ["EURUSDm"] and again == ["USDJPYm"]  # oil used all its attempts


def test_graduated_and_retraining_live_symbols_keep_trading(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    g.save_state({"symbols": {"EURUSDm": {"status": "graduated"},
                              "USDJPYm": {"status": "training", "was_live": True},
                              "XAGUSDm": {"status": "training"},
                              "USOILm": {"status": "needs_work"}}})
    assert sorted(g.graduated_symbols()) == ["EURUSDm", "USDJPYm"]


def _cfg():
    return {"candidates": ["EURUSDm"], "live": ["BTCUSDm"], "max_attempts": 3, "base_steps": 1000,
            "step_growth": 2.0, "scenarios": ["balanced", "patient", "hunter"],
            "scenario_about": {"balanced": "", "patient": "", "hunter": ""}}


def test_failed_attempts_rotate_scenarios_then_grow_steps(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    monkeypatch.setattr(g, "exam_passed", lambda s: (False, "no model passed"))
    monkeypatch.setattr(g, "last_exam", lambda s: {})
    monkeypatch.setattr(g, "log", lambda m: None)
    cfg = _cfg()
    st = {"symbols": {"EURUSDm": {"status": "queued", "attempts": 0, "history": []}}}
    seen = []
    for _ in range(3):
        v = st["symbols"]["EURUSDm"]
        scen, steps = g.scenario_for(v, cfg)
        seen.append((scen, steps))
        v.update(status="training", attempts=v["attempts"] + 1, scenario=scen, steps=steps)
        g.finish(st, "EURUSDm", cfg, 0, [])
    assert seen == [("balanced", 1000), ("patient", 1000), ("hunter", 1000)]
    assert st["symbols"]["EURUSDm"]["status"] == "needs_work"
    assert g.scenario_for(st["symbols"]["EURUSDm"], cfg) == ("balanced", 2000)  # next round trains longer


def test_passing_scenario_is_remembered_for_retrains(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    monkeypatch.setattr(g, "exam_passed", lambda s: (True, "canary"))
    monkeypatch.setattr(g, "last_exam", lambda s: {})
    monkeypatch.setattr(g, "log", lambda m: None)
    cfg = _cfg()
    st = {"symbols": {"EURUSDm": {"status": "training", "attempts": 2, "scenario": "patient", "scenario_idx": 1,
                                  "steps": 1000, "history": []}}}
    newly = []
    g.finish(st, "EURUSDm", cfg, 0, newly)
    v = st["symbols"]["EURUSDm"]
    assert newly == ["EURUSDm"] and v["status"] == "graduated" and v["best_scenario"] == "patient"
    assert g.scenario_for(v, cfg)[0] == "patient"


def test_live_market_failing_retrain_keeps_trading_and_tries_next(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    monkeypatch.setattr(g, "exam_passed", lambda s: (False, "no model passed"))
    monkeypatch.setattr(g, "last_exam", lambda s: {})
    monkeypatch.setattr(g, "log", lambda m: None)
    cfg = _cfg()
    st = {"symbols": {"BTCUSDm": {"status": "training", "attempts": 1, "scenario": "balanced", "was_live": True,
                                  "history": []}}}
    g.finish(st, "BTCUSDm", cfg, 0, [])
    assert st["symbols"]["BTCUSDm"]["status"] == "retrain_queued"
    assert g.scenario_for(st["symbols"]["BTCUSDm"], cfg)[0] == "patient"


def test_interrupted_attempt_is_retried_with_same_scenario(tmp_path, monkeypatch):
    g = _grad(tmp_path, monkeypatch)
    monkeypatch.setattr(g, "log", lambda m: None)
    cfg = _cfg()
    v = {"status": "training", "attempts": 2, "scenario": "patient", "scenario_idx": 1, "was_live": True, "history": []}
    g.requeue_interrupted(v, cfg)
    assert v["status"] == "retrain_queued" and v["attempts"] == 1
    assert g.scenario_for(v, cfg)[0] == "patient"
