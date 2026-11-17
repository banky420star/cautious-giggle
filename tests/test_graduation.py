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
