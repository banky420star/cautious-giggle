from Python import trade_scorecard as tc


def _entry(side):
    return {"ticket": 1, "symbol": "XAUUSDm", "side": side, "open_price": 100.0,
            "voters": {"a": {"name": "A", "vote": 0.8}, "b": {"name": "B", "vote": -0.6}, "c": {"name": "C", "vote": 0.0}}}


def test_backers_of_a_winner_are_right_and_opponents_wrong():
    r = tc.judge(_entry("BUY"), profit=2.5, exit_price=100.1)
    calls = {c["key"]: c for c in r["calls"]}
    assert r["outcome"] == "win" and r["move_bps"] == 10.0
    assert calls["a"]["call"] == "right" and calls["b"]["call"] == "wrong" and calls["c"]["call"] == "no view"


def test_opponents_of_a_loser_are_right():
    r = tc.judge(_entry("BUY"), profit=-1.0)
    calls = {c["key"]: c for c in r["calls"]}
    assert calls["a"]["call"] == "wrong" and calls["b"]["call"] == "right"
    assert calls["b"]["why"] == "warned against it and it lost"


def test_close_without_entry_ballot_is_skipped(tmp_path, monkeypatch):
    monkeypatch.setattr(tc, "ENTRIES", str(tmp_path / "none.jsonl"))
    assert tc.on_close({"ticket": 99, "profit": 1.0}) is None
