import json

from Python.autonomy_loop import _closed_since


def _ev(ts, sym, pnl, deal, backfill=False):
    p = {"deal_id": deal, "symbol": sym, "profit": pnl}
    if backfill:
        p["backfill"] = True
    return json.dumps({"ts": ts, "event": "trade_closed", "payload": p}) + "\n"


def test_trial_counts_only_this_market_after_start_once_per_deal(tmp_path):
    log = tmp_path / "trade_events.jsonl"
    log.write_text(
        _ev("2026-11-18T10:00:00+00:00", "XAUUSDm", -40.0, 1)       # before the trial started: not its trades
        + _ev("2026-11-18T12:00:00+00:00", "XAUUSDm", 5.0, 2)
        + _ev("2026-11-18T12:30:00+00:00", "XAUUSDm", 5.0, 2)       # same deal recorded again after a restart
        + _ev("2026-11-18T13:00:00+00:00", "EURUSDm", 9.0, 3)       # other market
        + _ev("2026-11-18T13:30:00+00:00", "XAUUSDm", -2.0, 4)
        + _ev("2026-11-18T14:00:00+00:00", "XAUUSDm", 50.0, 5, backfill=True)
    )
    import datetime as dt
    start = dt.datetime(2026, 11, 18, 11, 0, tzinfo=dt.timezone.utc).timestamp()
    assert _closed_since("XAUUSDm", start, str(log)) == (2, 3.0)


def test_trial_leaves_out_stand_alone_strategy_trades(tmp_path):
    """A Gemma / meta-brain / Gold Breakout loss is not the PPO model's and must not roll it back."""
    log = tmp_path / "trade_events.jsonl"
    ev = lambda deal, ticket, pnl: json.dumps({"ts": "2026-11-24T05:31:47+00:00", "event": "trade_closed",  # noqa: E731
                                               "payload": {"deal_id": deal, "ticket": ticket, "symbol": "USDJPYm",
                                                           "profit": pnl}}) + "\n"
    log.write_text(ev(1, 39607748, -86.72) + ev(2, 111, 4.0) + ev(3, 222, -1.0) + ev(4, 333, -9.0))
    (tmp_path / "gemma_trader.jsonl").write_text(json.dumps({"event": "open", "symbol": "USDJPYm", "ticket": "39607748"}) + "\n")
    (tmp_path / "meta_trader.jsonl").write_text(json.dumps({"event": "open", "ticket": "222"}) + "\n")
    (tmp_path / "gold_breakout.jsonl").write_text(json.dumps({"event": "range", "day": "x"}) + "\n"
                                                  + json.dumps({"event": "open", "ticket": 333}) + "\n")
    assert _closed_since("USDJPYm", 0, str(log)) == (1, 4.0)


def test_setup_gate_new_era_restarts_scores(tmp_path, monkeypatch):
    import json
    monkeypatch.setenv("AGI_LOG_DIR", str(tmp_path))
    from Python.setup_gate import SetupGate
    g = SetupGate({"min_trades_block": 2}, account_id="1")
    for _ in range(3):
        g.add_result("XAUUSDm|SELL", -5.0)
    g.save()
    assert g.verdict("XAUUSDm|SELL")[0] in ("block", "retest")
    g2 = SetupGate({"min_trades_block": 2, "era": "new rules"}, account_id="1")
    assert g2.stats("XAUUSDm|SELL")["n"] == 0 and g2.verdict("XAUUSDm|SELL")[0] == "probe"
    assert json.loads((tmp_path / "setups_1.before-new-rules.json").read_text())["stats"]
    g3 = SetupGate({"min_trades_block": 2, "era": "new rules"}, account_id="1")
    assert g3.state["era"] == "new rules"
