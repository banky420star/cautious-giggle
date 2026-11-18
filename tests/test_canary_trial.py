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
