import datetime as dt

from Python.daily_jobs import scout_due


def _t(h, m, day=24):
    return dt.datetime(2026, 11, day, h, m, tzinfo=dt.timezone.utc).timestamp()


def test_scout_runs_once_a_day_after_0715_utc():
    assert not scout_due(_t(7, 10), None, None)                                   # too early
    assert scout_due(_t(7, 15), None, None)                                       # due
    assert scout_due(_t(12, 0), "2026-11-23T07:20:00Z", "2026-11-23")             # yesterday's run: due again
    assert not scout_due(_t(12, 0), "2026-11-24T07:20:00Z", "2026-11-24")         # already ran today
    assert not scout_due(_t(12, 0), "2026-11-23T07:20:00Z", "2026-11-24")         # started today (crashed): wait
    assert scout_due(_t(7, 16, day=25), "2026-11-24T07:20:00Z", "2026-11-24")     # next day
