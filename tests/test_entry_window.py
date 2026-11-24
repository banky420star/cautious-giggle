import datetime as dt

from Python.entry_window import EntryWindow, market_opens, moves


def _t(s):
    return dt.datetime.fromisoformat(s).timestamp()


def test_news_windows_by_impact_and_market():
    t0 = _t("2026-11-25T13:30:00+00:00")                  # Wednesday, USD PCE
    cal = [{"t": t0, "title": "USD Core PCE", "impact": "High", "country": "USD"},
           {"t": t0 + 7200, "title": "AUD CPI", "impact": "High", "country": "AUD"},
           {"t": t0 - 7200, "title": "EUR ZEW", "impact": "Medium", "country": "EUR"}]
    w = EntryWindow({"open_before_min": 0, "open_after_min": 0}, calendar=cal)
    assert w.reason("BTCUSDm", t0 - 29 * 60) and w.reason("BTCUSDm", t0 + 29 * 60)
    assert w.reason("BTCUSDm", t0 - 31 * 60) is None
    assert w.reason("XAUUSDm", t0 + 7200) is None                         # AUD news does not move gold
    assert w.reason("EURUSDm", t0 - 7200 + 300) and w.reason("XAUUSDm", t0 - 7200 + 300) is None
    assert w.reason("EURUSDm", t0 - 7200 + 11 * 60) is None              # Medium: 10 min only


def test_market_opens_follow_daylight_saving():
    # November: London = UTC, New York = UTC-5 -> 08:00 and 14:30 UTC
    opens = dict((n, t) for n, t in market_opens(_t("2026-11-24T00:00:00+00:00"), 24))
    assert opens["London open"] == _t("2026-11-24T08:00:00+00:00")
    assert opens["New York stock open"] == _t("2026-11-24T14:30:00+00:00")
    # July: 07:00 and 13:30 UTC
    opens = dict((n, t) for n, t in market_opens(_t("2026-07-14T00:00:00+00:00"), 24))
    assert opens["London open"] == _t("2026-07-14T07:00:00+00:00")
    assert opens["New York stock open"] == _t("2026-07-14T13:30:00+00:00")
    sat = _t("2026-11-28T00:00:00+00:00")
    assert [t for _, t in market_opens(sat, 24) if t > sat] == []    # Saturday: nothing ahead


def test_open_window_and_off_switch():
    w = EntryWindow({}, calendar=[])
    assert "New York stock open" in w.reason("XAUUSDm", _t("2026-11-24T14:45:00+00:00"))
    assert w.reason("XAUUSDm", _t("2026-11-24T15:05:00+00:00")) is None
    assert w.reason("XAUUSDm", _t("2026-11-24T07:50:00+00:00")).startswith("London open in 10")
    assert EntryWindow({"enabled": False}, calendar=[]).reason("XAUUSDm", _t("2026-11-24T14:45:00+00:00")) is None


def test_oil_news_moves_crude():
    e = {"title": "USD Crude Oil Inventories", "country": "USD"}
    assert moves(e, "USOILm") and moves(e, "XAUUSDm")
    assert not moves({"title": "JPY BoJ", "country": "JPY"}, "XAUUSDm") and moves({"title": "x", "country": "JPY"}, "USDJPYm")
