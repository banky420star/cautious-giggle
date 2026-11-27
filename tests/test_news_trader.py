import numpy as np

from Python import news_trader as nt


def _bars(path, t0=1_000_000.0):
    """1-minute bars: 90 quiet bars before t0, then the given closes after it."""
    pre = [100 + 0.05 * np.sin(i) for i in range(90)]
    closes = np.array(pre + list(path), float)
    t = t0 - 90 * 60 + 60 * np.arange(len(closes))
    o = np.r_[closes[0], closes[:-1]]
    h = np.maximum(o, closes) + 0.02
    l = np.minimum(o, closes) - 0.02
    return t, o, h, l, closes, t0


def test_fade_triggers_on_failed_sweep_of_highs():
    t, o, h, l, c, t0 = _bars([100.4, 101.0, 99.95, 99.7, 99.5, 99.2, 99.0, 98.9] + [98.9] * 70)
    rng = nt.pre_range(t, h, l, t0)
    en = nt.find_entry("fade", t, o, h, l, c, t0, rng, a=0.1)
    assert en is not None and en["side"] == -1 and en["sl"] > 101.0


def test_breakout_triggers_on_strong_close_beyond_range():
    t, o, h, l, c, t0 = _bars([100.5, 101.0, 101.5, 102.0] + [102.5] * 70)
    en = nt.find_entry("breakout", t, o, h, l, c, t0, nt.pre_range(t, h, l, t0), a=0.1)
    assert en is not None and en["side"] == 1


def test_no_trigger_before_release_time():
    t, o, h, l, c, t0 = _bars([100.0] * 80)
    assert nt.find_entry("fade", t, o, h, l, c, t0, nt.pre_range(t, h, l, t0), a=0.1) is None


def test_choose_needs_samples_and_edge(monkeypatch):
    monkeypatch.setattr(nt, "ALWAYS_TRADE", False)
    pb = {}
    for _ in range(nt.MIN_SAMPLES - 1):
        nt.record(pb, "XAUUSDm", "inflation", "fade", 1.0, True)
    assert nt.choose(pb, "XAUUSDm", "inflation")["play"] is None  # too few
    nt.record(pb, "XAUUSDm", "inflation", "fade", 1.0, True)
    assert nt.choose(pb, "XAUUSDm", "inflation")["play"] == "fade"
    for _ in range(nt.MIN_SAMPLES):
        nt.record(pb, "BTCUSDm", "fed", "breakout", -0.5, True)
    assert nt.choose(pb, "BTCUSDm", "fed")["play"] is None  # no edge -> stand aside


def test_always_trade_takes_best_play_at_half_size(monkeypatch):
    monkeypatch.setattr(nt, "ALWAYS_TRADE", True)
    pb = {}
    for _ in range(nt.MIN_SAMPLES):
        nt.record(pb, "BTCUSDm", "fed", "breakout", -0.05, True)
    ch = nt.choose(pb, "BTCUSDm", "fed")
    assert ch["play"] == "breakout" and ch["proven"] is False
    assert nt.choose({}, "XAUUSDm", "jobs")["play"] == "fade"


# ---- surprise play (user 2026-11-25): the model reads the release from the headlines
def test_direction_follows_the_dollar():
    from Python.news_reader import direction
    hot = {"surprise": "dollar_up", "confidence": 0.8}
    assert direction("XAUUSDm", hot, 0.6) == -1 and direction("BTCUSDm", hot, 0.6) == -1
    assert direction("EURUSDm", hot, 0.6) == -1 and direction("USDJPYm", hot, 0.6) == 1
    assert direction("USOILm", hot, 0.6) == 0                                  # no fixed dollar link
    assert direction("XAUUSDm", {"surprise": "dollar_down", "confidence": 0.8}, 0.6) == 1
    assert direction("XAUUSDm", {"surprise": "dollar_up", "confidence": 0.4}, 0.6) == 0   # not sure enough
    assert direction("XAUUSDm", {"surprise": "in_line", "confidence": 0.9}, 0.6) == 0


def test_surprise_paper_trade_wins_when_the_read_was_right():
    t, o, h, l, c, t0 = _bars([100.0, 99.8, 99.5, 99.0, 98.5, 98.0, 97.5, 97.0] + [96.5] * 70)
    r = nt.simulate_surprise("XAUUSDm", t, o, h, l, c, t0, t_read=t0 + 120, side=-1)
    assert r is not None and r["traded"] and r["r"] > 0
    r2 = nt.simulate_surprise("XAUUSDm", t, o, h, l, c, t0, t_read=t0 + 120, side=1)
    assert r2["r"] < 0


def test_reader_reads_once_per_release(tmp_path):
    import json, time
    from Python.news_reader import ReleaseReader, recent_reads
    t0 = time.time() - 300
    with open(tmp_path / "news_items.jsonl", "w") as f:
        f.write("{}\n" + json.dumps({"kind": "headline", "t": t0 + 200, "source": "CNBC",
                                     "title": "Inflation gauge accelerated to 3.8%, hotter than expected"}) + "\n")
    calls = []
    def ask(prompt):
        calls.append(prompt)
        return {"actual": "3.8%", "surprise": "dollar_up", "confidence": 0.85, "reason": "hotter than forecast"}
    rd = ReleaseReader({"read_after_min": 2}, log_dir=str(tmp_path), ask=ask)
    rel = [{"title": "USD Core PCE Price Index m/m", "forecast": "0.3%", "previous": "0.3%"}]
    v = rd.verdict(t0, rel, wait=True)
    assert v["surprise"] == "dollar_up" and "forecast 0.3%" in calls[0] and "3.8%" in calls[0]
    assert rd.verdict(t0, rel) == v and len(calls) == 1                       # cached: one model call per release
    assert recent_reads(t0 - 10, str(tmp_path))[0]["actual"] == "3.8%"       # remembered


def test_reader_waits_for_headlines(tmp_path):
    import time
    from Python.news_reader import ReleaseReader
    rd = ReleaseReader({}, log_dir=str(tmp_path), ask=lambda p: {"surprise": "dollar_up", "confidence": 1})
    assert rd.verdict(time.time() - 30, [], wait=True) is None                # too early
    assert rd.verdict(time.time() - 300, [], wait=True) is None               # no headlines yet: retry later


def test_surprise_record_and_choose_ignore_each_other():
    pb = {}
    for _ in range(nt.MIN_SAMPLES):
        nt.record(pb, "XAUUSDm", "inflation", "surprise", 1.0, True)
    assert nt.surprise_record(pb, "XAUUSDm", "inflation")["proven"] is True
    assert nt.choose(pb, "XAUUSDm", "inflation")["play"] in ("fade", None)     # price play is chosen separately
