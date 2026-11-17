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
