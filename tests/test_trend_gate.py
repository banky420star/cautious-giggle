from Python.trend_gate import TrendGate, ema_trend, verdict


def test_ema_trend_up_down_and_short_history():
    up = [100 + i * 0.1 for i in range(200)]
    assert ema_trend(up, 50) == 1
    assert ema_trend(list(reversed(up)), 50) == -1
    assert ema_trend(up[:60], 50) == 0          # not enough bars: no view


def test_blocks_only_when_every_timeframe_is_against():
    assert verdict(1, {"4h": -1, "1h": -1})[0] == "block"
    assert verdict(-1, {"4h": 1, "1h": 1})[0] == "block"
    assert verdict(1, {"4h": -1, "1h": 1})[0] == "pass"      # mixed
    assert verdict(1, {"4h": 1, "1h": 1})[0] == "pass"       # with the trend
    assert verdict(-1, {"4h": -1, "1h": -1})[0] == "pass"
    assert verdict(1, {"4h": 0, "1h": -1})[0] == "pass"      # unknown -> no gate


class _FakeMT5:
    TIMEFRAME_H4, TIMEFRAME_H1 = 4, 1

    def __init__(self, closes):
        self.closes = closes

    def copy_rates_from_pos(self, symbol, tf, start, n):
        import numpy as np
        return np.array([(c,) for c in self.closes[tf][-n:]], dtype=[("close", "f8")])


def test_gate_reads_bars_and_can_be_switched_off():
    down = [100 - i * 0.1 for i in range(300)]
    g = TrendGate({}, _FakeMT5({4: down, 1: down}))
    assert g.check("XAUUSDm", 1)[0] == "block"
    assert g.check("XAUUSDm", -1)[0] == "pass"
    assert TrendGate({"enabled": False}, _FakeMT5({4: down, 1: down})).check("XAUUSDm", 1)[0] == "pass"


def test_bars_unavailable_never_blocks():
    class Broken(_FakeMT5):
        def copy_rates_from_pos(self, *a):
            raise RuntimeError("no feed")
    assert TrendGate({}, Broken({})).check("XAUUSDm", 1)[0] == "pass"
