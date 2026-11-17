from types import SimpleNamespace as P

import Python.Server_AGI as srv


class Ex:
    def __init__(self, longs=(), shorts=()):
        self.longs, self.shorts = list(longs), list(shorts)

    def get_positions(self, symbol):
        return self.longs, self.shorts


CFG = {"enabled": True, "max": 3, "cooldown_s": 0.0}


def test_adds_only_when_open_trades_are_in_profit():
    st = {}
    assert srv._stack_level("X", 0.5, Ex(), CFG, st) == 1                      # nothing open yet
    assert srv._stack_level("X", 0.5, Ex([P(profit=-1.0)]), CFG, st) == 1      # losing: no add
    assert srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), CFG, st) == 2       # winning: add one
    assert srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), CFG, st) == 2       # waits for the add to fill
    assert srv._stack_level("X", 0.5, Ex([P(profit=2.0), P(profit=1.0)]), CFG, st) == 3
    assert srv._stack_level("X", 0.5, Ex([P(profit=2.0)] * 3), CFG, st) == 3   # capped at max


def test_resets_on_flip_flat_or_closed_layer():
    st = {}
    srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), CFG, st)
    assert st["X"]["n"] == 2
    assert srv._stack_level("X", 0.5, Ex(), CFG, st) == 1                       # stacked trades stopped out
    srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), CFG, st)
    assert srv._stack_level("X", -0.5, Ex(shorts=[]), CFG, st) == 1             # direction flipped
    assert srv._stack_level("X", 0.0, Ex(), CFG, st) == 1 and "X" not in st     # flat


def test_disabled_or_cooldown():
    assert srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), {"enabled": False}, {}) == 1
    st = {}
    cfg = dict(CFG, cooldown_s=3600)
    srv._stack_level("X", 0.5, Ex([P(profit=2.0)]), cfg, st)
    assert st["X"]["n"] == 1
