from types import SimpleNamespace

from Python.action_translator import translate_trade_action
from Python.mt5_executor import MT5Executor


class _Pos:
    def __init__(self, symbol: str, volume: float, order_type: int):
        self.symbol = symbol
        self.volume = volume
        self.type = order_type


class _Risk:
    def __init__(self):
        self.trades = []

    def can_trade(self, symbol=None):
        return True

    def record_trade(self, symbol=None):
        self.trades.append(symbol)


class _Exec(MT5Executor):
    def __init__(self, risk, longs=None, shorts=None):
        super().__init__(risk)
        self._longs = longs or []
        self._shorts = shorts or []
        self.closed = []
        self.opened = []

    def get_positions(self, symbol):
        return self._longs, self._shorts

    def close_positions(self, positions):
        self.closed.append([(p.symbol, p.volume, p.type) for p in positions])

    def open_position(self, symbol, order_type, volume):
        self.opened.append((symbol, order_type, volume))


def test_reconcile_exposure_flattens_without_reopening():
    risk = _Risk()
    exec_ = _Exec(risk, shorts=[_Pos("EURUSDm", 0.15, 1)])

    exec_.reconcile_exposure("EURUSDm", 0.0, 1.0)

    assert len(exec_.closed) == 1
    assert exec_.opened == []


def test_reconcile_exposure_flips_to_target_not_preclose_delta():
    risk = _Risk()
    exec_ = _Exec(risk, shorts=[_Pos("EURUSDm", 0.15, 1)])

    exec_.reconcile_exposure("EURUSDm", 0.10, 1.0)

    assert len(exec_.closed) == 1
    assert exec_.opened == [("EURUSDm", 0, 0.1)]


def test_translate_trade_action_skips_zero_exposure():
    tick = SimpleNamespace(bid=1.1, ask=1.1002)
    action = {
        "direction": 1.0,
        "size": 0.4,
        "entry_mode": "market",
        "entry_offset_pct": 0.0,
        "tp_offset_pct": 0.01,
        "sl_offset_pct": 0.01,
    }

    out = translate_trade_action("EURUSDm", action, 0.0, 1.0, tick=tick)

    assert out is None
