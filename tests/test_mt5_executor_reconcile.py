from types import SimpleNamespace

from Python.action_translator import translate_trade_action
from Python.mt5_executor import MT5Executor


class _Pos:
    def __init__(self, symbol: str, volume: float, order_type: int):
        self.symbol = symbol
        self.volume = volume
        self.type = order_type


class _Risk:
    def __init__(self, can_trade=True):
        self.trades = []
        self._can_trade = can_trade

    def can_trade(self, symbol=None):
        return self._can_trade

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

    def close_positions(self, positions, order_meta=None, execution_context=None):
        self.closed.append([(p.symbol, p.volume, p.type) for p in positions])
        return {"request_action": "close", "executed": True}

    def open_position(self, symbol, order_type, volume, order_meta=None, execution_context=None):
        self.opened.append((symbol, order_type, volume))
        return {"request_action": "open", "executed": True}


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


def test_reconcile_exposure_allows_flatten_when_trade_halted():
    risk = _Risk(can_trade=False)
    exec_ = _Exec(risk, shorts=[_Pos("EURUSDm", 0.15, 1)])

    meta = exec_.reconcile_exposure("EURUSDm", 0.0, 1.0)

    assert meta["request_action"] == "close"
    assert len(exec_.closed) == 1
    assert exec_.opened == []
    assert risk.trades == ["EURUSDm"]


def test_reconcile_exposure_does_not_reverse_when_trade_halted():
    risk = _Risk(can_trade=False)
    exec_ = _Exec(risk, shorts=[_Pos("EURUSDm", 0.15, 1)])

    meta = exec_.reconcile_exposure("EURUSDm", 0.10, 1.0)

    assert meta["request_action"] == "close"
    assert len(exec_.closed) == 1
    assert exec_.opened == []
    assert risk.trades == ["EURUSDm"]


def test_reconcile_exposure_does_not_record_trade_on_rejected_order():
    risk = _Risk()
    exec_ = _Exec(risk)

    def _rejected(*_args, **_kwargs):
        return {"request_action": "open", "executed": False}

    exec_.open_position = _rejected

    meta = exec_.reconcile_exposure("EURUSDm", 0.10, 1.0)

    assert meta["executed"] is False
    assert risk.trades == []


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


def test_magic_number_is_stable_by_symbol_and_lane():
    exec_ = MT5Executor(_Risk())

    btc_canary = exec_._magic_for_order("BTCUSDm", {"lane": "canary"}, request_kind="open")
    btc_champion = exec_._magic_for_order("BTCUSDm", {"lane": "champion"}, request_kind="open")
    xau_canary = exec_._magic_for_order("XAUUSDm", {"lane": "canary"}, request_kind="open")

    assert btc_canary != btc_champion
    assert btc_canary != xau_canary
    assert btc_canary == exec_._magic_for_order("BTCUSDm", {"lane": "canary"}, request_kind="open")


def test_order_comment_includes_symbol_lane_and_version():
    exec_ = MT5Executor(_Risk())

    comment = exec_._order_comment(
        "BTCUSDm",
        {
            "lane": "canary",
            "model_family": "ppo",
            "model_version": "20260315_072910",
            "ppo_target": -0.3316,
        },
        request_kind="open",
    )

    assert "BTC" in comment
    assert "CA" in comment
    assert "072910" in comment
    assert len(comment) <= 31
