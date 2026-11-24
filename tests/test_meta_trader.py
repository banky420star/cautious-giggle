from types import SimpleNamespace

import numpy as np

from Python.meta_trader import MetaTrader


class FakeMT5:
    TIMEFRAME_M5, TRADE_ACTION_DEAL, ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TIME_GTC = 5, 1, 0, 1, 0

    def __init__(self):
        self.hide_foreign = True
        self.own = set()
        self.positions = []
        self.sent = []
        self.n = 0

    def copy_rates_from_pos(self, *a):
        c = 100 + np.cumsum(np.full(200, 0.01))
        return [{"open": x, "high": x + 0.2, "low": x - 0.2, "close": x} for x in c]   # ATR ~0.4

    def symbol_info_tick(self, s):
        return SimpleNamespace(bid=101.9, ask=102.0)

    def account_info(self):
        return SimpleNamespace(equity=1000.0)

    def symbol_info(self, s):          # $1 per 1.0 move per 0.01 lot
        return SimpleNamespace(trade_tick_size=0.01, trade_tick_value=1.0, point=0.01, volume_step=0.01,
                               volume_min=0.01, digits=2)

    def positions_get(self, symbol=None):
        return tuple(p for p in self.positions if not (self.hide_foreign and p.ticket not in self.own))

    def _own_load(self):
        return set(self.own)

    def _own_save(self, ids):
        self.own = set(ids)

    def order_send(self, req):
        self.sent.append(req)
        if req.get("position"):
            self.positions = [p for p in self.positions if p.ticket != req["position"]]
        else:
            self.n += 1
            t = f"T{self.n}"
            self.positions.append(SimpleNamespace(ticket=t, type=req["type"], volume=req["volume"], profit=1.5, magic=None))
            self.own.add(t)
        return SimpleNamespace(retcode=10009, comment="done")


class FakeExt:
    def __init__(self):
        self.meta = SimpleNamespace(margin=0.3)
        self.values, self.details = {}, {}

    def predict(self, sym, pl, ps, t):
        self.values[(sym, "meta_brain")] = (0.5, t)
        self.details[(sym, "meta_brain")] = {"pred_long": pl, "pred_short": ps}


def test_opens_above_margin_with_model_stops_hidden_and_once(tmp_path):
    m, x = FakeMT5(), FakeExt()
    tr = MetaTrader({"enabled": True}, str(tmp_path))
    x.predict("EURUSDm", 0.1, 0.2, 1000.0)                 # below the 0.3 margin: no trade
    tr.step(m, x, ["EURUSDm"], now=1001.0)
    assert m.sent == []
    x.predict("EURUSDm", 0.05, 0.45, 1300.0)               # short predicted +0.45R
    tr.step(m, x, ["EURUSDm"], now=1301.0)
    req = m.sent[-1]
    assert req["type"] == m.ORDER_TYPE_SELL and req["price"] == 101.9
    atr = 101.9 - req["tp"]
    assert abs((req["sl"] - 101.9) / atr - 2.0 / 3.2) < 0.05          # stop 2.0 ATR, target 3.2 ATR
    assert "T1" not in m.own and m.positions_get() == ()             # hidden from the council
    tr.step(m, x, ["EURUSDm"], now=1302.0)
    assert len(m.sent) == 1                                          # same prediction, already open


def test_closes_after_8h_and_notices_broker_exit(tmp_path):
    m, x = FakeMT5(), FakeExt()
    tr = MetaTrader({"enabled": True}, str(tmp_path))
    x.predict("XAUUSDm", 0.5, 0.0, 1000.0)
    tr.step(m, x, ["XAUUSDm"], now=1001.0)
    opened = tr.state["open"]["XAUUSDm"]["opened"]
    tr.step(m, x, ["XAUUSDm"], now=opened + 96 * 300 + 1)
    assert m.sent[-1].get("position") == "T1" and "XAUUSDm" not in tr.state["open"]
    x.predict("XAUUSDm", 0.6, 0.0, 2000.0)
    tr.step(m, x, ["XAUUSDm"], now=2001.0)
    m.positions = []                                                 # broker stop filled
    tr.step(m, x, ["XAUUSDm"], now=2002.0)
    assert "XAUUSDm" not in tr.state["open"]


def test_stale_or_halted(tmp_path):
    m, x = FakeMT5(), FakeExt()
    tr = MetaTrader({"enabled": True}, str(tmp_path))
    x.predict("BTCUSDm", 0.9, 0.0, 1000.0)
    tr.step(m, x, ["BTCUSDm"], now=1000.0 + 700)                     # prediction older than 10 min
    tr.step(m, x, ["BTCUSDm"], now=1000.0 + 701)
    assert m.sent == []
    x.predict("BTCUSDm", 0.9, 0.0, 5000.0)
    tr.step(m, x, ["BTCUSDm"], can_open=False, now=5001.0)
    assert m.sent == []
