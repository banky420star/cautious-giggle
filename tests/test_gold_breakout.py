from datetime import datetime, timezone
from types import SimpleNamespace

from Python.gold_breakout import GoldBreakout


def _ts(day, hour, minute=0, tz=3):
    """Unix time of a broker-clock (GMT+3) moment."""
    return datetime(2026, 11, day, hour, minute, tzinfo=timezone.utc).timestamp() - tz * 3600


class FakeMT5:
    TIMEFRAME_M5, TRADE_ACTION_DEAL, ORDER_TYPE_BUY, ORDER_TYPE_SELL, ORDER_TIME_GTC = 5, 1, 0, 1, 0
    TRADE_RETCODE_DONE = 10009

    def __init__(self, day=24):
        self.hide_foreign = True
        self.own = {"111"}                         # a council position that must stay the council's
        self.positions = [SimpleNamespace(ticket="111", type=0, volume=0.01, profit=0.0, magic=None, symbol="XAUUSDm")]
        self.bid = self.ask = 3630.0
        self.sent = []
        base = _ts(day, 0)
        self.rates = [{"time": int(base + i * 300), "high": 3640.0 + (i == 10) * 5, "low": 3620.0 - (i == 20) * 3}
                      for i in range(84)]

    def copy_rates_from_pos(self, *a):
        return self.rates

    def symbol_info_tick(self, s):
        return SimpleNamespace(bid=self.bid, ask=self.ask)

    def account_info(self):
        return SimpleNamespace(equity=2000.0)

    def symbol_info(self, s):   # gold: 1 lot = 100 oz -> $100 per $1 move per lot
        return SimpleNamespace(trade_tick_size=0.001, trade_tick_value=0.1, point=0.001, volume_step=0.01, volume_min=0.01)

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
            self.positions.append(SimpleNamespace(ticket="222", type=req["type"], volume=req["volume"], profit=0.0,
                                                  magic=None, symbol=req["symbol"]))
            self.own.add("222")                    # the Exness adapter claims every new position as the bot's
        return SimpleNamespace(retcode=10009, comment="done")


def _gb(tmp_path, **cfg):
    return GoldBreakout({"enabled": True, **cfg}, str(tmp_path))


def test_range_and_no_trade_inside(tmp_path):
    m, g = FakeMT5(), _gb(tmp_path)
    g.step(m, now=_ts(24, 8))
    assert g.state["hi"] == 3645.0 and g.state["lo"] == 3617.0
    assert m.sent == []
    g.step(m, now=_ts(24, 6))                      # before 07:00: nothing, same day
    assert m.sent == []


def test_breakout_long_sized_hidden_and_closed_at_20(tmp_path):
    m, g = FakeMT5(), _gb(tmp_path)
    g.step(m, now=_ts(24, 8))
    m.ask, m.bid = 3645.5, 3645.3
    g.step(m, now=_ts(24, 9))
    req = m.sent[-1]
    assert req["type"] == m.ORDER_TYPE_BUY and abs(req["sl"] - 3645.5 * 0.99) < 0.01
    # 1% of $2000 = $20; 1% stop = $36.455 per oz -> 0.01 lot (1 oz) risks 1.8% (<= 2.5% min-lot ceiling)
    assert req["volume"] == 0.01 and g.state["risk_pct"] == 1.82
    assert g.state["ticket"] == "222" and "222" not in m.own and "111" in m.own
    assert [p.ticket for p in m.positions_get()] == ["111"]       # the council cannot see it
    g.step(m, now=_ts(24, 12))
    assert len(m.sent) == 1                                       # one trade a day, held
    g.step(m, now=_ts(24, 20, 1))
    assert m.sent[-1].get("position") == "222" and g.state["closed"]
    assert [p.ticket for p in m.positions] == ["111"]             # council position untouched


def test_stop_hit_at_broker_is_noticed(tmp_path):
    m, g = FakeMT5(), _gb(tmp_path)
    g.step(m, now=_ts(24, 8))
    m.ask, m.bid = 3616.9, 3616.8
    g.step(m, now=_ts(24, 9))
    assert m.sent[-1]["type"] == m.ORDER_TYPE_SELL
    m.positions = [p for p in m.positions if p.ticket != "222"]   # broker stop filled
    g.step(m, now=_ts(24, 10))
    assert g.state["closed"] and len(m.sent) == 1


def test_no_chase_and_too_big_min_lot(tmp_path):
    m, g = FakeMT5(), _gb(tmp_path)
    g.step(m, now=_ts(24, 8))
    m.ask, m.bid = 3655.0, 3654.8                                 # 0.27% past the high
    g.step(m, now=_ts(24, 9))
    assert m.sent == [] and "not chased" in g.state["done"]
    m2, g2 = FakeMT5(), _gb(tmp_path / "x", min_lot_max_risk_pct=1.0)
    (tmp_path / "x").mkdir()
    g2.step(m2, now=_ts(24, 8))
    m2.ask, m2.bid = 3645.5, 3645.3
    g2.step(m2, now=_ts(24, 9))
    assert m2.sent == [] and "minimum lot" in g2.state["done"]


def test_weekend_and_disabled(tmp_path):
    m = FakeMT5(day=28)                                           # 2026-11-28 is a Saturday
    g = _gb(tmp_path)
    g.step(m, now=_ts(28, 9))
    assert m.sent == [] and "hi" not in g.state
    g2 = GoldBreakout({"enabled": False}, str(tmp_path))
    g2.step(FakeMT5(), now=_ts(24, 9))
