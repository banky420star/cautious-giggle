from types import SimpleNamespace as NS

import numpy as np

from Python.stop_manager import StopManager, new_stop


def test_rule_math_breakeven_then_trail():
    rule = {"be_at": 0.5, "be_lock": 0.0, "trail_from": 1.0, "trail_atr": 1.0}
    assert new_stop(1, 100.0, 2.0, 1.0, 100.5, rule) is None            # +0.25R: not yet
    assert new_stop(1, 100.0, 2.0, 1.0, 101.2, rule) == 100.0           # +0.6R: breakeven
    assert new_stop(1, 100.0, 2.0, 1.0, 103.0, rule) == 102.0           # +1.5R: trails 1 ATR behind the best
    assert new_stop(-1, 100.0, 2.0, 1.0, 97.0, rule) == 98.0            # short mirror


class Broker:
    TIMEFRAME_M5, TRADE_ACTION_SLTP, ORDER_TYPE_BUY = 5, 6, 0

    def __init__(self):
        self.hide_foreign = True
        self.px = 100.0
        self.pos = [NS(ticket="C1", symbol="EURUSDm", type=0, price_open=100.0, sl=98.0, tp=103.0),   # council
                    NS(ticket="G1", symbol="EURUSDm", type=1, price_open=100.0, sl=102.0, tp=97.0),   # Gemma (hidden)
                    NS(ticket="H1", symbol="EURUSDm", type=0, price_open=100.0, sl=98.0, tp=103.0)]   # by hand
        self.sent, self.hidden_during_send = [], []

    def _own_load(self):
        return {"C1"}

    def positions_get(self, symbol=None):
        return tuple(p for p in self.pos if not self.hide_foreign or p.ticket == "C1")

    def symbol_info_tick(self, s):
        return NS(bid=self.px, ask=self.px)

    def symbol_info(self, s):
        return NS(digits=3)

    def copy_rates_from_pos(self, *a):
        return [{"high": 100.5, "low": 99.5, "close": 100.0}] * 200          # ATR 1.0

    def order_send(self, req):
        self.sent.append(req)
        self.hidden_during_send.append(self.hide_foreign)
        p = next(p for p in self.pos if p.ticket == req["position"])
        p.sl, p.tp = req["sl"], req["tp"]
        return NS(retcode=10009, comment="done")


def test_moves_council_and_strategy_stops_never_manual(tmp_path):
    import json
    (tmp_path / "gemma_trader.jsonl").write_text(json.dumps({"event": "open", "ticket": "G1"}) + "\n")
    m = Broker()
    sm = StopManager({"rules": {"EURUSD": {"trail_from": 1.0, "trail_atr": 0.5, "keep_target": False}}}, str(tmp_path))
    sm.step(m)                                           # first sight: R = 2; stop stays, the target goes out to 10R
    assert [(r["position"], r["sl"], r["tp"]) for r in m.sent] == [("C1", 98.0, 120.0), ("G1", 102.0, 80.0)]
    m.sent.clear(); m.hidden_during_send.clear()
    m.px = 102.5                                         # long +1.25R, short -1.25R
    sm.step(m)
    c1 = next(p for p in m.pos if p.ticket == "C1")
    assert np.isclose(c1.sl, 102.0) and np.isclose(c1.tp, 120.0)      # trails 0.5 ATR behind; target out to 10R
    assert all(r["position"] != "H1" for r in m.sent)                  # the hand-opened trade is untouched
    assert m.hidden_during_send == [False]
    m.px = 97.0                                          # now the Gemma short is +1.5R
    sm.step(m)
    g1 = next(p for p in m.pos if p.ticket == "G1")
    assert np.isclose(g1.sl, 97.5)
    m.px = 101.0                                         # pulls back: stops never loosen
    sm.step(m)
    assert np.isclose(c1.sl, 102.0) or c1.sl >= 102.0


def test_wide_stop_trades_use_the_wide_rules(tmp_path):
    m = Broker()
    m.pos = [NS(ticket="C1", symbol="EURUSDm", type=0, price_open=100.0, sl=92.0, tp=112.8)]   # 8 ATR stop
    sm = StopManager({"rules": {"EURUSD": {"trail_from": 0.5, "trail_atr": 0.5}},
                      "rules_wide": {"EURUSD": {"be_at": 0.75, "be_lock": 0.1}}}, str(tmp_path))
    sm.step(m)
    m.px = 105.0                                      # +0.625R: the narrow rule would trail, the wide one waits
    sm.step(m)
    assert m.sent == []
    m.px = 106.5                                      # +0.81R: lock +0.1R
    sm.step(m)
    assert np.isclose(m.pos[0].sl, 100.8)
