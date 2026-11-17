import time
from types import SimpleNamespace as NS

import Python.setup_gate as sg


def gate(tmp_path, **cfg):
    g = sg.SetupGate(cfg, account_id="T1")
    g.path = str(tmp_path / "setups.json")
    g.state = {"stats": {}, "open": [], "retest": {}}
    return g


def test_key_and_sessions():
    assert sg.session_of(3) == "asia" and sg.session_of(8) == "london" and sg.session_of(14) == "newyork"
    k = sg.setup_key("XAUUSDm", -1, 0.8, -0.2, 14)
    assert k == "XAUUSDm|SELL|newyork|hi|ai-agree"
    assert sg.setup_key("XAUUSDm", 1, 0.5, -0.2, 3).endswith("lo|ai-against")
    assert sg.levels(k) == [k, "XAUUSDm|SELL|newyork", "XAUUSDm|SELL"]


def test_verdicts(tmp_path):
    g = gate(tmp_path, min_trades_full=4, min_trades_block=3, retest_hours=4)
    k = g.key("BTCUSDm", 1, 0.8, 0.0, 3)
    assert g.verdict(k)[0] == "probe"
    for p in (2, 3, -1, 1):
        g.add_result(k, p)
    assert g.verdict(k)[0] == "full"
    bad = g.key("USOILm", 1, 0.5, 0.0, 14)
    for p in (-5, -4, 1):
        g.add_result(bad, p)
    assert g.verdict(bad)[0] == "retest"          # one smallest-lot re-test allowed
    g.record_open("USOILm", 1, bad, "retest")
    assert g.verdict(bad)[0] == "block"           # then blocked until retest_hours pass
    # a new combination inherits its family's losing record
    sib = g.key("USOILm", 1, 0.9, 0.3, 14)
    g.state["retest"][sib] = time.time()
    assert g.verdict(sib)[0] == "block"


def test_off(tmp_path):
    g = gate(tmp_path, enabled=False)
    assert g.verdict("X|BUY|asia|hi|ai-none")[0] == "full"


def test_resolve_scores_closed_trades(tmp_path):
    g = gate(tmp_path)
    t = time.time() - 100
    g.state["open"] = [{"symbol": "EURUSDm", "side": -1, "key": "EURUSDm|SELL|asia|hi|ai-none", "t": t},
                       {"symbol": "EURUSDm", "side": 1, "key": "EURUSDm|BUY|asia|hi|ai-none", "t": t}]
    mt5 = NS(DEAL_ENTRY_OUT=1, ORDER_TYPE_BUY=0, positions_get=lambda: [NS(ticket="P2")],
             history_deals_get=lambda a, b: [NS(symbol="EURUSDm", entry=0, type=1, position_id="P1", time=t + 2, profit=0.0),
                                             NS(symbol="EURUSDm", entry=1, type=0, position_id="P1", time=t + 50, profit=-1.5),
                                             NS(symbol="EURUSDm", entry=0, type=0, position_id="P2", time=t + 3, profit=0.0)])
    assert g.resolve(mt5, every_s=0) == 1
    assert g.stats("EURUSDm|SELL")["net"] == -1.5
    assert len(g.state["open"]) == 1 and g.state["open"][0]["pid"] == "P2"   # still open, kept
