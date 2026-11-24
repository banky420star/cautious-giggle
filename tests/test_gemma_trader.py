import json
import time

import numpy as np
import pandas as pd

from Python.gemma_trader import GemmaTrader, council_brief
from test_meta_trader import FakeMT5  # noqa: E402  (tests/ is on sys.path under pytest)


def _m15(n=400):
    r = np.random.default_rng(3)
    c = 100 + np.cumsum(r.normal(0, 0.2, n))
    o = np.r_[c[0], c[:-1]]
    return pd.DataFrame({"time": pd.date_range("2026-11-01", periods=n, freq="15min", tz="UTC"), "open": o,
                         "high": np.maximum(o, c) + 0.1, "low": np.minimum(o, c) - 0.1, "close": c})


def test_brief_lists_council_and_models():
    cm = {"up": 30, "down": 15, "flat": 5, "target": 0.4,
          "votes": {"rsi": 0.8, "macd": -0.6, "kronos": 0.3, "smc_gemma": 0.7},
          "assess": {"rsi": {"mult": 1.2, "hit": 0.61}}}
    b = council_brief(cm, {"agi_direction": "BUY", "agi_confidence": 0.7, "ppo_target": 0.1, "target": 0.12,
                           "agi_proven": True})
    assert "30 say buy" in b and "rsi +0.80 (hit 61%)" in b and "Kronos forecaster +0.30" in b
    assert "Gemma SMC" not in b and "Regime model (LSTM): BUY" in b
    b = council_brief(cm, {"agi_direction": "BUY", "agi_confidence": 0.7, "ppo_target": 0.1, "target": 0.12})
    assert "LSTM): not proven" in b and "LSTM): BUY" not in b


def test_own_decision_own_position_and_flip(tmp_path):
    answers = iter([{"reason": "trend up", "decision": "buy", "confidence": 0.8},
                    {"reason": "reversal", "decision": "sell", "confidence": 0.9}])
    g = GemmaTrader({"enabled": False}, str(tmp_path), ask=lambda p: next(answers))
    g.enabled = True                                    # no worker thread: call decide directly
    m = FakeMT5()
    g.decisions["EURUSDm"] = g.decide("EURUSDm", "s1", "brief", _m15())
    g.step(m, ["EURUSDm"])
    assert m.sent[-1]["type"] == m.ORDER_TYPE_BUY and "T1" not in m.own        # hidden from the council
    g.step(m, ["EURUSDm"])
    assert len(m.sent) == 1                                                   # same call: no second trade
    g.decisions["EURUSDm"] = g.decide("EURUSDm", "s2", "brief", _m15())
    g.step(m, ["EURUSDm"])                                                    # opposite, sure: closes
    assert m.sent[-1].get("position") == "T1" and "EURUSDm" not in g.state["open"]
    g.step(m, ["EURUSDm"])                                                    # then opens the new side
    assert m.sent[-1]["type"] == m.ORDER_TYPE_SELL


def test_low_confidence_or_hold_or_stale_does_nothing(tmp_path):
    m = FakeMT5()
    g = GemmaTrader({"enabled": False}, str(tmp_path), ask=lambda p: {"reason": "x", "decision": "buy", "confidence": 0.4})
    g.enabled = True
    g.decisions["XAUUSDm"] = g.decide("XAUUSDm", "a", "brief", _m15())
    g.step(m, ["XAUUSDm"])
    g.decisions["XAUUSDm"] = {"t": time.time() - 5000, "stamp": "b", "decision": "buy", "confidence": 0.9, "reason": ""}
    g.step(m, ["XAUUSDm"])
    g.decisions["XAUUSDm"] = {"t": time.time(), "stamp": "c", "decision": "hold", "confidence": 0.9, "reason": ""}
    g.step(m, ["XAUUSDm"])
    assert m.sent == []


def test_desk_context_sees_account_positions_news_and_own_record(tmp_path):
    import json
    from types import SimpleNamespace as NS
    m = FakeMT5()
    m.account_info = lambda: NS(equity=900.0, balance=950.0)
    m.positions = [NS(ticket="G1", symbol="EURUSDm", type=1, volume=2.42, profit=-30.0),
                   NS(ticket="B7", symbol="XAUUSDm", type=0, volume=0.01, profit=4.0),
                   NS(ticket="S3", symbol="XAUUSDm", type=0, volume=0.05, profit=1.0)]
    g = GemmaTrader({"enabled": False}, str(tmp_path))
    g.state["open"]["EURUSDm"] = {"ticket": "G1", "side": -1, "risk_pct": 5.0}
    (tmp_path / "gemma_trader.jsonl").write_text(json.dumps({"event": "open", "ticket": "G0"}) + "\n"
                                                 + json.dumps({"event": "open", "ticket": "G1"}) + "\n")
    (tmp_path / "gold_breakout.jsonl").write_text(json.dumps({"event": "open", "ticket": "S3"}) + "\n")
    (tmp_path / "trade_events.jsonl").write_text(json.dumps({"event": "trade_closed", "payload": {
        "deal_id": 1, "ticket": "G0", "symbol": "EURUSDm", "profit": -86.7}}) + "\n")
    now = 1795500000.0                                     # a Tuesday, 05:20 UTC
    cal = [{"t": now + 3600, "title": "USD CPI m/m", "impact": "High", "country": "USD"},
           {"t": now + 3600, "title": "AUD Retail", "impact": "High", "country": "AUD"}]
    d = g.desk_context(m, "EURUSDm", _m15(), atr_m5=0.2, now=now, calendar=cal)
    assert "equity $900.00" in d and "open profit/loss $-25.00" in d
    assert "EUR/USD SELL 2.42 lots $-30.00 (yours)" in d and "(council bot)" in d and "(other strategy)" in d
    assert "risking 5.0% of equity" in d and "Asian session" in d
    assert "USD CPI m/m (High) in 60 min" in d and "AUD" not in d
    assert "London open in 120 min" in d
    assert "0 won / 1 lost" in d and "net $-86.70" in d and "ATR over 4 h" in d


def test_cost_uses_m5_atr(tmp_path):
    seen = {}
    g = GemmaTrader({"enabled": False}, str(tmp_path), ask=lambda p: seen.setdefault("p", p) and {"decision": "hold"})
    g.decide("EURUSDm", "s", "brief", _m15(), desk="DESK LINE", atr_m5=0.0001)   # tiny ATR -> spread is a big share
    assert "DESK LINE" in seen["p"]
    big = float(seen["p"].split("costs about ")[1].split(" ")[0])
    seen.clear()
    g.decide("EURUSDm", "s", "brief", _m15(), atr_m5=1.0)
    assert big > float(seen["p"].split("costs about ")[1].split(" ")[0])


def test_shared_worker_serves_every_trader(tmp_path):
    import Python.gemma_trader as gt
    seen = []
    a = gt.GemmaTrader({"enabled": False}, str(tmp_path / "a"), ask=lambda p: {"decision": "hold", "confidence": 0.5})
    b = gt.QwenTrader({"enabled": False}, str(tmp_path / "b"), ask=lambda p: {"decision": "hold", "confidence": 0.5})
    for t in (a, b):
        t._answer = (lambda name: (lambda *job: seen.append((name, job[0]))))(t.NAME)
    job = lambda s: (s, "1", "", None, "", None)
    gt._round([(a, job("X1")), (b, job("X1")), (a, job("X2"))])
    assert seen == [("gemma trader", "X1"), ("gemma trader", "X2"), ("qwen trader", "X1")]


def test_entry_window_blocks_new_trades_but_not_exits(tmp_path):
    from Python.entry_window import EntryWindow
    now = 1795500000.0                                     # Tuesday 2026-11-24 05:20 UTC
    cal = [{"t": now + 600, "title": "USD CPI m/m", "impact": "High", "country": "USD"}]
    win = EntryWindow({}, calendar=cal)
    assert "USD CPI" in win.reason("XAUUSDm", now) and "in 10 min" in win.reason("XAUUSDm", now)
    assert win.reason("XAUUSDm", now - 3600) is None                    # an hour before: free
    g = GemmaTrader({"enabled": False}, str(tmp_path), ask=lambda p: {"reason": "x", "decision": "buy", "confidence": 0.9})
    g.enabled = True
    m = FakeMT5()
    g.decisions["EURUSDm"] = g.decide("EURUSDm", "s1", "brief", _m15())
    g.step(m, ["EURUSDm"], block=lambda s: win.reason(s, now))
    assert m.sent == []


def test_jev_trader_uses_probabilities_and_backs_off_when_down(tmp_path):
    from Python.gemma_trader import JevTrader
    seen = {}

    def ask(state):
        seen.update(state)
        return {"choice": "sell", "probabilities": {"sell": 0.71, "buy": 0.19, "hold": 0.10}}
    j = JevTrader({"enabled": False}, str(tmp_path), ask=ask)
    j.enabled = True
    m = FakeMT5()
    j.decisions["XAUUSDm"] = j.decide("XAUUSDm", "s1", "council brief", _m15(), desk="desk lines")
    assert j.decisions["XAUUSDm"]["decision"] == "sell" and j.decisions["XAUUSDm"]["confidence"] == 0.71
    assert seen["council"] == "council brief" and seen["desk"] == "desk lines" and "ATR" in seen["stops"]
    j.step(m, ["XAUUSDm"])
    assert m.sent[-1]["type"] == m.ORDER_TYPE_SELL
    # unsure answer: hold, no trade
    j2 = JevTrader({"enabled": False}, str(tmp_path / "b"), ask=lambda s: {"choice": "buy", "probabilities": {"buy": 0.52, "sell": 0.48}})
    (tmp_path / "b").mkdir()
    j2.enabled = True
    m2 = FakeMT5()
    j2.decisions["EURUSDm"] = j2.decide("EURUSDm", "s1", "b", _m15())
    j2.step(m2, ["EURUSDm"])
    assert m2.sent == []
    # endpoint down: the call fails, then Jev is skipped for the cooldown (never stalls Gemma's shared worker)
    calls = []

    def down(state):
        calls.append(1)
        raise RuntimeError("HTTP 502")
    (tmp_path / "c").mkdir()
    j3 = JevTrader({"enabled": False}, str(tmp_path / "c"), ask=down)
    j3._answer("XAUUSDm", "s1", "b", _m15(), "", None)
    assert j3.decisions["XAUUSDm"]["decision"] == "hold" and j3.decisions["XAUUSDm"]["error"]
    j3._ask = None
    j3._answer("XAUUSDm", "s2", "b", _m15(), "", None)
    assert len(calls) == 1 and "cooling down" in j3.decisions["XAUUSDm"]["reason"]


def test_jev_asks_every_five_minutes_gemma_once_per_bar(tmp_path):
    from Python.gemma_trader import GemmaTrader, JevTrader
    g = GemmaTrader({"enabled": False}, str(tmp_path))
    j = JevTrader({"enabled": False}, str(tmp_path))
    bar = "2026-11-24 16:15:00"
    t0 = 1795536900.0                       # 16:15:00 UTC
    assert g.slot(bar, t0) == g.slot(bar, t0 + 600) == bar
    s1, s2, s3 = j.slot(bar, t0 + 10), j.slot(bar, t0 + 290), j.slot(bar, t0 + 310)
    assert s1 == s2 and s1 != s3 and s1.startswith(bar)
    j.enabled = True
    j._stamp["BTCUSDm"] = s1
    assert not j.wants("BTCUSDm", s2) and j.wants("BTCUSDm", s3)
    assert JevTrader.OWN_WORKER and not GemmaTrader.OWN_WORKER and j._jobs is not g._jobs


class _Pos:
    def __init__(self, typ, sym="XAUUSDm", ticket=1):
        self.type, self.symbol, self.ticket = typ, sym, ticket


class _CrowdMT5:
    ORDER_TYPE_BUY, ORDER_TYPE_SELL = 0, 1

    def __init__(self, types):
        self._p = [_Pos(t, ticket=i) for i, t in enumerate(types)]

    def positions_get(self, symbol=None):
        return [p for p in self._p if symbol in (None, p.symbol)]


def test_crowd_cap_counts_every_trader_per_direction():
    from Python import strategy_exec as sx
    mt5 = _CrowdMT5([1, 1, 1, 0])                        # 3 sells, 1 buy
    assert sx.crowd_reason(mt5, -1, 3) and "3 sell" in sx.crowd_reason(mt5, -1, 3)
    assert sx.crowd_reason(mt5, 1, 3) is None
    assert sx.crowd_reason(mt5, -1, 0) is None           # 0 = off


def test_crowd_cap_blocks_atr_trader_open(tmp_path, monkeypatch):
    from Python import strategy_exec as sx
    from Python.gemma_trader import GemmaTrader
    monkeypatch.setitem(sx.CROWD, "max_same_side", 3)
    g = GemmaTrader({"enabled": False}, str(tmp_path))
    g.open(_CrowdMT5([1, 1, 1]), "XAUUSDm", -1, confidence=0.7, why="x")
    with open(g.events) as f:
        rows = [json.loads(l) for l in f]
    assert rows[-1]["event"] == "skip" and "cap 3" in rows[-1]["reason"] and not g.state["open"]


def test_gemma_skip_symbols_and_daily_cap(tmp_path):
    import time as _t
    from Python.gemma_trader import GemmaTrader
    g = GemmaTrader({"enabled": True, "skip_symbols": ["USOILm"], "max_trades_per_market_day": 2}, str(tmp_path))
    assert not g.wants("USOILm", "s1") and g.wants("XAUUSDm", "s1")
    now = _t.time()
    with open(g.events, "w") as f:                       # two opens on gold already today (e.g. before a restart)
        for i in range(2):
            f.write(json.dumps({"t": now - 60 * (i + 1), "event": "open", "symbol": "XAUUSDm", "ticket": str(i)}) + "\n")
    g.state["day"] = {}
    opened = []
    g.open = lambda *a, **k: opened.append(a)
    g.decisions["XAUUSDm"] = {"t": now, "stamp": "b1", "decision": "sell", "confidence": 0.8, "reason": "r"}
    g.step(object(), ["XAUUSDm"], now=now)
    with open(g.events) as f:
        rows = [json.loads(l) for l in f]
    assert not opened and rows[-1]["event"] == "skip" and "max 2" in rows[-1]["reason"]


def test_ai_traders_use_their_own_wider_stops(tmp_path):
    from Python.gemma_trader import GemmaTrader
    from Python.meta_trader import MetaTrader
    g = GemmaTrader({"enabled": False, "stops": {"default": [4.0, 6.4]}}, str(tmp_path))
    assert g.stops_for("XAUUSDm") == (4.0, 6.4) and g.stops_for("BTCUSDm") == (4.0, 6.4)
    m = MetaTrader({"enabled": False}, str(tmp_path))      # meta-brain keeps the stops it was trained on
    assert m.stops_for("XAUUSDm") == (1.5, 2.4) and m.stops_for("BTCUSDm") == (2.0, 3.2)
