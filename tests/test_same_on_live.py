from types import SimpleNamespace

from Python import demo_exploration as de


def _mt5(demo: bool):
    return SimpleNamespace(account_info=lambda: SimpleNamespace(trade_mode=0 if demo else 2))


def test_live_runs_like_demo_when_same_on_live():
    cfg = {"trading": {"demo_learning": {"same_on_live": True}}}
    assert de.demo_rules_apply(_mt5(False), cfg) is True
    assert de.demo_rules_apply(_mt5(True), cfg) is True


def test_live_stays_conservative_without_the_flag():
    assert de.demo_rules_apply(_mt5(False), {}) is False
    assert de.demo_rules_apply(_mt5(True), {}) is True
