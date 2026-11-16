"""Regression tests for the Exness/MT5 execution, risk and feature-pipeline review fixes."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

import Python.mt5_executor as mx
from Python.mt5_executor import MT5Executor
from Python.risk_supervisor import RiskSupervisor


# --------------------------------------------------------------------------- fakes
class FakeMT5:
    ORDER_TYPE_BUY = 0
    ORDER_TYPE_SELL = 1
    ORDER_FILLING_FOK = 0
    ORDER_FILLING_IOC = 1
    ORDER_FILLING_RETURN = 2
    ORDER_TIME_GTC = 0
    TRADE_ACTION_DEAL = 1
    TRADE_ACTION_SLTP = 6
    TRADE_RETCODE_DONE = 10009
    TIMEFRAME_M5 = 5

    def __init__(self, positions=None, filling_mode=1, retcodes=None):
        self.positions = list(positions or [])
        self.filling_mode = filling_mode
        self.retcodes = list(retcodes or [])
        self.sent = []

    def symbol_info(self, symbol):
        return SimpleNamespace(
            point=0.01, digits=2, filling_mode=self.filling_mode,
            volume_step=0.01, volume_min=0.01, volume_max=5.0,
            trade_stops_level=0, trade_freeze_level=0,
            trade_tick_size=0.01, trade_tick_value=0.01,
        )

    def symbol_info_tick(self, symbol):
        return SimpleNamespace(bid=100.0, ask=100.1, time=0)

    def positions_get(self, symbol=None):
        return [p for p in self.positions if symbol is None or p.symbol == symbol]

    def copy_rates_from_pos(self, *a, **k):
        return None

    def account_info(self):
        return SimpleNamespace(equity=10_000.0)

    def last_error(self):
        return (0, "ok")

    def order_send(self, request):
        self.sent.append(dict(request))
        code = self.retcodes.pop(0) if self.retcodes else 10009
        return SimpleNamespace(retcode=code, order=len(self.sent), deal=0, volume=request.get("volume"))


class FakeRisk:
    def __init__(self, can_open=True):
        self._can_open = can_open
        self.trades = []
        self.errors = 0
        self.successes = 0
        self.max_lots = 1.0
        self.risk_percent = 0.0

    def can_open(self, symbol=None):
        return self._can_open

    def record_trade(self, symbol=None):
        self.trades.append(symbol)

    def record_error(self):
        self.errors += 1

    def record_success(self):
        self.successes += 1

    def get_symbol_profile(self, symbol):
        return {"entry_deviation": 50, "sl_points": 250, "tp_points": 450}


def _pos(ticket, volume, side, magic=51000, symbol="BTCUSDm", t=0):
    return SimpleNamespace(ticket=ticket, symbol=symbol, volume=volume, type=side, magic=magic,
                           price_open=100.0, sl=0.0, tp=0.0, time=t, profit=0.0)


@pytest.fixture
def fake(monkeypatch):
    def _make(**kw):
        f = FakeMT5(**kw)
        monkeypatch.setattr(mx, "mt5", f)
        return f
    return _make


# ----------------------------------------------------------------- execution
def test_filling_mode_bitmask_decoded(fake):
    ex = MT5Executor(FakeRisk())
    fake(filling_mode=1)
    assert ex._select_filling_mode("BTCUSDm") == FakeMT5.ORDER_FILLING_FOK
    fake(filling_mode=2)
    assert ex._select_filling_mode("BTCUSDm") == FakeMT5.ORDER_FILLING_IOC
    fake(filling_mode=3)
    assert ex._select_filling_mode("BTCUSDm") == FakeMT5.ORDER_FILLING_FOK
    fake(filling_mode=0)
    assert ex._select_filling_mode("BTCUSDm") == FakeMT5.ORDER_FILLING_RETURN


def test_volume_normalized_to_step_and_limits(fake):
    fake()
    ex = MT5Executor(FakeRisk())
    assert ex.normalize_volume("BTCUSDm", 0.129) == pytest.approx(0.12)
    assert ex.normalize_volume("BTCUSDm", 0.004) == 0.0
    assert ex.normalize_volume("BTCUSDm", 99.0) == pytest.approx(5.0)


def test_noop_reconcile_does_not_count_a_trade(fake):
    f = fake(positions=[_pos(1, 0.20, 0)])
    risk = FakeRisk()
    ex = MT5Executor(risk)
    for _ in range(100):
        meta = ex.reconcile_exposure("BTCUSDm", 0.20, 1.0)
    assert meta["executed"] is False
    assert f.sent == []
    assert risk.trades == []


def test_flatten_allowed_while_risk_halted(fake):
    f = fake(positions=[_pos(1, 0.20, 0)])
    risk = FakeRisk(can_open=False)
    ex = MT5Executor(risk)
    meta = ex.reconcile_exposure("BTCUSDm", 0.0, 1.0)
    assert meta["executed"] is True
    assert len(f.sent) == 1 and f.sent[0]["position"] == 1


def test_halt_blocks_new_risk(fake):
    f = fake()
    ex = MT5Executor(FakeRisk(can_open=False))
    meta = ex.reconcile_exposure("BTCUSDm", 0.3, 1.0)
    assert meta["request_action"] == "blocked"
    assert f.sent == []


def test_failed_close_aborts_flip(fake):
    f = fake(positions=[_pos(1, 0.20, 0)], retcodes=[10006])
    ex = MT5Executor(FakeRisk())
    meta = ex.reconcile_exposure("BTCUSDm", -0.2, 1.0)
    assert meta["request_action"] == "close_failed"
    assert len(f.sent) == 1  # no sell opened on top of the still-open long


def test_reduction_is_partial_close(fake):
    f = fake(positions=[_pos(1, 0.50, 0)])
    risk = FakeRisk()
    ex = MT5Executor(risk)
    ex.reconcile_exposure("BTCUSDm", 0.40, 1.0)
    assert len(f.sent) == 1
    assert f.sent[0]["position"] == 1
    assert f.sent[0]["volume"] == pytest.approx(0.10)
    assert risk.trades == ["BTCUSDm"]


def test_foreign_magic_positions_ignored(fake):
    f = fake(positions=[_pos(1, 0.30, 0, magic=12345)])
    ex = MT5Executor(FakeRisk())
    ex.reconcile_exposure("BTCUSDm", 0.0, 1.0)
    assert f.sent == []  # manual/other-EA trade is not closed


def test_market_closed_is_not_an_error(fake):
    fake(retcodes=[10018])
    risk = FakeRisk()
    ex = MT5Executor(risk)
    meta = ex.reconcile_exposure("BTCUSDm", 0.1, 1.0)
    assert meta["executed"] is False
    assert risk.errors == 0


def test_requote_is_retried(fake):
    f = fake(retcodes=[10004, 10009])
    risk = FakeRisk()
    ex = MT5Executor(risk)
    meta = ex.reconcile_exposure("BTCUSDm", 0.1, 1.0)
    assert meta["executed"] is True
    assert len(f.sent) == 2
    assert risk.errors == 0


def test_close_uses_profile_deviation(fake):
    f = fake(positions=[_pos(1, 0.20, 0)])
    ex = MT5Executor(FakeRisk())
    ex.reconcile_exposure("BTCUSDm", 0.0, 1.0)
    assert f.sent[0]["deviation"] == 50


def test_risk_percent_caps_lots(fake):
    fake()
    risk = FakeRisk()
    risk.risk_percent = 1.0  # 1% of 10k = $100 at risk
    ex = MT5Executor(risk)
    # SL 250 points * 0.01 = 2.5 price; tick 0.01 worth $0.01 => $2.5/lot => 40 lots allowed
    assert ex.risk_capped_lots("BTCUSDm", 1.0) == pytest.approx(1.0)
    risk.risk_percent = 0.01  # $1 => 0.4 lots
    assert ex.risk_capped_lots("BTCUSDm", 1.0) == pytest.approx(0.4)


# -------------------------------------------------------------- risk engine
def test_success_resets_consecutive_errors(monkeypatch):
    import builtins, io, yaml
    from Python.risk_engine import RiskEngine

    monkeypatch.setattr(builtins, "open", lambda *a, **kw: io.StringIO(""))
    monkeypatch.setattr(yaml, "safe_load", lambda f: {})
    eng = RiskEngine()
    eng.record_error()
    eng.record_error()
    eng.record_success()
    eng.record_error()
    eng.record_error()
    assert eng.error_halt is False


# --------------------------------------------------------------- supervisor
def _sup():
    s = RiskSupervisor({"risk": {"supervisor": {"enabled": True}}})
    s.halt_until = s._now().replace(year=s._now().year + 1)
    return s


def _allow(s, target, current):
    return s.allow_trade(
        symbol="BTCUSDm", target_exposure=target, confidence=0.9, spread_bps=1.0,
        snapshot={"pnl_today": 0.0}, symbol_positions=1, total_positions=1,
        current_symbol_exposure=current, total_exposure=abs(current), drawdown_pct=0.0,
    )


def test_supervisor_flip_is_not_risk_reduction():
    d = _allow(_sup(), -0.2, 0.3)
    assert not d.allowed
    assert "halt_until" in d.reason


def test_supervisor_flatten_and_same_side_reduce_allowed_when_halted():
    s = _sup()
    assert _allow(s, 0.0, 0.3).reason == "risk_reduction"
    assert _allow(s, 0.1, 0.3).reason == "risk_reduction"


# -------------------------------------------------------------------- blend
def test_blend_risk_scalar_zero_blocks():
    from Python.Server_AGI import _blend_symbol_decision

    out = _blend_symbol_decision("BTCUSDm", {"risk_scalar": 0.0, "trend_bias": 0.9, "confidence": 0.99}, {"target": 0.9}, None)
    assert out["target"] == 0.0


def test_blend_missing_ppo_does_not_hand_weight_to_lstm():
    from Python.Server_AGI import _blend_symbol_decision

    agi = {"risk_scalar": 1.0, "trend_bias": 0.5, "confidence": 0.9}
    out = _blend_symbol_decision("BTCUSDm", agi, None, None)
    assert out["agi_weight_used"] == pytest.approx(0.30)


def test_blend_confidence_threshold_applied():
    from Python.Server_AGI import _blend_symbol_decision

    agi = {"risk_scalar": 1.0, "trend_bias": 0.5, "confidence": 0.5}
    out = _blend_symbol_decision("BTCUSDm", agi, {"target": 0.0}, None, cfg={"trading": {"confidence_threshold": 0.8}})
    assert out["agi_bias"] == 0.0


# ---------------------------------------------------------- feature pipeline
def _ohlcv(n=2000, seed=0):
    rng = np.random.default_rng(seed)
    c = 100 + np.cumsum(rng.normal(0, 0.1, n))
    return pd.DataFrame({
        "time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
        "open": c, "high": c + 0.05, "low": c - 0.05, "close": c,
        "volume": rng.integers(100, 500, n).astype(float),
    })


def test_features_have_no_lookahead():
    from Python.feature_pipeline import build_env_feature_matrix, ULTIMATE_150

    df = _ohlcv()
    a = np.asarray(build_env_feature_matrix(df, feature_version=ULTIMATE_150))
    df2 = df.copy()
    k = 1500
    df2.loc[k:, ["open", "high", "low", "close"]] *= 1.5
    df2.loc[k:, "volume"] *= 3.0
    b = np.asarray(build_env_feature_matrix(df2, feature_version=ULTIMATE_150))
    leaking = np.where(np.abs(a[:k] - b[:k]).max(axis=0) > 1e-9)[0]
    assert leaking.tolist() == []


# --------------------------------------------------------------------- env
def test_env_flat_action_stays_flat():
    from drl.trading_env import TradingEnv

    env = TradingEnv(df=_ohlcv(400), window_size=30, feature_version="ultimate_150")
    env.reset()
    meta = {"direction": 0.0, "size": 0.0, "target": 0.0, "entry_mode": "market",
            "entry_offset_pct": 0.0, "tp_offset_pct": 0.01, "sl_offset_pct": 0.01}
    env._process_action(meta, 100.0)
    assert env.open_trade is None or not env.open_trade.get("is_open")
    assert env.position == 0.0


# ------------------------------------------------------------ PPO live obs
def test_ppo_live_observation_uses_latest_window():
    from Python.hybrid_brain import HybridBrain
    from drl.trading_env import TradingEnv

    df = _ohlcv(400)
    brain = HybridBrain.__new__(HybridBrain)
    brain.window_size = 30
    brain.executor = None
    brain.risk = SimpleNamespace(max_lots=1.0)
    brain._infer_portfolio_feature_count = lambda obs_dim, feature_version=None: 3
    bundle = {"model": SimpleNamespace(), "meta": {"feature_set_version": "ultimate_150", "window_size": 30}}
    obs = brain._build_ppo_observation(df, bundle)

    env = TradingEnv(df=df.copy(), window_size=30, portfolio_feature_count=3, feature_version="ultimate_150")
    latest = np.asarray(env.feature_data[-30:], dtype=np.float32).reshape(-1)
    oldest = np.asarray(env.feature_data[:30], dtype=np.float32).reshape(-1)
    assert np.allclose(obs[: latest.size], latest)
    assert not np.allclose(obs[: oldest.size], oldest)
