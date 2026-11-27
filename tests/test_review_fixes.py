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
def test_blend_risk_scalar_zero_blocks(monkeypatch):
    from Python import lstm_gate
    from Python.Server_AGI import _blend_symbol_decision

    monkeypatch.setattr(lstm_gate, "proven", lambda s: True)
    out = _blend_symbol_decision("BTCUSDm", {"risk_scalar": 0.0, "trend_bias": 0.9, "confidence": 0.99}, {"target": 0.9}, None)
    assert out["target"] == 0.0


def test_blend_ignores_unproven_lstm(monkeypatch):
    from Python import lstm_gate
    from Python.Server_AGI import _blend_symbol_decision

    monkeypatch.setattr(lstm_gate, "proven", lambda s: False)
    out = _blend_symbol_decision("BTCUSDm", {"risk_scalar": 0.0, "trend_bias": 0.9, "confidence": 0.99}, {"target": 0.9}, None)
    assert out["agi_bias"] == 0.0 and out["agi_risk_scalar"] == 1.0 and out["target"] != 0.0 and out["agi_proven"] is False


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


# ================================================================ redesign (round 2)
def test_target_mode_holds_position_and_charges_costs():
    from drl.trading_env import TradingEnv

    df = _ohlcv(600)
    env = TradingEnv(df=df, window_size=30, feature_version="ultimate_150", execution_mode="target",
                     spread_bps=2.0, commission_rate=0.0, slippage_bps=0.0, sl_atr_mult=1e6, tp_atr_mult=1e6)
    env.reset()
    long_action = np.array([1.0, 1.0, -1, 0, 0, 0], dtype=np.float32)  # direction 1, size 1
    _, _, _, _, info = env.step(long_action)
    assert info["position"] == pytest.approx(1.0)
    assert info["cost"] == pytest.approx(10000.0 * 2.0 / 10000.0, rel=1e-6)
    _, _, _, _, info2 = env.step(long_action)
    assert info2["cost"] == 0.0  # holding costs nothing
    flat = np.array([0.0, -1.0, -1, 0, 0, 0], dtype=np.float32)
    _, _, _, _, info3 = env.step(flat)
    assert info3["position"] == 0.0


def test_target_mode_stop_loss_hits_intrabar():
    from drl.trading_env import TradingEnv

    n = 300
    c = np.full(n, 100.0)
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
                       "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": np.full(n, 100.0)})
    env = TradingEnv(df=df, window_size=30, feature_version="ultimate_150", execution_mode="target",
                     spread_bps=0.0, commission_rate=0.0, slippage_bps=0.0, sl_atr_mult=1.0, tp_atr_mult=50.0)
    env.reset()
    t = env.current_step
    env.lows[t] = 99.0  # crash through the stop during the next bar
    _, _, _, _, info = env.step(np.array([1.0, 1.0, -1, 0, 0, 0], dtype=np.float32))
    assert info["exit_reason"] == "sl"
    assert info["position"] == 0.0
    # lost ~ATR (0.2) on 100, not the full 1.0 drop
    assert 10000 * (1 - 0.003) < info["equity"] < 10000


def test_random_start_varies_episode_start():
    from drl.trading_env import TradingEnv

    env = TradingEnv(df=_ohlcv(3000), window_size=30, feature_version="ultimate_150",
                     execution_mode="target", random_start=True, episode_length=200)
    starts = {env.reset(seed=s) and env.episode_start for s in range(8)}
    assert len(starts) > 1


def test_chronological_split_is_ordered_and_disjoint():
    from training.train_drl import chronological_split

    df = _ohlcv(1000)
    tr, va, te = chronological_split(df, {"drl": {"split": {"val": 0.15, "test": 0.15}}})
    assert len(tr) + len(va) + len(te) == 1000
    assert tr["time"].max() < va["time"].min() < te["time"].min()


def test_baselines_computed_with_costs():
    from Python.backtester import run_baselines

    df = _ohlcv(2000)
    free = run_baselines(df, "5m", cost_bps=0.0)
    costly = run_baselines(df, "5m", cost_bps=5.0)
    assert set(free) == {"ma_cross", "buy_hold"}
    assert costly["ma_cross"]["total_return"] < free["ma_cross"]["total_return"]


def test_holdout_slice_keeps_only_unseen_bars():
    from Python.backtester import _holdout_slice

    df = _ohlcv(1000)
    cut = str(df["time"].iloc[800])
    out = _holdout_slice(df, cut)
    assert len(out) == 200
    assert out["time"].min() == df["time"].iloc[800]


def _fake_multi(cand_ret, baseline_ret):
    def _run(symbols, model_dir, period="120d", interval="5m", reward_weights=None, holdout_start=None):
        row = {"symbol": symbols[0], "score": 5.0, "max_drawdown": 0.05, "sharpe": 1.2,
               "total_return": cand_ret, "steps": 900}
        return {"avg_score": 5.0, "avg_return": cand_ret, "worst_drawdown": 0.05, "avg_sharpe": 1.2,
                "baseline_avg_return": baseline_ret, "per_symbol": [row]}
    return _run


def test_evaluator_requires_holdout(monkeypatch):
    from Python import model_evaluator as me

    monkeypatch.setattr(me, "run_multi", _fake_multi(0.05, 0.0))
    monkeypatch.setattr(me, "_candidate_holdout_start", lambda _d: None)
    rep = me.evaluate_candidate_vs_champion("cand", None, ["BTCUSDm"])
    assert rep["passes_thresholds"] is False
    assert "holdout" in rep["error"]


def test_evaluator_requires_beating_baseline(monkeypatch):
    from Python import model_evaluator as me

    monkeypatch.setattr(me, "_candidate_holdout_start", lambda _d: "2026-10-01")
    monkeypatch.setattr(me, "run_multi", _fake_multi(0.02, 0.03))
    assert me.evaluate_candidate_vs_champion("cand", None, ["BTCUSDm"])["passes_thresholds"] is False
    monkeypatch.setattr(me, "run_multi", _fake_multi(0.04, 0.03))
    assert me.evaluate_candidate_vs_champion("cand", None, ["BTCUSDm"])["passes_thresholds"] is True


def test_supervisor_caps_instead_of_rejecting():
    s = RiskSupervisor({"risk": {"supervisor": {"enabled": True, "max_symbol_exposure": 0.35, "max_total_exposure": 1.2}}})
    s.halt_until = None
    s.last_trade_at_by_symbol.clear()
    d = s.allow_trade(symbol="BTCUSDm", target_exposure=0.9, confidence=0.9, spread_bps=1.0,
                      snapshot={"pnl_today": 0.0}, symbol_positions=0, total_positions=0,
                      current_symbol_exposure=0.0, total_exposure=0.0, drawdown_pct=0.0)
    assert d.allowed
    assert d.target == pytest.approx(0.35)


def test_lstm_labels_come_from_raw_prices():
    from Python.feature_pipeline import build_lstm_feature_frame
    from training.train_lstm import _raw_label_matrix, create_sequences

    df = _ohlcv(3000)
    fdf, _ = build_lstm_feature_frame(df, feature_version="ultimate_150")
    raw = _raw_label_matrix(df, fdf)
    assert np.allclose(raw[:, 0], df["close"].to_numpy())
    _, y = create_sequences(np.zeros((len(fdf), 3)), 0, 1, 2, 60, label_data=raw)
    counts = np.bincount(y, minlength=3)
    assert counts[1] > 0 and counts[2] > 0  # not degenerate all-HOLD


def test_target_mode_gap_through_stop_fills_at_open():
    from drl.trading_env import TradingEnv

    n = 300
    c = np.full(n, 100.0)
    df = pd.DataFrame({"time": pd.date_range("2026-01-01", periods=n, freq="5min", tz="UTC"),
                       "open": c, "high": c + 0.1, "low": c - 0.1, "close": c, "volume": np.full(n, 100.0)})
    env = TradingEnv(df=df, window_size=30, feature_version="ultimate_150", execution_mode="target",
                     spread_bps=0.0, commission_rate=0.0, slippage_bps=0.0, sl_atr_mult=1.0, tp_atr_mult=50.0)
    env.reset()
    t = env.current_step
    env.opens[t], env.lows[t], env.highs[t] = 98.0, 97.5, 98.2  # gaps 2.0 below a ~0.2 stop
    _, _, _, _, info = env.step(np.array([1.0, 1.0, -1, 0, 0, 0], dtype=np.float32))
    assert info["exit_reason"] == "sl"
    assert info["equity"] == pytest.approx(10000 * 0.98, rel=1e-6)


def test_target_mode_portfolio_state_only_exposes_position():
    from drl.trading_env import TradingEnv

    env = TradingEnv(df=_ohlcv(400), window_size=30, feature_version="ultimate_150", execution_mode="target",
                     trade_memory={"win_rate": 80.0, "expectancy": 5.0, "trades": 10, "losses": 2})
    env.reset()
    env.position = 0.4
    env.equity = 12345.0
    assert env._build_portfolio_state() == [1.0, 0.4, 0.0]


def test_evaluator_champion_failure_blocks_promotion(monkeypatch):
    from Python import model_evaluator as me

    def _run(symbols, model_dir, period="120d", interval="5m", reward_weights=None, holdout_start=None):
        if model_dir == "champ":
            return {"error": "obs shape mismatch"}
        return _fake_multi(0.05, 0.0)(symbols, model_dir)

    monkeypatch.setattr(me, "_candidate_holdout_start", lambda _d: "2026-10-01")
    monkeypatch.setattr(me, "run_multi", _run)
    monkeypatch.setattr(me.os.path, "isdir", lambda _p: True)
    rep = me.evaluate_candidate_vs_champion("cand", "champ", ["BTCUSDm"])
    assert rep["wins"] is False
    assert rep["champion_eval_failed"] is True


def test_evaluator_baseline_uses_best_of_buy_hold_and_ma(monkeypatch):
    from Python import model_evaluator as me

    def _run(symbols, model_dir, period="120d", interval="5m", reward_weights=None, holdout_start=None):
        out = _fake_multi(0.04, 0.0)(symbols, model_dir)
        out["per_symbol"][0]["baselines"] = {"ma_cross": {"total_return": 0.01}, "buy_hold": {"total_return": 0.06}}
        return out

    monkeypatch.setattr(me, "_candidate_holdout_start", lambda _d: "2026-10-01")
    monkeypatch.setattr(me, "run_multi", _run)
    rep = me.evaluate_candidate_vs_champion("cand", None, ["BTCUSDm"])
    assert rep["beats_baseline"] is False
