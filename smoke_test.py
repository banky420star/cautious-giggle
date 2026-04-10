#!/usr/bin/env python3
"""Quick smoke test for all fixed modules."""
import sys, os
os.chdir(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

passed = 0
failed = 0

def check(label, fn):
    global passed, failed
    try:
        fn()
        print(f"✅ {label}")
        passed += 1
    except Exception as e:
        print(f"❌ {label}: {e}")
        failed += 1

# ── Core Imports ────────────────────────────────────────────────────

def test_data_feed():
    from Python.data_feed import fetch_training_data, get_combined_training_df, get_latest_data

def test_risk_engine():
    from Python.risk_engine import RiskEngine
    r = RiskEngine()
    assert hasattr(r, 'can_trade')
    assert hasattr(r, 'current_dd')
    assert r.can_trade() is True

def test_model_registry():
    from Python.model_registry import ModelRegistry
    reg = ModelRegistry()
    assert hasattr(reg, 'save_candidate')
    assert hasattr(reg, 'evaluate_and_stage_canary')
    assert hasattr(reg, 'load_active_model')

def test_mt5_executor():
    from Python.mt5_executor import MT5Executor
    from Python.risk_engine import RiskEngine
    r = RiskEngine()
    ex = MT5Executor(r)
    assert ex._is_live is False  # macOS → dry-run

def test_trading_env():
    from drl.trading_env import TradingEnv
    env = TradingEnv()
    obs, info = env.reset()
    assert obs.shape == (100 * 5 + 3,), f"Unexpected obs shape: {obs.shape}"

def test_hybrid_brain():
    from Python.hybrid_brain import HybridBrain
    # Just test the import; full init loads models (may not exist)

def test_autonomy_loop():
    from Python.autonomy_loop import AutonomyLoop

def test_backtester():
    from Python.backtester import run_ppo_backtest, run_multi

def test_model_evaluator():
    from Python.model_evaluator import evaluate_candidate_vs_champion

def test_ppo_agent():
    from drl.ppo_agent import load_model, predict, DEVICE
    assert DEVICE in ("cpu", "mps", "cuda")

def test_lstm_feature_extractor():
    from drl.lstm_feature_extractor import LSTMFeatureExtractor

def test_gradient_analyzer():
    from analysis.gradient_flow_analyzer import LSTMGradientDiagnostics

def test_n8n_bridge():
    from Python.agi_n8n_bridge import main as _bridge_main

# ── Data Fetch (network-dependent) ─────────────────────────────────

def test_fetch_data():
    from Python.data_feed import fetch_training_data
    df = fetch_training_data("EURUSD", period="5d")
    if df is not None and not df.empty:
        assert len(df) > 10, f"Only {len(df)} bars returned"
        for col in ["open", "high", "low", "close", "volume"]:
            assert col in df.columns, f"Missing column: {col}"
    else:
        raise RuntimeError("fetch_training_data returned empty (network issue?)")

# ── Run All ─────────────────────────────────────────────────────────

print("=" * 60)
print("🧪 cautious-giggle smoke test")
print("=" * 60)

check("data_feed imports", test_data_feed)
check("RiskEngine", test_risk_engine)
check("ModelRegistry", test_model_registry)
check("MT5Executor (dry-run)", test_mt5_executor)
check("TradingEnv (synthetic)", test_trading_env)
check("HybridBrain import", test_hybrid_brain)
check("AutonomyLoop import", test_autonomy_loop)
check("Backtester import", test_backtester)
check("ModelEvaluator import", test_model_evaluator)
check("PPO Agent import", test_ppo_agent)
check("LSTMFeatureExtractor import", test_lstm_feature_extractor)
check("GradientAnalyzer import", test_gradient_analyzer)
check("n8n Bridge import", test_n8n_bridge)
check("fetch_training_data (EURUSD)", test_fetch_data)

print()
print(f"{'=' * 60}")
print(f"Results: {passed} passed, {failed} failed")
print(f"{'=' * 60}")

if failed > 0:
    sys.exit(1)
