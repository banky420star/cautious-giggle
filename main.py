import time
import pandas as pd
import numpy as np
import os
import json
from mt5_connector import MT5Connector
from neuro_evo_agent import TradingEnv, create_agent
from strategy_evolver import StrategyEvolver
from dynamic_symbol_manager import SymbolManager
from news_adjuster import NewsAdjuster
from risk_engine import RiskEngine

def main():
    print("🔥 NeuroTrader v3.1 Booting up...")
    
    # 1. Initialization
    connector = MT5Connector()
    # Try connecting, but allow execution for simulation if it fails
    connected = connector.connect()
    
    symbol_manager = SymbolManager()
    risk_engine = RiskEngine()
    news_adjuster = NewsAdjuster()
    
    # 2. Daily Evolution (Strategy Optimization)
    print("🧬 Starting Daily Evolution Gate...")
    # Fetch some data for evolution
    if connected:
        df = connector.get_rates("BTCUSD", "H1", 2000)
    else:
        # Generate dummy data for simulation
        print("⚠️ Simulation Mode: Generating synthetic data...")
        df = pd.DataFrame({
            'time': pd.date_range(start='2025-01-01', periods=2000, freq='h'),
            'open': 100 + np.random.randn(2000).cumsum(),
            'high': 105 + np.random.randn(2000).cumsum(),
            'low': 95 + np.random.randn(2000).cumsum(),
            'close': 100 + np.random.randn(2000).cumsum(),
            'tick_volume': np.random.randint(100, 1000, 2000)
        })
    
    evolver = StrategyEvolver(df)
    best_strategy = evolver.evolve()
    print(f"✅ Evolution complete. Best Params: {best_strategy}")
    
    with open("models/best_strategy.json", "w") as f:
        json.dump(best_strategy, f)

    # 3. RL Agent Training (PPO)
    print("🧠 Training RL Agents (PPO)...")
    env = TradingEnv(df)
    model = create_agent(env)
    
    print("Training for 10,000 steps...")
    model.learn(total_timesteps=10000)
    model.save("models/PPO_BTCUSD_Latest")
    print("✅ PPO Agent trained and saved.")

    # 4. Live Trading Loop Simulation/Start
    print("🚀 Transitioning to Live Monitoring Mode...")
    # In a real scenario, this would loop or hand off to trade_server_mt5.py
    # For this main script, we'll do one cycle
    
    equity = connector.get_account_info().get('equity', 10000) if connected else 10000
    active_symbols = symbol_manager.get_active_symbols(equity)
    print(f"📡 Active symbols based on equity: {active_symbols}")
    
    sentiment_factor = news_adjuster.get_market_sentiment()
    print(f"📰 Global Market Sentiment Factor: {sentiment_factor}")
    
    print("NeuroTrader v3.1 is now operational.")
    
    # Keep alive if running in a container
    try:
        while True:
            time.sleep(3600)
    except KeyboardInterrupt:
        print("Shutting down...")

if __name__ == "__main__":
    main()
