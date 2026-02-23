import pandas as pd
import numpy as np
import time
import os
import random

# Ensure data directory exists
os.makedirs("data", exist_ok=True)

def generate_mock_data():
    """Generate fake live trading data for the dashboard demonstration."""
    # Equity curve
    if os.path.exists("data/equity_curve.csv"):
        df_equity = pd.read_csv("data/equity_curve.csv")
        new_equity = df_equity['equity'].iloc[-1] * (1 + random.uniform(-0.01, 0.012))
        new_row = pd.DataFrame({'timestamp': [pd.Timestamp.now()], 'equity': [new_equity]})
        df_equity = pd.concat([df_equity, new_row]).tail(100)
    else:
        df_equity = pd.DataFrame({
            'timestamp': pd.date_range(end=pd.Timestamp.now(), periods=100, freq='H'),
            'equity': np.cumsum(np.random.normal(10, 50, 100)) + 10000
        })
    df_equity.to_csv("data/equity_curve.csv", index=False)

    # PnL per pair
    symbols = ["BTCUSD", "EURUSD", "XAUUSD", "ETHUSD", "GBPUSD"]
    pnl_data = {
        'symbol': symbols,
        'pnl': [random.uniform(-500, 2000) for _ in symbols],
        'win_rate': [random.uniform(0.4, 0.7) for _ in symbols],
        'trades': [random.randint(20, 100) for _ in symbols]
    }
    df_pnl = pd.DataFrame(pnl_data)
    df_pnl.to_csv("data/pnl_per_pair.csv", index=False)

    # Global Metrics
    metrics = {
        'total_profit': df_pnl['pnl'].sum(),
        'sharpe_ratio': 2.14,
        'max_drawdown': 0.084,
        'active_symbols': len(symbols),
        'status': 'HEALTHY'
    }
    with open("data/metrics.json", "w") as f:
        import json
        json.dump(metrics, f)

if __name__ == "__main__":
    print("📊 Dashboard Data Updater started...")
    while True:
        try:
            generate_mock_data()
            time.sleep(10) # Update every 10 seconds
        except Exception as e:
            print(f"Error updating dashboard data: {e}")
            time.sleep(5)
