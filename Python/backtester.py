"""VectorBT Backtester — runs on REAL market data."""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import vectorbt as vbt
import pandas as pd
import numpy as np
from loguru import logger
from Python.data_feed import fetch_training_data

# ── Logging ─────────────────────────────────────────────────────────
LOG_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "logs")
os.makedirs(LOG_DIR, exist_ok=True)
logger.add(os.path.join(LOG_DIR, "backtester.log"), rotation="10 MB", level="DEBUG")

def run_backtest(symbol: str = "EURUSD"):
    logger.info(f"Starting backtest for {symbol}...")

    # Fetch real data
    df = fetch_training_data(symbol, period="60d")
    if df.empty or len(df) < 50:
        logger.error(f"Insufficient data for {symbol}")
        return

    logger.info(f"Backtest data: {len(df)} candles | {df.index[0]} → {df.index[-1]}")

    # Strategy: Moving average crossover (SMA 10 vs SMA 30)
    fast_ma = df['close'].rolling(10).mean()
    slow_ma = df['close'].rolling(30).mean()
    entries = (fast_ma > slow_ma) & (fast_ma.shift(1) <= slow_ma.shift(1))
    exits = (fast_ma < slow_ma) & (fast_ma.shift(1) >= slow_ma.shift(1))

    pf = vbt.Portfolio.from_signals(
        df['close'], entries, exits, freq="1h", init_cash=10000
    )

    # Results
    total_return = pf.total_return()
    sharpe = pf.sharpe_ratio()
    max_dd = pf.max_drawdown()
    total_trades = pf.trades.count()
    win_rate = pf.trades.win_rate() if total_trades > 0 else 0

    logger.success(f"{'='*60}")
    logger.success(f"BACKTEST RESULTS — {symbol}")
    logger.success(f"{'='*60}")
    logger.success(f"  Period:       {df.index[0]} → {df.index[-1]}")
    logger.success(f"  Candles:      {len(df)}")
    logger.success(f"  Total Return: {total_return:.2%}")
    logger.success(f"  Sharpe Ratio: {sharpe:.2f}")
    logger.success(f"  Max Drawdown: {max_dd:.2%}")
    logger.success(f"  Total Trades: {total_trades}")
    logger.success(f"  Win Rate:     {win_rate:.2%}")
    logger.success(f"  Final Value:  ${pf.final_value():.2f}")
    logger.success(f"{'='*60}")

    return {
        "symbol": symbol,
        "return": total_return,
        "sharpe": sharpe,
        "max_dd": max_dd,
        "trades": total_trades,
        "win_rate": win_rate,
        "final_value": pf.final_value()
    }

if __name__ == "__main__":
    results = []
    for sym in ["EURUSD", "GBPUSD", "XAUUSD"]:
        r = run_backtest(sym)
        if r:
            results.append(r)

    if results:
        logger.success("\n📊 MULTI-PAIR BACKTEST SUMMARY:")
        for r in results:
            logger.success(f"  {r['symbol']}: Return={r['return']:.2%} | Sharpe={r['sharpe']:.2f} | MaxDD={r['max_dd']:.2%} | Trades={r['trades']} | WR={r['win_rate']:.0%}")
