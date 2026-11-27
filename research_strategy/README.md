 # Strategy tournament

Offline research only. No MT5 execution, no registry promotion, no changes to existing live engine.

The tournament engine (`tournament.py`) and the reviewer use only the Python 3.10+ standard library. Loading the cached Exness arrays needs `numpy`, which is already in `requirements.txt`.

## Run it

```sh
python -m unittest research_strategy.test_tournament

# every cached Exness symbol, M5 and H1, with per-symbol costs -> results/summary.md
python -m research_strategy.run_exness

# or one CSV of your own
python -m research_strategy.exness_data XAUUSD --resample-minutes 60 --output gold_h1.csv
python -m research_strategy.tournament gold_h1.csv --cost-bps 1 --folds 5 --output strategy_results.json

# optional advisory review (validation results only, never holdout)
ollama pull qwen3:8b
python -m research_strategy.ollama_review strategy_results.json --model qwen3:8b
```

CSV input: closed candles for one symbol and timeframe with columns `timestamp,open,high,low,close,volume` and ascending canonical UTC ISO timestamps. Set the per-side cost to include commission, spread and slippage.

## Method

- **Split test:** first 60% development, next 20% validation, final 20% holdout. Ranking uses validation return only. Do not repeatedly tune on holdout.
- **Walk-forward:** five expanding folds. Before each test window the rule with the best net return on all earlier bars is picked; if none made money the fold stays flat (`WAIT`). Buy-and-hold over the same windows is the benchmark, and the run is repeated at 3x cost.
- **Costs** (`run_exness`): half the round-trip commission plus half the standard spread from `data/zero_costs.json`, plus 0.5 bps slippage, per side. Charging both commission and spread overstates a Zero account on purpose.
- Signals are computed at close and filled at next open; positions are signed, unlevered exposure and flattened at split boundaries. Profit factor is bar-level, not trade-level. Rolling VWAP is a close-volume approximation; RSI and ATR use simple windows. These are original simplified rules, not exact TradingView ports. No funding, swaps, intrabar stops, margin or lot constraints are modeled.

## What the Exness cache showed (run 2026-11-27)

Full table: [`results/summary.md`](results/summary.md). About 17 months of M5 bars per symbol (June/December 2025 to 19 Nov 2026).

- **M5 is dead after costs.** On all seven symbols no rule made money in training, so every walk-forward fold stayed flat.
- **Gold H1 is the only strong result:** +50.7% walk-forward vs +1.1% buy-and-hold, positive in 4 of 5 folds (picks: SMA/EMA cross early, MACD later). It is fragile: one fold during the rally made most of it, one fold drew down 22%, and at 3x costs it falls to +9.9%. The MACD fold returns were reproduced independently with pandas.
- **USDJPY H1** EMA cross: +6.6% vs +5.9% buy-and-hold; negative at 3x costs. No real edge.
- **EURUSD, ETHUSD, USOIL H1** lost money in walk-forward; silver and BTC were roughly flat.

No profitability claim. Before any of this goes near the registry: more history (the daily CSVs go back to 2003), several brokers' feeds, a proper trade-level backtest with stops and sizing, and forward paper trading.
