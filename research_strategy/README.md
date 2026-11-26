 # Strategy tournament

Offline research only. No MT5 execution, no registry promotion, no changes to existing live engine.

Use Python 3.10+; no third-party dependencies. Export closed, equally spaced candles for one symbol and timeframe to CSV with columns timestamp,open,high,low,close,volume. Use ascending canonical UTC ISO timestamps. Adjust the per-side cost to include commission, spread and slippage estimates.

```sh
python -m unittest research_strategy.test_tournament
python -m research_strategy.tournament candles.csv --cost-bps 10 --output strategy_results.json
ollama pull qwen3:8b
python -m research_strategy.ollama_review strategy_results.json --model qwen3:8b
```

The first 60% is development, next 20% validation and final 20% holdout. Ranking uses validation return only. Do not repeatedly tune on holdout. Signals are computed at close and filled at next open; positions are signed, unlevered exposure and flattened at split boundaries. Profit factor is bar-level, not trade-level. Rolling VWAP is a close-volume approximation, RSI and ATR use simple windows; these are original simplified research rules, not exact TradingView ports. No funding, borrow, intrabar stops, margin or exchange-specific lot constraints are modeled. No profitability claims. Require multiple symbols, rolling walk-forward tests, real costs and forward paper validation before considering registry integration. Ollama is advisory only and receives validation results, not holdout.
