import argparse
import json
import os
import sys
from datetime import datetime, timedelta, timezone

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.backtester import run_multi


def _parse_args():
    parser = argparse.ArgumentParser(description="Compare backtest metrics to recent live trade drift.")
    parser.add_argument("--model-dir", default=os.path.join("models", "registry", "champion"))
    parser.add_argument("--symbols", nargs="+", default=["EURUSDm", "GBPUSDm", "XAUUSDm", "BTCUSDm"])
    parser.add_argument("--period", default="7d")
    parser.add_argument("--interval", default="5m")
    parser.add_argument("--trade-log", default=os.path.join("logs", "trade_events.jsonl"))
    parser.add_argument("--max-slippage-multiple", type=float, default=1.5)
    parser.add_argument("--max-win-rate-drop", type=float, default=0.08)
    return parser.parse_args()


def _recent_live_metrics(path: str, symbols: list[str], lookback_days: int = 7) -> dict:
    cutoff = datetime.now(timezone.utc) - timedelta(days=lookback_days)
    rows = []
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8", errors="replace") as f:
            for line in f:
                try:
                    payload = json.loads(line)
                except Exception:
                    continue
                if payload.get("event") != "trade_closed":
                    continue
                ts = datetime.fromisoformat(str(payload.get("ts")))
                if ts.tzinfo is None:
                    ts = ts.replace(tzinfo=timezone.utc)
                if ts < cutoff:
                    continue
                body = payload.get("payload", {}) if isinstance(payload.get("payload"), dict) else {}
                symbol = str(body.get("symbol", ""))
                if symbol in symbols:
                    rows.append({"symbol": symbol, "profit": float(body.get("profit", 0.0) or 0.0)})

    result = {}
    for symbol in symbols:
        symbol_rows = [row for row in rows if row["symbol"] == symbol]
        wins = sum(1 for row in symbol_rows if row["profit"] > 0)
        total = len(symbol_rows)
        pnl = sum(row["profit"] for row in symbol_rows)
        result[symbol] = {
            "trades": total,
            "win_rate": 0.0 if total == 0 else wins / total,
            "realized_pnl": pnl,
        }
    return result


def main():
    args = _parse_args()
    lookback_days = max(1, int(str(args.period).rstrip("d")))
    backtest = run_multi(args.symbols, args.model_dir, period=args.period, interval=args.interval)
    live = _recent_live_metrics(args.trade_log, args.symbols, lookback_days=lookback_days)

    drift = {"backtest": backtest, "live": live, "alerts": []}
    for item in backtest.get("per_symbol", []):
        symbol = str(item.get("symbol"))
        live_row = live.get(symbol, {})
        backtest_return = float(item.get("total_return", 0.0))
        live_pnl = float(live_row.get("realized_pnl", 0.0))
        live_trades = int(live_row.get("trades", 0))
        if live_trades > 0 and backtest_return < 0 and live_pnl < 0:
            drift["alerts"].append(f"{symbol}: backtest/live both negative")
        if live_trades > 0 and abs(live_pnl) > abs(backtest_return) * args.max_slippage_multiple * 10000:
            drift["alerts"].append(f"{symbol}: pnl drift exceeds slippage multiple")
        if live_trades > 0 and live_row.get("win_rate", 0.0) + args.max_win_rate_drop < 0.5:
            drift["alerts"].append(f"{symbol}: live win rate materially weak")

    print(json.dumps(drift, indent=2))


if __name__ == "__main__":
    main()
