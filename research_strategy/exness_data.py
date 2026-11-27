"""Load cached Exness candles (data/exness_cache/*.npy) as tournament rows.

Offline only: reads the on-disk cache written by Python/exness_api.py. No broker calls.
"""
import argparse
import csv
import json
import math
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CACHE = ROOT / "data" / "exness_cache"
COSTS = ROOT / "data" / "zero_costs.json"


def iso(ts):
    return datetime.fromtimestamp(int(ts), timezone.utc).isoformat()


def load_cache(symbol, timeframe="M5", cache=CACHE):
    """Return ascending OHLCV rows with tick volume as volume."""
    import numpy as np  # only needed for the cache; tournament itself stays stdlib-only

    arr = np.load(Path(cache) / f"{symbol}_{timeframe}.npy", allow_pickle=False)
    rows = []
    for r in arr:
        x = {"timestamp": iso(r["time"]), "_t": int(r["time"]),
             "open": float(r["open"]), "high": float(r["high"]),
             "low": float(r["low"]), "close": float(r["close"]),
             "volume": float(r["tick_volume"])}
        if rows and x["_t"] <= rows[-1]["_t"]:
            raise ValueError(f"{symbol} {timeframe}: timestamps not strictly ascending")
        rows.append(x)
    return rows


def resample(rows, minutes):
    """Aggregate to fixed UTC buckets. Empty buckets (market closed) are skipped."""
    size = minutes * 60
    out = []
    for r in rows:
        bucket = r["_t"] - r["_t"] % size
        if out and out[-1]["_t"] == bucket:
            b = out[-1]
            b["high"] = max(b["high"], r["high"]); b["low"] = min(b["low"], r["low"])
            b["close"] = r["close"]; b["volume"] += r["volume"]
        else:
            out.append({"timestamp": iso(bucket), "_t": bucket, "open": r["open"],
                        "high": r["high"], "low": r["low"], "close": r["close"],
                        "volume": r["volume"]})
    return out


def cost_per_side_bps(symbol, slippage_bps=0.5, costs=COSTS):
    """Conservative per-side cost: half the round-trip commission, half the
    standard spread, plus a slippage allowance. Charging both commission and
    spread overstates a Zero account's cost on purpose."""
    c = json.loads(Path(costs).read_text())[symbol]
    commission = c["commission_bps"]
    spread = commission * c["standard_spread_usd"] / c["commission_usd_round_trip"]
    value = commission / 2 + spread / 2 + slippage_bps
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"bad cost for {symbol}")
    return round(value, 3)


def write_csv(rows, path):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["timestamp", "open", "high", "low", "close", "volume"])
        for r in rows:
            w.writerow([r["timestamp"], r["open"], r["high"], r["low"], r["close"], r["volume"]])


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="Export cached Exness candles to tournament CSV")
    p.add_argument("symbol"); p.add_argument("--timeframe", default="M5")
    p.add_argument("--resample-minutes", type=int, default=0)
    p.add_argument("--output", required=True)
    a = p.parse_args()
    rows = load_cache(a.symbol, a.timeframe)
    if a.resample_minutes:
        rows = resample(rows, a.resample_minutes)
    write_csv(rows, a.output)
    print(f"wrote {len(rows)} bars to {a.output}")
