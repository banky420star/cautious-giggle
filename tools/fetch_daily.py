"""Download long daily price history (bid and ask candles) from Dukascopy to data/daily/<NAME>_D1.csv.
Both sides are kept so the spread can be measured; the bid series is the price. Read-only research data.
    PYTHONPATH=. .venv/bin/python tools/fetch_daily.py
"""
from __future__ import annotations

import datetime as dt
import os
import sys

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "data", "daily")
# name -> Dukascopy instrument. The first 7 are the markets the bot trades; the rest broaden the test.
MARKETS = {
    "XAUUSD": "XAU-USD", "BTCUSD": "BTC-USD", "EURUSD": "EUR-USD", "USDJPY": "USD-JPY", "XAGUSD": "XAG-USD",
    "ETHUSD": "ETH-USD", "USOIL": "LIGHT.CMD-USD",
    "GBPUSD": "GBP-USD", "AUDUSD": "AUD-USD", "USDCAD": "USD-CAD", "USDCHF": "USD-CHF", "NZDUSD": "NZD-USD",
    "EURJPY": "EUR-JPY", "GBPJPY": "GBP-JPY", "UKOIL": "BRENT.CMD-USD", "US500": "USA500.IDX-USD",
    "US30": "USA30.IDX-USD", "USTEC": "USATECH.IDX-USD", "DE40": "DEU.IDX-EUR", "XNGUSD": "GAS.CMD-USD",
    "XCUUSD": "COPPER.CMD-USD",
}


def fetch(inst: str, side) -> pd.DataFrame:
    import dukascopy_python as dk
    c = dk.fetch_candles(inst, 1, dk.TimeUnit.DAY, side, dt.datetime(2003, 1, 1, tzinfo=dt.timezone.utc),
                         dt.datetime.now(dt.timezone.utc))
    return dk.to_dataframe(c)[["open", "high", "low", "close"]]


def main():
    import dukascopy_python as dk
    os.makedirs(OUT, exist_ok=True)
    for name, inst in MARKETS.items():
        try:
            bid, ask = fetch(inst, dk.OfferSide.BID), fetch(inst, dk.OfferSide.ASK)
        except Exception as exc:
            print(f"{name}: failed ({exc})", flush=True)
            continue
        df = bid.join(ask.add_prefix("ask_"), how="inner")
        df = df[(df["close"] > 0) & (df["high"] >= df["low"])]
        df.index.name = "date"
        df.to_csv(os.path.join(OUT, f"{name}_D1.csv"))
        print(f"{name}: {len(df)} days, {df.index[0].date()} -> {df.index[-1].date()}", flush=True)


if __name__ == "__main__":
    sys.exit(main())
