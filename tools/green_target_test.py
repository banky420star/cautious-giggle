"""Per market: which small profit target (in $, at the smallest trade size) comes closest to 'always green'?

Real Exness ticks (October 2026), second by second. A trade opens every minute, BUY and SELL both counted,
at the real ask/bid. For each $ target: did the trade reach that profit (after the spread) within 1 hour,
with NO stop? Trades that didn't are marked at their 1-hour price (that is what they'd be worth if closed).
Dollar values use the account's real contract size and minimum lot (data/contracts.json); USD/JPY P&L is
converted from yen at the entry price.
"""
import json, os, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK = ["XAUUSD", "BTCUSD", "EURUSD", "USDJPY", "XAGUSD", "ETHUSD", "USOIL"]
TARGETS = [0.05, 0.10, 0.20, 0.30, 0.50, 1.00]
H = 3600
CON = json.load(open(os.path.join(ROOT, "data", "contracts.json")))

def load(sym):
    with zipfile.ZipFile(os.path.join(ROOT, "data", "ticks", f"Exness_{sym}m_2026_10.zip")) as z:
        n = next(x for x in z.namelist() if x.endswith(".csv"))
        df = pd.read_csv(z.open(n), usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    s = df.set_index("Timestamp").resample("1s").last().ffill().dropna()
    return s["Bid"].to_numpy(), s["Ask"].to_numpy()

out = {}
for sym in MK:
    bid, ask = load(sym); units = CON[sym]["contract"] * CON[sym]["vol_min"]
    spread_usd = []; hit = {t: [] for t in TARGETS}; pnl = {t: [] for t in TARGETS}; miss = {t: [] for t in TARGETS}
    for i in range(0, len(bid) - H, 60):
        fx = (1 / ask[i]) if sym == "USDJPY" else 1.0              # yen -> $
        if sym == "USDJPY": units_i = CON[sym]["contract"] * CON[sym]["vol_min"]
        spread_usd.append((ask[i] - bid[i]) * units * fx)
        for side in (1, -1):
            net = ((bid[i+1:i+H] - ask[i]) if side == 1 else (bid[i] - ask[i+1:i+H])) * units * fx   # $ if closed now
            for t in TARGETS:
                h = net >= t
                if h.any():
                    hit[t].append(1); pnl[t].append(t)
                else:
                    hit[t].append(0); pnl[t].append(float(net[-1])); miss[t].append(float(net[-1]))
    sp = float(np.median(spread_usd))
    rows = {}
    for t in TARGETS:
        rows[f"${t:.2f}"] = {"green_pct": round(100 * float(np.mean(hit[t])), 1),
                             "miss_avg_usd": round(float(np.mean(miss[t])), 2) if miss[t] else 0.0,
                             "per_trade_usd": round(float(np.mean(pnl[t])), 3),
                             "per_2000_trades_usd": round(2000 * float(np.mean(pnl[t])))}
    out[sym] = {"spread_cost_usd": round(sp, 3), "trade_size": f"{CON[sym]['vol_min']} lot", "targets": rows}
    print(f"\n{sym} ({CON[sym]['vol_min']} lot, spread costs ${sp:.3f} per trade)")
    for k, r in rows.items():
        print(f"  target {k}: green {r['green_pct']:5.1f}% | misses avg ${r['miss_avg_usd']:+7.2f} | per trade ${r['per_trade_usd']:+.3f} | per 2,000 trades ${r['per_2000_trades_usd']:+,}", flush=True)
json.dump(out, open(os.path.join(ROOT, "logs", "green_target_test.json"), "w"), indent=1)
