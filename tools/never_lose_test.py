"""'Always close in profit' test: many small trades on 30-second bars, take a small profit, NEVER close at a loss.

Real Exness ticks (October 2026). A trade opens on every 30-second bar (BUY at the ask, and separately SELL
at the bid). It closes the first time it can be closed with a profit of TARGET after the spread. There is no
stop: losing trades are held, and whatever is still open at the end of the month is marked at the last price.
Reports the share closed in profit, the trades left stuck, the total result, and the worst moment for the
account (sum of open losses), all in bps of position value (1 bps on a $1,000 position = $0.10).
"""
import json, os, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]

def load(sym):
    with zipfile.ZipFile(os.path.join(ROOT, "data", "ticks", f"Exness_{sym}_2026_10.zip")) as z:
        n = next(x for x in z.namelist() if x.endswith(".csv"))
        df = pd.read_csv(z.open(n), usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    s = df.set_index("Timestamp").resample("30s").last().dropna()
    return s["Bid"].to_numpy(), s["Ask"].to_numpy()

out = {}
for sym in MK:
    bid, ask = load(sym); N = len(bid); mid = (bid + ask) / 2
    sp = float(np.median((ask - bid) / mid * 1e4))
    for tgt_x in (1, 3):
        for side in (1, -1):
            tgt = tgt_x * sp
            res, hold, stuck_pnl = [], [], []
            open_iv = []                      # (entry index, close index) for drawdown accounting
            for i in range(N - 1):
                net = ((bid[i+1:] - ask[i]) if side == 1 else (bid[i] - ask[i+1:])) / mid[i] * 1e4
                hit = net >= tgt
                if hit.any():
                    j = int(np.argmax(hit)); res.append(float(net[j])); hold.append((j + 1) * 0.5); open_iv.append((i, i + 1 + j))
                else:
                    res.append(float(net[-1])); stuck_pnl.append(float(net[-1])); open_iv.append((i, N))
            # worst moment: sum of open (unrealised) P&L across all open trades, checked every hour (120 bars)
            worst, worst_n = 0.0, 0
            iv = np.array(open_iv)
            for t in range(0, N, 120):
                m = (iv[:, 0] < t) & (iv[:, 1] > t)
                if not m.any(): continue
                e = iv[m, 0]
                u = ((bid[t] - ask[e]) if side == 1 else (bid[e] - ask[t])) / mid[e] * 1e4
                if u.sum() < worst: worst, worst_n = float(u.sum()), int(m.sum())
            r = np.array(res); k = f"{sym} {'BUY' if side == 1 else 'SELL'} target {tgt_x}x spread"
            out[k] = {"spread_bps": round(sp, 2), "trades": len(r), "closed_in_profit_pct": round(100 * (len(r) - len(stuck_pnl)) / len(r), 1),
                      "median_minutes_to_profit": round(float(np.median(hold)), 1) if hold else None,
                      "stuck_at_month_end": len(stuck_pnl), "stuck_avg_bps": round(float(np.mean(stuck_pnl)), 1) if stuck_pnl else 0,
                      "total_bps": round(float(r.sum()), 0), "per_trade_bps": round(float(r.mean()), 3),
                      "worst_open_loss_bps": round(worst, 0), "open_trades_at_worst": worst_n}
            o = out[k]
            print(f"{k:34} closed in profit {o['closed_in_profit_pct']:5.1f}% (median {o['median_minutes_to_profit']} min) | stuck {o['stuck_at_month_end']:5} avg {o['stuck_avg_bps']:+7.1f} bps"
                  f" | TOTAL {o['total_bps']:+9.0f} bps = {o['per_trade_bps']:+.2f}/trade | worst open loss {o['worst_open_loss_bps']:+.0f} bps across {o['open_trades_at_worst']} open trades", flush=True)
json.dump(out, open(os.path.join(ROOT, "logs", "never_lose_test.json"), "w"), indent=1)
