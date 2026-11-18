"""How far does a trade go in your favour before it turns negative? (real Exness ticks, October 2026)

Entries at every minute, both BUY and SELL, at the real ask/bid. Price is followed second by second for up
to 1 hour. Measured per trade:
  draw-up (price)  = best favourable move of the MID price before the mid falls back below the entry mid
  draw-up (net)    = best profit you could have closed at (after paying the spread) before the trade is
                     worse off than when it opened by one more spread (i.e. a loss of 2x spread)
  ever in profit   = share of trades that were ever closable at a profit within the hour
Then: take-profit targets (as multiples of the spread) with a stop the same distance away: how often does
each hit, and what does it earn per trade after costs?
"""
import os, sys, zipfile, json
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
T = os.path.join(ROOT, "data", "ticks")
MK = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
H = 3600

def load(sym):
    with zipfile.ZipFile(os.path.join(T, f"Exness_{sym}_2026_10.zip")) as z:
        n = next(x for x in z.namelist() if x.endswith(".csv"))
        df = pd.read_csv(z.open(n), usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    s = df.set_index("Timestamp").resample("1s").last().ffill().dropna()
    return s["Bid"].to_numpy(), s["Ask"].to_numpy()

out = {}
for sym in MK:
    bid, ask = load(sym); mid = (bid + ask) / 2
    idx = np.arange(0, len(mid) - H, 60)
    du_mid, du_net, ever, spr = [], [], [], []
    tp = {k: [] for k in (1, 2, 3, 5, 10)}
    for i in idx:
        if ask[i] - bid[i] <= 0 or not np.isfinite(mid[i]): continue
        sp = (ask[i] - bid[i]) / mid[i] * 1e4; spr.append(sp)
        for side in (1, -1):
            m = (mid[i:i+H] - mid[i]) / mid[i] * 1e4 * side              # mid move in your favour, bps
            net = ((bid[i:i+H] - ask[i]) if side == 1 else (bid[i] - ask[i:i+H])) / mid[i] * 1e4   # closable P&L
            neg = np.argmax(m[1:] < 0) + 1 if (m[1:] < 0).any() else H
            du_mid.append(max(0.0, m[:neg].max()))
            stop = np.argmax(net < -2 * sp) if (net < -2 * sp).any() else H
            du_net.append(max(0.0, net[:max(stop, 1)].max())); ever.append(net.max() > 0)
            for k in tp:                                               # TP at +k spreads, stop at -k spreads (net)
                up = np.argmax(net >= k * sp) if (net >= k * sp).any() else H
                dn = np.argmax(net <= -k * sp) if (net <= -k * sp).any() else H
                tp[k].append(k * sp if up < dn else (-k * sp if dn < up else net[-1]))
    sp = float(np.median(spr))
    out[sym] = {"spread_bps": round(sp, 2), "trades": len(ever),
                "drawup_mid_bps_avg": round(float(np.mean(du_mid)), 2), "drawup_mid_bps_median": round(float(np.median(du_mid)), 2),
                "drawup_net_bps_avg": round(float(np.mean(du_net)), 2), "ever_in_profit_pct": round(100 * float(np.mean(ever)), 1),
                "tp_sl": {f"{k}x": {"win_pct": round(100 * float(np.mean(np.array(v) > 0)), 1), "bps_per_trade": round(float(np.mean(v)), 2)} for k, v in tp.items()}}
    o = out[sym]
    print(f"{sym:8} spread {sp:5.2f} bps | draw-up before negative: avg {o['drawup_mid_bps_avg']:5.2f} median {o['drawup_mid_bps_median']:5.2f} bps"
          f" = {o['drawup_mid_bps_avg']/sp:4.2f}x spread | net draw-up avg {o['drawup_net_bps_avg']:5.2f} | ever profitable {o['ever_in_profit_pct']}%")
    print("         TP=SL at " + "  ".join(f"{k}: win {v['win_pct']}% {v['bps_per_trade']:+.2f}bps" for k, v in o["tp_sl"].items()), flush=True)
json.dump(out, open(os.path.join(ROOT, "logs", "drawup_test.json"), "w"), indent=1)
