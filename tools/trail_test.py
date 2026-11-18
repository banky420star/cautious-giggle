"""Stop-loss + close trailing stop that never lets a winner go back below break-even; re-enter after every close.

Real Exness ticks (October 2026), followed second by second. One trade at a time per market:
  * opens at the real ask (BUY) / bid (SELL); direction is a coin flip (nothing tested so far predicts it)
  * initial stop: SL x spread below the entry (net, after the spread)
  * once the trade can be closed at a net profit >= TRIG x spread, the stop jumps to break-even (net 0)
    and then trails TRAIL x spread behind the best net profit seen
  * stops are checked every second and filled at that second's real price, so a fast move can jump past
    the stop (slippage), exactly as in live trading
  * after a close the bot re-enters on the next second
Reports trades, win rate, average win / loss, how often a "protected" trade still closed negative, and the
total in bps of position value. 1 bps on a $1,000 position = $0.10.
"""
import json, os, zipfile
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MK = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
CONFIGS = [(5, 1, 0.5), (5, 1, 1), (10, 1, 1), (10, 2, 1)]     # (SL, TRIG, TRAIL) in spreads

def load(sym):
    with zipfile.ZipFile(os.path.join(ROOT, "data", "ticks", f"Exness_{sym}_2026_10.zip")) as z:
        n = next(x for x in z.namelist() if x.endswith(".csv"))
        df = pd.read_csv(z.open(n), usecols=["Timestamp", "Bid", "Ask"])
    df["Timestamp"] = pd.to_datetime(df["Timestamp"], utc=True, format="mixed")
    s = df.set_index("Timestamp").resample("1s").last().ffill().dropna()
    return s["Bid"].to_numpy(), s["Ask"].to_numpy()

def one_trade(bid, ask, i, side, sp, SL, TRIG, TRAIL):
    """Returns (exit index, net bps, was_protected)."""
    w = 900
    while True:
        end = min(len(bid), i + 1 + w)
        if side == 1: net = (bid[i+1:end] - ask[i]) / ask[i] * 1e4
        else: net = (bid[i] - ask[i+1:end]) / bid[i] * 1e4
        best = np.maximum.accumulate(np.concatenate([[-sp], net]))[:-1]         # best net seen BEFORE this second
        prot = best >= TRIG * sp
        stop = np.where(prot, np.maximum(0.0, best - TRAIL * sp), -SL * sp)
        hit = net <= stop
        if hit.any():
            j = int(np.argmax(hit)); return i + 1 + j, float(net[j]), bool(prot[j])
        if end >= len(bid): return len(bid) - 1, float(net[-1]), bool(prot[-1])
        w *= 4

rng = np.random.default_rng(7)
out = {}
for sym in MK:
    bid, ask = load(sym); sp = float(np.median((ask - bid) / ask * 1e4))
    for SL, TRIG, TRAIL in CONFIGS:
        i, res, prot, slip = 0, [], [], []
        while i < len(bid) - 2:
            side = 1 if rng.random() < 0.5 else -1
            j, r, p = one_trade(bid, ask, i, side, sp, SL, TRIG, TRAIL)
            res.append(r); prot.append(p); i = j + 1
        r, p = np.array(res), np.array(prot)
        k = f"{sym} SL {SL}x, break-even at +{TRIG}x, trail {TRAIL}x"
        o = {"spread_bps": round(sp, 2), "trades": len(r), "win_pct": round(100 * float((r > 0).mean()), 1),
             "avg_win_bps": round(float(r[r > 0].mean()), 2) if (r > 0).any() else 0,
             "avg_loss_bps": round(float(r[r <= 0].mean()), 2) if (r <= 0).any() else 0,
             "protected_but_closed_negative_pct": round(100 * float((r[p] < 0).mean()), 1) if p.any() else 0,
             "per_trade_bps": round(float(r.mean()), 3), "per_trade_in_spreads": round(float(r.mean()) / sp, 2),
             "total_bps": round(float(r.sum()))}
        out[k] = o
        print(f"{k:46} {o['trades']:6} trades | win {o['win_pct']:5.1f}% avg win {o['avg_win_bps']:+6.2f} avg loss {o['avg_loss_bps']:+6.2f} | "
              f"protected yet negative {o['protected_but_closed_negative_pct']:4.1f}% | per trade {o['per_trade_bps']:+.2f} bps ({o['per_trade_in_spreads']:+.2f} spreads) | total {o['total_bps']:+,}", flush=True)
json.dump(out, open(os.path.join(ROOT, "logs", "trail_test.json"), "w"), indent=1)
