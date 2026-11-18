"""From the trades the bot actually took: how far did each go in our favour before it turned negative?

Replays every logged trade (logs/trade_events.jsonl: entry time, side, price, stop, target; close time from
position_removed) against the 1-minute price cache. Per trade:
  draw-up before negative = best favourable move (bar highs for a BUY, lows for a SELL) before the first
                            minute whose price went past the entry against us (bps of entry price)
  max draw-up             = best favourable move at any point while the trade was open
  went negative first     = the very first minute already traded below (BUY) / above (SELL) the entry
Compared with the trade's own target and stop distance and with the typical spread.
Approximation: 1-minute bars from the first full minute after entry (the entry minute itself is unseen; order inside a minute is unknown), bid-side prices.
"""
import json, os
from collections import defaultdict
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SPREAD = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
opens, closes = {}, {}
for ln in open(os.path.join(ROOT, "logs", "trade_events.jsonl")):
    e = json.loads(ln); p = e["payload"]; t = pd.Timestamp(e["ts"]).timestamp()
    if e["event"] == "trade_open": opens[p["ticket"]] = {**p, "t": t}
    elif e["event"] == "position_removed": closes[p["ticket"]] = t
cache = {}
def bars(sym):
    if sym not in cache:
        f = os.path.join(ROOT, "data", "exness_cache", f"{sym}_M1.npy")
        cache[sym] = np.load(f, allow_pickle=True) if os.path.exists(f) else None
    return cache[sym]
rows = []
for tk, o in opens.items():
    if tk not in closes: continue
    sym = o["symbol"].rstrip("m"); b = bars(sym)
    if b is None: continue
    t0, t1 = o["t"], closes[tk]
    sel = b[(b["time"] >= t0) & (b["time"] <= t1)]   # first FULL minute after entry onward
    if len(sel) == 0: continue
    s = 1 if o["side"] == "BUY" else -1; px = float(o["open_price"])
    fav = ((sel["high"] - px) if s == 1 else (px - sel["low"])) / px * 1e4
    adv = ((sel["low"] - px) if s == 1 else (px - sel["high"])) / px * 1e4
    negi = int(np.argmax(adv < 0)) if (adv < 0).any() else len(sel)
    rows.append({"ticket": tk, "sym": sym, "side": o["side"], "minutes": round((t1 - t0) / 60, 1),
                 "drawup_before_neg": max(0.0, float(fav[:negi].max())) if negi > 0 else 0.0,
                 "max_drawup": max(0.0, float(fav.max())), "neg_first": negi == 0,
                 "tp_bps": abs(float(o.get("tp") or px) - px) / px * 1e4, "sl_bps": abs(float(o.get("sl") or px) - px) / px * 1e4,
                 "spread": SPREAD.get(sym, 1.0)})
df = pd.DataFrame(rows)
df["reached_tp"] = df.max_drawup >= df.tp_bps
pd.set_option("display.width", 200)
g = df.groupby("sym").agg(trades=("ticket", "size"), mins=("minutes", "median"), spread=("spread", "first"),
                          neg_first_pct=("neg_first", lambda x: round(100 * x.mean())),
                          drawup_before_neg_avg=("drawup_before_neg", "mean"), drawup_before_neg_med=("drawup_before_neg", "median"),
                          max_drawup_avg=("max_drawup", "mean"), tp_avg=("tp_bps", "mean"), sl_avg=("sl_bps", "mean"),
                          reached_tp_pct=("reached_tp", lambda x: round(100 * x.mean()))).round(2)
print(g.to_string())
a = df
print(f"\nALL {len(a)} trades: went against us in the first minute {100*a.neg_first.mean():.0f}% | draw-up before negative avg "
      f"{a.drawup_before_neg.mean():.2f} bps, median {a.drawup_before_neg.median():.2f} | best move while open avg {a.max_drawup.mean():.2f} bps"
      f" | target avg {a.tp_bps.mean():.1f} bps, stop avg {a.sl_bps.mean():.1f} bps | reached target {100*a.reached_tp.mean():.0f}%")
for q in (0.25, 0.5, 0.75, 0.9):
    print(f"  {int(q*100)}% of trades got at most {a.drawup_before_neg.quantile(q):.2f} bps before going negative ({a.drawup_before_neg.quantile(q)/a.spread.median():.1f}x a typical spread)")
df.to_csv(os.path.join(ROOT, "logs", "trade_drawup.csv"), index=False)
