"""Is Kronos predictive? Score its 30-minute forecasts on past 5-minute bars, after costs.

For random past moments (last ~60 days of the broker's M5 cache), Kronos sees the 400 bars before that
moment (exactly what the bot gives it) and forecasts the next 6 bars (30 min). Two ways:
  single   one sampled path (what the bot does today: sample_count=1)
  mean16   the average of 16 sampled paths (a consensus forecast)
Scored against what really happened over those 30 minutes:
  hit      share of times the forecast direction was right
  IC       correlation between forecast move and actual move
  edge     average actual move in the forecast's direction (bps), compared with the cost of one trade
Only strong calls (forecast move above half its typical size) are also scored separately.
    PYTHONPATH=. .venv/bin/python tools/kronos_test.py [--n 200] [--markets XAUUSD ...]
"""
import argparse, json, os, sys, time
import numpy as np, pandas as pd
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.expanduser("~/.cautious-giggle/kronos"))
COST = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
H, CTX = 6, 400

def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--n", type=int, default=200)
    ap.add_argument("--markets", nargs="*", default=["XAUUSD", "BTCUSD", "EURUSD", "XAGUSD"])
    a = ap.parse_args()
    import torch
    from model import Kronos, KronosPredictor, KronosTokenizer
    tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
    mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
    pred = KronosPredictor(mdl, tok, max_context=512, device="cpu")
    rng = np.random.default_rng(5)
    out = {}
    for sym in a.markets:
        b = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        df = pd.DataFrame(b); df["ts"] = pd.to_datetime(df["time"], unit="s")
        n = len(df); lo = max(CTX, n - 60 * 288)
        cand = [i for i in range(lo, n - H) if (df["time"][i + H] - df["time"][i]) <= H * 300 * 1.5
                and (df["time"][i] - df["time"][i - CTX + 1]) <= CTX * 300 * 3]
        pts = sorted(rng.choice(cand, size=min(a.n, len(cand)), replace=False))
        rows = []; t0 = time.time()
        for k, i in enumerate(pts):
            d = df.iloc[i - CTX + 1:i + 1]
            x = pd.DataFrame({"open": d["open"].values, "high": d["high"].values, "low": d["low"].values,
                              "close": d["close"].values, "volume": d["tick_volume"].astype(float).values})
            ts = d["ts"].reset_index(drop=True)
            fut = pd.Series(pd.date_range(ts.iloc[-1] + pd.Timedelta(minutes=5), periods=H, freq="5min"))
            last = float(x["close"].iloc[-1]); sd = float(np.log(x["close"]).diff().tail(200).std()) * np.sqrt(H) * 1e4
            r = {"actual": (float(df["close"][i + H]) / last - 1) * 1e4, "typical": sd}
            for name, sc in (("single", 1), ("mean16", 16)):
                with torch.random.fork_rng(devices=[]), torch.inference_mode():
                    torch.manual_seed(17 + k)
                    o = pred.predict(df=x, x_timestamp=ts, y_timestamp=fut, pred_len=H, T=1.0, top_p=0.9,
                                     sample_count=sc, verbose=False)
                r[name] = (float(o["close"].tail(2).mean()) / last - 1) * 1e4
            rows.append(r)
            if k == 4:
                print(f"  {sym}: ~{(time.time() - t0) / 5:.1f}s per point", flush=True)
        R = pd.DataFrame(rows); c = COST[sym]; res = {"points": len(R), "cost_bps": c,
              "actual_abs_move_bps": round(float(R.actual.abs().median()), 2)}
        for name in ("single", "mean16"):
            s = np.sign(R[name]); edge = s * R.actual
            strong = R[name].abs() >= 0.5 * R.typical
            res[name] = {"hit_pct": round(100 * float((edge > 0).mean()), 1),
                         "IC": round(float(np.corrcoef(R[name], R.actual)[0, 1]), 3),
                         "edge_bps": round(float(edge.mean()), 2), "edge_t": round(float(edge.mean() / (edge.std() / np.sqrt(len(edge)))), 2),
                         "net_after_cost_bps": round(float(edge.mean()) - c, 2),
                         "strong_calls": int(strong.sum()),
                         "strong_hit_pct": round(100 * float((edge[strong] > 0).mean()), 1) if strong.any() else None,
                         "strong_edge_bps": round(float(edge[strong].mean()), 2) if strong.any() else None}
        out[sym] = res
        for name in ("single", "mean16"):
            q = res[name]
            print(f"{sym:7} {name:7} hit {q['hit_pct']:5.1f}% IC {q['IC']:+.3f} | edge {q['edge_bps']:+6.2f} bps (t={q['edge_t']:+.1f}) vs cost {c} -> net {q['net_after_cost_bps']:+.2f} | "
                  f"strong calls {q['strong_calls']}: hit {q['strong_hit_pct']}% edge {q['strong_edge_bps']}", flush=True)
        json.dump(out, open(os.path.join(ROOT, "logs", "kronos_test.json"), "w"), indent=1)

if __name__ == "__main__":
    main()
