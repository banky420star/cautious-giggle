"""Does the SMC read predict price? Replays cached Exness bars through Python/smc.py.

    .venv/bin/python tools/smc_eval.py                       # fixed rules, all cached markets
    .venv/bin/python tools/smc_eval.py --gemma 60 --symbols XAUUSD   # also ask Gemma on 60 sampled bars

For each sampled M15 bar it scores the bar with only the bars up to it (window of --window bars), then
measures the move over the next 4 and 8 M15 bars in ATRs. "edge" = average move in the vote's direction;
it has to beat the spread (~0.05-0.15 ATR on these markets) to be worth trading.
Writes logs/smc_eval.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python import smc  # noqa: E402
from Python import external_voters as xv  # noqa: E402

CACHE = os.path.join(ROOT, "data", "exness_cache")


def load_m15(sym: str) -> pd.DataFrame | None:
    p = os.path.join(CACHE, f"{sym}_M5.npy")
    if not os.path.exists(p):
        return None
    d = pd.DataFrame(np.load(p))
    d.index = pd.to_datetime(d["time"], unit="s", utc=True)
    m = d[["open", "high", "low", "close"]].resample("15min", label="left", closed="left").agg(
        {"open": "first", "high": "max", "low": "min", "close": "last"}).dropna()
    m["time"] = m.index
    return m.reset_index(drop=True)


def stats(v: np.ndarray, f: np.ndarray, thr: float) -> dict:
    m = np.abs(v) > thr
    if m.sum() < 10:
        return {"n": int(m.sum())}
    s = np.sign(v[m]) * f[m]
    return {"n": int(m.sum()), "hit": round(float((s > 0).mean()), 3), "edge_atr": round(float(s.mean()), 4),
            "edge_t": round(float(s.mean() / (s.std(ddof=1) / np.sqrt(len(s)) + 1e-12)), 2)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--symbols", default="XAUUSD,BTCUSD,ETHUSD,EURUSD,USDJPY,XAGUSD,USOIL")
    ap.add_argument("--window", type=int, default=600)
    ap.add_argument("--step", type=int, default=4)
    ap.add_argument("--bars", type=int, default=12000, help="how many recent M15 bars to replay")
    ap.add_argument("--gemma", type=int, default=0, help="bars per market to also ask Gemma about")
    ap.add_argument("--model", default=xv.DEFAULTS["smc_gemma"]["model"])
    ap.add_argument("--url", default=xv.DEFAULTS["smc_gemma"]["url"])
    a = ap.parse_args()
    report = {"t": time.time(), "window": a.window, "step": a.step, "markets": {}}
    for sym in a.symbols.split(","):
        m = load_m15(sym)
        if m is None or len(m) < a.window + 50:
            print(f"{sym}: no data")
            continue
        h, l, c = (m[x].to_numpy(float) for x in ("high", "low", "close"))
        atr = smc._atr(h, l, c)
        start = max(a.window, len(m) - a.bars)
        rows = []
        t0 = time.time()
        for t in range(start, len(m) - 8, a.step):
            an = smc.analyze(m.iloc[t - a.window + 1:t + 1])
            rv = smc.rules_vote(an)
            if rv is None:
                continue
            rows.append((t, rv[0], (c[t + 4] - c[t]) / atr[t], (c[t + 8] - c[t]) / atr[t], an))
        v = np.array([r[1] for r in rows])
        f4 = np.array([r[2] for r in rows])
        f8 = np.array([r[3] for r in rows])
        res = {"bars_scored": len(rows), "days": round((m["time"].iloc[-1] - m["time"].iloc[start]).days, 1),
               "rules_1h": stats(v, f4, 0.2), "rules_2h": stats(v, f8, 0.2),
               "rules_strong_2h": stats(v, f8, 0.45)}
        print(f"{sym}: {len(rows)} bars in {time.time() - t0:.0f}s  1h {res['rules_1h']}  2h {res['rules_2h']}  "
              f"strong 2h {res['rules_strong_2h']}")
        if a.gemma:
            rng = np.random.default_rng(7)
            pick = rng.choice(len(rows), size=min(a.gemma, len(rows)), replace=False)
            gv, rv_, g8 = [], [], []
            for i in sorted(pick):
                t, r, _, fwd8, an = rows[i]
                try:
                    js = xv.ollama_json(a.url, a.model, xv.SMC_PROMPT.format(asset=sym, facts=smc.describe(an, sym)),
                                        xv.SMC_SCHEMA)
                    gv.append(xv.smc_score(js))
                    rv_.append(r)
                    g8.append(fwd8)
                except Exception as e:
                    print(f"  gemma failed: {e}")
                    break
            if gv:
                gv, rv_, g8 = map(np.array, (gv, rv_, g8))
                res["gemma"] = {"model": a.model, "n": len(gv), "gemma_2h": stats(gv, g8, 0.2),
                                "rules_same_bars_2h": stats(rv_, g8, 0.2),
                                "agree_sign": round(float((np.sign(gv) == np.sign(rv_)).mean()), 3)}
                print(f"  gemma {res['gemma']}")
        report["markets"][sym] = res
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)
    with open(os.path.join(ROOT, "logs", "smc_eval.json"), "w") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    main()
