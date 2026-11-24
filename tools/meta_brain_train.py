"""Build, walk-forward test and save the meta-brain (Python/meta_brain.py).

    .venv/bin/python tools/meta_brain_train.py            # build features (cached), walk-forward, save model
    .venv/bin/python tools/meta_brain_train.py --rebuild  # recompute the cached features

Walk-forward: for every calendar month after the first four, the model is trained only on bars that
closed more than a day before that month starts, then trades that month. Every strategy is simulated
the same way: one position per market at a time, live stops, spread paid, results in R (1R = the stop).
Writes logs/meta_brain_report.json and models/meta_brain/{long,short}.txt + meta.json.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from multiprocessing import get_context

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python import meta_brain as mb
from Python.lux_smc import lux_smc  # noqa: E402
from Python.indicator_council import resample_ohlc  # noqa: E402

CACHE = os.path.join(ROOT, "data", "exness_cache")
OUT = os.path.join(ROOT, "data", "meta_brain")
PARAMS = dict(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=300, subsample=0.8,
              subsample_freq=1, colsample_bytree=0.7, reg_lambda=5.0, verbose=-1)


def load_m5(sym):
    d = pd.DataFrame(np.load(os.path.join(CACHE, f"{sym}_M5.npy")))
    d.index = pd.to_datetime(d["time"], unit="s", utc=True)
    return d[["open", "high", "low", "close"]].astype(float)


def build(sym):
    t0 = time.time()
    m5 = load_m5(sym)
    m15 = resample_ohlc(m5, 15)
    m15["time"] = m15.index
    sm = mb.smc_frame(m15)
    f = mb.features(m5, sym, sm)
    y = mb.labels(m5, sym)
    d = f.join(y)
    d["symbol"] = sym
    d["bar"] = np.arange(len(d))
    d = d.iloc[2000:]
    d.to_pickle(os.path.join(OUT, f"{sym}.pkl"))
    return sym, len(d), round(time.time() - t0)


def simulate(df: pd.DataFrame, sig: np.ndarray) -> dict:
    """One position per market at a time. sig: +1 long, -1 short, 0 none (aligned to df rows)."""
    rs, per = [], {}
    for sym, g in df.assign(_sig=sig).groupby("symbol", sort=False):
        g = g.sort_values("bar")
        busy_until = -1
        bars, s = g["bar"].to_numpy(), g["_sig"].to_numpy()
        rl, rsh = g["r_long"].to_numpy(), g["r_short"].to_numpy()
        el, es = g["end_long"].to_numpy(), g["end_short"].to_numpy()
        out = []
        for i in range(len(g)):
            if bars[i] <= busy_until or s[i] == 0:
                continue
            r, e = (rl[i], el[i]) if s[i] > 0 else (rsh[i], es[i])
            if not np.isfinite(r) or e < 0:
                continue
            out.append(r)
            busy_until = e
        per[sym] = {"n": len(out), "R": round(float(np.sum(out)), 1),
                    "avgR": round(float(np.mean(out)), 3) if out else None}
        rs += out
    rs = np.array(rs)
    if not len(rs):
        return {"n": 0}
    eq = np.cumsum(rs)
    return {"n": int(len(rs)), "totalR": round(float(rs.sum()), 1), "avgR": round(float(rs.mean()), 3),
            "win": round(float((rs > 0).mean()), 3),
            "pf": round(float(rs[rs > 0].sum() / max(1e-9, -rs[rs < 0].sum())), 2),
            "maxDD_R": round(float((np.maximum.accumulate(eq) - eq).max()), 1), "markets": per}


def baselines(d: pd.DataFrame) -> dict:
    s = lambda x: np.sign(x).astype(float)  # noqa: E731
    return {
        "council (all 48 indicators, equal)": np.where(d["council"].abs() > 0.35, s(d["council"]), 0.0),
        "higher-timeframe trend only (H1+H4)": np.where(d["htf_mean"].abs() > 0.5, s(d["htf_mean"]), 0.0),
        "bankbot M15 crossover": np.where(d["bank_age@M15"] == 0, d["bank_state@M15"], 0.0),
        "bankbot H1 crossover": np.where(d["bank_age@H1"] == 0, d["bank_state@H1"], 0.0),
        "ICT FVG->iFVG entry": d["ict_fresh"].to_numpy(float),
        "Sweep + FVG reversal": d["sfr_fresh"].to_numpy(float),
        "LuxAlgo internal CHoCH with swing trend": d["lux_int_fresh"].to_numpy(float),
        "LuxAlgo swing BOS/CHoCH": np.where(d["lux_swing_age"] == 0, np.sign(d["lux_swing_ev"]), 0.0),
        "Gold Breakout signal (first range break)": d["gbo_fresh"].to_numpy(float),
        "SMC fixed rules": np.where(d["smc_rules"].abs() > 0.3, s(d["smc_rules"]), 0.0),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--sample", type=int, default=3, help="train on every Nth M5 bar")
    a = ap.parse_args()
    os.makedirs(OUT, exist_ok=True)
    todo = [s for s in mb.SYMBOLS if a.rebuild or not os.path.exists(os.path.join(OUT, f"{s}.pkl"))]
    if todo:
        print(f"building features for {todo} ...", flush=True)
        with get_context("spawn").Pool(min(7, len(todo))) as p:
            for r in p.imap_unordered(build, todo):
                print("  built", r, flush=True)
    import lightgbm as lgb
    parts = []
    for s in mb.SYMBOLS:
        p = pd.read_pickle(os.path.join(OUT, f"{s}.pkl"))
        if "sfr_fresh" not in p.columns:      # strategies added after the cache was built: cheap, add in place
            p = p.join(mb.sweep_fvg(load_m5(s)))
            p.to_pickle(os.path.join(OUT, f"{s}.pkl"))
        if "lux_int_fresh" not in p.columns:
            m5 = load_m5(s)
            if not isinstance(m5.index, pd.DatetimeIndex):
                m5.index = pd.to_datetime(m5["time"], unit="s", utc=True)
            p = p.join(lux_smc(m5[["open", "high", "low", "close"]].astype(float)))
            p.to_pickle(os.path.join(OUT, f"{s}.pkl"))
        if "gbo_dir" not in p.columns:
            p = p.join(mb.range_breakout(load_m5(s)))
            p.to_pickle(os.path.join(OUT, f"{s}.pkl"))
        parts.append(p)
    d = pd.concat(parts)
    d = d[np.isfinite(d["r_long"]) & np.isfinite(d["r_short"])]
    drop = {"r_long", "r_short", "end_long", "end_short", "symbol", "bar"}
    cols = [c for c in d.columns if c not in drop and not c.startswith("_")]
    hi = max(mb.STOPS["default"][1] / mb.STOPS["default"][0], 1.6)
    months = d.index.to_period("M").unique().sort_values()
    margins = [0.0, 0.05, 0.1, 0.2, 0.3]
    preds = []
    for m in months[4:]:
        start = m.to_timestamp().tz_localize("UTC")
        end = (m + 1).to_timestamp().tz_localize("UTC")
        tr = d[d.index < start - pd.Timedelta(days=1)].iloc[::a.sample]
        te = d[(d.index >= start) & (d.index < end)]
        if len(tr) < 20000 or not len(te):
            continue
        ml = lgb.LGBMRegressor(**PARAMS).fit(tr[cols], tr["r_long"].clip(-1.5, hi))
        ms = lgb.LGBMRegressor(**PARAMS).fit(tr[cols], tr["r_short"].clip(-1.5, hi))
        te = te.assign(p_long=ml.predict(te[cols]), p_short=ms.predict(te[cols]))
        preds.append(te)
        print(f"{m}: trained on {len(tr)} rows, tested {len(te)}", flush=True)
    wf = pd.concat(preds)
    report = {"t": time.time(), "period": [str(wf.index.min()), str(wf.index.max())], "strategies": {}}
    for name, sig in baselines(wf).items():
        report["strategies"][name] = simulate(wf, sig)
    for mg in margins:
        sig = np.where((wf["p_long"] > mg) & (wf["p_long"] >= wf["p_short"]), 1.0,
                       np.where(wf["p_short"] > mg, -1.0, 0.0))
        report["strategies"][f"META-BRAIN (trade if predicted > {mg:+.2f}R)"] = simulate(wf, sig)
    print("\nwalk-forward", report["period"])
    for k, v in report["strategies"].items():
        print(f"  {k:45s} n {v.get('n'):5d}  avgR {v.get('avgR')}  totalR {v.get('totalR')}  win {v.get('win')}  "
              f"pf {v.get('pf')}  maxDD {v.get('maxDD_R')}")
    # pick the margin by walk-forward result (needs enough trades), then train on everything
    ok = [(v["avgR"], mg) for mg in margins
          for k, v in [(f"META-BRAIN (trade if predicted > {mg:+.2f}R)", report["strategies"][f"META-BRAIN (trade if predicted > {mg:+.2f}R)"])]
          if v.get("n", 0) >= 300]
    margin = max(ok)[1] if ok else 0.2
    full = d.iloc[::a.sample]
    ml = lgb.LGBMRegressor(**PARAMS).fit(full[cols], full["r_long"].clip(-1.5, hi))
    ms = lgb.LGBMRegressor(**PARAMS).fit(full[cols], full["r_short"].clip(-1.5, hi))
    os.makedirs(mb.MODEL_DIR, exist_ok=True)
    ml.booster_.save_model(os.path.join(mb.MODEL_DIR, "long.txt"))
    ms.booster_.save_model(os.path.join(mb.MODEL_DIR, "short.txt"))
    # the live bot reads these JSON dumps (numpy evaluator): lightgbm + torch in one process segfaults on macOS
    json.dump(ml.booster_.dump_model(), open(os.path.join(mb.MODEL_DIR, "long.json"), "w"))
    json.dump(ms.booster_.dump_model(), open(os.path.join(mb.MODEL_DIR, "short.json"), "w"))
    imp = pd.Series(ml.booster_.feature_importance("gain") + ms.booster_.feature_importance("gain"), index=cols)
    report["margin"] = margin
    report["top_features"] = {k: round(float(v), 1) for k, v in imp.sort_values(ascending=False).head(25).items()}
    json.dump({"features": cols, "margin": margin, "trained_rows": len(full), "stops": mb.STOPS,
               "horizon": mb.HORIZON, "walk_forward": report["strategies"]},
              open(os.path.join(mb.MODEL_DIR, "meta.json"), "w"), indent=1)
    json.dump(report, open(os.path.join(ROOT, "logs", "meta_brain_report.json"), "w"), indent=1)
    print("\nmargin", margin, "\ntop features", list(report["top_features"])[:15])


if __name__ == "__main__":
    main()
