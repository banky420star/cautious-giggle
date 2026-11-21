"""Does the US Dollar Index (DXY) add information? (asked 2026-11-21)

Exness has no DXY symbol, so it is rebuilt from its 6 currencies with the ICE formula:
  DXY = 50.14348112 * EURUSD^-0.576 * USDJPY^0.136 * GBPUSD^-0.119 * USDCAD^0.091 * USDSEK^0.042 * USDCHF^0.036
Two tests per market, train first 70% / judge the unseen 30% (1-hour gap), past-only features:
  A. direction of the next hour: LightGBM up/down, base indicators vs base + DXY; AUC and a trading check
     (trade the 15% most confident each way, net of the measured round-trip spread)
  B. the big-move filter (same label/settings as tools/train_move_filter.py): AUC base vs base + DXY
Run: PYTHONPATH=. .venv/bin/python tools/dxy_test.py
"""
import lightgbm as lgb  # noqa: I001  (must load before torch)
import json
import os
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from tools.train_move_filter import COST, H, dataset  # noqa: E402

W = {"EURUSD": -0.576, "USDJPY": 0.136, "GBPUSD": -0.119, "USDCAD": 0.091, "USDSEK": 0.042, "USDCHF": 0.036}


def closes(sym):
    a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
    s = pd.Series(a["close"].astype(float), index=pd.to_datetime(a["time"], unit="s", utc=True))
    return s[~s.index.duplicated()].sort_index()


def dxy_series():
    c = pd.concat({k: closes(k) for k in W}, axis=1).dropna()      # only bars where all 6 traded
    return np.exp(np.log(50.14348112) + sum(w * np.log(c[k]) for k, w in W.items()))


def dxy_features(dxy, mkt_close, idx):
    lr = np.log(dxy).diff()
    f = pd.DataFrame(index=dxy.index)
    for k in (1, 6, 12, 48):
        f[f"dxy_ret_{k}"] = (np.log(dxy) - np.log(dxy.shift(k))) * 1e4
    f["dxy_vol_48"] = lr.rolling(48).std() * 1e4
    f["dxy_vol_ratio"] = f["dxy_vol_48"] / (lr.rolling(576).std() * 1e4)
    f["dxy_z_48"] = (dxy - dxy.rolling(48).mean()) / dxy.rolling(48).std()
    f["dxy_trend_50h"] = np.sign(dxy - dxy.ewm(span=600, adjust=False).mean())
    # bars are stamped with OPEN time; both series close at the same moment, so same-stamp alignment is past-only
    f = f.reindex(idx)                                               # no fill: crypto weekend bars get NaN
    m = np.log(mkt_close.reindex(idx))
    mr, dr = m.diff(), np.log(dxy.reindex(idx)).diff()
    beta = (mr.rolling(2000, min_periods=500).cov(dr) / dr.rolling(2000, min_periods=500).var()).shift(1)
    f["dxy_corr_288"] = mr.rolling(288, min_periods=100).corr(dr)
    f["dxy_gap_12"] = ((m - m.shift(12)) - beta * (np.log(dxy.reindex(idx)) - np.log(dxy.reindex(idx).shift(12)))) * 1e4
    return f.replace([np.inf, -np.inf], np.nan).astype("float32")


def fit_auc(Xtr, ytr, Xte, yte):
    v = int(len(Xtr) * 0.85)
    m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200, subsample=0.8,
                           subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)
    m.fit(Xtr.iloc[:v], ytr[:v], eval_set=[(Xtr.iloc[v:], ytr[v:])], callbacks=[lgb.early_stopping(40, verbose=False)])
    p = m.predict_proba(Xte)[:, 1]
    return roc_auc_score(yte, p), p, m


def run():
    dxy = dxy_series()
    out = {}
    print(f"{'market':7} | DIRECTION auc base -> +DXY | bps/trade net base -> +DXY (n) | BIG MOVE auc base -> +DXY | top DXY input")
    for sym, cost in COST.items():
        f, fwd, typ = dataset(sym)
        g = dxy_features(dxy, closes(sym), f.index)
        fx = pd.concat([f, g], axis=1)
        n = len(f); cut = int(n * 0.7); te0 = cut + H
        r = {"dxy_coverage_test": round(float(g["dxy_ret_12"].iloc[te0:].notna().mean()), 3)}
        fw = fwd.to_numpy()
        # A. direction
        yd = (fw > 0).astype(int)
        res = {}
        for name, X in (("base", f), ("dxy", fx)):
            auc, p, m = fit_auc(X.iloc[:cut], yd[:cut], X.iloc[te0:], yd[te0:])
            hi, lo = np.quantile(p, 0.85), np.quantile(p, 0.15)
            mv = fw[te0:]
            trades = np.concatenate([mv[p >= hi], -mv[p <= lo]]) - cost
            res[name] = (auc, float(trades.mean()), len(trades), m)
        # B. big-move filter
        yb = (np.abs(fw) > 2 * typ.to_numpy()).astype(int)
        b0, _, _ = fit_auc(f.iloc[:cut], yb[:cut], f.iloc[te0:], yb[te0:])
        b1, _, mb = fit_auc(fx.iloc[:cut], yb[:cut], fx.iloc[te0:], yb[te0:])
        imp = dict(zip(fx.columns, mb.booster_.feature_importance("gain")))
        dimp = sorted(((k, v) for k, v in imp.items() if k.startswith("dxy")), key=lambda x: -x[1])
        share = sum(v for _, v in dimp) / max(1e-9, sum(imp.values()))
        r.update(dir_auc_base=round(res["base"][0], 4), dir_auc_dxy=round(res["dxy"][0], 4),
                 bps_base=round(res["base"][1], 2), bps_dxy=round(res["dxy"][1], 2), n_trades=res["dxy"][2],
                 big_auc_base=round(b0, 4), big_auc_dxy=round(b1, 4), dxy_gain_share_big=round(share, 3),
                 top_dxy=[k for k, _ in dimp[:3]], cost=cost, test_from=str(f.index[te0])[:10])
        out[sym] = r
        print(f"{sym:7} | {r['dir_auc_base']:.3f} -> {r['dir_auc_dxy']:.3f} | {r['bps_base']:+6.2f} -> {r['bps_dxy']:+6.2f} "
              f"({r['n_trades']}) | {r['big_auc_base']:.3f} -> {r['big_auc_dxy']:.3f} | {r['top_dxy'][0]} "
              f"({share:.0%} of filter gain, DXY known {r['dxy_coverage_test']:.0%} of test bars)", flush=True)
    json.dump(out, open(os.path.join(ROOT, "logs", "dxy_test.json"), "w"), indent=1)


if __name__ == "__main__":
    run()
