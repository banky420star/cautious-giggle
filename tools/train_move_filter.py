"""Train the per-market "big move coming" filter, graded train -> unseen test.

Question it answers at each 5-minute close: will the NEXT HOUR move more than twice the usual hour, either way?
Inputs: the same 164 indicator features the LSTM sees (build_feature_frame). Model: LightGBM, one per market.
TRAIN = first 70% of the history (the threshold is also chosen there), TEST = last 30%, never used to choose.
The filter says GO for the top 30% of moments by predicted chance (threshold fixed on TRAIN).
Reported on TEST:
  auc            0.5 = no skill, 1.0 = perfect
  big_rate       share of hours that are big moves: all hours vs GO hours vs STOP hours
  caught         share of all big moves that fell in GO hours (GO is 30% of the time)
  move_go/stop   average absolute 1-hour move in GO vs STOP hours, in bps, vs the round-trip cost
Saves models/move_filter/<SYM>.pkl  (model, feature list, threshold, test report)
    PYTHONPATH=. .venv/bin/python tools/train_move_filter.py
"""
import json
import os
import pickle
import sys

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.agi_brain import build_feature_frame  # noqa: E402

COST = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
H = 12
GO_SHARE = 0.30


def dataset(sym):
    a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
    df = pd.DataFrame(a)
    df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
    df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
    f = build_feature_frame(df)
    f.index = df["time"].iloc[-len(f):].to_numpy()
    c = df.set_index("time")["close"].reindex(f.index)
    ts = pd.Series(pd.DatetimeIndex(f.index).as_unit("s").asi8, index=f.index)
    fwd = (c.shift(-H) / c - 1) * 1e4
    typical = fwd.abs().rolling(2000, min_periods=500).median().shift(H)   # only past, finished hours
    ok = fwd.notna() & typical.notna() & ((ts.shift(-H) - ts) <= H * 300 * 1.5)
    f = f[ok].replace([np.inf, -np.inf], np.nan).astype("float32")
    return f, fwd[ok], typical[ok]


def run():
    report = {}
    print(f"{'market':8} {'auc':>5} | big-move share: all  GO  STOP | caught | avg |1h move| GO / STOP / cost (bps)")
    for sym, cost in COST.items():
        f, fwd, typ = dataset(sym)
        y = (fwd.abs() > 2 * typ).to_numpy().astype(int)
        n = len(f)
        cut = int(n * 0.7)
        te0 = cut + H                                   # gap so no training label overlaps the test period
        Xtr, ytr, Xte, yte = f.iloc[:cut], y[:cut], f.iloc[te0:], y[te0:]
        m = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.03, num_leaves=31, min_child_samples=200,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.6, reg_lambda=5.0, verbose=-1)
        # last 15% of TRAIN for early stopping, still before the test period
        v = int(cut * 0.85)
        m.fit(Xtr.iloc[:v], ytr[:v], eval_set=[(Xtr.iloc[v:], ytr[v:])], callbacks=[lgb.early_stopping(40, verbose=False)])
        p_tr, p_te = m.predict_proba(Xtr)[:, 1], m.predict_proba(Xte)[:, 1]
        thr = float(np.quantile(p_tr[v:], 1 - GO_SHARE))
        go = p_te >= thr
        mv = fwd.abs().to_numpy()[te0:]
        r = {"auc": round(float(roc_auc_score(yte, p_te)), 3), "threshold": thr,
             "go_share_test": round(float(go.mean()), 3),
             "big_all": round(float(yte.mean()), 3), "big_go": round(float(yte[go].mean()), 3), "big_stop": round(float(yte[~go].mean()), 3),
             "caught": round(float(yte[go].sum() / max(1, yte.sum())), 3),
             "move_go": round(float(mv[go].mean()), 2), "move_stop": round(float(mv[~go].mean()), 2), "cost": cost,
             "test_from": str(f.index[te0])[:10], "n_test": int(len(yte))}
        imp = sorted(zip(f.columns, m.booster_.feature_importance("gain")), key=lambda x: -x[1])[:8]
        r["top_inputs"] = [k for k, _ in imp]
        report[sym] = r
        with open(os.path.join(ROOT, "models", "move_filter", f"{sym}.pkl"), "wb") as fh:
            pickle.dump({"model": m, "features": list(f.columns), "threshold": thr, "horizon_bars": H, "report": r}, fh)
        print(f"{sym:8} {r['auc']:5.3f} | {r['big_all']:5.1%} {r['big_go']:5.1%} {r['big_stop']:5.1%} | {r['caught']:5.1%} | "
              f"{r['move_go']:6.1f} / {r['move_stop']:6.1f} / {cost:4.2f}   top: {', '.join(r['top_inputs'][:4])}", flush=True)
    with open(os.path.join(ROOT, "logs", "move_filter_report.json"), "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=1)


if __name__ == "__main__":
    run()
