"""Indicator map: for every market and every LSTM input (164 indicator features), what did it tell us
BEFORE the market moved? Graded the same way as everything else: learned on the first 70% of the history,
checked on the last 30%.

Per market, per feature (value at the close of a 5-minute bar), against the next hour (12 bars):
  dir_train / dir_test   rank correlation with the next-hour move (direction signal)
  big_test               how well it flags an IMPACTFUL move in either direction (AUC - 0.5; size signal)
  up_test / dn_test      AUC - 0.5 for impactful UP moves / impactful DOWN moves specifically
  holds                  direction signal significant on BOTH halves with the same sign (strict bar: many tests)
Timeline (last 10 days per market): 15-minute closes, every impactful move, and the 6 features that were
most unusual (largest |z| vs their own recent history) just before it.
Writes logs/indicator_map.json for the war-room / artifact chart.
    PYTHONPATH=. .venv/bin/python tools/indicator_map.py
"""
import json
import math
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from Python.agi_brain import build_feature_frame  # noqa: E402

MARKETS = {"XAUUSD": "Gold", "BTCUSD": "Bitcoin", "EURUSD": "EUR/USD", "USDJPY": "USD/JPY", "XAGUSD": "Silver",
           "ETHUSD": "Ether", "USOIL": "Oil"}
H = 12
N_TESTS = 164 * 7
Z_BAR = abs(float(__import__("statistics").NormalDist().inv_cdf(0.025 / N_TESTS)))   # ~3.9 two-sided 5% after Bonferroni


def auc(score: np.ndarray, label: np.ndarray) -> float:
    pos = label.sum()
    neg = len(label) - pos
    if pos < 20 or neg < 20:
        return 0.5
    r = pd.Series(score).rank().to_numpy()
    return float((r[label].sum() - pos * (pos + 1) / 2) / (pos * neg))


def spearman(x: np.ndarray, y: np.ndarray) -> float:
    rx, ry = pd.Series(x).rank().to_numpy(), pd.Series(y).rank().to_numpy()
    return float(np.corrcoef(rx, ry)[0, 1])


def group(name: str) -> str:
    for g in ("rsi", "bb_width", "ema_rel", "atr_rel", "breakout", "slope", "momentum", "close_z", "realized_vol",
              "volume", "range", "ret", "close_ma_rel", "wick", "body", "gap", "hour", "dow", "month"):
        if g in name:
            return g
    return name.split("_")[0]


def run():
    out = {"z_bar": round(Z_BAR, 2), "markets": {}}
    for sym, label in MARKETS.items():
        a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        df = pd.DataFrame(a)
        df["time"] = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df.drop_duplicates("time").sort_values("time").reset_index(drop=True)
        f = build_feature_frame(df)
        f.index = df["time"].iloc[-len(f):].to_numpy() if len(f) != len(df) else df["time"].to_numpy()
        c = df.set_index("time")["close"].reindex(f.index)
        t_s = pd.Series(f.index.asi8 // 10**9, index=f.index)
        fwd = (c.shift(-H) / c - 1) * 1e4
        ok = fwd.notna() & ((t_s.shift(-H) - t_s) <= H * 300 * 1.5)
        typical = fwd.abs().rolling(2000, min_periods=500).median().shift(H)
        ok &= typical.notna()
        f, fwd, typical = f[ok], fwd[ok], typical[ok]
        big_up = (fwd > 2 * typical).to_numpy()
        big_dn = (fwd < -2 * typical).to_numpy()
        big = big_up | big_dn
        n = len(f)
        cut = int(n * 0.7)
        tr, te = slice(0, cut), slice(cut + H, n)
        y = fwd.to_numpy()
        feats = []
        for col in f.columns:
            x = f[col].to_numpy(float)
            if not np.isfinite(x).all() or np.nanstd(x) == 0:
                continue
            d_tr, d_te = spearman(x[tr], y[tr]), spearman(x[te], y[te])
            # overlapping 1-hour outcomes: about n/12 independent observations
            z_tr = d_tr * math.sqrt(cut / H)
            z_te = d_te * math.sqrt((n - cut - H) / H)
            feats.append({"f": col, "g": group(col),
                          "dir_train": round(d_tr, 4), "dir_test": round(d_te, 4), "z_test": round(z_te, 2),
                          "big_test": round(auc(np.abs(x[te] - np.median(x[tr])), big[te]) - 0.5, 4),
                          "up_test": round(auc(x[te], big_up[te]) - 0.5, 4),
                          "dn_test": round(auc(-x[te], big_dn[te]) - 0.5, 4),
                          "holds": bool(abs(z_tr) >= Z_BAR and abs(z_te) >= Z_BAR and np.sign(d_tr) == np.sign(d_te))})
        # combinations: the 25 most informative features (on TRAIN only), every pair, each in its low (<20th pct)
        # or high (>80th pct) state -> average next-hour move when both conditions hold. Graded train -> test.
        top = [x["f"] for x in sorted(feats, key=lambda x: -(abs(x["dir_train"]) + 0))[:25]]
        q = {k: (np.quantile(f[k].to_numpy()[tr], 0.2), np.quantile(f[k].to_numpy()[tr], 0.8)) for k in top}
        states = {}
        for k in top:
            x = f[k].to_numpy()
            states[(k, "low")] = x <= q[k][0]
            states[(k, "high")] = x >= q[k][1]
        combos, n_tests = [], 0
        keys = list(states)
        for i in range(len(keys)):
            for j in range(i + 1, len(keys)):
                if keys[i][0] == keys[j][0]:
                    continue
                m = states[keys[i]] & states[keys[j]]
                mtr, mte = m[tr], m[te]
                if mtr.sum() < 200 or mte.sum() < 60:
                    continue
                n_tests += 1
                ytr, yte = y[tr][mtr], y[te][mte]
                z = ytr.mean() / (ytr.std() / math.sqrt(max(1, len(ytr) / H)))
                combos.append({"a": f"{keys[i][0]} {keys[i][1]}", "b": f"{keys[j][0]} {keys[j][1]}",
                               "n_train": int(mtr.sum()), "train_bps": round(float(ytr.mean()), 2), "z_train": round(float(z), 2),
                               "n_test": int(mte.sum()), "test_bps": round(float(yte.mean()), 2),
                               "test_big_pct": round(100 * float(big[te][mte].mean()), 1)})
        zc = abs(float(__import__("statistics").NormalDist().inv_cdf(0.025 / max(1, n_tests * 7))))
        for cmb in combos:
            cmb["passes_train"] = abs(cmb["z_train"]) >= zc
            cmb["holds"] = bool(cmb["passes_train"] and np.sign(cmb["test_bps"]) == np.sign(cmb["train_bps"]))
        combos.sort(key=lambda x: -abs(x["z_train"]))
        combo_summary = {"tested": n_tests, "z_bar": round(zc, 2), "pass_train": sum(x["passes_train"] for x in combos),
                         "hold_test": sum(x["holds"] for x in combos), "top": combos[:30]}
        # timeline: last 10 days
        last = f.index[-1]
        win = f.index >= last - pd.Timedelta(days=10)
        mu = f.rolling(2000, min_periods=300).mean()
        sd = f.rolling(2000, min_periods=300).std().replace(0, np.nan)
        z = ((f - mu) / sd)[win]
        ev = []
        bu, bd = pd.Series(big_up, index=f.index)[win], pd.Series(big_dn, index=f.index)[win]
        last_t = None
        for t in z.index:
            if not (bu[t] or bd[t]):
                continue
            if last_t is not None and (t - last_t) < pd.Timedelta(minutes=60):
                continue                                   # one marker per move, not one per bar
            last_t = t
            row = z.loc[t].dropna()
            top = row.abs().sort_values(ascending=False).head(6)
            ev.append({"t": t.isoformat(), "dir": "up" if bu[t] else "down", "move_bps": round(float(fwd[t]), 1),
                       "price": float(c.loc[t]),
                       "top": [{"f": k, "z": round(float(row[k]), 2)} for k in top.index]})
        px = c.reindex(f.index)[win].resample("15min").last().dropna()
        out["markets"][sym] = {
            "name": label, "bars": int(n), "train_until": f.index[cut].isoformat(), "test_from": f.index[cut + H].isoformat(),
            "impactful_up_pct": round(100 * float(big_up.mean()), 1), "impactful_dn_pct": round(100 * float(big_dn.mean()), 1),
            "features": feats, "combos": combo_summary,
            "timeline": {"t": [x.isoformat() for x in px.index], "p": [round(float(v), 6) for v in px.to_numpy()], "events": ev},
        }
        hold = [x["f"] for x in feats if x["holds"]]
        best_big = sorted(feats, key=lambda x: -x["big_test"])[:3]
        best_dir = sorted(feats, key=lambda x: -abs(x["dir_test"]))[:3]
        print(f"{label:8} {n:6} bars | direction signals that hold on both halves: {len(hold)} {hold[:4]} | "
              f"best direction (test): {[(x['f'], x['dir_test']) for x in best_dir]} | "
              f"best big-move flag (test): {[(x['f'], x['big_test']) for x in best_big]} | {len(ev)} big moves in last 10 days | "
              f"combos: {n_tests} tested, {combo_summary['pass_train']} pass on train, {combo_summary['hold_test']} keep direction on test",
              flush=True)
    with open(os.path.join(ROOT, "logs", "indicator_map.json"), "w", encoding="utf-8") as fh:
        json.dump(out, fh)


if __name__ == "__main__":
    run()
