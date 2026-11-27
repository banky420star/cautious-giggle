"""Does each market's LSTM have an edge? Scores its BUY / SELL calls on the latest 30 days, after the spread.

    .venv/bin/python tools/lstm_eval.py      # writes logs/lstm_eval.json (the bot reads it: Python/lstm_gate.py)

A market's LSTM counts as PROVEN only if, at the horizon it was trained for (meta horizon_bars; 1 bar for the old
label), its calls were right more often than wrong (hit > 51 %) AND made money on average after the spread, over at
least MIN_CALLS calls. Until then the bot ignores it (no vote, no size cut, the LLM traders are told it is unproven),
and it switches itself back on the day a retrained model passes. Run daily by the dashboard service.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "logs", "lstm_eval.json")
DAYS = 30
MIN_CALLS = 200
MIN_HIT = 0.51


def evaluate(sym_m: str) -> dict:
    import joblib
    import torch
    from Python.agi_brain import AGIModel
    from Python.feature_pipeline import build_lstm_feature_frame
    from Python.meta_brain import SPREAD_BPS, base

    d = os.path.join(ROOT, "models", "per_symbol")
    paths = [os.path.join(d, f) for f in (f"lstm_{sym_m}.pt", f"lstm_scaler_{sym_m}.pkl", f"lstm_{sym_m}.meta.json")]
    if not all(os.path.exists(p) for p in paths):
        return {"proven": False, "why": "no trained LSTM for this market"}
    meta = json.load(open(paths[2], encoding="utf-8"))
    h = int(meta.get("horizon_bars", 1) or 1)
    inst = base(sym_m) if base(sym_m) != sym_m else (sym_m[:-1] if sym_m.endswith("m") else sym_m)
    cache = os.path.join(ROOT, "data", "exness_cache", f"{inst}_M5.npy")
    if not os.path.exists(cache):
        return {"proven": False, "why": "no price history cached"}
    a = np.load(cache)[-(DAYS * 288 + 3000):]
    df = pd.DataFrame({k: a[k].astype(float) for k in ("open", "high", "low", "close")})
    df["volume"] = a["tick_volume"].astype(float)
    df["tick_volume"] = df["volume"]
    df["time"] = pd.to_datetime(a["time"], unit="s", utc=True)
    fdf, cols = build_lstm_feature_frame(df.copy(), meta.get("feature_version", "ultimate_150"))
    cols = meta.get("feature_columns") or cols
    X = joblib.load(paths[1]).transform(fdf[cols].values.astype(np.float32)).astype(np.float32)
    seq = int(meta.get("seq_len", 60))
    m = AGIModel(input_dim=len(cols))
    m.load_state_dict(torch.load(paths[0], map_location="cpu", weights_only=True))
    m.eval()
    idx = np.arange(max(seq, len(X) - DAYS * 288), len(X) - h - 1, 3)
    with torch.no_grad():
        p = torch.cat([torch.softmax(m(torch.tensor(np.stack([X[i - seq:i] for i in idx[k:k + 512]]))), 1)
                       for k in range(0, len(idx), 512)]).numpy()
    c = df["close"].to_numpy()[-len(X):]
    cls = p.argmax(1)
    side = np.where(cls == 1, 1, np.where(cls == 2, -1, 0))
    sel = side != 0
    spread = SPREAD_BPS.get(inst, 1.0)
    r = ((c[idx - 1 + h] / c[idx - 1] - 1) * 1e4 * side - spread)[sel]
    out = {"horizon_bars": h, "labels": meta.get("labels"), "calls": int(sel.sum()),
           "buy_sell_share": round(float(sel.mean()), 3), "hit": round(float((r > 0).mean()), 3) if len(r) else None,
           "avg_bps": round(float(r.mean()), 2) if len(r) else None, "spread_bps": spread,
           "trained": time.strftime("%Y-%m-%d", time.localtime(os.path.getmtime(paths[0])))}
    ok = len(r) >= MIN_CALLS and out["hit"] > MIN_HIT and out["avg_bps"] > 0
    out["proven"] = bool(ok)
    out["why"] = "" if ok else (f"only {len(r)} calls" if len(r) < MIN_CALLS else
                                f"right {out['hit']:.0%} of the time, {out['avg_bps']:+.2f} bps per call after the spread")
    return out


def main() -> int:
    from Python.config_utils import load_project_config
    cfg = load_project_config(ROOT)
    t = cfg.get("trading") or {}
    syms = list(dict.fromkeys(list(t.get("symbols") or []) + list((t.get("news_trading") or {}).get("symbols") or [])))
    res = {"at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "days": DAYS, "markets": {}}
    for s in syms:
        try:
            res["markets"][s] = evaluate(s)
        except Exception as exc:
            res["markets"][s] = {"proven": False, "why": f"evaluation failed: {type(exc).__name__}: {exc}"[:200]}
        r = res["markets"][s]
        print(f"{s:9s} {'PROVEN' if r['proven'] else 'ignored'}  {r.get('why') or ''}", flush=True)
    with open(OUT + ".tmp", "w", encoding="utf-8") as f:
        json.dump(res, f, indent=1)
    os.replace(OUT + ".tmp", OUT)
    return 0


if __name__ == "__main__":
    sys.exit(main())
