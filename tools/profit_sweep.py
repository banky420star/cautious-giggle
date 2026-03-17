import itertools
import json
import os
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from Python.Server_AGI import _signal_to_exposure
from Python.agi_brain import SmartAGI
from Python.config_utils import DEFAULT_TRADING_SYMBOLS
from Python.data_feed import fetch_training_data
from Python.hybrid_brain import HybridBrain


def _to_1d_close(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        return arr[:, 0]
    return arr.reshape(-1)


def precompute_steps(df, agi, hb, start_idx=120):
    closes = _to_1d_close(df["close"])
    steps = []
    for i in range(start_idx, len(df) - 1):
        sl = df.iloc[: i + 1]
        pred = agi.predict(sl, production=True)
        conf = float(pred.get("confidence", 0.0))
        sig = str(pred.get("signal", "LOW_VOLATILITY"))
        close_vals = _to_1d_close(sl["close"])
        agi_base = float(_signal_to_exposure(sig, conf, close_vals))

        ppo_exp = hb.predict_ppo_exposure(str(sl["symbol"].iloc[-1]), sl)
        ppo_exp = None if ppo_exp is None else float(ppo_exp)

        c0 = float(closes[i])
        c1 = float(closes[i + 1])
        r = (c1 - c0) / (c0 + 1e-12)
        steps.append((conf, agi_base, ppo_exp, r))
    return steps


def score_combo(steps, hb, threshold, blend, cost_bps=1.5):
    hb.ppo_blend = float(blend)

    equity = 1.0
    peak = 1.0
    pos = 0.0
    trades = 0
    rets = []

    for conf, agi_base, ppo_exp, r in steps:
        agi_exp = agi_base if conf >= threshold else 0.0
        exp = float(hb.blend_exposure(float(agi_exp), ppo_exp, conf))

        dpos = float(abs(exp - pos))
        if dpos > 0.02:
            trades += 1

        cost = dpos * (cost_bps / 10000.0)
        step = float((exp * r) - cost)

        prev = float(equity)
        equity = float(equity * (1.0 + step))
        pos = exp
        peak = float(max(peak, equity))
        rets.append((equity - prev) / (prev + 1e-12))

    arr = np.asarray(rets, dtype=np.float64)
    vol = float(np.std(arr) + 1e-12)
    sharpe = float(np.mean(arr) / vol)
    max_dd = float((peak - equity) / (peak + 1e-12)) if peak > 0 else 1.0
    total_return = float(equity - 1.0)
    score = float((total_return * 100.0) - (max_dd * 100.0 * 2.0) + (sharpe * 5.0))

    return {
        "threshold": float(threshold),
        "blend": float(blend),
        "return": total_return,
        "max_drawdown": max_dd,
        "sharpe": sharpe,
        "trades": int(trades),
        "score": score,
    }


def main():
    os.environ["AGI_AUTONOMY_ENABLED"] = "false"

    symbol = os.environ.get("SWEEP_SYMBOL", DEFAULT_TRADING_SYMBOLS[0])
    period = os.environ.get("SWEEP_PERIOD", "7d")
    interval = os.environ.get("SWEEP_INTERVAL", "5m")

    df = fetch_training_data(symbol, period=period, interval=interval)
    if df is None or df.empty or len(df) < 400:
        rows = 0 if df is None else len(df)
        raise RuntimeError(f"insufficient data for sweep: symbol={symbol} rows={rows}")

    agi = SmartAGI()

    class _R:
        def can_trade(self):
            return True

    class _E:
        def reconcile_exposure(self, *args, **kwargs):
            return None

    hb = HybridBrain(_R(), _E())

    steps = precompute_steps(df, agi, hb)

    thresholds = [0.50, 0.55, 0.60, 0.65]
    blends = [0.35, 0.50, 0.65, 0.80]
    rows = [score_combo(steps, hb, th, bl) for th, bl in itertools.product(thresholds, blends)]
    rows.sort(key=lambda x: x["score"], reverse=True)

    out = {
        "symbol": symbol,
        "period": period,
        "interval": interval,
        "bars": len(df),
        "steps": len(steps),
        "best": rows[0],
        "top5": rows[:5],
    }

    out_path = PROJECT_ROOT / "logs" / "profit_sweep_latest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2), encoding="utf-8")

    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
