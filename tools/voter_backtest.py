"""Test the new method on history: does each voter (and the council with them) call the next 30 minutes?

Read-only: pulls past bars from the broker, replays every voter *causally* (only data up to each bar),
and scores its call against the move that followed.

  hit rate   share of active calls (|vote| > 0.05) that made money after paying the spread
  edge       average of (vote x move - |vote| x spread) in bps: > 0 means following it made money AFTER costs
  follow     average per trade taken (sign of vote x move - spread), the bot's view of one trade
  gross      the same edge before costs, for reference
Horizon = trading.model_memory.horizon_bars (the bot's trades are held ~8 min), same as live scoring.

The news reader is not replayed (old headlines weren't stored with enough history); it is judged live.
Result: logs/voter_backtest.json (shown in the war room).

    python tools/voter_backtest.py [--days 14] [--kronos-every 6] [--horizon BARS]
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from Python import external_voters as xv  # noqa: E402
from Python.indicator_council import council_vote, votes_mtf  # noqa: E402

H = 2  # forward horizon in M5 bars; run() takes it from trading.model_memory.horizon_bars


def _rates(mt5, sym, tf, n):
    r = mt5.copy_rates_from_pos(sym, tf, 1, n)
    if r is None or len(r) == 0:
        return None
    d = pd.DataFrame(r)
    d["time"] = pd.to_datetime(d["time"], unit="s", utc=True)
    return d


def ob_vote_series(m15: pd.DataFrame) -> np.ndarray:
    """Vote of the gold strategy at the close of every M15 bar (one causal pass of the frozen executor)."""
    o, h, l_, c = (m15[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    atr = xv._atr_wilder(h, l_, c)
    by_arm = {}
    for b, d, sp, lim, tg in xv.ob_events(o, c, h, l_):
        by_arm.setdefault(b + 1, []).append((d, sp, lim, tg))
    armed, poss = [], []
    out = np.zeros(len(o))
    for t in range(len(o)):
        keep = []
        for (d, sp, lim, tg, ei) in poss:
            done = (l_[t] <= sp or h[t] >= tg) if d == 1 else (h[t] >= sp or l_[t] <= tg)
            if not done and t - ei < 96:
                keep.append((d, sp, lim, tg, ei))
        poss = keep
        still = []
        for (d, sp, lim, tg) in armed:
            if (l_[t] <= lim) if d == 1 else (h[t] >= lim):
                hit = (l_[t] <= sp or h[t] >= tg) if d == 1 else (h[t] >= sp or l_[t] <= tg)
                if not hit:
                    poss.append((d, sp, lim, tg, t))
            elif not ((d == 1 and c[t] < sp) or (d == -1 and c[t] > sp)):
                still.append((d, sp, lim, tg))
        armed = still + list(by_arm.get(t, []))
        if poss:
            out[t] = float(np.clip(sum(p[0] for p in poss), -1, 1))
        else:
            near = [z for z in armed if atr[t] == atr[t] and abs(z[2] - c[t]) <= atr[t]]
            out[t] = 0.5 * float(np.sign(sum(z[0] for z in near))) if near else 0.0
    return out


def _spread_bps(mt5, sym, last_close):
    """Live bid/ask spread; falls back to the symbol's quoted spread (points) when there is no tick."""
    try:
        mt5.symbol_select(sym, True)
        tick = mt5.symbol_info_tick(sym)
        if tick is not None and float(tick.bid) > 0 and float(tick.ask) > float(tick.bid):
            return (float(tick.ask) - float(tick.bid)) / float(tick.bid) * 1e4
        info = mt5.symbol_info(sym)
        if info is not None and float(info.spread or 0) > 0 and last_close > 0:
            return float(info.spread) * float(info.point) / last_close * 1e4
    except Exception:
        pass
    return None


def score(v: np.ndarray, fwd: np.ndarray, cost: float = 0.0) -> dict:
    """Net of `cost` (bps per round trip, charged in proportion to the position taken)."""
    ok = np.isfinite(v) & np.isfinite(fwd)
    v, fwd = v[ok], fwd[ok]
    act = np.abs(v) > 0.05
    if act.sum() == 0:
        return {"n": 0, "active": 0, "hit": None, "edge_bps": None, "follow_bps": None, "gross_edge_bps": None}
    per_trade = np.sign(v[act]) * fwd[act] - cost
    return {"n": int(len(v)), "active": int(act.sum()), "hit": round(float((per_trade > 0).mean()), 3),
            "edge_bps": round(float((v * fwd - np.abs(v) * cost).mean()), 3),
            "follow_bps": round(float(per_trade.mean()), 3),
            "gross_edge_bps": round(float((v * fwd).mean()), 3), "cost_bps": round(float(cost), 3)}


def run(days: int = 14, kronos_every: int = 6, symbols=None, log=print, horizon: int | None = None) -> dict:
    global H
    if horizon is None:
        try:
            import yaml
            with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as fh:
                horizon = (((yaml.safe_load(fh) or {}).get("trading") or {}).get("model_memory") or {}).get("horizon_bars")
        except Exception:
            horizon = None
    H = max(1, int(horizon or 2))
    from Python.mt5_compat import mt5
    import torch
    torch.set_num_threads(2)
    mt5.initialize()
    cfg = xv._merge(xv.DEFAULTS, {})
    symbols = symbols or ["BTCUSDm", "XAUUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
    kr = xv._Kronos(cfg["kronos"])
    cb = xv._CatBoost(cfg["catboost"])
    w = cfg["weights"]
    n5 = days * 288
    res = {"t": time.time(), "days": days, "horizon_min": H * 5, "net_of_cost": True, "markets": {}, "note":
           "news reader is judged live only; scores are after the market's spread"}
    for sym in symbols:
        t0 = time.time()
        m5 = _rates(mt5, sym, mt5.TIMEFRAME_M5, n5 + 1700)
        m15 = _rates(mt5, sym, mt5.TIMEFRAME_M15, n5 // 3 + 900)
        h1 = _rates(mt5, sym, mt5.TIMEFRAME_H1, 900)
        h4 = _rates(mt5, sym, mt5.TIMEFRAME_H4, 400)
        if m5 is None or m15 is None or len(m5) < 2000:
            log(f"{sym}: not enough history")
            continue
        spread_bps = _spread_bps(mt5, sym, float(m5["close"].iloc[-1]))
        if spread_bps is None:
            log(f"{sym}: spread unknown, scoring before costs")
        close = m5["close"].to_numpy(float)
        fwd = np.full(len(close), np.nan)
        fwd[:-H] = np.log(close[H:] / close[:-H]) * 1e4
        m5i = m5.set_index("time")
        htf = {k: d.set_index("time")[["open", "high", "low", "close"]] for k, d in (("M15", m15), ("H1", h1), ("H4", h4))
               if d is not None}
        vt = votes_mtf(m5i[["open", "high", "low", "close"]], htf)
        start = len(m5) - n5
        idx = np.arange(start, len(m5) - H)
        # M15 bar that has closed by each M5 decision time (an M15 bar opened at T closes at T+15m)
        m15_close_t = (m15["time"] + pd.Timedelta(minutes=15)).to_numpy()
        dec_t = (m5["time"] + pd.Timedelta(minutes=5)).to_numpy()
        j15 = np.searchsorted(m15_close_t, dec_t, side="right") - 1
        votes = {}
        if sym in cfg["ob_gold"]["symbols"]:
            obs = ob_vote_series(m15)
            votes["ob_gold"] = np.array([obs[j15[i]] if j15[i] >= 0 else np.nan for i in idx])
        if sym in cfg["catboost"]["pairs"]:
            lg = np.full(len(idx), np.nan)
            model, trained_at = None, -10 ** 9
            pair = cfg["catboost"]["pairs"][sym]
            for k, i in enumerate(idx):
                j = j15[i]
                if j < 700:
                    continue
                if j - trained_at >= 24:  # retrain every 6 hours of M15 bars, like the live voter
                    model = cb._retrain(sym, m15.iloc[: max(0, j - 8) + 1].reset_index(drop=True))
                    trained_at = j
                if model is None:
                    continue
                win = m15.iloc[max(0, j - 400): j + 1].reset_index(drop=True)
                f = cb.features(win, pair)
                for col in [c for c in f.columns if "%-ema-period" in c]:
                    f[col] = f[col] / win["close"] - 1.0
                for col in [c for c in f.columns if "%-volume_gen" in c]:
                    f[col] = f[col] / win["tick_volume"].rolling(200, 20).mean()
                x = f[model["cols"]].tail(1)
                if not x.isna().any(axis=None):
                    lg[k] = math.tanh(float(model["model"].predict(x.to_numpy())[0]) / max(model["std"], 1e-6))
            votes["catboost"] = lg
        kv = np.full(len(idx), np.nan)
        for k, i in enumerate(idx):
            if k % kronos_every:
                continue
            try:
                v = kr.vote(m5.iloc[max(0, i - 400): i + 1])
                kv[k] = np.nan if v is None else v
            except Exception as exc:
                log(f"{sym} kronos error: {exc}")
                break
        votes["kronos"] = kv
        ind = {c: vt[c].to_numpy(float)[idx] for c in vt.columns}
        f_idx = fwd[idx]
        base = np.array([council_vote({c: ind[c][k] for c in ind}, {})["target"] for k in range(len(idx))])
        withm = np.array([council_vote({**{c: ind[c][k] for c in ind},
                                        **{n: votes[n][k] for n in votes if np.isfinite(votes[n][k])}}, {}, weights=w)["target"]
                          for k in range(len(idx))])
        kmask = np.isfinite(kv)  # compare on the bars where every voter had a say
        cost = float(spread_bps or 0.0)
        out = {name: score(v, f_idx, cost) for name, v in votes.items()}
        out["indicators_only"] = score(base[kmask], f_idx[kmask], cost)
        out["indicators_plus_models"] = score(withm[kmask], f_idx[kmask], cost)
        res["markets"][sym] = {"voters": out, "spread_bps": None if spread_bps is None else round(spread_bps, 2),
                               "bars": int(len(idx)), "from": str(m5["time"].iloc[start]), "to": str(m5["time"].iloc[-1]),
                               "seconds": round(time.time() - t0)}
        log(f"{sym}: " + json.dumps(res["markets"][sym]["voters"]))
        with open(os.path.join(ROOT, "logs", "voter_backtest.json"), "w", encoding="utf-8") as fh:
            json.dump(res, fh, indent=1)
    res["done"] = True
    with open(os.path.join(ROOT, "logs", "voter_backtest.json"), "w", encoding="utf-8") as fh:
        json.dump(res, fh, indent=1)
    return res


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=14)
    ap.add_argument("--kronos-every", type=int, default=6)
    ap.add_argument("--symbols", nargs="*")
    ap.add_argument("--horizon", type=int, help="bars ahead to score (default: trading.model_memory.horizon_bars)")
    a = ap.parse_args()
    run(a.days, a.kronos_every, a.symbols, log=lambda m: print(m, flush=True), horizon=a.horizon)
