"""Meta-brain: one model that learns how much to trust every signal the bot has, from results.

Instead of fixed weights, a gradient-boosted model reads, on every closed M5 bar:

    council     all 12 indicator strategies on M5/M15/H1/H4 (Python/indicator_council.py, 48 votes)
    bankbot     the user's open/close SMMA(8) crossover strategy (Pine "bankbot by bank", alternate
                resolution 3x), ported WITHOUT the repainting lookahead: only closed M15/H1 bars
    ict         ICT Entry V2 [TS_Indie] style FVG -> inverse FVG (iFVG) entries, ATR(50) stop, pivot-10 trend
    sweep_fvg   the user's "Sweep + FVG Reversal": liquidity sweep -> structure break -> FVG limit entry
    smc         Smart Money Concepts facts from Python/smc.py on M15 + H1
    regime      volatility, trend strength, session, cost of the spread in ATRs

and predicts the result, in R (multiples of the stop distance, after the spread), of a long and of a
short opened at that close with the live stops (trading.stops in config.yaml). The bot should only
trade where the predicted R beats zero by a margin. tools/meta_brain_train.py builds the data,
runs the walk-forward test against the current council, and saves the model to models/meta_brain/.
"""
from __future__ import annotations

import json
import os

import numpy as np
import pandas as pd

from Python import smc
from Python.lux_smc import lux_smc
from Python.indicator_council import _atr, _ema, align_to_m5, resample_ohlc, votes_mtf

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(ROOT, "models", "meta_brain")
SYMBOLS = ["XAUUSD", "BTCUSD", "ETHUSD", "EURUSD", "USDJPY", "XAGUSD", "USOIL"]
SPREAD_BPS = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42,
              "ETHUSD": 4.1, "USOIL": 2.09}          # measured live (logs/drawup_test.json)
STOPS = {"default": (2.0, 3.2), "XAUUSD": (1.5, 2.4)}  # (sl, tp) in M5 ATR(14), as trading.stops
HORIZON = 96                                           # bars (8 h) before a trade is closed at market


def base(symbol: str) -> str:
    return symbol[:-1] if symbol.endswith("m") and symbol[:-1] in SPREAD_BPS else symbol


def _rma(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(alpha=1.0 / n, adjust=False).mean()


# ------------------------------------------------------------------------------- bankbot (user's Pine)
def bankbot(src: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """Open/close SMMA crossover on one timeframe: gap in ATRs, state, bars since the last cross."""
    cs, os_ = _rma(src["close"], n), _rma(src["open"], n)
    atr = _atr(src)
    gap = ((cs - os_) / atr.replace(0, np.nan)).clip(-3, 3).fillna(0)
    state = np.sign(gap)
    flip = state.ne(state.shift()).cumsum()
    age = state.groupby(flip).cumcount().clip(upper=50)
    return pd.DataFrame({"gap": gap, "state": state, "age": age.astype(float)}, index=src.index)


# ------------------------------------------------------------------------------- ICT FVG -> iFVG
def _trend_series(o, h, l, c, k):
    """Per-bar structure trend (+1/-1/0) with pivots of strength k, confirmed k bars later."""
    ph, pl = smc._pivots(h, l, k)
    n, hi_i, lo_i = len(c), 0, 0
    last_hi = last_lo = None
    tr, out = 0, np.zeros(n)
    for t in range(n):
        while hi_i < len(ph) and ph[hi_i][0] + k <= t:
            last_hi = ph[hi_i][1]
            hi_i += 1
        while lo_i < len(pl) and pl[lo_i][0] + k <= t:
            last_lo = pl[lo_i][1]
            lo_i += 1
        if last_hi is not None and c[t] > last_hi:
            tr, last_hi = 1, None
        elif last_lo is not None and c[t] < last_lo:
            tr, last_lo = -1, None
        out[t] = tr
    return out


def ict(df: pd.DataFrame, gap_min=0.01, gap_max=0.2, fvg_age=30, trend_k=10, hold=6) -> pd.DataFrame:
    """Buy: a bearish FVG is closed through upward (it becomes a bullish iFVG) while a bullish FVG formed in
    the last `fvg_age` bars sits just below it (gap between the boxes in [gap_min, gap_max] x ATR(50)),
    in a bullish pivot-10 structure. Sell is the mirror. The Pine script's entry/stop: FVG edge, 0.5x ATR(50)."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    atr50 = _rma(pd.Series(np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))), 50).to_numpy()
    trend = _trend_series(o, h, l, c, trend_k)
    bull, bear = [], []          # active FVGs: (bar, low, high)
    sig = np.zeros(n)
    idir, iage, idist = np.zeros(n), np.full(n, 99.0), np.zeros(n)
    last_i = None                # (dir, bar, low, high)
    for t in range(2, n):
        if l[t] > h[t - 2]:
            bull.append((t, h[t - 2], l[t]))
        if h[t] < l[t - 2]:
            bear.append((t, h[t], l[t - 2]))
        new_bull_i = [z for z in bear if z[0] < t and c[t] > z[2]]
        new_bear_i = [z for z in bull if z[0] < t and c[t] < z[1]]
        bear = [z for z in bear if z not in new_bull_i and t - z[0] <= 300][-30:]
        bull = [z for z in bull if z not in new_bear_i and t - z[0] <= 300][-30:]
        a = atr50[t]
        if new_bull_i:
            z = new_bull_i[-1]
            last_i = (1, t, z[1], z[2])
            fv = [f for f in bull if t - f[0] <= fvg_age and gap_min * a <= z[1] - f[2] <= gap_max * a]
            if fv and trend[t] >= 0:
                sig[t] = 1.0
        if new_bear_i:
            z = new_bear_i[-1]
            last_i = (-1, t, z[1], z[2])
            fv = [f for f in bear if t - f[0] <= fvg_age and gap_min * a <= f[1] - z[2] <= gap_max * a]
            if fv and trend[t] <= 0:
                sig[t] = -1.0
        if last_i is not None and a > 0:
            idir[t], iage[t] = last_i[0], min(99, t - last_i[1])
            idist[t] = np.clip((c[t] - (last_i[2] + last_i[3]) / 2) / a, -10, 10)
    held = pd.Series(sig).replace(0, np.nan).ffill(limit=hold).fillna(0).to_numpy()
    return pd.DataFrame({"ict_sig": held, "ict_fresh": sig, "ict_ifvg_dir": idir, "ict_ifvg_age": iage,
                         "ict_ifvg_dist": idist, "ict_trend10": trend}, index=df.index)


# ------------------------------------------------------------------------------- Sweep + FVG reversal
def sweep_fvg(df: pd.DataFrame, k=3, max_rev=10, min_fvg=0.15, min_pierce=0.05, buf=0.1, max_stop=3.0,
              window=20) -> pd.DataFrame:
    """Port of the user's "Sweep + FVG Reversal [Strategy]" (liquidity-sweep mode, reclaim window 1).

    Short: with swing lows still rising, a swing high is pierced by >= min_pierce ATR and the same candle closes
    back below it in its lower half; within max_rev bars price closes below the last swing low (structure break)
    and a bearish FVG >= min_fvg ATR formed since the sweep -> sell limit at the FVG's near edge, stop
    buf ATR beyond the sweep extreme (skipped if > max_stop ATR). Long is the mirror."""
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    atr = _rma(pd.Series(np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))), 14).to_numpy()
    ph, pl = smc._pivots(h, l, k)
    ph_at = {p + k: (p, v) for p, v in ph}     # confirmed at bar p+k
    pl_at = {p + k: (p, v) for p, v in pl}
    last_h = prior_h = last_l = prior_l = None
    h_swept = l_swept = False
    setup = None          # (dir, sweep_bar, break_level, extreme)
    fresh = np.zeros(n)
    entry = np.full(n, np.nan)
    stop_atr = np.full(n, np.nan)
    stop_px = np.full(n, np.nan)
    for t in range(2, n):
        if t in ph_at:
            prior_h, last_h, h_swept = last_h, ph_at[t], False
        if t in pl_at:
            prior_l, last_l, l_swept = last_l, pl_at[t], False
        a = atr[t]
        hl2 = (h[t] + l[t]) / 2
        if last_h is not None and not h_swept and last_h[0] < t and h[t] > last_h[1]:
            if c[t] < last_h[1]:
                if h[t] - last_h[1] >= min_pierce * a and c[t] < hl2:
                    h_swept = True
                    if last_l is not None and prior_l is not None and last_l[1] > prior_l[1]:
                        setup = (-1, t, last_l[1], h[t])
            else:
                h_swept = True          # accepted above: breakout, level retired
        if last_l is not None and not l_swept and last_l[0] < t and l[t] < last_l[1]:
            if c[t] > last_l[1]:
                if last_l[1] - l[t] >= min_pierce * a and c[t] > hl2:
                    l_swept = True
                    if last_h is not None and prior_h is not None and last_h[1] < prior_h[1]:
                        setup = (1, t, last_h[1], l[t])
            else:
                l_swept = True
        if setup is not None:
            d, sb, lvl, ext = setup
            if t - sb > max_rev:
                setup = None
            elif (d == -1 and c[t] < lvl) or (d == 1 and c[t] > lvl):
                gaps = []
                for i in range(max(sb, 2), t + 1):
                    if d == -1 and l[i - 2] - h[i] >= min_fvg * atr[i]:
                        gaps.append(h[i])            # near edge of a bearish FVG = its bottom
                    if d == 1 and l[i] - h[i - 2] >= min_fvg * atr[i]:
                        gaps.append(l[i])            # near edge of a bullish FVG = its top
                if gaps:
                    e = gaps[-1]
                    s = ext + d * -1 * buf * a       # short: above the high; long: below the low
                    risk = abs(e - s) / a
                    if 0 < risk <= max_stop:
                        fresh[t], entry[t], stop_atr[t], stop_px[t] = d, e, risk, s
                setup = None
    sig = pd.Series(fresh).replace(0, np.nan).ffill(limit=window).fillna(0).to_numpy()
    ent = pd.Series(entry).ffill(limit=window).to_numpy()
    return pd.DataFrame({"sfr_sig": sig, "sfr_fresh": fresh,
                         "sfr_entry_dist": np.clip((c - ent) / atr, -10, 10),
                         "sfr_stop_atr": pd.Series(stop_atr).ffill(limit=window).to_numpy(),
                         # raw prices for the strategy's own backtest; leading "_" = not a model feature
                         "_sfr_entry_px": entry, "_sfr_stop_px": stop_px}, index=df.index)


# ------------------------------------------------------------------------------- Gold Breakout (Asian range)
def range_breakout(df: pd.DataFrame, tz_hours: int = 3, range_end: int = 7, close_hour: int = 20) -> pd.DataFrame:
    """The Gold Breakout strategy's state per M5 bar (any market): the 00:00-07:00 broker-time (GMT+3) range,
    then the FIRST break of it until 20:00. No lookahead: the range is only used once it is complete, and the
    break direction only from the bar that broke it.

      gbo_dir    +1 / -1 after today's first break up / down (07:00-20:00), else 0
      gbo_fresh  +1 / -1 on the breaking bar itself
      gbo_pos    close inside the range: 0 = low, 1 = high (beyond = outside), 07:00-20:00 only
      gbo_width  range height in ATR"""
    idx = df.index
    bt = idx + pd.Timedelta(hours=tz_hours)
    day = pd.Series(bt.floor("D"), index=idx)
    hour = pd.Series(bt.hour, index=idx)
    h, l, c = df["high"], df["low"], df["close"]
    in_rng = hour < range_end
    hi = h.where(in_rng).groupby(day).transform("max")
    lo = l.where(in_rng).groupby(day).transform("min")
    full = in_rng.groupby(day).transform("sum") >= 60          # the live strategy needs >= 60 range bars
    after = (hour >= range_end) & (hour < close_hour) & full
    up, dn = after & (h > hi), after & (l < lo)
    ev = up.astype(float) - dn.astype(float)
    both = up & dn
    ev[both] = np.sign(c[both] - (hi[both] + lo[both]) / 2)
    nz = ev != 0
    first = ev.where(nz & (nz.astype(int).groupby(day).cumsum() == 1))
    direction = first.groupby(day).ffill().fillna(0.0).where(after, 0.0)
    width = (hi - lo)
    atr = _atr(df[["open", "high", "low", "close"]].astype(float))
    return pd.DataFrame({"gbo_dir": direction, "gbo_fresh": first.fillna(0.0),
                         "gbo_pos": ((c - lo) / width.replace(0, np.nan)).clip(-3, 4).where(after),
                         "gbo_width": (width / atr).where(after)}, index=idx)


# ------------------------------------------------------------------------------- SMC on M15
def smc_frame(m15: pd.DataFrame, window: int = 600, start: int = 0) -> pd.DataFrame:
    """SMC facts per closed M15 bar (indexed by M15 open time)."""
    rows, idx = [], []
    m15 = m15.reset_index(drop=True) if "time" in m15.columns else m15
    for t in range(max(start, 150), len(m15)):
        a = smc.analyze(m15.iloc[max(0, t - window + 1):t + 1])
        idx.append(pd.Timestamp(m15["time"].iloc[t]))
        rows.append(smc_row(a))
    return pd.DataFrame(rows, index=pd.DatetimeIndex(idx))


def smc_row(a: dict | None) -> dict:
    if not a:
        return {}
    ev = a.get("m15_last_event") or {}
    rv = smc.rules_vote(a)
    obs = [z for z in a.get("order_blocks") or [] if z["dist_low"] <= 0.5 and z["dist_high"] >= -0.5]
    gaps = [g for g in a.get("fvgs") or [] if g["dist_low"] <= 0.3 and g["dist_high"] >= -0.3]
    sw = (a.get("sweeps") or [{}])[-1] if a.get("sweeps") else {}
    up = [q["dist"] for q in a.get("liquidity_pools") or [] if q["dist"] > 0]
    dn = [q["dist"] for q in a.get("liquidity_pools") or [] if q["dist"] < 0]
    return {"smc_rules": rv[0] if rv else 0.0, "smc_h1": float((a.get("h1") or {}).get("trend", 0)),
            "smc_m15": float(a["m15_trend"]), "smc_ev_dir": float(ev.get("dir", 0)),
            "smc_ev_choch": float(ev.get("kind") == "CHoCH"), "smc_ev_age": float(min(ev.get("age", 99), 99)),
            "smc_range_pos": float((a.get("range") or {}).get("pos", 0.5)),
            "smc_in_ob": float(obs[0]["dir"]) if obs else 0.0, "smc_in_fvg": float(gaps[0]["dir"]) if gaps else 0.0,
            "smc_sweep": float(sw.get("dir", 0)), "smc_sweep_age": float(sw.get("age", 99)),
            "smc_liq_up": float(min(up)) if up else 10.0, "smc_liq_dn": float(max(dn)) if dn else -10.0}


# ------------------------------------------------------------------------------- all features
def features(m5: pd.DataFrame, symbol: str, smc_m15: pd.DataFrame | None = None, htf: dict | None = None) -> pd.DataFrame:
    """Feature row per M5 bar (index = M5 open time, UTC). `smc_m15` = smc_frame output (computed from M15 if None)."""
    if not isinstance(m5.index, pd.DatetimeIndex):
        m5 = m5.set_index(pd.to_datetime(m5["time"], unit="s", utc=True) if np.issubdtype(m5["time"].dtype, np.number)
                          else pd.to_datetime(m5["time"], utc=True))
    m5 = m5[["open", "high", "low", "close"]].astype(float)
    sym = base(symbol)
    v = votes_mtf(m5, htf)
    f = v.copy()
    c = m5["close"]
    atr = _atr(m5)
    f["council"] = np.tanh(2.0 * v.mean(axis=1))
    htf_cols = [x for x in v.columns if x.endswith("@H1") or x.endswith("@H4")]
    f["htf_mean"] = v[htf_cols].mean(axis=1) if htf_cols else 0.0
    # bankbot on its alternate resolution (3 x M5 = M15) and on H1
    for tf, minutes in (("M15", 15), ("H1", 60)):
        src = (htf or {}).get(tf)
        if src is None or len(src) < 60:
            src = resample_ohlc(m5, minutes)
        b = bankbot(src)
        b.columns = [f"bank_{x}@{tf}" for x in b.columns]
        f = f.join(align_to_m5(b, minutes, m5.index))
    f = f.join(ict(m5))
    sw = sweep_fvg(m5)
    f = f.join(sw[[x for x in sw.columns if not x.startswith("_")]])
    f = f.join(lux_smc(m5))                         # LuxAlgo Smart Money Concepts (swing 50 / internal 5)
    f = f.join(range_breakout(m5))                  # Gold Breakout's Asian-range break, on every market
    if smc_m15 is None:
        m15 = resample_ohlc(m5, 15)
        m15["time"] = m15.index
        smc_m15 = smc_frame(m15, start=max(0, len(m15) - 40))
    if len(smc_m15):
        f = f.join(align_to_m5(smc_m15, 15, m5.index))
    # regime
    f["atr_bps"] = (atr / c * 1e4)
    f["atr_rel"] = atr / atr.rolling(300, min_periods=50).median()
    f["trend_gap"] = ((_ema(c, 20) - _ema(c, 100)) / atr).clip(-6, 6)
    f["ret_12"] = ((c / c.shift(12) - 1) * 1e4 / f["atr_bps"]).clip(-10, 10)
    f["hour"] = m5.index.hour + m5.index.minute / 60.0
    f["dow"] = m5.index.dayofweek.astype(float)
    f["cost_atr"] = SPREAD_BPS.get(sym, 1.0) / f["atr_bps"]
    f["sym"] = float(SYMBOLS.index(sym)) if sym in SYMBOLS else -1.0
    return f.replace([np.inf, -np.inf], np.nan)


# ------------------------------------------------------------------------------- labels
def labels(m5: pd.DataFrame, symbol: str, horizon: int = HORIZON) -> pd.DataFrame:
    """Result in R (after the spread) of a long and of a short opened at each close with the live stops,
    and the bar it ends on. A bar that touches both stop and target counts as the stop (conservative)."""
    sym = base(symbol)
    sl_m, tp_m = STOPS.get(sym, STOPS["default"])
    h, l, c = (m5[x].to_numpy(float) for x in ("high", "low", "close"))
    atr = _atr(m5[["open", "high", "low", "close"]].astype(float)).to_numpy()
    n = len(c)
    cost_r = SPREAD_BPS.get(sym, 1.0) * c / 1e4 / (sl_m * atr)
    out = {}
    for side in (1, -1):
        sl = c - side * sl_m * atr
        tp = c + side * tp_m * atr
        res = np.full(n, np.nan)
        end = np.full(n, -1)
        open_ = np.ones(n, bool)
        for j in range(1, horizon + 1):
            idx = np.arange(n - j)
            hh, lll = h[idx + j], l[idx + j]
            stop = (lll <= sl[idx]) if side == 1 else (hh >= sl[idx])
            targ = (hh >= tp[idx]) if side == 1 else (lll <= tp[idx])
            o = open_[idx]
            s_hit, t_hit = o & stop, o & targ & ~stop
            res[idx[s_hit]] = -1.0
            res[idx[t_hit]] = tp_m / sl_m
            end[idx[s_hit | t_hit]] = idx[s_hit | t_hit] + j
            open_[idx[s_hit | t_hit]] = False
        idx = np.arange(n - horizon)
        still = open_[idx]
        res[idx[still]] = side * (c[idx[still] + horizon] - c[idx[still]]) / (sl_m * atr[idx[still]])
        end[idx[still]] = idx[still] + horizon
        res = res - cost_r
        key = "long" if side == 1 else "short"
        out[f"r_{key}"] = res
        out[f"end_{key}"] = end
    return pd.DataFrame(out, index=m5.index if isinstance(m5.index, pd.DatetimeIndex) else None)


# ------------------------------------------------------------------------------- live use
class TreeModel:
    """LightGBM model evaluated in numpy from its JSON dump (Booster.dump_model()).

    The live bot must NOT import lightgbm: on macOS its OpenMP runtime and PyTorch's crash the process
    (segfault) when both are loaded. Only numerical '<=' splits are used by this model."""

    def __init__(self, dump: dict):
        self.trees = []
        for t in dump["tree_info"]:
            feat, thr, left, right, dleft, miss, leaf = [], [], [], [], [], [], []

            def walk(node):
                i = len(feat)
                feat.append(-1), thr.append(0.0), left.append(-1), right.append(-1)
                dleft.append(True), miss.append(0), leaf.append(0.0)
                if "leaf_value" in node:
                    leaf[i] = float(node["leaf_value"])
                    return i
                if node.get("decision_type", "<=") != "<=":
                    raise ValueError(f"unsupported split {node.get('decision_type')}")
                feat[i], thr[i] = int(node["split_feature"]), float(node["threshold"])
                dleft[i] = bool(node["default_left"])
                miss[i] = {"None": 0, "Zero": 1, "NaN": 2}[node["missing_type"]]
                left[i] = walk(node["left_child"])
                right[i] = walk(node["right_child"])
                return i

            walk(t["tree_structure"])
            self.trees.append((np.array(feat), np.array(thr), np.array(left), np.array(right),
                               np.array(dleft), np.array(miss), np.array(leaf)))

    def predict(self, x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(np.asarray(x, float))
        out = np.zeros(len(x))
        for r, row in enumerate(x):
            s = 0.0
            for feat, thr, left, right, dleft, miss, leaf in self.trees:
                i = 0
                while feat[i] >= 0:
                    v = row[feat[i]]
                    m = miss[i]
                    if np.isnan(v):
                        if m == 2:
                            i = left[i] if dleft[i] else right[i]
                            continue
                        v = 0.0
                    if m == 1 and abs(v) <= 1e-35:
                        i = left[i] if dleft[i] else right[i]
                        continue
                    i = left[i] if v <= thr[i] else right[i]
                s += leaf[i]
            out[r] = s
        return out


class MetaBrain:
    """Loads the trained model; vote() turns the predicted long/short R into a council vote."""

    def __init__(self, model_dir: str = MODEL_DIR, margin: float | None = None):
        meta = json.load(open(os.path.join(model_dir, "meta.json")))
        self.cols = meta["features"]
        self.margin = float(margin if margin is not None else meta.get("margin", 0.1))
        self.long = TreeModel(json.load(open(os.path.join(model_dir, "long.json"))))
        self.short = TreeModel(json.load(open(os.path.join(model_dir, "short.json"))))
        self.meta = meta

    def predict(self, f: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
        x = f.reindex(columns=self.cols).astype(float).to_numpy()
        return self.long.predict(x), self.short.predict(x)

    def vote(self, symbol: str, m5: pd.DataFrame, m15: pd.DataFrame | None = None, htf: dict | None = None):
        if m5 is None or len(m5) < 600:
            return None
        sm = None
        if m15 is not None and len(m15) >= 200:
            m15 = m15.copy()
            m15["time"] = pd.to_datetime(m15["time"], utc=True)
            sm = smc_frame(m15.tail(700).reset_index(drop=True), start=len(m15.tail(700)) - 40)
            if htf is None:   # long M15 history -> warmed-up M15/H1/H4 votes, as in training
                k = m15.set_index("time")[["open", "high", "low", "close"]].astype(float)
                htf = {"M15": k, "H1": resample_ohlc(k, 60), "H4": resample_ohlc(k, 240)}
        f = features(m5.tail(1500), symbol, sm, htf).tail(1)
        pl, ps = (float(x[0]) for x in self.predict(f))
        best = max(pl, ps)
        if best < self.margin:
            v = 0.0
        else:
            v = float(np.clip((pl - ps) / (2 * self.margin + 1e-9), -1, 1))
        return v, {"reason": f"Predicted result long {pl:+.2f}R, short {ps:+.2f}R (trades only above {self.margin:+.2f}R).",
                   "pred_long": round(pl, 3), "pred_short": round(ps, 3)}
