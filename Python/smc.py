"""Smart Money Concepts (SMC) read of the market, for the council and for the local Gemma analyst.

The structure is found by plain code so every level is real and reproducible; the language model only
weighs the confluence. Nothing here looks ahead: a swing point counts only once the k bars after it have
closed, and the live feed passes closed bars only.

    structure   swing highs/lows, BOS (break of structure, with the trend) and CHoCH (change of character,
                the first break against it), on M15 and on H1 built from the same bars
    order block the last opposite candle before the move that broke structure; dropped once price closes
                through it
    FVG         three-candle fair value gap (bar i's low above bar i-2's high, or the mirror), until filled
    liquidity   equal highs/lows (resting stops) not yet taken, and recent sweeps (wick through a swing,
                close back inside)
    range       where price sits between the last swing high and low: discount (<0.5) or premium (>0.5)

`analyze` returns the facts, `rules_vote` scores them with fixed rules, `describe` writes them as text.
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(h, l, c, n=14):
    tr = np.maximum(h - l, np.maximum(np.abs(h - np.r_[c[0], c[:-1]]), np.abs(l - np.r_[c[0], c[:-1]])))
    return pd.Series(tr).ewm(alpha=1.0 / n, adjust=False).mean().to_numpy()


def _pivots(h, l, k):
    """Swing highs/lows as (bar, price); bar p is confirmed at bar p+k."""
    ph, pl = [], []
    for p in range(k, len(h) - k):
        if h[p] >= h[p - k:p + k + 1].max() and h[p] > h[p - k:p].max():
            ph.append((p, float(h[p])))
        if l[p] <= l[p - k:p + k + 1].min() and l[p] < l[p - k:p].min():
            pl.append((p, float(l[p])))
    return ph, pl


def structure(o, h, l, c, k=3, lookback=400):
    """Walk the bars once: trend, BOS/CHoCH events, order blocks still valid, swings."""
    n = len(c)
    start = max(0, n - lookback)
    ph, pl = _pivots(h, l, k)
    hi_i = lo_i = 0
    last_hi = last_lo = None          # latest confirmed, not yet broken
    swing_hi = swing_lo = None        # latest confirmed (broken or not), for the dealing range
    trend, events, obs = 0, [], []
    for t in range(start, n):
        while hi_i < len(ph) and ph[hi_i][0] + k <= t:
            last_hi = swing_hi = ph[hi_i]
            hi_i += 1
        while lo_i < len(pl) and pl[lo_i][0] + k <= t:
            last_lo = swing_lo = pl[lo_i]
            lo_i += 1
        # order blocks die when a close goes through them
        obs = [z for z in obs if not ((z["dir"] == 1 and c[t] < z["low"]) or (z["dir"] == -1 and c[t] > z["high"]))]
        for z in obs:
            if (z["dir"] == 1 and l[t] <= z["high"]) or (z["dir"] == -1 and h[t] >= z["low"]):
                if t > z["bar"]:
                    z["tested"] = True
        for d, lvl in ((1, last_hi), (-1, last_lo)):
            if lvl is None or not ((d == 1 and c[t] > lvl[1]) or (d == -1 and c[t] < lvl[1])):
                continue
            kind = "BOS" if trend == d else "CHoCH"
            events.append({"bar": t, "dir": d, "kind": kind, "level": lvl[1]})
            trend = d
            origin = (swing_lo if d == 1 else swing_hi)
            a = origin[0] if origin is not None and origin[0] < t else max(0, t - 12)
            opp = [i for i in range(a, t) if (c[i] < o[i] if d == 1 else c[i] > o[i])]
            if opp:
                i = opp[-1]
                obs.append({"dir": d, "bar": i, "low": float(l[i]), "high": float(h[i]), "tested": False})
            if d == 1:
                last_hi = None
            else:
                last_lo = None
    return {"trend": trend, "events": events, "obs": obs[-6:], "ph": ph, "pl": pl,
            "swing_hi": swing_hi, "swing_lo": swing_lo}


def fvgs(h, l, c, lookback=200):
    """Fair value gaps not yet filled: {dir, bar, low, high, filled (0-1)}."""
    n, out = len(c), []
    for i in range(max(2, n - lookback), n):
        if l[i] > h[i - 2]:
            out.append({"dir": 1, "bar": i, "low": float(h[i - 2]), "high": float(l[i])})
        elif h[i] < l[i - 2]:
            out.append({"dir": -1, "bar": i, "low": float(h[i]), "high": float(l[i - 2])})
    keep = []
    for g in out:
        after = slice(g["bar"] + 1, n)
        size = g["high"] - g["low"]
        if g["dir"] == 1:
            lo = l[after].min() if g["bar"] + 1 < n else np.inf
            filled = 0.0 if lo >= g["high"] else min(1.0, (g["high"] - lo) / size)
        else:
            hi = h[after].max() if g["bar"] + 1 < n else -np.inf
            filled = 0.0 if hi <= g["low"] else min(1.0, (hi - g["low"]) / size)
        if filled < 1.0:
            g["filled"] = round(float(filled), 2)
            keep.append(g)
    return keep


def liquidity(h, l, c, ph, pl, atr, k=3, tol=0.15, sweep_bars=8):
    """Equal highs/lows not yet taken, and sweeps in the last few bars."""
    n = len(c)
    pools = []
    for side, piv, arr in ((1, ph, h), (-1, pl, l)):
        piv = [p for p in piv if p[0] + k < n][-12:]
        for a in range(len(piv)):
            for b in range(a + 1, len(piv)):
                (i, x), (j, y) = piv[a], piv[b]
                if abs(x - y) > tol * atr:
                    continue
                lvl = max(x, y) if side == 1 else min(x, y)
                later = arr[j + 1:]
                taken = (later.max() > lvl) if side == 1 else (later.min() < lvl)
                if not taken:
                    pools.append({"side": side, "level": float(lvl), "bars": [i, j]})
    sweeps = []
    for t in range(max(1, n - sweep_bars), n):
        for p, v in ph:
            if p + k <= t - 1 and p >= t - 120 and h[t] > v and c[t] < v and h[p + 1:t].max(initial=-np.inf) <= v:
                sweeps.append({"bar": t, "dir": -1, "level": v, "what": "buy-side liquidity (a swing high)"})
                break
        for p, v in pl:
            if p + k <= t - 1 and p >= t - 120 and l[t] < v and c[t] > v and l[p + 1:t].min(initial=np.inf) >= v:
                sweeps.append({"bar": t, "dir": 1, "level": v, "what": "sell-side liquidity (a swing low)"})
                break
    uniq = {}
    for q in pools:
        uniq[(q["side"], round(q["level"], 8))] = q
    return list(uniq.values()), sweeps


def resample(m15: pd.DataFrame, minutes: int = 60) -> pd.DataFrame:
    d = m15.copy()
    d.index = pd.to_datetime(d["time"], utc=True) if "time" in d.columns else d.index
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = d[["open", "high", "low", "close"]].resample(f"{minutes}min", label="left", closed="left").agg(agg).dropna()
    # the last H1 bar is only complete if the M15 bars cover all of it
    if len(d) and len(out):
        last_end = out.index[-1] + pd.Timedelta(minutes=minutes)
        if d.index[-1] + pd.Timedelta(minutes=15) < last_end:
            out = out.iloc[:-1]
    return out


def analyze(m15: pd.DataFrame, k: int = 3) -> dict | None:
    """SMC facts at the last closed M15 bar, in prices and in ATR multiples."""
    if m15 is None or len(m15) < 120:
        return None
    o, h, l, c = (m15[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    atr = float(_atr(h, l, c)[-1])
    if not np.isfinite(atr) or atr <= 0:
        return None
    px = float(c[-1])
    n = len(c)
    st = structure(o, h, l, c, k=k)
    htf = None
    if "time" in m15.columns:
        h1 = resample(m15, 60)
        if len(h1) >= 60:
            ho, hh, hl, hc = (h1[x].to_numpy(float) for x in ("open", "high", "low", "close"))
            hs = structure(ho, hh, hl, hc, k=3)
            htf = {"trend": hs["trend"], "last_event": hs["events"][-1] if hs["events"] else None,
                   "bars": len(h1)}
    gaps = fvgs(h, l, c)
    pools, sweeps = liquidity(h, l, c, st["ph"], st["pl"], atr, k=k)
    # dealing range = the current leg: from the swing it started at to the extreme reached since
    rng = None
    if st["trend"] >= 0 and st["swing_lo"]:
        i0, lo = st["swing_lo"]
        hi = float(h[i0:].max())
    elif st["trend"] < 0 and st["swing_hi"]:
        i0, hi = st["swing_hi"]
        lo = float(l[i0:].min())
    else:
        lo = hi = None
    if lo is not None and hi - lo > 0.5 * atr:
        rng = {"low": lo, "high": hi, "pos": float(np.clip((px - lo) / (hi - lo), 0, 1))}

    def dist(x):
        return round((x - px) / atr, 2)

    near_obs = [dict(z, dist_low=dist(z["low"]), dist_high=dist(z["high"]), age=n - 1 - z["bar"])
                for z in st["obs"] if min(abs(dist(z["low"])), abs(dist(z["high"]))) <= 6
                or z["low"] <= px <= z["high"]]
    near_gaps = sorted([dict(g, dist_low=dist(g["low"]), dist_high=dist(g["high"]), age=n - 1 - g["bar"])
                        for g in gaps if min(abs(dist(g["low"])), abs(dist(g["high"]))) <= 4],
                       key=lambda g: min(abs(g["dist_low"]), abs(g["dist_high"])))[:6]
    near_pools = sorted([dict(q, dist=dist(q["level"])) for q in pools], key=lambda q: abs(q["dist"]))[:4]
    last_ev = st["events"][-1] if st["events"] else None
    return {"price": px, "atr": atr, "bar_time": str(m15["time"].iloc[-1]) if "time" in m15.columns else None,
            "m15_trend": st["trend"], "m15_last_event": None if last_ev is None else dict(last_ev, age=n - 1 - last_ev["bar"]),
            "m15_events": [dict(e, age=n - 1 - e["bar"]) for e in st["events"][-4:]],
            "h1": htf, "range": rng, "order_blocks": near_obs, "fvgs": near_gaps,
            "liquidity_pools": near_pools, "sweeps": [dict(s, age=n - 1 - s["bar"]) for s in sweeps]}


def rules_vote(a: dict | None) -> tuple[float, list[str]] | None:
    """Fixed-rule SMC score in [-1, 1] (+ = up) and the reasons. The baseline the Gemma analyst must beat."""
    if not a:
        return None
    s, why = 0.0, []
    h1 = (a.get("h1") or {}).get("trend", 0)
    if h1:
        s += 0.3 * h1
        why.append(f"H1 structure {'bullish' if h1 > 0 else 'bearish'}")
    if a["m15_trend"]:
        s += 0.15 * a["m15_trend"]
    ev = a.get("m15_last_event")
    if ev and ev["kind"] == "CHoCH" and ev["age"] <= 12:
        s += 0.2 * ev["dir"]
        why.append(f"fresh M15 CHoCH {'up' if ev['dir'] > 0 else 'down'} ({ev['age']} bars ago)")
    rng = a.get("range")
    if rng:
        if rng["pos"] < 0.4 and s > 0:
            s += 0.1
            why.append("price in discount")
        elif rng["pos"] > 0.6 and s < 0:
            s -= 0.1
            why.append("price in premium")
        elif (rng["pos"] > 0.75 and s > 0) or (rng["pos"] < 0.25 and s < 0):
            s *= 0.6
            why.append("chasing: buying premium / selling discount")
    for z in a.get("order_blocks") or []:
        inside_or_near = z["dist_low"] <= 0.5 and z["dist_high"] >= -0.5
        if inside_or_near:
            s += 0.15 * z["dir"]
            why.append(f"at a {'bullish' if z['dir'] > 0 else 'bearish'} order block")
            break
    for g in a.get("fvgs") or []:
        if g["dist_low"] <= 0.3 and g["dist_high"] >= -0.3:
            s += 0.1 * g["dir"]
            why.append(f"in a {'bullish' if g['dir'] > 0 else 'bearish'} FVG")
            break
    for sw in a.get("sweeps") or []:
        s += 0.2 * sw["dir"]
        why.append(f"swept {sw['what']} {sw['age']} bars ago")
        break
    # don't push straight into resting liquidity that is very close: it tends to get run first, then reverse
    for q in a.get("liquidity_pools") or []:
        if abs(q["dist"]) <= 0.5 and np.sign(q["dist"]) == np.sign(s) and s != 0:
            s *= 0.7
            why.append("resting liquidity just ahead")
            break
    return float(np.clip(s, -1, 1)), why


def _fmt(x: float, px: float) -> str:
    dec = 2 if px >= 100 else 3 if px >= 10 else 5
    return f"{x:.{dec}f}"


def describe(a: dict, asset: str = "") -> str:
    """The facts as short lines for the language model (prices plus distance in ATRs)."""
    px, f = a["price"], (lambda x: _fmt(x, a["price"]))
    word = {1: "bullish", -1: "bearish", 0: "none yet"}
    L = [f"Market: {asset}" if asset else "Market:",
         f"Price {f(px)}, M15 ATR {f(a['atr'])} (distances below are in ATRs; + above price, - below)."]
    h1 = a.get("h1")
    if h1:
        e = h1.get("last_event")
        L.append(f"H1 structure: {word[h1['trend']]}" + (f", last {e['kind']} {'up' if e['dir'] > 0 else 'down'} at {f(e['level'])}" if e else ""))
    L.append(f"M15 structure: {word[a['m15_trend']]}")
    for e in a.get("m15_events") or []:
        L.append(f"  {e['kind']} {'up' if e['dir'] > 0 else 'down'} through {f(e['level'])}, {e['age']} bars ago")
    r = a.get("range")
    if r:
        zone = "discount" if r["pos"] < 0.5 else "premium"
        L.append(f"Dealing range {f(r['low'])}-{f(r['high'])}: price at {r['pos']:.0%} ({zone}).")
    for z in a.get("order_blocks") or []:
        L.append(f"{'Bullish' if z['dir'] > 0 else 'Bearish'} order block {f(z['low'])}-{f(z['high'])} "
                 f"({z['dist_low']:+.1f} to {z['dist_high']:+.1f} ATR, {'tested' if z['tested'] else 'untested'}, {z['age']} bars old)")
    for g in a.get("fvgs") or []:
        L.append(f"{'Bullish' if g['dir'] > 0 else 'Bearish'} FVG {f(g['low'])}-{f(g['high'])} "
                 f"({g['dist_low']:+.1f} to {g['dist_high']:+.1f} ATR, {g['filled']:.0%} filled)")
    for q in a.get("liquidity_pools") or []:
        L.append(f"Equal {'highs' if q['side'] > 0 else 'lows'} (resting {'buy' if q['side'] > 0 else 'sell'}-side liquidity) "
                 f"at {f(q['level'])} ({q['dist']:+.1f} ATR)")
    for s in a.get("sweeps") or []:
        L.append(f"Sweep: took {s['what']} at {f(s['level'])} and closed back inside, {s['age']} bars ago")
    if len(L) <= 4:
        L.append("No order blocks, gaps or liquidity pools nearby.")
    return "\n".join(L)
