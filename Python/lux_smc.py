"""Port of LuxAlgo "Smart Money Concepts" (CC BY-NC-SA 4.0, (c) LuxAlgo) to per-bar features, no repainting.

Defaults follow the indicator: swing structure length 50, internal structure length 5, order blocks from the
leg's extreme candle with ATR(200) volatility filter (bars >= 2 ATR have high/low swapped, as LuxAlgo does),
High/Low mitigation, equal highs/lows on 3-bar pivots within 0.1 ATR, premium/discount from trailing extremes.

Leg rule (LuxAlgo `leg(size)`): the bar `size` ago is a swing high if its high is above every high since;
the pivot is only known `size` bars later, so it is stored at confirmation time. BOS/CHoCH: close crosses a
pivot not crossed before; CHoCH if it flips that structure's trend, else BOS.

Output columns (prefix lux_), all known at the close of their bar:
  lux_swing_trend, lux_int_trend      +1 / -1 / 0
  lux_int_ev, lux_swing_ev            signed last break (+1 bull, -1 bear), +-2 for a CHoCH
  lux_int_age, lux_swing_age          bars since that break (capped 500)
  lux_int_fresh                       +-1 on the bar of an internal CHoCH that agrees with the swing trend
  lux_ob                              +1 / -1 if price is inside an unmitigated internal order block, else 0
  lux_ob_dist                         ATRs from close to the nearest unmitigated internal OB edge (signed by side)
  lux_pd                              0 = trailing low (discount) .. 1 = trailing high (premium)
  lux_eqh, lux_eql                    ATRs to the latest unswept equal high (+) / equal low (-), 10 if none
"""
from __future__ import annotations

import numpy as np
import pandas as pd


def _atr(h, l, c, n):
    pc = np.r_[c[0], c[:-1]]
    tr = np.maximum(h - l, np.maximum(np.abs(h - pc), np.abs(l - pc)))
    return pd.Series(tr).ewm(alpha=1 / n, adjust=False).mean().to_numpy()


def _legs(h, l, size):
    """Confirmed pivots: dict confirm_bar -> (pivot_bar, price) for highs and lows (LuxAlgo leg switches)."""
    n = len(h)
    hi_max = pd.Series(h).rolling(size).max().to_numpy()     # max of bars t-size+1..t
    lo_min = pd.Series(l).rolling(size).min().to_numpy()
    ph, pl, leg = {}, {}, 0
    for t in range(size, n):
        p = t - size
        new = leg
        if h[p] > hi_max[t]:
            new = 0                       # bearish leg begins -> swing high at p
        elif l[p] < lo_min[t]:
            new = 1
        if new != leg:
            if new == 0:
                ph[t] = (p, h[p])
            else:
                pl[t] = (p, l[p])
            leg = new
    return ph, pl


class _Structure:
    def __init__(self):
        self.top = self.bot = None        # [bar, price, crossed]
        self.trend = 0
        self.ev = 0
        self.ev_bar = -10 ** 9


def lux_smc(df: pd.DataFrame, swing: int = 50, internal: int = 5, eq_len: int = 3, eq_thr: float = 0.1,
            max_obs: int = 20) -> pd.DataFrame:
    o, h, l, c = (df[x].to_numpy(float) for x in ("open", "high", "low", "close"))
    n = len(c)
    atr200 = _atr(h, l, c, 200)
    # LuxAlgo parsed high/low: very volatile bars count inverted for OB selection
    vol = (h - l) >= 2 * atr200
    ph_ = np.where(vol, l, h)
    pl_ = np.where(vol, h, l)
    piv = {"s": _legs(h, l, swing), "i": _legs(h, l, internal)}
    eqp = _legs(h, l, eq_len)
    st = {"s": _Structure(), "i": _Structure()}
    obs: list[list] = []                  # [dir, top, bottom]
    trail_top = trail_bot = None
    last_eqh = last_eql = None            # previous eq pivots
    eqh = eql = None                      # active equal-high / equal-low levels
    out = {k: np.zeros(n) for k in ("lux_swing_trend", "lux_int_trend", "lux_int_ev", "lux_swing_ev",
                                     "lux_int_age", "lux_swing_age", "lux_int_fresh", "lux_ob", "lux_ob_dist",
                                     "lux_pd", "lux_eqh", "lux_eql")}
    for t in range(n):
        a = atr200[t] or 1e-12
        for key in ("s", "i"):
            ph, pl = piv[key]
            s = st[key]
            if t in ph:
                p, v = ph[t]
                if key == "i" and t in piv["s"][0] and piv["s"][0][t][1] == v:
                    pass                  # LuxAlgo skips internal pivots that equal the swing pivot
                else:
                    s.top = [p, v, False]
                if key == "s":
                    trail_top = v
            if t in pl:
                p, v = pl[t]
                if key == "i" and t in piv["s"][1] and piv["s"][1][t][1] == v:
                    pass
                else:
                    s.bot = [p, v, False]
                if key == "s":
                    trail_bot = v
        trail_top = h[t] if trail_top is None else max(trail_top, h[t])
        trail_bot = l[t] if trail_bot is None else min(trail_bot, l[t])
        for key in ("s", "i"):
            s = st[key]
            prev = c[t - 1] if t else c[t]
            if s.top and not s.top[2] and c[t] > s.top[1] and prev <= s.top[1]:
                s.ev = 2 if s.trend == -1 else 1
                s.trend, s.top[2], s.ev_bar = 1, True, t
                if key == "i":
                    lo = s.top[0] + int(np.argmin(pl_[s.top[0]:t + 1])) if t > s.top[0] else t
                    obs.append([1, ph_[lo], pl_[lo]])
                    if s.ev == 2 and st["s"].trend == 1:
                        out["lux_int_fresh"][t] = 1
            if s.bot and not s.bot[2] and c[t] < s.bot[1] and prev >= s.bot[1]:
                s.ev = -2 if s.trend == 1 else -1
                s.trend, s.bot[2], s.ev_bar = -1, True, t
                if key == "i":
                    hi = s.bot[0] + int(np.argmax(ph_[s.bot[0]:t + 1])) if t > s.bot[0] else t
                    obs.append([-1, ph_[hi], pl_[hi]])
                    if s.ev == -2 and st["s"].trend == -1:
                        out["lux_int_fresh"][t] = -1
        # mitigation (High/Low): a bull OB dies when low trades below its bottom, a bear OB when high above its top
        obs = [b for b in obs if not ((b[0] == 1 and l[t] < b[2]) or (b[0] == -1 and h[t] > b[1]))][-max_obs:]
        inside = [b for b in obs if b[2] <= c[t] <= b[1]]
        if inside:
            out["lux_ob"][t] = inside[-1][0]
        if obs:
            d = [((b[1] - c[t]) if c[t] > b[1] else (b[2] - c[t]) if c[t] < b[2] else 0.0) for b in obs]
            out["lux_ob_dist"][t] = d[int(np.argmin(np.abs(d)))] / a
        else:
            out["lux_ob_dist"][t] = 10.0
        # equal highs / lows
        if t in eqp[0]:
            v = eqp[0][t][1]
            if last_eqh is not None and abs(v - last_eqh) < eq_thr * a:
                eqh = max(v, last_eqh)
            last_eqh = v
        if t in eqp[1]:
            v = eqp[1][t][1]
            if last_eql is not None and abs(v - last_eql) < eq_thr * a:
                eql = min(v, last_eql)
            last_eql = v
        if eqh is not None and h[t] > eqh:
            eqh = None
        if eql is not None and l[t] < eql:
            eql = None
        out["lux_eqh"][t] = min((eqh - c[t]) / a, 10.0) if eqh is not None else 10.0
        out["lux_eql"][t] = max((eql - c[t]) / a, -10.0) if eql is not None else -10.0
        rng = trail_top - trail_bot
        out["lux_pd"][t] = (c[t] - trail_bot) / rng if rng > 0 else 0.5
        out["lux_swing_trend"][t], out["lux_int_trend"][t] = st["s"].trend, st["i"].trend
        out["lux_swing_ev"][t], out["lux_int_ev"][t] = st["s"].ev, st["i"].ev
        out["lux_swing_age"][t] = min(t - st["s"].ev_bar, 500)
        out["lux_int_age"][t] = min(t - st["i"].ev_bar, 500)
    out["lux_ob_dist"] = np.clip(out["lux_ob_dist"], -10, 10)
    return pd.DataFrame(out, index=df.index)
