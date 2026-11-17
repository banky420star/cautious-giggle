"""Indicator council: classic indicator strategies vote every bar, weighted by their track record.

Each strategy maps raw OHLC to a vote in [-1, 1] (+1 = long, -1 = short, 0 = no view) using only
data up to and including the closed bar. The model memory scores every strategy separately per
market context (key "ind:<name>"), so the council leans on what has actually worked in markets
like the current one and ignores what hasn't:

    weight_i = memory multiplier (0.25x .. 1.5x), dropped entirely if the memory says "doesn't work"
    council  = tanh(2 * sum(w_i * vote_i) / sum(w_i))

Everything is vectorised so the same code computes the live vote (last row) and replays years of
history to warm up the memory (tools/warm_memory.py).
"""
from __future__ import annotations

import numpy as np
import pandas as pd

# name -> plain-English description (shown in the dashboard)
STRATEGIES = {
    "ema_cross": "EMA 20 vs 50 trend",
    "ema200": "price vs 200-EMA",
    "macd": "MACD histogram",
    "rsi_trend": "RSI momentum (>55 up, <45 down)",
    "rsi_revert": "RSI extremes fade (<30 buy, >70 sell)",
    "bb_revert": "Bollinger band fade",
    "bb_break": "Bollinger breakout",
    "donchian": "20-bar high/low breakout",
    "adx_di": "ADX trend strength + direction",
    "stoch": "Stochastic 14 cross",
    "roc": "12-bar rate of change",
    "supertrend": "ATR trailing trend",
}


def _ema(x: pd.Series, n: int) -> pd.Series:
    return x.ewm(span=n, adjust=False).mean()


def _atr(df: pd.DataFrame, n: int = 14) -> pd.Series:
    h, l, c = df["high"], df["low"], df["close"]
    tr = pd.concat([(h - l), (h - c.shift()).abs(), (l - c.shift()).abs()], axis=1).max(axis=1)
    return tr.ewm(alpha=1 / n, adjust=False).mean()


def votes(df: pd.DataFrame) -> pd.DataFrame:
    """Vote per strategy per bar (columns = STRATEGIES keys)."""
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    out = pd.DataFrame(index=df.index)
    atr = _atr(df)

    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    out["ema_cross"] = np.clip((e20 - e50) / (0.5 * atr), -1, 1)
    out["ema200"] = np.clip((c - e200) / (2.0 * atr), -1, 1)

    macd = _ema(c, 12) - _ema(c, 26)
    hist = macd - _ema(macd, 9)
    out["macd"] = np.clip(hist / (0.15 * atr), -1, 1)

    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + gain / loss.replace(0, np.nan))
    rsi = rsi.fillna(50)
    out["rsi_trend"] = np.where(rsi > 55, np.clip((rsi - 55) / 15, 0, 1), np.where(rsi < 45, -np.clip((45 - rsi) / 15, 0, 1), 0.0))
    out["rsi_revert"] = np.where(rsi < 30, np.clip((30 - rsi) / 10, 0, 1), np.where(rsi > 70, -np.clip((rsi - 70) / 10, 0, 1), 0.0))

    mid = c.rolling(20).mean()
    sd = c.rolling(20).std()
    z = (c - mid) / sd.replace(0, np.nan)
    out["bb_revert"] = np.where(z.abs() > 1.5, -np.clip(z / 2.5, -1, 1), 0.0)
    out["bb_break"] = np.where(z.abs() > 2.0, np.clip(z / 2.5, -1, 1), 0.0)

    hi20, lo20 = h.rolling(20).max().shift(1), l.rolling(20).min().shift(1)
    don = pd.Series(np.where(c > hi20, 1.0, np.where(c < lo20, -1.0, np.nan)), index=df.index)
    out["donchian"] = don.ffill(limit=12).fillna(0.0)  # hold the breakout view for an hour

    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    pdi, mdi = 100 * pdm / atr, 100 * mdm / atr
    dx = 100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)
    adx = dx.ewm(alpha=1 / 14, adjust=False).mean().fillna(0)
    out["adx_di"] = np.where(adx > 20, np.sign(pdi - mdi) * np.clip((adx - 20) / 20, 0, 1), 0.0)

    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    dline = k.rolling(3).mean()
    out["stoch"] = np.clip((k - dline) / 20, -1, 1).fillna(0)

    roc = c / c.shift(12) - 1
    out["roc"] = np.clip(roc / (3 * atr / c), -1, 1)

    # Supertrend-like: direction flips when price closes beyond an ATR band around the median price.
    band = 3.0 * atr
    st = np.zeros(len(c))
    hl2 = ((h + l) / 2).to_numpy()
    cc, bb = c.to_numpy(), band.to_numpy()
    upper, lower, dirn = np.nan, np.nan, 1.0
    for i in range(len(cc)):
        if not np.isfinite(bb[i]):
            continue
        u, lo_ = hl2[i] + bb[i], hl2[i] - bb[i]
        upper = u if not np.isfinite(upper) or u < upper or cc[i - 1] > upper else upper
        lower = lo_ if not np.isfinite(lower) or lo_ > lower or cc[i - 1] < lower else lower
        if cc[i] > upper:
            dirn = 1.0
        elif cc[i] < lower:
            dirn = -1.0
        st[i] = dirn
    out["supertrend"] = st

    return out.replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)


def context_series(df: pd.DataFrame) -> pd.Series:
    """Market context key per bar: trend|volatility|session (same buckets as model_memory)."""
    c = df["close"].astype(float)
    atr = _atr(df)
    gap = (_ema(c, 20) - _ema(c, 100)) / atr.replace(0, np.nan)
    trend = np.where(gap > 0.75, "up", np.where(gap < -0.75, "down", "range"))
    med = atr.rolling(300, min_periods=50).median()
    vol = np.where(atr > 1.3 * med, "high", np.where(atr < 0.75 * med, "low", "normal"))
    if isinstance(df.index, pd.DatetimeIndex):
        hour = df.index.hour
    elif "time" in df.columns:
        t = df["time"]
        t = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
        hour = t.dt.hour.to_numpy()
    else:
        hour = np.full(len(df), pd.Timestamp.now(tz="UTC").hour)
    session = np.where(hour < 7, "asia", np.where(hour < 13, "europe", np.where(hour < 21, "us", "late")))
    return pd.Series([f"{t}|{v}|{s}" for t, v, s in zip(trend, vol, session)], index=df.index)


def council_vote(latest_votes: dict, assessments: dict, weights: dict | None = None) -> dict:
    """Combine the latest votes using the memory's verdict per strategy for the current context.
    `weights` gives some voters (the external models) more say than a single indicator."""
    num = den = 0.0
    used, ignored = [], []
    for name, v in latest_votes.items():
        a = assessments.get(name) or {}
        mult = float(a.get("mult", 1.0)) * float((weights or {}).get(name, 1.0))
        if a.get("verdict") == "doesn't work":
            ignored.append(name)
            continue
        num += mult * float(v)
        den += mult
        if abs(v) > 0.05:
            used.append(name)
    raw = num / den if den > 0 else 0.0
    return {"target": float(np.tanh(2.0 * raw)), "raw": raw, "used": used, "ignored": ignored,
            "up": sum(1 for v in latest_votes.values() if v > 0.05),
            "down": sum(1 for v in latest_votes.values() if v < -0.05),
            "flat": sum(1 for v in latest_votes.values() if abs(v) <= 0.05)}


# --------------------------------------------------------------------------- multi-timeframe
TIMEFRAMES = {"M5": 5, "M15": 15, "H1": 60, "H4": 240}


def resample_ohlc(m5: pd.DataFrame, minutes: int) -> pd.DataFrame:
    """M5 -> higher timeframe, bars labelled by their open time."""
    agg = {"open": "first", "high": "max", "low": "min", "close": "last"}
    out = m5[["open", "high", "low", "close"]].resample(f"{minutes}min", label="left", closed="left").agg(agg)
    return out.dropna()


def align_to_m5(htf_votes: pd.DataFrame, minutes: int, m5_index: pd.DatetimeIndex) -> pd.DataFrame:
    """Make a higher-timeframe vote visible only from the M5 bar that closes with the HTF bar.

    M5 bars are labelled by open time, so the HTF bar opened at T (closing at T+minutes) becomes
    usable on the M5 bar opened at T+minutes-5. Earlier M5 bars see the previous HTF bar: no lookahead.
    """
    shifted = htf_votes.copy()
    shifted.index = shifted.index + pd.Timedelta(minutes=minutes - 5)
    shifted = shifted[~shifted.index.duplicated(keep="last")].sort_index()
    return shifted.reindex(m5_index, method="ffill").fillna(0.0)


def votes_mtf(m5: pd.DataFrame, htf: dict | None = None) -> pd.DataFrame:
    """Votes for every strategy on every timeframe, aligned to the M5 index.

    Columns: '<strategy>' for M5 and '<strategy>@<TF>' for higher timeframes. `htf` may supply
    broker bars per timeframe ({'H1': df, ...}, closed bars, indexed by open time); missing
    timeframes are resampled from the M5 frame.
    """
    if not isinstance(m5.index, pd.DatetimeIndex):
        m5 = m5.copy()
        if "time" in m5.columns:
            t = m5["time"]
            m5.index = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
        else:
            return votes(m5)  # no timestamps: M5 only
    elif m5.index.tz is None:
        m5 = m5.tz_localize("UTC")
    frames = [votes(m5)]
    for tf, minutes in TIMEFRAMES.items():
        if minutes == 5:
            continue
        src = (htf or {}).get(tf)
        if src is None or len(src) < 60:
            src = resample_ohlc(m5, minutes)
        if len(src) < 30:
            continue
        v = votes(src)
        v.columns = [f"{c}@{tf}" for c in v.columns]
        frames.append(align_to_m5(v, minutes, m5.index))
    out = pd.concat(frames, axis=1)
    return out


def describe(key: str) -> str:
    try:
        from Python.external_voters import NAMES as _EXT
        if key in _EXT:
            return _EXT[key]
    except Exception:
        pass
    name, _, tf = key.partition("@")
    return f"{STRATEGIES.get(name, name)} ({tf or 'M5'})"


# --------------------------------------------------------------------------- why each strategy votes
def _r(v, lo, hi, buy, sell, short, text, fmt=2):
    v = float(v) if v is not None and np.isfinite(v) else 0.0
    return {"v": round(v, 4), "min": lo, "max": hi, "buy": buy, "sell": sell, "fmt": fmt, "short": short, "text": text}


def readings(df: pd.DataFrame) -> dict:
    """For the last bar: the actual indicator reading behind each strategy's vote, the zone where it votes
    buy or sell, and a one-line reason. Mirrors votes() exactly; for the war room only."""
    c, h, l = df["close"].astype(float), df["high"].astype(float), df["low"].astype(float)
    if len(c) < 30:
        return {}
    atr = _atr(df)
    a, px = float(atr.iloc[-1]) or 1e-9, float(c.iloc[-1])
    out = {}
    e20, e50, e200 = _ema(c, 20), _ema(c, 50), _ema(c, 200)
    g = float((e20.iloc[-1] - e50.iloc[-1]) / a)
    out["ema_cross"] = _r(g, -1.5, 1.5, [0, 1.5], [-1.5, 0], f"{g:+.2f} ATR",
                          f"EMA20 is {abs(g):.2f} ATR {'above' if g >= 0 else 'below'} EMA50: short-term trend points "
                          f"{'up' if g >= 0 else 'down'} (full vote at ±0.5 ATR).")
    g2 = float((px - e200.iloc[-1]) / a)
    out["ema200"] = _r(g2, -4, 4, [0, 4], [-4, 0], f"{g2:+.1f} ATR",
                       f"Price is {abs(g2):.1f} ATR {'above' if g2 >= 0 else 'below'} the 200-EMA ({float(e200.iloc[-1]):.5g}): "
                       f"the long trend is {'up' if g2 >= 0 else 'down'} (full vote at ±2 ATR).")
    macd = _ema(c, 12) - _ema(c, 26)
    hist = float((macd - _ema(macd, 9)).iloc[-1] / a)
    out["macd"] = _r(hist, -0.4, 0.4, [0, 0.4], [-0.4, 0], f"{hist:+.2f} ATR",
                     f"MACD histogram is {'positive' if hist >= 0 else 'negative'} ({hist:+.2f} ATR): momentum is "
                     f"{'building up' if hist >= 0 else 'building down'} (full vote at ±0.15 ATR).")
    d = c.diff()
    gain = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean()
    loss = (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = float((100 - 100 / (1 + gain / loss.replace(0, np.nan))).fillna(50).iloc[-1])
    out["rsi_trend"] = _r(rsi, 0, 100, [55, 100], [0, 45], f"RSI {rsi:.0f}",
                          f"RSI is {rsi:.0f}: " + ("above 55, upward momentum." if rsi > 55 else
                                                   "below 45, downward momentum." if rsi < 45 else
                                                   "between 45 and 55, no clear momentum, so no vote."), 0)
    out["rsi_revert"] = _r(rsi, 0, 100, [0, 30], [70, 100], f"RSI {rsi:.0f}",
                           f"RSI is {rsi:.0f}: " + ("below 30, oversold, so it bets on a bounce." if rsi < 30 else
                                                    "above 70, overbought, so it bets on a pullback." if rsi > 70 else
                                                    "not stretched (30-70), so no vote."), 0)
    mid, sd = c.rolling(20).mean(), c.rolling(20).std()
    z = float(((c - mid) / sd.replace(0, np.nan)).fillna(0).iloc[-1])
    out["bb_revert"] = _r(z, -3, 3, [-3, -1.5], [1.5, 3], f"z {z:+.1f}",
                          f"Price is {abs(z):.1f} standard deviations {'above' if z >= 0 else 'below'} its 20-bar average: " +
                          ("stretched, so it bets on a snap back." if abs(z) > 1.5 else "inside the bands, so no vote."), 1)
    out["bb_break"] = _r(z, -3, 3, [2, 3], [-3, -2], f"z {z:+.1f}",
                         f"Price is {abs(z):.1f} standard deviations {'above' if z >= 0 else 'below'} its 20-bar average: " +
                         ("past 2, a breakout it follows." if abs(z) > 2 else "not past 2, so no breakout vote."), 1)
    hi20, lo20 = float(h.rolling(20).max().shift(1).iloc[-1]), float(l.rolling(20).min().shift(1).iloc[-1])
    pos = (px - lo20) / (hi20 - lo20) if hi20 > lo20 else 0.5
    out["donchian"] = _r(pos * 100, -20, 120, [100, 120], [-20, 0], f"{pos * 100:.0f}% of range",
                         f"Close {px:.5g} vs the 20-bar range {lo20:.5g}–{hi20:.5g}: " +
                         ("broke above the high." if px > hi20 else "broke below the low." if px < lo20 else
                          "inside the range; it keeps a breakout view for an hour after one."), 0)
    up, dn = h.diff(), -l.diff()
    pdm = pd.Series(np.where((up > dn) & (up > 0), up, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    mdm = pd.Series(np.where((dn > up) & (dn > 0), dn, 0.0), index=df.index).ewm(alpha=1 / 14, adjust=False).mean()
    pdi, mdi = 100 * pdm / atr, 100 * mdm / atr
    adx = float((100 * (pdi - mdi).abs() / (pdi + mdi).replace(0, np.nan)).ewm(alpha=1 / 14, adjust=False).mean().fillna(0).iloc[-1])
    p_, m_ = float(pdi.iloc[-1]), float(mdi.iloc[-1])
    out["adx_di"] = _r(adx if p_ >= m_ else -adx, -60, 60, [20, 60], [-60, -20], f"ADX {adx:.0f} {'↑' if p_ >= m_ else '↓'}",
                       f"ADX {adx:.0f} ({'trending, above 20' if adx > 20 else 'no real trend, under 20'}); +DI {p_:.0f} vs "
                       f"−DI {m_:.0f}, so buyers are {'stronger' if p_ >= m_ else 'weaker'}.", 0)
    ll, hh = l.rolling(14).min(), h.rolling(14).max()
    k = 100 * (c - ll) / (hh - ll).replace(0, np.nan)
    kv, dv = float(k.fillna(50).iloc[-1]), float(k.rolling(3).mean().fillna(50).iloc[-1])
    out["stoch"] = _r(kv - dv, -40, 40, [0, 40], [-40, 0], f"%K−%D {kv - dv:+.0f}",
                      f"Stochastic %K {kv:.0f} is {'above' if kv >= dv else 'below'} its signal line %D {dv:.0f}: "
                      f"short-term turn {'up' if kv >= dv else 'down'} (full vote at a 20-point gap).", 0)
    roc = float(c.iloc[-1] / c.iloc[-13] - 1) if len(c) > 13 else 0.0
    full = 3 * a / px
    out["roc"] = _r(roc / full if full else 0, -1.5, 1.5, [0, 1.5], [-1.5, 0], f"{roc * 100:+.2f}%",
                    f"Price moved {roc * 100:+.2f}% over the last 12 bars ({'up' if roc >= 0 else 'down'}); a full vote "
                    f"needs ±{full * 100:.2f}% (3 ATR).")
    band = (3.0 * atr).to_numpy()
    hl2, cc = ((h + l) / 2).to_numpy(), c.to_numpy()
    upper, lower, dirn = np.nan, np.nan, 1.0
    for i in range(len(cc)):
        if not np.isfinite(band[i]):
            continue
        u, lo_ = hl2[i] + band[i], hl2[i] - band[i]
        upper = u if not np.isfinite(upper) or u < upper or cc[i - 1] > upper else upper
        lower = lo_ if not np.isfinite(lower) or lo_ > lower or cc[i - 1] < lower else lower
        dirn = 1.0 if cc[i] > upper else -1.0 if cc[i] < lower else dirn
    trail = lower if dirn > 0 else upper
    gap = (px - trail) / a if np.isfinite(trail) else 0.0
    out["supertrend"] = _r(gap, -6, 6, [0, 6], [-6, 0], f"{'UP' if dirn > 0 else 'DOWN'} trail",
                           f"Trend is {'up' if dirn > 0 else 'down'}: price {px:.5g} is {abs(gap):.1f} ATR "
                           f"{'above' if gap >= 0 else 'below'} its ATR trailing stop at {trail:.5g}; it flips if price closes past it.", 1)
    return out


def readings_mtf(m5: pd.DataFrame, htf: dict | None = None) -> dict:
    """readings() for every timeframe, keyed like votes_mtf ('<strategy>' for M5, '<strategy>@<TF>')."""
    if not isinstance(m5.index, pd.DatetimeIndex):
        m5 = m5.copy()
        if "time" not in m5.columns:
            return readings(m5)
        t = m5["time"]
        m5.index = pd.to_datetime(t, unit="s", utc=True) if np.issubdtype(t.dtype, np.number) else pd.to_datetime(t, utc=True)
    elif m5.index.tz is None:
        m5 = m5.tz_localize("UTC")
    out = dict(readings(m5))
    last = m5.index[-1]
    for tf, minutes in TIMEFRAMES.items():
        if minutes == 5:
            continue
        src = (htf or {}).get(tf)
        if src is None or len(src) < 60:
            src = resample_ohlc(m5, minutes)
        if len(src) < 30:
            continue
        # same visibility as align_to_m5: only HTF bars that have closed by the last M5 bar
        idx = src.index if isinstance(src.index, pd.DatetimeIndex) else pd.to_datetime(src.index, utc=True)
        if idx.tz is None:
            idx = idx.tz_localize("UTC")
        src = src[(idx + pd.Timedelta(minutes=minutes - 5)) <= last]
        if len(src) < 30:
            continue
        for k, r in readings(src).items():
            out[f"{k}@{tf}"] = r
    return out
