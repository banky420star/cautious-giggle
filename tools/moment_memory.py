"""Moment memory: do recurring indicator EVENT combinations come before big moves, often enough to trade?

The idea (from the user): the model should remember situations like "EMA cross + bounce off the lower Bollinger
band" and what the market did next, especially when an impactful move followed.

For every 5-minute bar of the last ~year (broker M5 cache), per market:
  fingerprint = which events just happened (last 3 bars):
      EMA9/21 cross up/down, bounce off lower Bollinger band, rejection at upper band, RSI <30 / >70,
      MACD cross up/down, 20-bar breakout up/down
    + session (Asia / London / New York) + 1-hour trend (up/down) + volatility (calm / normal / wild)
  outcome     = the move over the next 60 minutes (12 bars), in bps, and whether it was IMPACTFUL
                (bigger than 2x the usual 60-minute move)
MEMORY: on the first 70% of the history, every fingerprint seen >= 40 times gets its record: average next-hour
move, share of impactful up / down moves. A fingerprint becomes a RULE when its average move beats the
cost of a trade with t >= 2.5 (strict, because many fingerprints are tried).
TEST: on the last 30% (never used to pick rules), every rule trades its direction whenever its fingerprint
appears; result = actual next-hour move minus the real cost.
    PYTHONPATH=. .venv/bin/python tools/moment_memory.py
"""
import json
import os
from collections import defaultdict

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
COST = {"XAUUSD": 0.59, "BTCUSD": 0.79, "EURUSD": 0.71, "USDJPY": 0.63, "XAGUSD": 4.42, "ETHUSD": 4.10, "USOIL": 2.09}
H = 12          # 60 minutes ahead
MIN_N = 40
T_RULE = 2.5


def ema(s, n):
    return s.ewm(span=n, adjust=False).mean()


def fingerprints(df: pd.DataFrame) -> pd.Series:
    c, h, l = df["close"], df["high"], df["low"]
    e9, e21 = ema(c, 9), ema(c, 21)
    up_x = (e9 > e21) & (e9.shift(1) <= e21.shift(1))
    dn_x = (e9 < e21) & (e9.shift(1) >= e21.shift(1))
    ma, sd = c.rolling(20).mean(), c.rolling(20).std()
    lo_b, hi_b = ma - 2 * sd, ma + 2 * sd
    bounce = (l.shift(1) <= lo_b.shift(1)) & (c > lo_b) & (c > c.shift(1))      # touched lower band, closed back up
    reject = (h.shift(1) >= hi_b.shift(1)) & (c < hi_b) & (c < c.shift(1))
    d = c.diff()
    rs = d.clip(lower=0).ewm(alpha=1 / 14, adjust=False).mean() / (-d.clip(upper=0)).ewm(alpha=1 / 14, adjust=False).mean()
    rsi = 100 - 100 / (1 + rs)
    macd = ema(c, 12) - ema(c, 26)
    sig = ema(macd, 9)
    m_up = (macd > sig) & (macd.shift(1) <= sig.shift(1))
    m_dn = (macd < sig) & (macd.shift(1) >= sig.shift(1))
    bo_up = c > h.shift(1).rolling(20).max()
    bo_dn = c < l.shift(1).rolling(20).min()
    recent = lambda x: x.rolling(3, min_periods=1).max().astype(bool)  # noqa: E731  event within the last 3 bars
    ev = {"ema_x_up": recent(up_x), "ema_x_dn": recent(dn_x), "bb_bounce": recent(bounce), "bb_reject": recent(reject),
          "rsi_lo": rsi < 30, "rsi_hi": rsi > 70, "macd_up": recent(m_up), "macd_dn": recent(m_dn),
          "break_up": recent(bo_up), "break_dn": recent(bo_dn)}
    hr = df.index.hour
    sess = np.where(hr < 7, "asia", np.where(hr < 13, "london", np.where(hr < 21, "ny", "late")))
    h1 = ema(c, 12 * 20)                      # ~20-hour EMA on 5-min bars: the 1-hour trend's direction
    trend = np.where(c > h1, "h1up", "h1dn")
    atr = (h - l).rolling(48).mean()
    q = atr.rolling(2000, min_periods=500).rank(pct=True)
    vol = np.where(q < 0.33, "calm", np.where(q > 0.67, "wild", "normal"))
    names = np.array(list(ev))
    E = np.column_stack([ev[k].fillna(False).to_numpy() for k in ev])
    fp = []
    for i in range(len(df)):
        act = names[E[i]]
        fp.append("+".join(act) + f"|{sess[i]}|{trend[i]}|{vol[i]}" if len(act) else None)
    return pd.Series(fp, index=df.index)


def run():
    out = {}
    tot_n, tot_pnl = 0, 0.0
    for sym, cost in COST.items():
        a = np.load(os.path.join(ROOT, "data", "exness_cache", f"{sym}_M5.npy"), allow_pickle=True)
        df = pd.DataFrame(a)
        df.index = pd.to_datetime(df["time"], unit="s", utc=True)
        df = df[~df.index.duplicated()].sort_index()
        c = df["close"]
        fwd = (c.shift(-H) / c - 1) * 1e4
        gap_ok = (df["time"].shift(-H) - df["time"]) <= H * 300 * 1.5
        typical = fwd.abs().rolling(2000, min_periods=500).median().shift(H)   # usual 60-min move, known in advance
        fp = fingerprints(df)
        ok = fp.notna() & fwd.notna() & gap_ok & typical.notna()
        split = df.index[int(len(df) * 0.7)]
        tr, te = ok & (df.index < split), ok & (df.index >= split + pd.Timedelta(hours=1))
        mem = defaultdict(list)
        for k, v, t in zip(fp[tr], fwd[tr], typical[tr]):
            mem[k].append((v, t))
        rules = {}
        for k, rows in mem.items():
            if len(rows) < MIN_N:
                continue
            v = np.array([r[0] for r in rows]); t = np.array([r[1] for r in rows])
            m, s = v.mean(), v.std() / np.sqrt(len(v))
            if abs(m) > cost and abs(m) / (s or 1e-9) >= T_RULE:
                rules[k] = {"side": int(np.sign(m)), "train_n": len(v), "train_move_bps": round(float(m), 2),
                            "impact_up_pct": round(100 * float((v > 2 * t).mean()), 1),
                            "impact_dn_pct": round(100 * float((v < -2 * t).mean()), 1)}
        test_pnl = []
        per_rule = defaultdict(list)
        for k, v in zip(fp[te], fwd[te]):
            if k in rules:
                r = rules[k]["side"] * v - cost
                test_pnl.append(r)
                per_rule[k].append(r)
        tp = np.array(test_pnl)
        base_imp = float((fwd[tr].abs() > 2 * typical[tr]).mean() * 100)
        res = {"cost_bps": cost, "fingerprints_seen": len(mem), "fingerprints_with_40+": sum(len(r) >= MIN_N for r in mem.values()),
               "rules_found": len(rules), "impactful_base_rate_pct": round(base_imp, 1),
               "test_trades": len(tp), "test_net_bps_per_trade": round(float(tp.mean()), 2) if len(tp) else None,
               "test_t": round(float(tp.mean() / (tp.std() / np.sqrt(len(tp)))), 2) if len(tp) > 2 else None,
               "test_win_pct": round(100 * float((tp > 0).mean()), 1) if len(tp) else None,
               "rules": {k: {**rules[k], "test_n": len(per_rule[k]),
                             "test_net_bps": round(float(np.mean(per_rule[k])), 2) if per_rule[k] else None}
                         for k in sorted(rules, key=lambda k: -rules[k]["train_n"])[:12]}}
        out[sym] = res
        tot_n += len(tp); tot_pnl += float(tp.sum()) if len(tp) else 0.0
        print(f"{sym:7} cost {cost:4.2f} | {res['fingerprints_seen']:5} fingerprints, {res['fingerprints_with_40+']:4} seen 40+ times | "
              f"impactful moves are {res['impactful_base_rate_pct']}% of hours | RULES {res['rules_found']:3} | "
              f"TEST {res['test_trades']:5} trades, net {res['test_net_bps_per_trade']} bps/trade (t={res['test_t']}), win {res['test_win_pct']}%", flush=True)
        for k, r in list(res["rules"].items())[:5]:
            print(f"        {'BUY ' if r['side'] > 0 else 'SELL'} {k:70.70} train {r['train_n']:4}x {r['train_move_bps']:+6.2f} bps"
                  f" (impact up {r['impact_up_pct']}% / dn {r['impact_dn_pct']}%) -> test {r['test_n']:3}x {r['test_net_bps']} bps")
    print(f"\nALL MARKETS: {tot_n} test trades, net {tot_pnl / max(1, tot_n):+.2f} bps per trade")
    with open(os.path.join(ROOT, "logs", "moment_memory.json"), "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    run()
