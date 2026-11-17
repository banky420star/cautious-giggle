"""Backtest the news sweep plays on past US release times and seed logs/news_playbook.json.

There is no free historical calendar, so release minutes are found from the schedule plus the
market itself: every weekday at 08:30 and 10:00 New York time (when US data is published), kept
only when the first 3 minutes moved >= 3x the range of the 2 hours before. That filter looks at
how big the reaction was, never its direction, so it does not favour either play.

    python tools/news_backtest.py            # report only
    python tools/news_backtest.py --seed     # report and write the playbook
"""
from __future__ import annotations

import datetime as dt
import os
import sys
from zoneinfo import ZoneInfo

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from Python import news_trader as nt  # noqa: E402

NY = ZoneInfo("America/New_York")
CACHE = os.path.join(nt.ROOT, "data", "exness_cache")


def load(sym):
    a = np.load(os.path.join(CACHE, f"{sym.rstrip('m')}_M1.npy"), allow_pickle=True)
    return (a["time"].astype(float), a["open"], a["high"], a["low"], a["close"])


def release_minutes(t, h, l):
    days = sorted({dt.datetime.fromtimestamp(x, NY).date() for x in t[:: 60]})
    out = []
    for d in days:
        if d.weekday() >= 5:
            continue
        for hh, mm in ((8, 30), (10, 0)):
            t0 = dt.datetime(d.year, d.month, d.day, hh, mm, tzinfo=NY).timestamp()
            i0 = int(np.searchsorted(t, t0))
            if i0 < 130 or i0 + 3 >= len(t) or t[i0] != t0:
                continue
            first = h[i0:i0 + 3].max() - l[i0:i0 + 3].min()
            prev = [h[k:k + 3].max() - l[k:k + 3].min() for k in range(i0 - 120, i0, 3)]
            if first >= 3 * float(np.mean(prev)):
                out.append(t0)
    return out


def main(seed: bool):
    pb = {}
    for sym in ("XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "USOILm", "XAGUSDm", "ETHUSDm"):
        t, o, h, l, c = load(sym)
        evs = release_minutes(t, h, l)
        print(f"\n{sym}: {len(evs)} release reactions found "
              f"({dt.datetime.fromtimestamp(t[0], NY):%d %b} – {dt.datetime.fromtimestamp(t[-1], NY):%d %b})")
        for play in ("fade", "breakout"):
            rs = []
            for t0 in evs:
                r = nt.simulate(play, sym, t, o, h, l, c, t0)
                if r is None:
                    continue
                rs.append(r["r"] if r["traded"] else None)
                nt.record(pb, sym, nt.GENERIC, play, r["r"], r["traded"])
            s = nt.stats(rs)
            print(f"  {play:9s} trades {s['n']:3d}/{s['events']:3d} events | win {s['win']:.0%} | "
                  f"avg {s['avg_r']:+.2f}R | total {s['total_r']:+.1f}R (after costs)")
        # controls: same rules at random non-release times
        rng = np.random.default_rng(7)
        ctrl = rng.choice(t[200:-200], size=min(300, len(t) // 400), replace=False)
        for play in ("fade", "breakout"):
            rs = [x for x in (nt.simulate(play, sym, t, o, h, l, c, float(x)) for x in ctrl) if x and x["traded"]]
            if rs:
                print(f"  control {play:9s} (random times) trades {len(rs)} | avg {np.mean([x['r'] for x in rs]):+.2f}R")
        ch = nt.choose(pb, sym, nt.GENERIC)
        print(f"  → plan: {ch['play'] or 'stand aside'} — {ch['why']}")
    if seed:
        old = nt.load_playbook()
        for sym, d in pb.items():
            old.setdefault(sym, {})[nt.GENERIC] = d[nt.GENERIC]
        nt.save_playbook(old)
        print(f"\nseeded {nt.PLAYBOOK}")


if __name__ == "__main__":
    main("--seed" in sys.argv)
