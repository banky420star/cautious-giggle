"""Pipeline audit: is every part of the decision pipeline getting correct, fresh, complete information?

Checks, per market where it applies:
  feed      price caches (M1/M5/M15): last bar age, gaps, duplicates, bad OHLC, missing spread, clock offset
  live      live quote vs the cache's last close (wrong symbol / stale cache shows up here)
  council   48 indicator readings present, fresh, finite
  kronos    forecast fresh, built on the newest bar, its time stamps in UTC, how many samples it uses
  ob_gold   order-book gold model: producing anything other than 0?
  catboost  LightGBM model: trained rows, whether it accepts today's data
  news      headlines present and recent
  signal    each market decided recently; LSTM counted?; PPO contributing?; council sane
  learning  closed-trade results reaching trade memory and the scorecard
Each line is PASS / WARN / FAIL with the reason. Writes logs/pipeline_audit.json.
    source scripts/mac/account_env.sh; PYTHONPATH=. .venv/bin/python tools/pipeline_audit.py
"""
from __future__ import annotations

import json
import os
import sys
import time
from collections import defaultdict

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LOG = os.path.join(ROOT, "logs")
CACHE = os.path.join(ROOT, "data", "exness_cache")
MARKETS = ["XAUUSDm", "BTCUSDm", "EURUSDm", "USDJPYm", "XAGUSDm", "ETHUSDm", "USOILm"]
CRYPTO = {"BTCUSDm", "ETHUSDm"}
results: list[dict] = []


def rep(area, market, status, what):
    results.append({"area": area, "market": market, "status": status, "what": what})


def jload(name):
    try:
        with open(os.path.join(LOG, name), encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return {}


def tail_events(path, n_bytes=3_000_000):
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - n_bytes))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            out.append(json.loads(ln))
        except ValueError:
            pass
    return out


def market_open(sym, now):
    if sym in CRYPTO:
        return True
    wd, hr = time.gmtime(now).tm_wday, time.gmtime(now).tm_hour
    return not (wd == 5 or (wd == 6 and hr < 22) or (wd == 4 and hr >= 21))


def check_feed(now, live_px):
    for sym in MARKETS:
        inst = sym.rstrip("m")
        for tf, step in (("M1", 60), ("M5", 300), ("M15", 900)):
            p = os.path.join(CACHE, f"{inst}_{tf}.npy")
            if not os.path.exists(p):
                rep("feed", sym, "FAIL", f"{tf} cache missing")
                continue
            a = np.load(p, allow_pickle=True)
            t = a["time"].astype(np.int64)
            age = now - (t[-1] + step)
            # These files are the TRAINING cache (only large history requests use them); live decisions fetch
            # fresh bars directly. So a stale file only matters for training, and only when very old.
            if market_open(sym, now) and age > 2 * 86400:
                rep("feed", sym, "WARN", f"{tf} training cache last bar is {age / 86400:.1f} days old "
                                         "(refreshed on the next training run)")
            if age < -120:
                rep("feed", sym, "FAIL", f"{tf} last bar is {-age / 60:.0f} min IN THE FUTURE: bar times are not UTC "
                                         f"(Kronos time features and freshness checks would be shifted)")
            d = np.diff(t)
            if (d <= 0).any():
                rep("feed", sym, "FAIL", f"{tf} has {(d <= 0).sum()} duplicate/out-of-order bars")
            recent = t >= now - 86400
            if recent.sum() > 10:
                dd = np.diff(t[recent])
                big = dd[dd > step * 5]
                if market_open(sym, now) and len(big) and big.max() > 3600 * 2 and sym in CRYPTO:
                    rep("feed", sym, "WARN", f"{tf} has a {big.max() / 3600:.1f} h gap in the last day")
            o, h, l, c = (a[k].astype(float) for k in ("open", "high", "low", "close"))
            bad = (~np.isfinite(c)) | (c <= 0) | (h < np.maximum(o, c) - 1e-9) | (l > np.minimum(o, c) + 1e-9)
            if bad.any():
                rep("feed", sym, "FAIL", f"{tf} has {bad.sum()} impossible bars (NaN/zero or high<close etc)")
            if tf == "M5":
                sp = a["spread"]
                if (sp == 0).all():
                    rep("feed", sym, "WARN", "M5 cache has no spread data (all 0): anything sizing costs from bars "
                                             "uses a fixed guess, not the real spread")
                if live_px.get(sym) and age < 1800:
                    gap = abs(live_px[sym] / c[-1] - 1) * 1e4
                    vol = float(np.nanstd(np.diff(np.log(c[-300:])))) * 1e4
                    if gap > max(25.0, 6 * vol) and age < 1800:
                        rep("live", sym, "FAIL", f"live price differs from the cache's last close by {gap:.0f} bps "
                                                 "(wrong symbol mapping or stale cache)")
                    else:
                        rep("live", sym, "PASS", f"live price matches the cache ({gap:.1f} bps apart)")
                if len(c) < 400:
                    rep("feed", sym, "FAIL", f"only {len(c)} M5 bars: Kronos needs 400, the indicators ~250")


def check_council(now):
    cr = jload("council_readings.json")
    # the live bot's own bars: indicator readings are rebuilt from them every cycle
    for sym in MARKETS:
        r = cr.get(sym) or {}
        if r and market_open(sym, now) and now - float(r.get("t", 0)) <= 600:
            rep("feed", sym, "PASS", "live decisions use fresh bars (indicator readings rebuilt within 10 min)")
    for sym in MARKETS:
        r = cr.get(sym)
        if not r:
            rep("council", sym, "FAIL", "no indicator readings")
            continue
        age = now - float(r.get("t", 0))
        rd = r.get("readings") or {}
        nonfin = [k for k, v in rd.items() if v.get("v") is None or not np.isfinite(float(v.get("v")))]
        if age > 600:
            rep("council", sym, "FAIL", f"readings {age / 60:.0f} min old")
        elif len(rd) != 48:
            rep("council", sym, "WARN", f"{len(rd)} readings, expected 48 (12 indicators x 4 timeframes)")
        elif nonfin:
            rep("council", sym, "WARN", f"{len(nonfin)} readings not a number: {', '.join(nonfin[:4])}")
        else:
            rep("council", sym, "PASS", f"48 readings, {age:.0f}s old")


def check_external(now):
    ev = jload("external_voters.json")
    if not ev:
        rep("voters", "-", "FAIL", "external_voters.json missing")
        return
    age = now - float(ev.get("t", 0))
    if age > 900:
        rep("voters", "-", "FAIL", f"external voters last updated {age / 60:.0f} min ago")
    vals, det, mk = ev.get("values") or {}, ev.get("details") or {}, ev.get("markets") or {}
    for e in (ev.get("errors") or {}).items():
        rep("voters", "-", "FAIL", f"error {e[0]}: {str(e[1])[:120]}")
    # Kronos
    if ev.get("kronos") != "ready":
        rep("kronos", "-", "FAIL", f"Kronos not ready: {ev.get('kronos')}")
    try:
        src = open(os.path.join(ROOT, "Python", "external_voters.py"), encoding="utf-8").read()
        if "sample_count=1," in src:
            rep("kronos", "-", "WARN", "votes from ONE random sampled path (sample_count=1): a noisy draw, not a "
                                       "consensus forecast; average many samples")
    except OSError:
        pass
    for sym in MARKETS:
        k = det.get(f"{sym}:kronos")
        if f"{sym}:kronos" not in vals or not k:
            rep("kronos", sym, "FAIL", "no Kronos vote/forecast")
            continue
        import pandas as pd
        t_last = pd.Timestamp(k["t_last"]).tz_localize("UTC") if pd.Timestamp(k["t_last"]).tzinfo is None else pd.Timestamp(k["t_last"])
        lag = now - (t_last.timestamp() + 300)
        a = np.load(os.path.join(CACHE, f"{sym.rstrip('m')}_M5.npy"), allow_pickle=True)
        newest = int(a["time"][-1])
        behind = (newest - t_last.timestamp()) / 300
        if lag < -120:
            rep("kronos", sym, "FAIL", f"forecast's last input bar is stamped {-lag / 60:.0f} min in the future: "
                                       "time stamps given to Kronos are not UTC")
        elif behind > 2:
            rep("kronos", sym, "WARN", f"forecast built on a bar {behind:.0f} bars older than the newest one")
        elif market_open(sym, now) and lag > 1800:
            rep("kronos", sym, "WARN", f"forecast input is {lag / 60:.0f} min old")
        else:
            mv, v = float(k.get("move_bps", 0)), float(vals[f"{sym}:kronos"])
            if mv and np.sign(mv) != np.sign(v) and abs(v) > 0.05:
                rep("kronos", sym, "FAIL", f"vote {v:+.2f} points the other way from its own forecast ({mv:+.1f} bps)")
            else:
                rep("kronos", sym, "PASS", f"fresh forecast ({lag / 60:.0f} min), vote {v:+.2f} matches {mv:+.1f} bps")
    # ob_gold: has it ever said anything?
    hist = []
    try:
        for ln in open(os.path.join(LOG, "audit_events.jsonl"), encoding="utf-8"):
            if '"council"' in ln and "XAUUSDm" in ln and "ob_gold" in ln:
                e = json.loads(ln)
                if e.get("event") == "council":
                    hist.append(float((e["payload"].get("votes") or {}).get("ob_gold", 0) or 0))
    except (OSError, ValueError):
        pass
    why = (det.get("XAUUSDm:ob_gold") or {}).get("reason", "")
    nz = sum(abs(x) > 1e-9 for x in hist)
    if hist and nz == 0:
        rep("ob_gold", "XAUUSDm", "FAIL", f"voted 0 on all {len(hist)} gold decisions ever: input missing or model idle")
    elif hist:
        rep("ob_gold", "XAUUSDm", "PASS", f"speaks on {nz}/{len(hist)} gold decisions (only near an order-block zone, "
                                          f"by design). Now: {why[:90]}")
    # catboost
    for sym, c in (ev.get("catboost") or {}).items():
        if c.get("original_model_accepts_today") is False:
            rep("catboost", sym, "WARN", f"the original model rejects today's data; running a stand-in trained on only "
                                         f"{c.get('trained_rows')} rows")
        else:
            rep("catboost", sym, "PASS", f"trained rows {c.get('trained_rows')}")
    # news
    ns = jload("news_state.json")
    for sym in MARKETS:
        d = det.get(f"{sym}:news_llm") or {}
        n = int(d.get("n_headlines", 0) or 0)
        if n == 0:
            rep("news", sym, "WARN", "no headlines behind the news vote (it can only say 0)")
        else:
            rep("news", sym, "PASS", f"{n} headlines")
    if ns and now - float(ns.get("t", ns.get("updated", now)) or now) > 3600:
        rep("news", "-", "WARN", "news state older than an hour")


def check_signals(now):
    ev = tail_events(os.path.join(LOG, "audit_events.jsonl"))
    sig = defaultdict(list)
    for e in ev:
        if e.get("event") == "signal":
            p = e.get("payload") or {}
            sig[p.get("symbol")].append(p)
    for sym in MARKETS:
        s = sig.get(sym) or []
        if not s:
            rep("signal", sym, "FAIL", "no decisions recorded recently")
            continue
        last = s[-1]
        lstm = sum(float(x.get("agi_counted", 0) or 0) != 0 for x in s)
        ppo_w = sum(float((x.get("weights_used") or {}).get("ppo", 0) or 0) > 0 for x in s)
        ppo_nz = sum(abs(float(x.get("ppo_exposure", 0) or 0)) > 1e-9 for x in s)
        rep("signal", sym, "PASS" if lstm else "WARN",
            f"LSTM counted in {lstm}/{len(s)} recent decisions" + ("" if lstm else
            " (its confidence never reaches the 0.85 bar: it is shown but never used)"))
        rep("ppo", sym, "PASS" if ppo_w and ppo_nz else "WARN",
            f"PPO weight > 0 in {ppo_w}/{len(s)}, non-zero position in {ppo_nz}/{len(s)}"
            + ("" if ppo_w and ppo_nz else " (the trader model is not contributing to decisions)"))
        tm = last.get("trade_memory") or {}
        if int(tm.get("trades", 0) or 0) == 0:
            rep("learning", sym, "FAIL", "trade memory has 0 trades although the bot has traded: closed-trade results "
                                         "never reach it")


def check_learning():
    te = os.path.join(LOG, "trade_events.jsonl")
    n_open = n_close = 0
    try:
        for ln in open(te, encoding="utf-8"):
            n_open += '"trade_open"' in ln
            n_close += '"trade_closed"' in ln
    except OSError:
        pass
    if n_open and not n_close:
        rep("learning", "-", "FAIL", f"{n_open} trades opened but 0 'trade_closed' events: exit price and profit are never "
                                     "recorded (the API's deal history is not being read on close), so the scorecard, "
                                     "setup gate and trade memory learn nothing from results")
    sc = os.path.join(LOG, "voter_scorecard.jsonl")
    if not os.path.exists(sc):
        rep("learning", "-", "WARN", "no trade has been scored yet in 'Who called it'")


def check_mapping():
    try:
        src = open(os.path.join(ROOT, "Python", "data_feed.py"), encoding="utf-8").read()
        if "USO-ILM" in src or "USOILM" in src.replace("-", ""):
            rep("feed", "USOILm", "WARN", "Dukascopy fallback maps oil to a wrong instrument (should be LIGHT.CMD-USD)")
    except OSError:
        pass


def main():
    now = time.time()
    live = {}
    try:
        from Python.mt5_compat import mt5
        mt5.initialize()
        for s in MARKETS:
            t = mt5.symbol_info_tick(s)
            if t:
                live[s] = (float(t.bid) + float(t.ask)) / 2
    except Exception as e:  # noqa: BLE001
        rep("live", "-", "WARN", f"could not read live prices: {str(e)[:80]}")
    check_feed(now, live)
    check_mapping()
    check_council(now)
    check_external(now)
    check_signals(now)
    check_learning()
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    results.sort(key=lambda r: (order[r["status"]], r["area"], r["market"]))
    with open(os.path.join(LOG, "pipeline_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"t": now, "results": results}, f, indent=1)
    for r in results:
        print(f"{r['status']:4} {r['area']:9} {r['market']:8} {r['what']}")
    c = {k: sum(r["status"] == k for r in results) for k in order}
    print(f"\n{c['FAIL']} FAIL, {c['WARN']} WARN, {c['PASS']} PASS")


if __name__ == "__main__":
    main()
