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
  filter    the baseline big-move filter: trained, skill on unseen data, live and fresh, GO share
  learning  (process) live closes recorded, model memory updating, GO/STOP grades, what each model has learned
  lstm/ppo  does each trained model beat doing nothing; is the canary trial measuring correctly; training logs
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
        # the audit log rotates (audit_events.jsonl.1 holds the older part): read both, oldest first
        paths = [os.path.join(LOG, n) for n in ("audit_events.jsonl.1", "audit_events.jsonl")]
        for ln in (x for pth in paths if os.path.exists(pth) for x in open(pth, encoding="utf-8")):
            if '"event": "council"' in ln and "XAUUSDm" in ln and "ob_gold" in ln:
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


def check_move_filter(now):
    """The baseline: per-market big-move filter. Trained? good enough on unseen data? live and fresh? sane GO share?"""
    mdir = os.path.join(ROOT, "models", "move_filter")
    ev = [e for e in tail_events(os.path.join(LOG, "audit_events.jsonl"), 20_000_000) if e.get("event") == "move_filter"]
    by = defaultdict(list)
    for e in ev:
        by[e["payload"].get("symbol")].append(e)
    rep_path = os.path.join(LOG, "move_filter_report.json")
    report = json.load(open(rep_path)) if os.path.exists(rep_path) else {}
    for s in MARKETS:
        base = s[:-1]
        mp = os.path.join(mdir, f"{base}.npz.pkl")
        if not os.path.exists(mp):
            rep("filter", s, "FAIL", "no trained move filter (run tools/train_move_filter.py then export_move_filter.py)")
            continue
        age_d = (now - os.path.getmtime(mp)) / 86400
        r = report.get(base, {})
        auc = r.get("auc")
        if auc is None or auc < 0.6:
            rep("filter", s, "WARN", f"unseen-data skill {auc} is weak (want >= 0.60)")
        if age_d > 30:
            rep("filter", s, "WARN", f"model is {age_d:.0f} days old: retrain on recent data")
        rows = by.get(s, [])
        if not rows:
            rep("filter", s, "WARN" if market_open(s, now) else "PASS",
                "no live readings yet" + ("" if market_open(s, now) else " (market closed)"))
            continue
        last_age = now - _ts(rows[-1]["ts"])
        day = [x for x in rows if now - _ts(x["ts"]) < 86400]
        go = np.mean([x["payload"]["go"] for x in day]) if day else float("nan")
        ps = [x["payload"]["p"] for x in day]
        bad = [p for p in ps if not (0 <= p <= 1)]
        if bad:
            rep("filter", s, "FAIL", f"{len(bad)} readings outside 0..1")
        elif market_open(s, now) and last_age > 900:
            rep("filter", s, "FAIL", f"last reading {last_age/60:.0f} min old while the market is open")
        elif day and not (0.10 <= go <= 0.60) and len(day) >= 24:
            rep("filter", s, "WARN", f"GO {go:.0%} of the last day (trained for ~30%): live inputs may differ from training")
        else:
            rep("filter", s, "PASS", f"skill {auc} on unseen data, model {age_d:.0f} d old, last reading {last_age/60:.0f} min ago, "
                                     f"GO {go:.0%} of {len(day)} readings today (watch-only)")


def _ts(iso):
    import datetime as _dt
    return _dt.datetime.fromisoformat(str(iso).replace("Z", "+00:00")).timestamp()


def check_learning_process(now):
    """Does each learner actually learn from correct results, and does what it learned beat doing nothing?"""
    # 1. results reach the learners: live closes recorded since the deal-id fix
    te = os.path.join(LOG, "trade_events.jsonl")
    opens, closes, live_closes = [], [], []
    for e in tail_events(te, 50_000_000):
        if e.get("event") == "trade_open":
            opens.append(e)
        elif e.get("event") == "trade_closed":
            closes.append(e)
            if not (e.get("payload") or e).get("backfill") and "backfill" not in json.dumps(e):
                live_closes.append(e)
    day_open = [e for e in opens if now - _ts(e.get("ts")) < 86400]
    day_close = [e for e in live_closes if now - _ts(e.get("ts")) < 86400]
    if day_open and not day_close:
        rep("learning", "-", "FAIL", f"{len(day_open)} trades opened in 24 h but no live close recorded")
    else:
        rep("learning", "-", "PASS", f"results reach the learners: {len(day_close)} live closes recorded in 24 h "
                                     f"({len(live_closes)} live + {len(closes) - len(live_closes)} backfilled overall)")
    # 2. model memory (grades every model's calls after costs) is updating, and the GO/STOP baseline is in it
    mm = os.path.join(LOG, "memory", "model_memory.json")
    try:
        mem = json.load(open(mm))
        age = now - os.path.getmtime(mm)
        stats = mem.get("stats", {})
        rep("learning", "-", "PASS" if age < 900 else "FAIL",
            f"model memory updated {age/60:.0f} min ago, {len(stats)} grade buckets, horizon {mem.get('horizon_bars')} bars")
        n_mf = sum("|mf:" in k for k in stats)
        rep("learning", "-", "PASS" if n_mf else "WARN",
            f"{n_mf} GO/STOP grade buckets: every model is graded per big-move state" if n_mf
            else "no GO/STOP grades yet (they appear one scoring horizon after the first filter readings)")
        # 3. what each learner has learned, per market, after costs (all contexts)
        for s in MARKETS:
            parts = []
            for m in ("lstm", "ppo", "council", "blend", "trade"):
                st = stats.get(f"{s}|{m}|*")
                if st and st.get("w", 0) > 30:
                    parts.append(f"{m} {st['m']:+.2f}")
            inds = sorted(((k.split("|")[1][4:], v["m"]) for k, v in stats.items()
                           if k.startswith(f"{s}|ind:") and k.endswith("|*") and v.get("w", 0) > 200), key=lambda x: -x[1])
            if inds:
                parts.append(f"best voter {inds[0][0]} {inds[0][1]:+.2f}, worst {inds[-1][0]} {inds[-1][1]:+.2f}")
            tr = stats.get(f"{s}|trade|*")
            status = "WARN" if tr and tr.get("w", 0) > 30 and tr["m"] < 0 else "PASS"
            rep("learning", s, status, "learned edge per call after costs (bps): " + (", ".join(parts) or "not enough calls yet")
                + (" -> the bot's own trades lose after costs" if status == "WARN" else ""))
    except (OSError, ValueError) as e:
        rep("learning", "-", "FAIL", f"model memory unreadable: {e}")
    # 4. LSTM: does each trained model beat always-HOLD on its own validation data?
    pdir = os.path.join(ROOT, "models", "per_symbol")
    for s in MARKETS:
        mp = os.path.join(pdir, f"lstm_{s}.meta.json")
        if not os.path.exists(mp):
            rep("lstm", s, "WARN", "no per-market LSTM trained; the shared one is used")
            continue
        m = json.load(open(mp))
        acc, base = m.get("val_acc"), m.get("val_majority_acc")
        if acc is not None and base is not None and acc < base:
            rep("lstm", s, "FAIL", f"validation accuracy {acc:.1f}% is below always-HOLD {base:.1f}%: it learned nothing usable "
                                   "(it predicts the next 5-min bar from price/indicators only, which the indicator map shows "
                                   "carries no direction signal)")
        else:
            rep("lstm", s, "PASS", f"validation accuracy {acc} vs always-HOLD {base}")
    # 5. PPO model registry: new models graduate only through a canary trial; is that trial measuring correctly?
    try:
        act = json.load(open(os.path.join(ROOT, "models", "registry", "active.json")))
        for s, v in (act.get("symbols") or {}).items():
            cs = v.get("canary_state") or {}
            champ = v.get("champion")
            if not champ and not v.get("canary"):
                rep("ppo", s, "WARN", "no champion and no canary model")
                continue
            msg = (f"champion {'none' if not champ else os.path.basename(champ)}; canary {os.path.basename(v.get('canary') or '-')}: "
                   f"{cs.get('trades')} trades, P/L {cs.get('realized_pnl')}, {cs.get('reason')}")
            if cs.get("reason") == "no_metrics":
                rep("ppo", s, "WARN", msg + " -> the trial has never been measured")
            elif cs.get("trades") == 0 and (cs.get("realized_pnl") or 0) != 0:
                rep("ppo", s, "FAIL", msg + " -> P/L without trades: the trial counts trades from the risk engine's DAILY counter "
                                         "(resets at midnight and on restart) and P/L over 7 days (includes trades made before "
                                         "this model), so it can neither graduate nor be judged fairly")
            else:
                rep("ppo", s, "WARN" if not champ else "PASS", msg)
    except (OSError, ValueError) as e:
        rep("ppo", "-", "FAIL", f"registry unreadable: {e}")
    # 6. training runs: did the last one get its data, and are the logs trustworthy?
    for name, area in (("ppo_training.log", "ppo"), ("lstm_training.log", "lstm")):
        pth = os.path.join(LOG, name)
        if not os.path.exists(pth):
            continue
        with open(pth, "rb") as f:
            f.seek(0, os.SEEK_END)
            f.seek(max(0, f.tell() - 2_000_000))
            tail = f.read().decode("utf-8", "replace")
        if "pytest-of-" in tail or "test_reason" in tail or "registry_symbol_test" in tail:
            rep(area, "-", "WARN", f"{name} contains lines written by the test suite (fake promotions, test halts): "
                                   "tests must log to a temp file, or this log can't be trusted when reading training history")
        dns = tail.count("NameResolutionError")
        if dns:
            rep(area, "-", "WARN", f"{name}: {dns} data downloads failed on name lookup (no internet/DNS) in recent runs: "
                                   "training may have used old cached bars or skipped markets")


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
    check_move_filter(now)
    check_learning_process(now)
    order = {"FAIL": 0, "WARN": 1, "PASS": 2}
    sections = ["feed", "live", "council", "kronos", "ob_gold", "catboost", "news", "signal", "filter", "learning", "lstm", "ppo"]
    results.sort(key=lambda r: (sections.index(r["area"]) if r["area"] in sections else 99, order[r["status"]], r["market"]))
    with open(os.path.join(LOG, "pipeline_audit.json"), "w", encoding="utf-8") as f:
        json.dump({"t": now, "results": results}, f, indent=1)
    last = None
    for r in results:
        if r["area"] != last:
            sec = [x for x in results if x["area"] == r["area"]]
            print(f"\n== {r['area'].upper()}  ({sum(x['status'] == 'FAIL' for x in sec)} fail, "
                  f"{sum(x['status'] == 'WARN' for x in sec)} warn, {sum(x['status'] == 'PASS' for x in sec)} pass)")
            last = r["area"]
        print(f"  {r['status']:4} {r['market']:8} {r['what']}")
    c = {k: sum(r["status"] == k for r in results) for k in order}
    print(f"\n{c['FAIL']} FAIL, {c['WARN']} WARN, {c['PASS']} PASS")


if __name__ == "__main__":
    main()
