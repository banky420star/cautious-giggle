"""Financial news x market reaction.

Pulls the economic calendar (ForexFactory weekly JSON) and headline RSS feeds (CNBC, MarketWatch,
Cointelegraph), then measures how each market moved after every item using Exness 1-minute bars:

    move_15 = % change from the last bar before the item to 15 minutes later
    normal  = median absolute 15-minute move over the previous 24 h
    size    = |move_15| / normal          (>= 2 means a clearly unusual move)

Each item gets a topic (inflation, Fed/rates, jobs, growth, crypto, geopolitics, oil, currencies),
the markets that usually react to that topic and why, and a check of whether the measured moves
matched that usual pattern. Timing is evidence, not proof: when several things happen at once the
text says so. Read-only; nothing here places trades.
"""
from __future__ import annotations

import datetime as dt
import email.utils
import hashlib
import json
import os
import re
import threading
import time
import xml.etree.ElementTree as ET

import numpy as np
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STORE = os.path.join(os.environ.get("AGI_LOG_DIR") or os.path.join(ROOT, "logs"), "news_items.jsonl")
UA = {"User-Agent": "Mozilla/5.0 (compatible; trading-dashboard/1.0)"}

CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
FEEDS = {
    "CNBC": "https://www.cnbc.com/id/100003114/device/rss/rss.html",
    "MarketWatch": "https://feeds.content.dowjones.io/public/rss/mw_topstories",
    "Cointelegraph": "https://cointelegraph.com/rss",
}
MARKETS = {  # symbol -> (label, what a rise means)
    "XAUUSDm": ("Gold", "gold up"),
    "BTCUSDm": ("Bitcoin", "bitcoin up"),
    "EURUSDm": ("EUR/USD", "dollar weaker"),
    "USDJPYm": ("USD/JPY", "dollar stronger"),
    "USOILm": ("Oil", "oil up"),
    "XAGUSDm": ("Silver", "silver up"),
    "ETHUSDm": ("Ethereum", "ether up"),
}

TOPICS = [
    ("inflation", r"\b(cpi|pce|ppi|inflation|prices? index|core prices)\b",
     ["XAUUSDm", "EURUSDm", "USDJPYm", "BTCUSDm"],
     "Inflation data changes how long rates stay high: hotter than expected → dollar up, gold and bitcoin down; cooler → the reverse."),
    ("fed", r"\b(fed|fomc|powell|rate (cut|hike|decision)|interest rates?|treasury yields?|bond yields?|central bank|ecb|boj|lagarde)\b",
     ["XAUUSDm", "EURUSDm", "USDJPYm", "BTCUSDm"],
     "Rate expectations drive the dollar: hawkish (higher for longer) → dollar up, gold and bitcoin down; dovish → the reverse."),
    ("jobs", r"\b(payrolls?|nfp|jobless|unemployment|employment|jobs report|labor market|labour market)\b",
     ["XAUUSDm", "EURUSDm", "USDJPYm"],
     "Strong jobs data → rates stay higher → dollar up, gold down; weak jobs → the reverse."),
    ("growth", r"\b(gdp|pmi|retail sales|ism|consumer confidence|sentiment|recession|growth)\b",
     ["EURUSDm", "USDJPYm", "XAUUSDm", "USOILm"],
     "Growth surprises move the dollar and risk appetite; weak growth tends to help gold."),
    ("crypto", r"\b(bitcoin|btc|crypto|ether(eum)?|stablecoin|sec|etf|binance|coinbase|blockchain|token)\b",
     ["BTCUSDm", "ETHUSDm"],
     "Crypto-specific news (regulation, ETF flows, exchanges) moves bitcoin and ether directly."),
    ("geopolitics", r"\b(war|missile|strike|attack|sanction|tariff|trade war|iran|israel|russia|ukraine|china|taiwan|conflict|ceasefire)\b",
     ["XAUUSDm", "USOILm", "USDJPYm"],
     "Geopolitical stress sends money to safe havens (gold, yen) and can lift oil on supply fears; easing does the reverse."),
    ("oil", r"\b(opec|crude|oil|brent|wti|barrels?)\b",
     ["USOILm"], "Supply or demand news for crude moves oil directly."),
    ("currencies", r"\b(dollar|yen|euro|currency|fx|forex|yuan)\b",
     ["EURUSDm", "USDJPYm", "XAUUSDm"],
     "Dollar moves feed straight into gold (priced in dollars): stronger dollar → gold usually lower."),
]
COUNTRY_TOPIC = {"USD": None, "EUR": "currencies", "JPY": "currencies", "CNY": "growth", "GBP": "currencies"}


def _topic(text: str):
    t = text.lower()
    for name, rx, mkts, why in TOPICS:
        if re.search(rx, t):
            return name, mkts, why
    return None, [], ""


def _id(*parts) -> str:
    return hashlib.sha1("|".join(map(str, parts)).encode()).hexdigest()[:16]


CAL_CACHE = os.path.join(os.environ.get("AGI_LOG_DIR") or os.path.join(ROOT, "logs"), "calendar_cache.json")
CAL_MAX_AGE = 1800  # the calendar host rate-limits hard: one fetch per 30 min shared by all processes


def get_calendar() -> list[dict]:
    """High/medium-impact events this week as [{t, title, impact, country, forecast, previous}].
    Cached on disk so the bot and the dashboard share one fetch."""
    cached = None
    try:
        with open(CAL_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if time.time() - float(cached.get("fetched", 0)) < CAL_MAX_AGE:
            return cached["events"]
    except Exception:
        pass
    try:
        r = requests.get(CALENDAR_URL, headers=UA, timeout=15)
        raw = r.json() if r.status_code == 200 else None
    except Exception:
        raw = None
    if raw is None:
        if cached:  # keep using the old copy; retry after another half hour
            cached["fetched"] = time.time() - CAL_MAX_AGE + 600
            _write_json(CAL_CACHE, cached)
            return cached["events"]
        _write_json(CAL_CACHE, {"fetched": time.time() - CAL_MAX_AGE + 900, "events": []})  # back off 15 min
        return []
    events = []
    for e in raw:
        try:
            t = dt.datetime.fromisoformat(e["date"]).timestamp()
        except Exception:
            continue
        if str(e.get("impact")) not in ("High", "Medium"):
            continue
        events.append({"t": t, "title": f"{e.get('country', '')} {e.get('title', '')}".strip(), "impact": e.get("impact"),
                       "country": e.get("country"), "forecast": e.get("forecast") or "", "previous": e.get("previous") or ""})
    _write_json(CAL_CACHE, {"fetched": time.time(), "events": events})
    return events


def _write_json(path, obj):
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(obj, f)
    os.replace(tmp, path)


def topic_of(title: str, country: str | None = None) -> str:
    t, _, _ = _topic(title)
    return t or COUNTRY_TOPIC.get(country or "") or "growth"


class NewsMarkets:
    def __init__(self, mt5, poll_sec: int = 300):
        self.mt5 = mt5
        self.poll_sec = poll_sec
        self.items: dict[str, dict] = {}
        self.calendar: list[dict] = []
        self.bars: dict[str, np.ndarray] = {}
        self.lock = threading.Lock()
        self.status = {"last_fetch": None, "errors": []}
        self._load()

    # ------------------------------------------------------------------ persistence
    def _load(self):
        try:
            with open(STORE, "r", encoding="utf-8") as f:
                for line in f:
                    try:
                        it = json.loads(line)
                        self.items[it["id"]] = it
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

    def _save(self):
        cutoff = time.time() - 7 * 86400
        rows = [it for it in self.items.values() if it["t"] >= cutoff]
        tmp = STORE + ".tmp"
        os.makedirs(os.path.dirname(STORE), exist_ok=True)
        with open(tmp, "w", encoding="utf-8") as f:
            for it in sorted(rows, key=lambda x: x["t"]):
                f.write(json.dumps(it) + "\n")
        os.replace(tmp, STORE)

    # ------------------------------------------------------------------ fetching
    def fetch(self):
        errs = []
        try:
            now = time.time()
            events = get_calendar()
            for e in events:
                t, title, impact = e["t"], e["title"], e["impact"]
                if t <= now and (impact == "High" or e.get("country") == "USD"):
                    iid = _id("cal", title, t)
                    if iid not in self.items:
                        topic = topic_of(title, e.get("country"))
                        mkts = next((m for n, _, m, _ in TOPICS if n == topic), [])
                        why = next((w for n, _, _, w in TOPICS if n == topic), "")
                        self.items[iid] = {"id": iid, "t": t, "source": "Economic calendar", "kind": "calendar",
                                           "title": title, "impact": impact, "country": e.get("country"),
                                           "detail": f"forecast {e.get('forecast') or '—'} · previous {e.get('previous') or '—'}",
                                           "topic": topic, "watch": mkts, "mechanism": why}
            self.calendar = sorted(events, key=lambda x: x["t"])
        except Exception as exc:
            errs.append(f"calendar: {exc}")
        for src, url in FEEDS.items():
            try:
                xml = requests.get(url, headers=UA, timeout=15).content
                root = ET.fromstring(xml)
                for it in root.iter("item"):
                    title = (it.findtext("title") or "").strip()
                    pub = it.findtext("pubDate")
                    if not title or not pub:
                        continue
                    try:
                        t = email.utils.parsedate_to_datetime(pub).timestamp()
                    except Exception:
                        continue
                    if time.time() - t > 3 * 86400:
                        continue
                    iid = _id(src, title)
                    if iid in self.items:
                        continue
                    desc = re.sub(r"<[^>]+>", "", it.findtext("description") or "").strip()[:280]
                    topic, mkts, why = _topic(title + " " + desc)
                    if not topic:
                        continue  # keep only market-relevant headlines
                    self.items[iid] = {"id": iid, "t": t, "source": src, "kind": "headline", "title": title,
                                       "impact": None, "detail": desc, "link": it.findtext("link"),
                                       "topic": topic, "watch": mkts, "mechanism": why}
            except Exception as exc:
                errs.append(f"{src}: {exc}")
        self.status = {"last_fetch": time.time(), "errors": errs}

    def refresh_bars(self):
        for sym in MARKETS:
            try:
                r = self.mt5.copy_rates_from_pos(sym, self.mt5.TIMEFRAME_M1, 0, 3000)
                if r is not None and len(r):
                    self.bars[sym] = np.array([(int(x["time"]), float(x["close"])) for x in r])
            except Exception:
                continue

    # ------------------------------------------------------------------ measuring
    def _reaction(self, sym: str, t0: float) -> dict | None:
        b = self.bars.get(sym)
        if b is None or len(b) < 200:
            return None
        ts, px = b[:, 0], b[:, 1]
        i0 = int(np.searchsorted(ts, t0, side="right")) - 1
        if i0 < 60 or ts[-1] < t0 + 15 * 60:
            return None
        def at(mins):
            j = int(np.searchsorted(ts, t0 + mins * 60, side="right")) - 1
            return px[j] if j > i0 else None
        p0 = px[i0]
        out = {}
        for m in (5, 15, 60):
            p = at(m)
            if p is not None:
                out[f"m{m}"] = (p / p0 - 1) * 100
        win = px[max(0, i0 - 1440):i0 + 1]
        if len(win) > 30:
            r15 = np.abs(win[15:] / win[:-15] - 1) * 100
            normal = float(np.median(r15)) if len(r15) else 0.0
        else:
            normal = 0.0
        if "m15" not in out or normal <= 0:
            return None
        out["normal15"] = normal
        out["size"] = abs(out["m15"]) / normal
        return out

    def measure(self):
        changed = False
        for it in self.items.values():
            if it.get("reactions_final"):
                continue
            if time.time() - it["t"] < 16 * 60:
                continue
            reacts = {}
            for sym in MARKETS:
                r = self._reaction(sym, it["t"])
                if r:
                    reacts[sym] = {k: round(v, 4) for k, v in r.items()}
            if not reacts:
                continue
            it["reactions"] = reacts
            it["reactions_final"] = time.time() - it["t"] > 65 * 60 and all("m60" in r for r in reacts.values())
            it["explain"] = self._explain(it)
            changed = True
        if changed:
            self._save()

    # ------------------------------------------------------------------ explaining
    def _explain(self, it: dict) -> str:
        re_ = it.get("reactions") or {}
        watch = set(it.get("watch") or [])
        moved_all = sorted(((s, r) for s, r in re_.items() if r["size"] >= 2.0), key=lambda x: -x[1]["size"])
        moved = [(s, r) for s, r in moved_all if s in watch]
        others = [(s, r) for s, r in moved_all if s not in watch]
        fmt = lambda s, r: f"{MARKETS[s][0]} {r['m15']:+.2f}% in 15 min ({r['size']:.1f}× normal)"
        if not moved:
            quiet = ", ".join(f"{MARKETS[s][0]} {re_[s]['m15']:+.2f}%" for s in it.get("watch", []) if s in re_)
            text = (f"The markets this kind of news usually moves stayed in their normal range ({quiet or 'n/a'}): "
                    f"it was expected, already priced in, or not important enough.")
            if others:
                text += " Unrelated moves at the same time: " + "; ".join(fmt(s, r) for s, r in others[:2]) + "."
            if it.get("kind") == "headline":
                text += " (Headlines often report moves that already happened, so the calendar items are the cleaner test.)"
            return text
        parts = [fmt(s, r) for s, r in moved[:4]]
        # dollar direction from EURUSD (down = dollar up) and USDJPY (up = dollar up)
        usd = 0.0
        if "EURUSDm" in re_:
            usd -= re_["EURUSDm"]["m15"] / max(re_["EURUSDm"]["normal15"], 1e-9)
        if "USDJPYm" in re_:
            usd += re_["USDJPYm"]["m15"] / max(re_["USDJPYm"]["normal15"], 1e-9)
        dollar = "stronger" if usd > 2 else ("weaker" if usd < -2 else None)
        text = "After this: " + "; ".join(parts) + "."
        if others:
            text += " Also moving (probably for other reasons): " + "; ".join(fmt(s, r) for s, r in others[:2]) + "."
        if dollar:
            text += f" The dollar got {dollar}."
        gold = re_.get("XAUUSDm")
        topic = it.get("topic")
        if topic in ("inflation", "fed", "jobs", "currencies") and dollar and gold and gold["size"] >= 1.5:
            fits = (dollar == "stronger") == (gold["m15"] < 0)
            text += (" That fits the usual pattern (dollar and gold moving opposite ways), so this item likely drove it."
                     if fits else " Gold and the dollar moved the same way, which is unusual; something else was probably moving markets too.")
        elif topic == "geopolitics" and gold:
            text += (" Gold rose as a safe haven, consistent with risk-off." if gold["m15"] > 0 and gold["size"] >= 1.5
                     else " Gold did not act as a safe haven this time.")
        elif topic == "crypto" and "BTCUSDm" in re_:
            b = re_["BTCUSDm"]
            other = [s for s, r in moved if s not in ("BTCUSDm", "ETHUSDm")]
            text += (" Only crypto moved, so this was crypto-specific." if b["size"] >= 2 and not other else
                     " Other markets moved too, so the move was probably broader than this headline.")
        near = [o for o in self.items.values() if o is not it and abs(o["t"] - it["t"]) <= 10 * 60]
        if near:
            text += f" Note: {len(near)} other item(s) landed within 10 minutes, so the move can't be pinned on this alone."
        return text

    # ------------------------------------------------------------------ outputs
    def recent(self, n: int = 40) -> list[dict]:
        rows = sorted(self.items.values(), key=lambda x: -x["t"])
        out = []
        for it in rows:
            r = it.get("reactions") or {}
            top = max((v["size"] for v in r.values()), default=0)
            out.append({**it, "max_size": round(top, 2),
                        "chips": [{"symbol": s, "label": MARKETS[s][0], "m15": v["m15"], "size": round(v["size"], 1)}
                                  for s, v in sorted(r.items(), key=lambda kv: -kv[1]["size"])][:5]})
            if len(out) >= n:
                break
        return out

    def upcoming(self, hours: int = 36) -> list[dict]:
        now = time.time()
        out = []
        for e in self.calendar:
            if now <= e["t"] <= now + hours * 3600:
                topic, mkts, why = _topic(e["title"])
                if not topic:
                    topic = COUNTRY_TOPIC.get(e.get("country")) or "growth"
                    mkts = next((m for n, _, m, _ in TOPICS if n == topic), [])
                    why = next((w for n, _, _, w in TOPICS if n == topic), "")
                out.append({**e, "in_s": int(e["t"] - now), "watch": [MARKETS[m][0] for m in mkts if m in MARKETS],
                            "mechanism": why})
        return out[:12]

    def loop(self):
        while True:
            try:
                self.fetch()
                self.refresh_bars()
                self.measure()
            except Exception as exc:
                self.status["errors"] = [str(exc)]
            time.sleep(self.poll_sec)
