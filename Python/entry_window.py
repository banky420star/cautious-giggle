"""Hard "no new trades now" windows for the stand-alone traders (Gemma, Qwen, meta-brain).

Added 2026-11-24 after the LLM traders lost ~$680 in a day: they only *read* about news in their prompt and the
council's news block (news_trader.blocks_regular) never reached them, so nothing stopped them opening into a
release or the first minutes of a market open. This gate does, in code:

    news   no new trade from `high_before_min` before to `high_after_min` after a High-impact release that moves
           the market (USD for everything, plus EUR / JPY for the majors, oil inventories for crude); Medium-impact
           releases block a shorter window (`medium_*`). Reads logs/calendar_cache.json (news_markets.get_calendar).
    opens  no new trade from `open_before_min` before to `open_after_min` after the London open (08:00 London)
           and the New York stock open (09:30 New York). Local times, so daylight saving is handled.

Managing / closing open trades is never blocked. Config: trading.entry_window (all keys optional).
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time
from zoneinfo import ZoneInfo

DEFAULTS = {"enabled": True,
            "high_before_min": 30, "high_after_min": 30,
            "medium_before_min": 10, "medium_after_min": 10,
            "open_before_min": 15, "open_after_min": 30}

OPENS = (("London open", "Europe/London", 8, 0), ("New York stock open", "America/New_York", 9, 30))

CURRENCIES = {"EURUSD": ("USD", "EUR"), "USDJPY": ("USD", "JPY"), "GBPUSD": ("USD", "GBP"), "AUDUSD": ("USD", "AUD")}


def currencies(symbol: str) -> tuple:
    return CURRENCIES.get(symbol.upper()[:6], ("USD",))


def moves(event: dict, symbol: str) -> bool:
    """Does this calendar event move `symbol`?"""
    title = str(event.get("title", "")).lower()
    if symbol.upper().startswith(("USOIL", "UKOIL", "WTI", "BRENT")) and ("oil" in title or "crude" in title):
        return True
    return event.get("country") in currencies(symbol)


def market_opens(now: float, within_h: float = 24.0) -> list[tuple[str, float]]:
    """[(name, unix time)] of weekday market opens from yesterday to `within_h` hours ahead."""
    out = []
    for name, tz, hh, mm in OPENS:
        z = ZoneInfo(tz)
        today = dt.datetime.fromtimestamp(now, z).date()
        for d in range(-1, int(within_h // 24) + 2):
            day = today + dt.timedelta(days=d)
            if day.weekday() >= 5:
                continue
            t = dt.datetime(day.year, day.month, day.day, hh, mm, tzinfo=z).timestamp()
            if now - 86400 <= t <= now + within_h * 3600:
                out.append((name, t))
    return sorted(out, key=lambda x: x[1])


class EntryWindow:
    def __init__(self, cfg: dict | None = None, log_dir: str | None = None, calendar=None):
        self.cfg = {**DEFAULTS, **(cfg or {})}
        self.enabled = bool(self.cfg["enabled"])
        self.log_dir = log_dir
        self._calendar = calendar          # list, or None to read the shared cache file
        self._cache = ([], 0.0)

    def calendar(self) -> list:
        if self._calendar is not None:
            return self._calendar
        if time.time() - self._cache[1] < 60:
            return self._cache[0]
        events = []
        try:
            with open(os.path.join(self.log_dir or "logs", "calendar_cache.json"), encoding="utf-8") as f:
                events = json.load(f).get("events") or []
        except (OSError, ValueError):
            pass
        self._cache = (events, time.time())
        return events

    def reason(self, symbol: str, now: float | None = None) -> str | None:
        """Why no new trade may open on `symbol` now, or None."""
        if not self.enabled:
            return None
        now = time.time() if now is None else now
        c = self.cfg
        for e in self.calendar():
            try:
                t = float(e["t"])
            except (KeyError, TypeError, ValueError):
                continue
            imp = str(e.get("impact"))
            if imp == "High":
                before, after = c["high_before_min"], c["high_after_min"]
            elif imp == "Medium":
                before, after = c["medium_before_min"], c["medium_after_min"]
            else:
                continue
            if t - before * 60 <= now <= t + after * 60 and moves(e, symbol):
                m = int(round((t - now) / 60))
                return f"news: {e.get('title')} ({imp}) " + (f"in {m} min" if m > 0 else f"{-m} min ago")
        for name, t in market_opens(now, 1.0):
            if t - c["open_before_min"] * 60 <= now <= t + c["open_after_min"] * 60:
                m = int(round((t - now) / 60))
                return f"{name} " + (f"in {m} min" if m > 0 else f"{-m} min ago")
        return None
