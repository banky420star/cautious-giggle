"""Read a high-impact release from the headlines and decide which way it pushes the dollar (user 2026-11-25).

The calendar feed only has forecast / previous, never the actual figure, but the headline feed
(logs/news_items.jsonl) reports the result within a few minutes of the release ("inflation gauge accelerated to
3.8%..."). After a release the local model (Gemma, via Ollama) reads those headlines with the forecast and says
whether the result is dollar-positive (hotter inflation, stronger jobs / growth, hawkish Fed), dollar-negative, in
line, or unclear. news_trader turns that into a direction per market ("surprise" play) and records how it did.

Memory: every read and every outcome goes to logs/news_memory.jsonl, so the bot (and Gemma's desk) keep what each
kind of release did to each market.
"""
from __future__ import annotations

import json
import os
import threading
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# How each market moves when the dollar strengthens. Oil has no fixed dollar link, so the surprise play skips it.
DOLLAR_SIGN = {"XAUUSDm": -1, "XAGUSDm": -1, "BTCUSDm": -1, "ETHUSDm": -1, "EURUSDm": -1, "USDJPYm": 1}

SCHEMA = {"type": "object", "properties": {
    "actual": {"type": "string"},
    "surprise": {"type": "string", "enum": ["dollar_up", "dollar_down", "in_line", "unclear"]},
    "confidence": {"type": "number"},
    "reason": {"type": "string"}},
    "required": ["actual", "surprise", "confidence", "reason"]}

PROMPT = """You read US economic releases for a trading desk. Decide whether this release surprised in a way that
strengthens or weakens the US dollar.

Release (just out, {mins} min ago): {releases}

Headlines since the release:
{headlines}

Rules:
- Compare the ACTUAL figure (find it in the headlines) with the FORECAST.
- dollar_up: inflation hotter than forecast, jobs / growth / spending stronger than forecast, or headlines say the Fed
  is more likely to raise rates or less likely to cut.
- dollar_down: the opposite (cooler inflation, weaker jobs / growth, rate cuts more likely).
- in_line: the result matches the forecast and headlines show no clear shift.
- unclear: the headlines don't report this release's result, or they conflict.
- When several releases came out together, judge the overall effect on the dollar.
- confidence 0-1: how sure you are about the direction (use below 0.5 for weak or mixed evidence).
Answer in JSON: actual (the figure(s) you found, or "unknown"), surprise, confidence, reason (one sentence)."""


def _log_dir() -> str:
    return os.environ.get("AGI_LOG_DIR") or os.path.join(ROOT, "logs")


def headlines_since(t_from: float, t_to: float, log_dir: str | None = None, limit: int = 15) -> list[str]:
    path = os.path.join(log_dir or _log_dir(), "news_items.jsonl")
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            f.seek(max(0, f.tell() - 600_000))
            lines = f.read().decode("utf-8", "replace").splitlines()[1:]
    except OSError:
        return []
    out = []
    for ln in lines:
        try:
            e = json.loads(ln)
        except ValueError:
            continue
        if e.get("kind") != "headline" or not (t_from <= float(e.get("t") or 0) <= t_to):
            continue
        out.append(f"[{e.get('source', '')}] {e.get('title', '')}. {(e.get('detail') or '')[:220]}")
    return out[-limit:]


def direction(symbol: str, verdict: dict | None, min_confidence: float) -> int:
    """+1 buy / -1 sell / 0 none for `symbol` from a release verdict."""
    if not verdict or float(verdict.get("confidence") or 0) < min_confidence:
        return 0
    usd = {"dollar_up": 1, "dollar_down": -1}.get(str(verdict.get("surprise")), 0)
    return usd * DOLLAR_SIGN.get(symbol, 0)


def remember(row: dict, log_dir: str | None = None) -> None:
    try:
        with open(os.path.join(log_dir or _log_dir(), "news_memory.jsonl"), "a", encoding="utf-8") as f:
            f.write(json.dumps({"t": time.time(), **row}) + "\n")
    except OSError:
        pass


def recent_reads(since: float, log_dir: str | None = None) -> list[dict]:
    """Release reads made since `since` (newest last), for Gemma's desk."""
    out = []
    try:
        with open(os.path.join(log_dir or _log_dir(), "news_memory.jsonl"), encoding="utf-8") as f:
            for ln in f:
                try:
                    e = json.loads(ln)
                except ValueError:
                    continue
                if e.get("kind") == "read" and float(e.get("t") or 0) >= since:
                    out.append(e)
    except OSError:
        pass
    return out


class ReleaseReader:
    """One model read per release time, done on a background thread so the trading loop never waits on it."""

    def __init__(self, cfg: dict | None, log_dir: str | None = None, ask=None, logger=None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.model = str(cfg.get("model", "gemma4:e4b-it-qat"))
        self.url = str(cfg.get("url", "http://127.0.0.1:11434"))
        self.timeout = int(cfg.get("timeout_s", 120))
        self.read_after = float(cfg.get("read_after_min", 2)) * 60      # headlines need a minute or two
        self.retry_s = float(cfg.get("retry_s", 60))
        self.min_confidence = float(cfg.get("min_confidence", 0.6))
        self.log_dir = log_dir or _log_dir()
        self.log = logger
        self._ask = ask
        self._lock = threading.Lock()
        self.cache: dict[int, dict] = {}        # t0 -> {"state": pending|done|retry, "verdict": ..., "t": ...}

    def verdict(self, t0: float, releases: list[dict], now: float | None = None, wait: bool = False) -> dict | None:
        """The read for the release at `t0` once it exists; starts or retries the read when due."""
        if not self.enabled:
            return None
        now = time.time() if now is None else now
        key = int(t0)
        with self._lock:
            st = self.cache.get(key)
            if st and st["state"] == "done":
                return st["verdict"]
            if now < t0 + self.read_after or (st and (st["state"] == "pending" or now - st["t"] < self.retry_s)):
                return None
            self.cache[key] = {"state": "pending", "t": now, "verdict": None}
        th = threading.Thread(target=self._read, args=(key, t0, list(releases), now), daemon=True, name="release-read")
        th.start()
        if wait:
            th.join()
            return (self.cache.get(key) or {}).get("verdict")
        return None

    def _read(self, key: int, t0: float, releases: list[dict], now: float) -> None:
        heads = headlines_since(t0 - 120, now, self.log_dir)
        if not heads:
            with self._lock:
                self.cache[key] = {"state": "retry", "t": now, "verdict": None}
            return
        rel = "; ".join(f"{r.get('title')} (forecast {r.get('forecast') or '?'}, previous {r.get('previous') or '?'})"
                        for r in releases) or "US high-impact release"
        prompt = PROMPT.format(mins=max(0, int((now - t0) // 60)), releases=rel, headlines="\n".join(f"- {h}" for h in heads))
        try:
            if self._ask:
                js = self._ask(prompt)
            else:
                from Python.external_voters import ollama_json
                js = ollama_json(self.url, self.model, prompt, SCHEMA, num_predict=300, timeout=self.timeout)
            v = {"surprise": str(js.get("surprise", "unclear")), "confidence": max(0.0, min(1.0, float(js.get("confidence") or 0))),
                 "actual": str(js.get("actual") or "unknown")[:120], "reason": str(js.get("reason") or "")[:300],
                 "t": now, "t0": t0, "releases": releases, "n_headlines": len(heads)}
        except Exception as exc:
            if self.log:
                self.log.warning(f"release read failed ({exc}); retrying in {self.retry_s:.0f} s")
            with self._lock:
                self.cache[key] = {"state": "retry", "t": now, "verdict": None}
            return
        # unclear / in-line with no headline about it yet: try again a little later while the window is open
        state = "done" if v["surprise"] in ("dollar_up", "dollar_down", "in_line") or now - t0 > 15 * 60 else "retry"
        with self._lock:
            self.cache[key] = {"state": state, "t": now, "verdict": v if state == "done" else None}
        if state == "done":
            remember({"kind": "read", **{k: v[k] for k in ("t0", "releases", "surprise", "confidence", "actual", "reason",
                                                         "n_headlines")}}, self.log_dir)
        if self.log:
            self.log.info(f"NEWS READ {', '.join(r.get('title', '') for r in releases)} | {v['surprise']} "
                          f"({v['confidence']:.2f}) actual {v['actual']} | {v['reason']}"
                          + ("" if state == "done" else " | will read again"))
