"""Per-model memory of what works and what doesn't.

Every closed bar, each model's call (target exposure in [-1, 1]) is stored with the
market context it was made in. On the next closed bar the call is scored against what
price actually did:

    score = target * log_return(next bar) * 1e4         (bps earned by following the model)

Scores are kept as exponentially-decayed statistics per
    symbol x model x context        (context = trend | volatility | session)
and per symbol x model overall. From them the memory produces a weight multiplier per
model for the *current* context:

    edge   = context mean, shrunk toward the model's overall mean (few samples -> overall)
    t      = edge / standard error
    mult   = clip(1 + gain * tanh(t / 2), floor, cap)    default floor 0.25, cap 1.5

so a model that is consistently wrong in this kind of market gets its say cut, one that is
consistently right gets more, and nothing changes until there is real evidence. Old results
fade (half-life ~1200 bars, about 4 days of M5) so the memory follows regime changes.

State persists in logs/memory/model_memory.json.
"""
from __future__ import annotations

import json
import math
import os
import threading
import time
from dataclasses import dataclass

import numpy as np

MODELS = ("ppo", "lstm", "dreamer", "blend")


def market_context(df) -> dict:
    """Coarse market context from raw OHLC: trend, volatility, session."""
    try:
        close = df["close"].astype(float).to_numpy()
        high = df["high"].astype(float).to_numpy()
        low = df["low"].astype(float).to_numpy()
        n = len(close)
        if n < 60:
            return {"trend": "range", "vol": "normal", "session": "na", "key": "range|normal|na"}

        def ema(x, span):
            a = 2.0 / (span + 1.0)
            out = np.empty_like(x)
            out[0] = x[0]
            for i in range(1, len(x)):
                out[i] = a * x[i] + (1 - a) * out[i - 1]
            return out

        tail = close[-min(n, 400):]
        fast, slow = ema(tail, 20), ema(tail, 100 if len(tail) >= 100 else len(tail) // 2)
        tr = np.maximum(high[1:] - low[1:], np.maximum(abs(high[1:] - close[:-1]), abs(low[1:] - close[:-1])))
        atr = float(np.mean(tr[-14:]))
        atr_hist = np.convolve(tr[-300:], np.ones(14) / 14, mode="valid")
        gap = (fast[-1] - slow[-1]) / max(atr, 1e-12)
        trend = "up" if gap > 0.75 else ("down" if gap < -0.75 else "range")
        med = float(np.median(atr_hist)) if len(atr_hist) else atr
        vol = "high" if atr > 1.3 * med else ("low" if atr < 0.75 * med else "normal")
        try:
            hour = int(df.index[-1].hour)
        except Exception:
            hour = int(time.gmtime().tm_hour)
        session = "asia" if hour < 7 else ("europe" if hour < 13 else ("us" if hour < 21 else "late"))
        return {"trend": trend, "vol": vol, "session": session, "key": f"{trend}|{vol}|{session}",
                "atr": atr, "trend_gap_atr": float(gap)}
    except Exception:
        return {"trend": "range", "vol": "normal", "session": "na", "key": "range|normal|na"}


@dataclass
class _Stat:
    w: float = 0.0      # decayed count
    m: float = 0.0      # decayed mean
    s: float = 0.0      # decayed second moment
    hits: float = 0.0   # decayed count of correct-direction calls
    calls: float = 0.0  # decayed count of non-flat calls
    last: float = 0.0

    def update(self, x: float, active: bool, decay: float) -> None:
        self.w *= decay
        self.hits *= decay
        self.calls *= decay
        new_w = self.w + 1.0
        self.m = (self.m * self.w + x) / new_w
        self.s = (self.s * self.w + x * x) / new_w
        self.w = new_w
        if active:
            self.calls += 1.0
            if x > 0:
                self.hits += 1.0
        self.last = time.time()

    @property
    def var(self) -> float:
        return max(self.s - self.m * self.m, 1e-12)

    def to_dict(self) -> dict:
        return {"w": round(self.w, 4), "m": self.m, "s": self.s, "hits": round(self.hits, 4),
                "calls": round(self.calls, 4), "last": self.last}

    @classmethod
    def from_dict(cls, d: dict) -> "_Stat":
        st = cls()
        for k in ("w", "m", "s", "hits", "calls", "last"):
            setattr(st, k, float(d.get(k, 0.0) or 0.0))
        return st


class ModelMemory:
    def __init__(self, path: str, half_life_bars: float = 1200.0, prior_bars: float = 150.0,
                 gain: float = 0.6, floor: float = 0.25, cap: float = 1.5, min_samples: float = 60.0):
        self.path = path
        self.decay = 0.5 ** (1.0 / max(1.0, float(half_life_bars)))
        self.prior = float(prior_bars)
        self.gain, self.floor, self.cap, self.min_samples = float(gain), float(floor), float(cap), float(min_samples)
        self.stats: dict[str, _Stat] = {}
        self.pending: dict[str, dict] = {}
        self._lock = threading.Lock()
        self._load()

    @classmethod
    def from_cfg(cls, cfg: dict, log_dir: str) -> "ModelMemory":
        mc = ((cfg or {}).get("trading", {}) or {}).get("model_memory", {}) or {}
        path = os.path.join(log_dir, "memory", "model_memory.json")
        return cls(path, half_life_bars=float(mc.get("half_life_bars", 1200)), prior_bars=float(mc.get("prior_bars", 150)),
                   gain=float(mc.get("gain", 0.6)), floor=float(mc.get("floor", 0.25)), cap=float(mc.get("cap", 1.5)),
                   min_samples=float(mc.get("min_samples", 60)))

    # ------------------------------------------------------------------ persistence
    def _load(self) -> None:
        try:
            with open(self.path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            self.stats = {k: _Stat.from_dict(v) for k, v in (raw.get("stats") or {}).items()}
            self.pending = dict(raw.get("pending") or {})
        except Exception:
            self.stats, self.pending = {}, {}

    def save(self) -> None:
        with self._lock:
            data = {"updated": time.time(), "stats": {k: v.to_dict() for k, v in self.stats.items()},
                    "pending": self.pending}
        try:
            os.makedirs(os.path.dirname(self.path), exist_ok=True)
            tmp = self.path + ".tmp"
            with open(tmp, "w", encoding="utf-8") as f:
                json.dump(data, f)
            os.replace(tmp, self.path)
        except Exception:
            pass

    # ------------------------------------------------------------------ learning
    @staticmethod
    def _key(symbol: str, model: str, ctx: str) -> str:
        return f"{symbol}|{model}|{ctx}"

    def observe(self, symbol: str, bar_time, close: float, context: dict, targets: dict) -> list[dict]:
        """Score the previous bar's calls for `symbol` against the move to this bar, then store
        this bar's calls. Returns the scored rows (for logging/UI)."""
        bar_ts = int(bar_time.timestamp()) if hasattr(bar_time, "timestamp") else int(bar_time or 0)
        scored = []
        with self._lock:
            prev = self.pending.get(symbol)
            if prev and bar_ts > int(prev.get("bar_ts", 0)) and float(prev.get("close", 0)) > 0 and close > 0:
                ret_bps = math.log(close / float(prev["close"])) * 1e4
                # Ignore gaps across long closures (weekend) — not something a model can be judged on.
                if bar_ts - int(prev["bar_ts"]) <= 4 * 3600:
                    for model, tgt in (prev.get("targets") or {}).items():
                        if tgt is None:
                            continue
                        x = float(tgt) * ret_bps
                        active = abs(float(tgt)) > 0.05
                        for ctx in (prev.get("ctx", "na"), "*"):
                            self.stats.setdefault(self._key(symbol, model, ctx), _Stat()).update(x, active, self.decay)
                        scored.append({"model": model, "target": float(tgt), "ret_bps": ret_bps, "score": x,
                                       "ctx": prev.get("ctx")})
            if not prev or bar_ts > int(prev.get("bar_ts", 0)):
                self.pending[symbol] = {"bar_ts": bar_ts, "close": float(close), "ctx": context.get("key", "na"),
                                        "targets": {k: (None if v is None else float(v)) for k, v in targets.items()}}
        if scored:
            self.save()
        return scored

    # ------------------------------------------------------------------ using it
    def assess(self, symbol: str, model: str, ctx: str) -> dict:
        with self._lock:
            c = self.stats.get(self._key(symbol, model, ctx), _Stat())
            o = self.stats.get(self._key(symbol, model, "*"), _Stat())
        n = c.w
        overall = o.m if o.w >= self.min_samples else 0.0
        edge = (c.m * n + overall * self.prior) / (n + self.prior)
        var = c.var if n > 5 else (o.var if o.w > 5 else 1.0)
        se = math.sqrt(var / max(n + self.prior * 0.5, 1.0))
        t = edge / se if se > 0 else 0.0
        if n + o.w < self.min_samples:
            mult = 1.0
        else:
            mult = 1.0 + self.gain * math.tanh(t / 2.0)
            mult = min(self.cap, max(self.floor, mult))
        hit = (c.hits / c.calls) if c.calls > 1 else None
        return {"mult": round(mult, 3), "edge_bps": round(edge, 3), "t": round(t, 2), "n": round(n, 1),
                "n_all": round(o.w, 1), "hit_rate": None if hit is None else round(hit, 3),
                "verdict": ("works" if mult > 1.1 else ("doesn't work" if mult < 0.9 else "neutral"))}

    def multipliers(self, symbol: str, ctx: str) -> dict:
        return {m: self.assess(symbol, m, ctx) for m in MODELS}

    def table(self, symbol: str) -> list[dict]:
        """All contexts seen for a symbol, for the dashboard."""
        with self._lock:
            ctxs = sorted({k.split("|", 2)[2] for k in self.stats if k.startswith(symbol + "|")})
        out = []
        for ctx in ctxs:
            row = {"ctx": ctx}
            for m in MODELS:
                row[m] = self.assess(symbol, m, ctx)
            out.append(row)
        return out


def load_snapshot(log_dir: str) -> dict:
    """Read-only view for the dashboard."""
    try:
        with open(os.path.join(log_dir, "memory", "model_memory.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}
