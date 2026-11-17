"""External voters: the user's own local models, added to the indicator council as extra votes.

    ob_gold   gold order-block strategy (Gold_M1_Dataset, OB-FWD-001 variant V2 "m15_only"; the
              detector and executor below are a line-for-line port of iter83_ob_forward.py)
    kronos    Kronos-small candle forecaster (NeoQuasar/Kronos-small, code pinned at 67b630e)
    catboost  FreqAI CatBoost return models for BTC and ETH (Trading_stack.bank, bank-freqai-spot-v1)
    news_llm  local Ollama model scoring the latest headlines per market

Every vote is a number in [-1, 1] (+ = up). Votes are computed in one background thread so a slow
model never delays trading; the loop always reads the latest finished value. Each voter's record is
kept by the model memory like any indicator (ind:<name>), so a voter that keeps being wrong is
ignored automatically, and the setup scorecard judges the trades it took part in.
"""
from __future__ import annotations

import glob
import json
import os
import queue
import threading
import time
import urllib.request

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DEFAULTS = {
    "enabled": True,
    "weights": {"ob_gold": 8.0, "kronos": 8.0, "catboost": 4.0, "news_llm": 4.0},
    "ob_gold": {"symbols": ["XAUUSDm"]},
    "kronos": {"code_dir": "~/.cautious-giggle/kronos", "device": "cpu", "horizon": 6, "context": 400},
    "catboost": {"model_dir": "/Volumes/AI_DRIVE/Trading_stack.bank/user_data/models/bank-freqai-spot-v1",
                 "pairs": {"BTCUSDm": "BTC/USDT", "ETHUSDm": "ETH/USDT"}, "retrain_hours": 6},
    "news_llm": {"model": "qwen3:4b", "url": "http://127.0.0.1:11434", "every_min": 20, "lookback_h": 6},
}
NAMES = {
    "ob_gold": "Gold order-block strategy (your research)",
    "kronos": "Kronos candle forecaster",
    "catboost": "LightGBM return model (your FreqAI recipe, retrained on Exness bars)",
    "news_llm": "Local AI news reader (Ollama)",
}


def _merge(a: dict, b: dict) -> dict:
    out = dict(a)
    for k, v in (b or {}).items():
        out[k] = _merge(out[k], v) if isinstance(v, dict) and isinstance(out.get(k), dict) else v
    return out


# =============================================================================== gold order blocks
# Port of Gold_M1_Dataset/iter83_ob_forward.py (frozen OB-FWD-001, V2: m15 only, no width cap).

def _atr_wilder(hh, ll, cc, ln=14):
    tr = np.maximum(hh[1:] - ll[1:], np.maximum(np.abs(hh[1:] - cc[:-1]), np.abs(ll[1:] - cc[:-1])))
    trf = np.concatenate([[hh[0] - ll[0]], tr])
    a = np.full(len(hh), np.nan)
    a[ln - 1] = trf[:ln].mean()
    for i in range(ln, len(hh)):
        a[i] = (a[i - 1] * (ln - 1) + trf[i]) / ln
    return a


def _find_pivots(hh, ll, k):
    ph, pl = [], []
    for p in range(k, len(hh) - k):
        if hh[p] > hh[p - k:p].max() and hh[p] > hh[p + 1:p + k + 1].max():
            ph.append((p, float(hh[p])))
        if ll[p] < ll[p - k:p].min() and ll[p] < ll[p + 1:p + k + 1].min():
            pl.append((p, float(ll[p])))
    return ph, pl


def _pv(lst):
    if not lst:
        return np.array([], dtype=float), np.array([], dtype=float)
    return np.array([p for p, _ in lst], dtype=float), np.array([v for _, v in lst], dtype=float)


def _ref_pivot(bars, vals, t, jlim, k):
    idx = int(np.searchsorted(bars, jlim)) - 1
    while idx >= 0 and bars[idx] + k > t:
        idx -= 1
    return float(vals[idx]) if idx >= 0 else np.nan


def _target(bars, vals, t, entry, k, above):
    best = np.nan
    for i in range(len(bars) - 1, -1, -1):
        if bars[i] >= t or bars[i] + k > t:
            continue
        ok = vals[i] > entry if above else vals[i] < entry
        if ok and (not (best == best) or (vals[i] < best if above else vals[i] > best)):
            best = float(vals[i])
    return best


def ob_events(oo, cc, hh, ll, disp_mult=1.5, k_int=3, k_swing=8, sweep_win=6):
    atr = _atr_wilder(hh, ll, cc)
    iph_b, iph_v = _pv(_find_pivots(hh, ll, k_int)[0])
    ipl_b, ipl_v = _pv(_find_pivots(hh, ll, k_int)[1])
    sph, spl = _find_pivots(hh, ll, k_swing)
    sph_b, sph_v = _pv(sph)
    spl_b, spl_v = _pv(spl)
    events = []
    for j in range(14, len(oo) - 1):
        if cc[j] < oo[j]:  # bullish OB
            pl_ref = _ref_pivot(ipl_b, ipl_v, j, j - sweep_win, k_int)
            if pl_ref == pl_ref and ll[j - sweep_win:j].min() < pl_ref and cc[j] > pl_ref:
                b = next((bb for bb in range(j + 1, min(j + 7, len(oo)))
                          if (lambda pr: pr == pr and cc[bb] > pr)(_ref_pivot(iph_b, iph_v, bb, j, k_int))), -1)
                if b > 0 and any((cc[i] - oo[i]) >= disp_mult * atr[i] or ll[i] > hh[i - 2]
                                 for i in range(j + 1, min(j + 3, b) + 1)):
                    entry, stop = float(hh[j]), float(ll[j])
                    tgt = _target(sph_b, sph_v, b, entry, k_swing, True)
                    if tgt != tgt:
                        tgt = entry + (entry - stop) * 2.0
                    events.append((b, 1, stop, entry, tgt))
        if cc[j] > oo[j]:  # bearish OB
            ph_ref = _ref_pivot(iph_b, iph_v, j, j - sweep_win, k_int)
            if ph_ref == ph_ref and hh[j - sweep_win:j].max() > ph_ref and cc[j] < ph_ref:
                b = next((bb for bb in range(j + 1, min(j + 7, len(oo)))
                          if (lambda pr: pr == pr and cc[bb] < pr)(_ref_pivot(ipl_b, ipl_v, bb, j, k_int))), -1)
                if b > 0 and any((oo[i] - cc[i]) >= disp_mult * atr[i] or hh[i] < ll[i - 2]
                                 for i in range(j + 1, min(j + 3, b) + 1)):
                    entry, stop = float(ll[j]), float(hh[j])
                    tgt = _target(spl_b, spl_v, b, entry, k_swing, False)
                    if tgt != tgt:
                        tgt = entry - (stop - entry) * 2.0
                    events.append((b, -1, stop, entry, tgt))
    return events


def ob_state(oo, cc, hh, ll, cap=96):
    """Run the frozen executor over the bars (events arm at the close of their BOS bar, so they can fill
    from the next bar on); return the positions still open and the zones still armed at the last bar."""
    by_arm = {}
    for b, d, sp, lim, tg in ob_events(oo, cc, hh, ll):
        by_arm.setdefault(b + 1, []).append((d, sp, lim, tg))
    armed, poss, trades = [], [], []
    for t in range(len(oo)):
        keep = []
        for (d, sp, lim, tg, ei, ep) in poss:
            xp = None
            if d == 1:
                xp = min(sp, oo[t]) if ll[t] <= sp else (max(tg, oo[t]) if hh[t] >= tg else None)
            else:
                xp = max(sp, oo[t]) if hh[t] >= sp else (min(tg, oo[t]) if ll[t] <= tg else None)
            if xp is None and t - ei >= cap:
                xp = cc[t]
            if xp is None:
                keep.append((d, sp, lim, tg, ei, ep))
            else:
                trades.append(d * (xp - ep))
        poss = keep
        still = []
        for (d, sp, lim, tg) in armed:
            if (ll[t] <= lim) if d == 1 else (hh[t] >= lim):
                ep = min(oo[t], lim) if d == 1 else max(oo[t], lim)
                hit = (ll[t] <= sp or hh[t] >= tg) if d == 1 else (hh[t] >= sp or ll[t] <= tg)
                if hit:
                    trades.append(0.0)
                else:
                    poss.append((d, sp, lim, tg, t, ep))
            elif not ((d == 1 and cc[t] < sp) or (d == -1 and cc[t] > sp)):
                still.append((d, sp, lim, tg))
        armed = still + list(by_arm.get(t, []))
    return {"open": poss, "armed": armed, "closed": len(trades)}


def ob_vote(m15: pd.DataFrame) -> float | None:
    if m15 is None or len(m15) < 200:
        return None
    o, h, l_, c = (m15[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    st = ob_state(o, c, h, l_)
    if st["open"]:
        return float(np.clip(sum(p[0] for p in st["open"]), -1, 1))
    px = c[-1]
    atr = _atr_wilder(h, l_, c)[-1]
    near = [z for z in st["armed"] if abs(z[2] - px) <= 1.0 * atr]
    if near:
        return 0.5 * float(np.sign(sum(z[0] for z in near)))
    return 0.0


def ob_vote_detail(m15: pd.DataFrame):
    """ob_vote plus the order-block levels behind it (for the war room)."""
    v = ob_vote(m15)
    if v is None:
        return None
    o, h, l_, c = (m15[k].to_numpy(float) for k in ("open", "high", "low", "close"))
    st = ob_state(o, c, h, l_)
    px = float(c[-1])
    side = lambda d: "BUY" if d > 0 else "SELL"  # noqa: E731
    if st["open"]:
        d, sp, lim, tg, _, ep = st["open"][-1]
        why = (f"In a {side(d)} from the order block at {ep:.2f}: stop {sp:.2f}, target {tg:.2f}. "
               f"The strategy is holding it, so it votes {side(d)}.")
        lv = {"side": side(d), "entry": ep, "stop": sp, "target": tg, "state": "in trade"}
    elif v:
        atr = _atr_wilder(h, l_, c)[-1]
        near = [z for z in st["armed"] if abs(z[2] - px) <= 1.0 * atr]
        d, sp, lim, tg = near[-1]
        why = (f"Price {px:.2f} is within one ATR of an armed {side(d)} order block at {lim:.2f} "
               f"(stop {sp:.2f}, target {tg:.2f}). Half vote until it fills.")
        lv = {"side": side(d), "entry": lim, "stop": sp, "target": tg, "state": "armed, waiting"}
    else:
        why = (f"No order block in play: {len(st['armed'])} armed zone(s), none within one ATR of price {px:.2f}. "
               f"Votes 0 (no opinion).")
        lv = None
    return float(v), {"reason": why, "levels": lv}


# =============================================================================== Kronos
class _Kronos:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.pred = None
        self.error = None

    def _load(self):
        if self.pred is not None or self.error:
            return self.pred
        try:
            import sys
            code = os.path.expanduser(self.cfg["code_dir"])
            if code not in sys.path:
                sys.path.insert(0, code)
            from model import Kronos, KronosPredictor, KronosTokenizer
            tok = KronosTokenizer.from_pretrained("NeoQuasar/Kronos-Tokenizer-base")
            mdl = Kronos.from_pretrained("NeoQuasar/Kronos-small")
            tok.eval()
            mdl.eval()
            self.pred = KronosPredictor(mdl, tok, max_context=512, device=self.cfg.get("device", "cpu"))
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        return self.pred

    def vote(self, m5: pd.DataFrame) -> float | None:
        r = self.forecast(m5)
        return None if r is None else r[0]

    def forecast(self, m5: pd.DataFrame):
        """(vote, forecast): the next `horizon` 5-minute candles Kronos predicts, and what they imply for
        an entry now: how far price is expected to go against a buy or a sell first, the best price to
        enter at, and the expected best exit. One sampled path, so treat levels as a guess, not a promise."""
        p = self._load()
        if p is None or m5 is None or len(m5) < 120:
            return None
        import torch
        n = int(self.cfg.get("context", 400))
        h = int(self.cfg.get("horizon", 6))
        d = m5.tail(n)
        ts = pd.to_datetime(d["time"], utc=True) if "time" in d.columns else pd.to_datetime(d.index, utc=True)
        ts = pd.Series(ts.tz_localize(None) if getattr(ts, "tz", None) is not None else ts).reset_index(drop=True)
        vol = d["tick_volume"] if "tick_volume" in d.columns else d.get("volume", pd.Series(0.0, index=d.index))
        x = pd.DataFrame({"open": d["open"].values, "high": d["high"].values, "low": d["low"].values,
                          "close": d["close"].values, "volume": np.asarray(vol, float)})
        fut = pd.Series(pd.date_range(ts.iloc[-1] + pd.Timedelta(minutes=5), periods=h, freq="5min"))
        with torch.random.fork_rng(devices=[]), torch.inference_mode():
            torch.manual_seed(17)
            out = p.predict(df=x, x_timestamp=ts, y_timestamp=fut, pred_len=h, T=1.0, top_p=0.9,
                            sample_count=1, verbose=False)
        last = float(x["close"].iloc[-1])
        ret = float(out["close"].tail(2).mean()) / last - 1.0
        sd = float(np.log(x["close"]).diff().tail(200).std() or 1e-4) * np.sqrt(h)
        vote = float(np.tanh(ret / max(sd, 1e-6)))
        hi, lo = float(out["high"].max()), float(out["low"].min())
        bp = lambda a: round((a / last - 1.0) * 1e4, 2)  # noqa: E731
        i_lo, i_hi = int(np.argmin(out["low"].to_numpy())), int(np.argmax(out["high"].to_numpy()))
        fc = {
            "t_last": str(ts.iloc[-1]), "last": last, "horizon_min": h * 5,
            "candles": [{"t": str(fut.iloc[i]), "o": float(out["open"].iloc[i]), "h": float(out["high"].iloc[i]),
                         "l": float(out["low"].iloc[i]), "c": float(out["close"].iloc[i])} for i in range(len(out))],
            "recent": [{"t": str(ts.iloc[i]), "o": float(x["open"].iloc[i]), "h": float(x["high"].iloc[i]),
                        "l": float(x["low"].iloc[i]), "c": float(x["close"].iloc[i])} for i in range(len(x) - 24, len(x))],
            "move_bps": round(ret * 1e4, 2), "typical_bps": round(sd * 1e4, 2),
            # a BUY now: the dip it may sit through first, the best price to buy at, the high it may reach
            "buy": {"drawdown_bps": bp(lo), "best_entry": lo, "best_entry_in_min": (i_lo + 1) * 5,
                    "target": hi, "upside_bps": bp(hi)},
            "sell": {"drawdown_bps": -bp(hi), "best_entry": hi, "best_entry_in_min": (i_hi + 1) * 5,
                     "target": lo, "upside_bps": -bp(lo)},
        }
        nxt = fc["candles"][0]
        fc["reason"] = (f"Forecasts the next {h * 5} min: close {fc['candles'][-1]['c']:.5g} ({fc['move_bps']:+.1f} bps; a normal "
                        f"{h * 5}-min move is ±{fc['typical_bps']:.1f}). Next candle {nxt['o']:.5g} → {nxt['c']:.5g} "
                        f"(range {nxt['l']:.5g}–{nxt['h']:.5g}).")
        return vote, fc


# =============================================================================== CatBoost (FreqAI)
class _CatBoost:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.last_detail: dict = {}
        self.cache = {}
        self.error = {}

    def _load(self, pair: str):
        if pair in self.cache or pair in self.error:
            return self.cache.get(pair)
        try:
            import joblib
            import pickle
            base = self.cfg["model_dir"]
            pd_ = json.load(open(os.path.join(base, "pair_dictionary.json")))[pair]
            name = pd_["model_filename"]
            sub = os.path.join(base, os.path.basename(pd_["data_path"]))
            meta = json.load(open(os.path.join(sub, f"{name}_metadata.json")))
            with open(os.path.join(sub, f"{name}_feature_pipeline.pkl"), "rb") as f:
                fp = pickle.load(f)
            with open(os.path.join(sub, f"{name}_label_pipeline.pkl"), "rb") as f:
                lp = pickle.load(f)
            self.cache[pair] = {"model": joblib.load(os.path.join(sub, f"{name}_model.joblib")), "fp": fp, "lp": lp,
                                "features": meta["training_features_list"],
                                "std": float((meta.get("labels_std") or {}).get("&-future_return", 0.0045) or 0.0045)}
        except Exception as exc:
            self.error[pair] = f"{type(exc).__name__}: {exc}"
        return self.cache.get(pair)

    @staticmethod
    def features(m15: pd.DataFrame, pair: str) -> pd.DataFrame:
        """BankFreqAI.py feature engineering, with FreqAI's column naming (15m, periods 14/50, 1 shift)."""
        d = m15.copy()
        close = d["close"]
        vol = d["tick_volume"] if "tick_volume" in d.columns else d["volume"]
        f = pd.DataFrame(index=d.index)
        sfx = f"{pair}_15m"
        for n in (14, 50):
            delta = close.diff()
            up = delta.clip(lower=0).ewm(alpha=1 / n, adjust=False).mean()
            dn = (-delta.clip(upper=0)).ewm(alpha=1 / n, adjust=False).mean()
            f[f"%-rsi-period_{n}_{sfx}"] = 100 - 100 / (1 + up / dn.replace(0, np.nan))
            f[f"%-ema-period_{n}_{sfx}"] = close.ewm(span=n, adjust=False).mean()
            f[f"%-relative_volume-period_{n}_{sfx}"] = vol / vol.rolling(n).mean()
        f[f"%-change_gen_{sfx}"] = close.pct_change()
        f[f"%-volume_gen_{sfx}"] = vol.astype(float)
        base_cols = list(f.columns)
        for col in base_cols:
            f[col.replace(f"_{sfx}", f"_shift-1_{sfx}")] = f[col].shift(1)
        t = pd.to_datetime(d["time"], utc=True) if "time" in d.columns else pd.to_datetime(d.index, utc=True)
        f["%-weekday"] = np.asarray(t.dt.dayofweek if hasattr(t, "dt") else t.dayofweek)
        f["%-hour"] = np.asarray(t.dt.hour if hasattr(t, "dt") else t.hour)
        return f

    def original_accepts(self, symbol: str, m15: pd.DataFrame) -> bool | None:
        """Would the user's original FreqAI model accept today's bars? Its own dissimilarity check
        (DI_threshold 0.9) answers; False means the bars are outside what it was trained on."""
        pair = (self.cfg.get("pairs") or {}).get(symbol)
        b = self._load(pair) if pair else None
        if b is None or m15 is None or len(m15) < 80:
            return None
        X = self.features(m15, pair)[b["features"]].tail(1)
        if X.isna().any(axis=None):
            return None
        out = b["fp"].transform(X.to_numpy())
        return bool(len(out[0] if isinstance(out, tuple) else out))

    def _retrain(self, symbol: str, m15: pd.DataFrame):
        """The same recipe (features, LightGBM n_estimators 160 / lr 0.04 / num_leaves 20, label = return
        over the next 8 bars) trained on this broker's own recent bars, refreshed every retrain_hours."""
        from lightgbm import LGBMRegressor
        pair = (self.cfg.get("pairs") or {}).get(symbol, symbol)
        lab_n = 8
        f = self.features(m15, pair)
        y = m15["close"].shift(-lab_n) / m15["close"] - 1.0
        # price-level features don't carry over between price regimes: use them relative to the close
        for col in [c for c in f.columns if "%-ema-period" in c]:
            f[col] = f[col] / m15["close"] - 1.0
        for col in [c for c in f.columns if "%-volume_gen" in c]:
            f[col] = f[col] / m15[("tick_volume" if "tick_volume" in m15.columns else "volume")].rolling(200, 20).mean()
        ok = f.notna().all(axis=1) & y.notna()
        Xtr, ytr = f[ok], y[ok]
        if len(Xtr) < 500:
            return None
        # n_jobs=1: LightGBM's OpenMP threads crash (segfault) next to PyTorch on macOS
        m = LGBMRegressor(n_estimators=160, learning_rate=0.04, num_leaves=20, verbose=-1, n_jobs=1)
        m.fit(Xtr.to_numpy(), ytr.to_numpy())
        return {"model": m, "t": time.time(), "std": float(ytr.std() or 0.0045), "cols": list(f.columns),
                "n": int(len(Xtr)), "rel": True}

    def vote(self, symbol: str, m15: pd.DataFrame) -> float | None:
        if symbol not in (self.cfg.get("pairs") or {}) or m15 is None or len(m15) < 600:
            return None
        key = ("retrained", symbol)
        b = self.cache.get(key)
        if b is None or time.time() - b["t"] > float(self.cfg.get("retrain_hours", 6)) * 3600:
            b = self._retrain(symbol, m15)
            if b is None:
                return None
            self.cache[key] = b
            try:
                b["original_ok"] = self.original_accepts(symbol, m15)
            except Exception:
                b["original_ok"] = None
        pair = (self.cfg.get("pairs") or {}).get(symbol, symbol)
        f = self.features(m15, pair)
        for col in [c for c in f.columns if "%-ema-period" in c]:
            f[col] = f[col] / m15["close"] - 1.0
        for col in [c for c in f.columns if "%-volume_gen" in c]:
            f[col] = f[col] / m15[("tick_volume" if "tick_volume" in m15.columns else "volume")].rolling(200, 20).mean()
        x = f[b["cols"]].tail(1)
        if x.isna().any(axis=None):
            return None
        r = float(b["model"].predict(x.to_numpy())[0])
        z = r / max(b["std"], 1e-6)
        self.last_detail[symbol] = {"reason": (
            f"Predicts {'a rise' if r > 0 else 'a fall' if r < 0 else 'no move'} over the next few 15-min bars, "
            f"{abs(z):.1f}× the size of its usual prediction (retrained every {self.cfg.get('retrain_hours', 6)} h on "
            f"{b.get('n', '?')} Exness bars)."), "z": round(z, 2)}
        return float(np.tanh(z))


# =============================================================================== news via local LLM
class _NewsLLM:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.error = None

    def _headlines(self, symbol: str) -> list[str]:
        path = os.path.join(ROOT, "logs", "news_items.jsonl")
        cutoff = time.time() - float(self.cfg.get("lookback_h", 6)) * 3600
        base = symbol.rstrip("m")
        words = {"BTCUSD": ["bitcoin", "btc", "crypto"], "ETHUSD": ["ether", "eth", "crypto"],
                 "XAUUSD": ["gold"], "XAGUSD": ["silver"], "USOIL": ["oil", "crude", "opec"],
                 "EURUSD": ["euro", "ecb", "eur"], "USDJPY": ["yen", "boj", "japan"]}.get(base, [])
        out = []
        try:
            with open(path, "rb") as f:
                f.seek(0, 2)
                f.seek(max(0, f.tell() - 400_000))
                lines = f.read().decode("utf-8", "replace").splitlines()[1:]
        except OSError:
            return []
        for ln in lines:
            try:
                e = json.loads(ln)
            except ValueError:
                continue
            if float(e.get("t") or 0) < cutoff:
                continue
            text = f"{e.get('title', '')}. {e.get('detail', '') or ''}"
            if symbol in (e.get("watch") or []) or any(w in text.lower() for w in words):
                out.append(text[:300])
        return out[-12:]

    def vote(self, symbol: str) -> float | None:
        heads = self._headlines(symbol)
        if not heads:
            return 0.0, {"reason": "No recent headlines about this market, so no opinion (0).", "headlines": [], "n_headlines": 0}
        asset = {"BTCUSDm": "Bitcoin", "ETHUSDm": "Ether", "XAUUSDm": "gold", "XAGUSDm": "silver",
                 "USOILm": "US crude oil", "EURUSDm": "EUR/USD (the euro against the dollar)",
                 "USDJPYm": "USD/JPY (the dollar against the yen)"}.get(symbol, symbol)
        prompt = (f"You are a markets analyst. Read these recent headlines and judge the likely direction of "
                  f"{asset} over the next few hours. Reply with JSON only: {{\"score\": number from -1 (strongly down) "
                  f"to 1 (strongly up), 0 if unclear or unrelated, \"reason\": one short sentence naming the headline "
                  f"that matters most and why}}.\n\n" + "\n".join(f"- {h}" for h in heads))
        # a schema, not "json": with plain json mode the model echoes the headlines and runs out of tokens
        schema = {"type": "object", "properties": {"score": {"type": "number", "minimum": -1, "maximum": 1},
                                                   "reason": {"type": "string", "maxLength": 220}},
                  "required": ["score", "reason"]}
        body = json.dumps({"model": self.cfg["model"], "prompt": prompt, "stream": False, "format": schema,
                           "think": False, "options": {"temperature": 0, "num_predict": 140}}).encode()
        req = urllib.request.Request(self.cfg["url"].rstrip("/") + "/api/generate", data=body,
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=120) as r:
            txt = json.loads(r.read().decode()).get("response", "")
        reason = None
        try:
            js = json.loads(txt)
            score = float(js.get("score", 0.0))
            reason = str(js.get("reason") or "")[:240] or None
        except (ValueError, AttributeError, TypeError):
            import re
            m = re.search(r'"?score"?\s*[:=]\s*(-?\d+(?:\.\d+)?)', txt)
            if not m:
                raise ValueError("no score in the model's reply")
            score = float(m.group(1))
        return float(np.clip(score, -1, 1)), {"reason": reason or "No reason given.", "headlines": heads[-3:],
                                              "n_headlines": len(heads)}


# =============================================================================== runner
class ExternalVoters:
    def __init__(self, cfg: dict | None = None, logger=None):
        self.cfg = _merge(DEFAULTS, cfg or {})
        self.enabled = bool(self.cfg.get("enabled", True))
        self.log = logger
        self.kronos = _Kronos(self.cfg["kronos"])
        self.cat = _CatBoost(self.cfg["catboost"])
        self.news = _NewsLLM(self.cfg["news_llm"])
        self.values: dict = {}      # (symbol, voter) -> (vote, computed_at)
        self.details: dict = {}     # (symbol, voter) -> why it voted that way (reason, levels, forecast)
        self.errors: dict = {}
        self._stamp: dict = {}      # (symbol, voter) -> input stamp last queued
        # two workers: the price models are fast; the news model can take a minute and must not hold them up
        self._q: queue.Queue = queue.Queue()
        self._q_slow: queue.Queue = queue.Queue()
        if self.enabled:
            for q_, nm_ in ((self._q, "external-voters"), (self._q_slow, "external-voters-news")):
                threading.Thread(target=self._run, args=(q_,), name=nm_, daemon=True).start()

    def weights(self) -> dict:
        return dict(self.cfg.get("weights") or {})

    def _run(self, q=None):
        q = q or self._q
        while True:
            name, symbol, fn = q.get()
            try:
                v = fn()
                det = None
                if isinstance(v, tuple):
                    v, det = v
                if v is not None and np.isfinite(v):
                    self.values[(symbol, name)] = (float(v), time.time())
                    if det is not None:
                        self.details[(symbol, name)] = det
                self.errors.pop((symbol, name), None)
            except Exception as exc:
                msg = f"{type(exc).__name__}: {exc}"[:200]
                if self.errors.get((symbol, name)) != msg and self.log:
                    self.log.warning(f"voter {name} failed on {symbol}: {msg}")
                self.errors[(symbol, name)] = msg

    def _queue(self, name, symbol, stamp, fn):
        if self._stamp.get((symbol, name)) != stamp:
            self._stamp[(symbol, name)] = stamp
            (self._q_slow if name == "news_llm" else self._q).put((name, symbol, fn))

    def votes(self, symbol: str, m5: pd.DataFrame, m15: pd.DataFrame | None) -> dict:
        """Latest finished votes for this market (fresh enough to use), and schedule the next ones."""
        if not self.enabled:
            return {}
        stamp5 = str(m5["time"].iloc[-1]) if m5 is not None and "time" in m5.columns and len(m5) else str(time.time() // 300)
        stamp15 = str(m15["time"].iloc[-1]) if m15 is not None and "time" in m15.columns and len(m15) else str(time.time() // 900)
        if symbol in (self.cfg["ob_gold"].get("symbols") or []) and m15 is not None:
            self._queue("ob_gold", symbol, stamp15, lambda d=m15.copy(): ob_vote_detail(d))
        if self.cfg["weights"].get("kronos") and m5 is not None:
            self._queue("kronos", symbol, stamp5, lambda d=m5.copy(): self.kronos.forecast(d))
        if symbol in (self.cfg["catboost"].get("pairs") or {}) and m15 is not None:
            self._queue("catboost", symbol, stamp15,
                        lambda d=m15.copy(): (lambda v: None if v is None else (v, self.cat.last_detail.get(symbol)))(self.cat.vote(symbol, d)))
        if self.cfg["weights"].get("news_llm"):
            every = float(self.cfg["news_llm"].get("every_min", 20)) * 60
            self._queue("news_llm", symbol, str(time.time() // every), lambda: self.news.vote(symbol))
        out, now = {}, time.time()
        max_age = {"ob_gold": 1800, "kronos": 900, "catboost": 1800, "news_llm": 3600}
        for (s, name), (v, t) in list(self.values.items()):
            if s == symbol and now - t <= max_age.get(name, 900):
                out[name] = v
        return out

    def dump(self, path: str) -> None:
        """Live votes and health for the war room (the dashboard runs in another process)."""
        st = self.status()
        st.update({"t": time.time(), "weights": self.weights(), "names": NAMES,
                   "markets": {"ob_gold": self.cfg["ob_gold"].get("symbols"), "catboost": list(self.cfg["catboost"].get("pairs") or {}),
                               "kronos": "all", "news_llm": "all"},
                   "news_model": self.cfg["news_llm"].get("model"),
                   "details": {f"{s}:{n}": d for (s, n), d in list(self.details.items())}})
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump(st, f)
        os.replace(tmp, path)

    def status(self) -> dict:
        return {"kronos": self.kronos.error or ("ready" if self.kronos.pred is not None else "loading"),
                "catboost": {f"{k[1]}": {"trained_rows": v.get("n"), "original_model_accepts_today": v.get("original_ok")}
                             for k, v in self.cat.cache.items() if isinstance(k, tuple)},
                "errors": {f"{s}:{n}": e for (s, n), e in self.errors.items()},
                "values": {f"{s}:{n}": round(v, 3) for (s, n), (v, _) in self.values.items()}}
