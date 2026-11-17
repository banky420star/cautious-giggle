"""Live 'what changed and why' feed.

Watches training (PPO stats, inline evals, LSTM epochs, Dreamer loss), the indicator memory,
signals and closed trades, and turns every change into a plain-English line that says what
moved, by how much, and the most likely reason. Items are kept in memory and appended to
logs/narration.jsonl so the history survives restarts.
"""
from __future__ import annotations

import json
import os
import re
import time
from collections import deque

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FEED_FILE = os.path.join(ROOT, "logs", "narration.jsonl")
CYCLE_LOG = os.path.join(ROOT, "logs", "champion_cycle_run.log")
NAMES = {"BTCUSDm": "Bitcoin", "XAUUSDm": "Gold"}


def _nm(s):
    return NAMES.get(str(s), str(s))


def _f(v, nd=2, sign=True):
    return (f"{v:+.{nd}f}" if sign else f"{v:.{nd}f}")


class Narrator:
    def __init__(self, maxlen: int = 300):
        self.items: deque = deque(maxlen=maxlen)
        self._load()
        self._ppo_last = None          # last PPO stats block
        self._ppo_first = None
        self._ppo_sym = None
        self._ppo_early = 0
        self._eval_seen = 0
        self._lstm_epochs_seen = {}
        self._dreamer_mark = None
        self._mem_snap = {}            # symbol -> (ts, {key: mult})
        self._last_dir = {}            # symbol -> "BUY"/"SELL"/"HOLD"
        self._deals_seen = set()
        self._first = True

    # ------------------------------------------------------------------ storage
    def _load(self):
        try:
            with open(FEED_FILE, "r", encoding="utf-8") as f:
                for line in f.readlines()[-300:]:
                    try:
                        self.items.append(json.loads(line))
                    except Exception:
                        pass
        except FileNotFoundError:
            pass

    def say(self, kind: str, title: str, text: str, tone: str = "info", numbers: list | None = None):
        it = {"t": time.time(), "kind": kind, "title": title, "text": text, "tone": tone, "numbers": numbers or []}
        self.items.append(it)
        try:
            with open(FEED_FILE, "a", encoding="utf-8") as f:
                f.write(json.dumps(it) + "\n")
        except Exception:
            pass

    def feed(self, n: int = 60) -> list[dict]:
        return list(self.items)[-n:][::-1]

    # ------------------------------------------------------------------ main tick
    def tick(self, logs, prog: dict, broker, tail_lines) -> None:
        try:
            self._ppo(tail_lines(CYCLE_LOG, 1500, max_bytes=600_000))
        except Exception:
            pass
        for fn in (lambda: self._lstm(prog), lambda: self._dreamer(prog), lambda: self._memory(logs),
                   lambda: self._signals(logs), lambda: self._trades(broker)):
            try:
                fn()
            except Exception:
                pass
        self._first = False

    # ------------------------------------------------------------------ PPO
    def _ppo(self, lines: list[str]):
        # which symbol is PPO training right now
        for l in reversed(lines):
            m = re.search(r"PPO start \| symbol=(\S+)", l)
            if m:
                if m.group(1) != self._ppo_sym:
                    self._ppo_sym, self._ppo_first, self._ppo_last, self._ppo_early = m.group(1), None, None, 0
                break
        if not self._ppo_sym:
            return
        start = max((i for i, l in enumerate(lines) if f"PPO start | symbol={self._ppo_sym}" in l), default=0)
        block, blocks, early = {}, [], 0
        for l in lines[start:]:
            if "Early stopping at step" in l:
                early += 1
            m = re.match(r"\|\s+(\w+)\s+\|\s+([-+0-9.e]+)\s+\|", l)
            if m:
                try:
                    block[m.group(1)] = float(m.group(2))
                except ValueError:
                    pass
            elif l.startswith("-----") and block.get("total_timesteps"):
                blocks.append(block)
                block = {}
        if block.get("total_timesteps"):
            blocks.append(block)
        blocks = [b for b in blocks if "ep_rew_mean" in b]
        if not blocks:
            return
        cur = blocks[-1]
        if self._ppo_first is None:
            self._ppo_first = blocks[0]
        if self._ppo_last is not None and cur.get("total_timesteps") == self._ppo_last.get("total_timesteps"):
            return
        prev = self._ppo_last or (blocks[-2] if len(blocks) > 1 else None)
        self._ppo_last = cur
        if prev is None or self._first and len(self.items) and self.items[-1].get("kind") == "ppo":
            return
        sym = _nm(self._ppo_sym)
        r0, r1 = prev["ep_rew_mean"], cur["ep_rew_mean"]
        d = r1 - r0
        steps = int(cur["total_timesteps"])
        why = []
        ev0, ev1 = prev.get("explained_variance"), cur.get("explained_variance")
        if ev0 is not None and ev1 is not None and abs(ev1 - ev0) >= 0.02:
            why.append(f"it {'predicts' if ev1 > ev0 else 'is worse at predicting'} which situations pay off "
                       f"(explained variance {ev0:.2f} → {ev1:.2f})")
        en0, en1 = prev.get("entropy_loss"), cur.get("entropy_loss")
        if en0 is not None and en1 is not None and abs(en1 - en0) >= 0.2:
            why.append("it is becoming more decisive (less random)" if en1 > en0 else
                       "it is exploring more (more random actions)")
        l0, l1 = prev.get("ep_len_mean"), cur.get("ep_len_mean")
        if l0 and l1 and abs(l1 - l0) / l0 > 0.05:
            why.append(f"practice runs last {'longer' if l1 > l0 else 'shorter'} ({l0:,.0f} → {l1:,.0f} bars"
                       f"{', i.e. it hits the loss limit less often' if l1 > l0 else ''})")
        if early > self._ppo_early:
            frac = early / max(1, int(cur.get("iterations", 1)))
            if frac > 0.5:
                why.append(f"its learning updates are being cut short for stability in {frac:.0%} of rounds")
        self._ppo_early = early
        total_d = r1 - self._ppo_first["ep_rew_mean"]
        tone = "good" if d > 0.05 else ("bad" if d < -0.05 else "info")
        verb = "improved" if d > 0.05 else ("got worse" if d < -0.05 else "held steady")
        text = (f"Practice score {verb}: {r0:.1f} → {r1:.1f} ({_f(d, 1)}) at {steps:,} steps; "
                f"{_f(total_d, 1)} since this run started. "
                + (("Why: " + "; ".join(why) + ".") if why else "No clear driver yet; normal noise between rounds.")
                + (" Still below zero, so it is still losing to trading costs in practice." if r1 < 0 else
                   " Above zero: it makes money after costs in practice."))
        self.say("ppo", f"PPO trader · {sym}", text, tone,
                 [{"label": "score", "from": round(r0, 2), "to": round(r1, 2)},
                  {"label": "steps", "to": steps}])
        # inline validation tests
        evals = [l for l in lines[start:] if "PPO inline eval done" in l]
        for l in evals[self._eval_seen:]:
            m = re.search(r"mean_reward=([-0-9.]+) \| best=([-0-9.]+)", l)
            if m:
                mr, best = float(m.group(1)), float(m.group(2))
                is_best = abs(mr - best) < 1e-6
                self.say("eval", f"Validation test · {sym}",
                         f"On data it never trained on it scored {mr:.2f}"
                         + (" — its best so far, so this version is saved as the candidate." if is_best else
                            f", below its best ({best:.2f}); the best version stays the candidate."),
                         "good" if is_best else "info", [{"label": "test", "to": mr}])
        self._eval_seen = len(evals)

    # ------------------------------------------------------------------ LSTM
    def _lstm(self, prog: dict):
        if not prog or prog.get("kind") != "lstm":
            return
        sym, hist = prog.get("symbol"), prog.get("history") or []
        seen = self._lstm_epochs_seen.get(sym, len(hist) if self._first else 0)
        for i in range(seen, len(hist)):
            h = hist[i]
            prev = hist[i - 1] if i else None
            if prev is None:
                self.say("lstm", f"LSTM · {_nm(sym)}", f"First pass done: error on unseen bars {h['val_loss']:.3f}, "
                         f"{h['val_acc']:.1f}% of next-bar calls right.", "info")
                continue
            dv, da = h["val_loss"] - prev["val_loss"], h["val_acc"] - prev["val_acc"]
            dt_ = h["loss"] - prev["loss"]
            if dv < 0 and dt_ < 0:
                why = "it improved on both practice and unseen data, so it is learning real patterns"
                tone = "good"
            elif dv >= 0 and dt_ < 0:
                why = "it got better on practice data but worse on unseen data: early sign of memorising"
                tone = "bad"
            else:
                why = "normal wobble between passes"
                tone = "info"
            self.say("lstm", f"LSTM · {_nm(sym)} · pass {h['epoch']}",
                     f"Unseen-data error {prev['val_loss']:.3f} → {h['val_loss']:.3f} ({_f(-dv / prev['val_loss'] * 100, 1)}% better), "
                     f"accuracy {prev['val_acc']:.1f}% → {h['val_acc']:.1f}% ({_f(da, 1)} pts). Why: {why}."
                     + (" ★ Best so far: this version is kept." if h.get("best") else ""), tone)
        self._lstm_epochs_seen[sym] = len(hist)

    # ------------------------------------------------------------------ Dreamer
    def _dreamer(self, prog: dict):
        if not prog or prog.get("kind") != "dreamer" or prog.get("world_model_loss") is None:
            return
        step, wm = int(prog.get("step", 0)), float(prog["world_model_loss"])
        if self._dreamer_mark is None or self._dreamer_mark[0] != prog.get("symbol"):
            self._dreamer_mark = (prog.get("symbol"), step, wm)
            return
        _, s0, w0 = self._dreamer_mark
        if step - s0 < 1000:
            return
        d = (w0 - wm) / w0 * 100 if w0 else 0
        self.say("dreamer", f"Dreamer · {_nm(prog.get('symbol'))}",
                 f"World-model error {w0:.0f} → {wm:.0f} ({_f(d, 0)}% better) over the last {step - s0:,} steps: "
                 f"{'its simulation of the market is getting more accurate' if d > 0 else 'no gain this stretch; it is still adjusting'}.",
                 "good" if d > 0 else "info")
        self._dreamer_mark = (prog.get("symbol"), step, wm)

    # ------------------------------------------------------------------ indicator memory
    def _memory(self, logs):
        from Python.indicator_council import describe

        now = time.time()
        for sym, ev in logs.last_council.items():
            assess = ((ev or {}).get("payload") or {}).get("assess") or {}
            cur = {k: float((v or {}).get("mult", 1.0)) for k, v in assess.items()}
            hit = {k: (v or {}).get("hit_rate") for k, v in assess.items()}
            snap = self._mem_snap.get(sym)
            if snap is None:
                self._mem_snap[sym] = (now, cur)
                continue
            if now - snap[0] < 900:  # compare every 15 min
                continue
            old = snap[1]
            moves = sorted(((k, old.get(k, 1.0), v) for k, v in cur.items() if abs(v - old.get(k, 1.0)) >= 0.03),
                           key=lambda x: -abs(x[2] - x[1]))[:3]
            self._mem_snap[sym] = (now, cur)
            if not moves:
                continue
            parts = []
            for k, a, b in moves:
                h = hit.get(k)
                parts.append(f"{describe(k)} ×{a:.2f} → ×{b:.2f} ({'more' if b > a else 'less'} trust"
                             + (f"; right {h * 100:.0f}% of the time lately" if h is not None else "") + ")")
            self.say("memory", f"Memory update · {_nm(sym)}",
                     "In the last 15 min the bot re-scored its indicators after seeing what price did: "
                     + "; ".join(parts) + ". Indicators that keep being right get more say in the next decision.",
                     "info")

    # ------------------------------------------------------------------ signals
    def _signals(self, logs):
        from Python.indicator_council import describe

        for sym, ev in logs.last_signal.items():
            p = (ev or {}).get("payload") or {}
            prof = p.get("decision_profile") if isinstance(p.get("decision_profile"), dict) else {}
            thr = float(prof.get("min_trade_threshold", 0.15) or 0.15)
            raw = float(p.get("raw_target", 0) or 0) * float(p.get("risk_scalar", 1) or 1)
            d = "BUY" if raw >= thr else ("SELL" if raw <= -thr else "HOLD")
            prev = self._last_dir.get(sym)
            self._last_dir[sym] = d
            if prev is None or prev == d:
                continue
            flips = []
            cur_v = ((logs.last_council.get(sym) or {}).get("payload") or {}).get("votes") or {}
            old_v = ((logs.prev_council.get(sym) or {}).get("payload") or {}).get("votes") or {}
            for k, v in cur_v.items():
                o = old_v.get(k)
                if o is None:
                    continue
                if (o > 0.05) != (v > 0.05) or (o < -0.05) != (v < -0.05):
                    flips.append((k, o, v))
            flips.sort(key=lambda x: -abs(x[2] - x[1]))
            why = ""
            if flips:
                why = " What changed: " + ", ".join(
                    f"{describe(k)} turned {'up' if v > 0.05 else ('down' if v < -0.05 else 'neutral')}" for k, o, v in flips[:3])
                why += f" ({len(flips)} indicator view{'s' if len(flips) != 1 else ''} changed since the last check)."
            self.say("signal", f"{_nm(sym)} · {prev} → {d}",
                     f"Signal is now {raw:+.2f} (trade line ±{thr:.2f})." + why,
                     "good" if d != "HOLD" else "info", [{"label": "signal", "to": round(raw, 3)}])

    # ------------------------------------------------------------------ trades
    def _trades(self, broker):
        out_entry = int(getattr(broker.mt5, "DEAL_ENTRY_OUT", 1)) if broker.mt5 else 1
        for d in broker.deals:
            key = int(getattr(d, "ticket", 0) or 0) or (getattr(d, "time", 0), getattr(d, "price", 0))
            if key in self._deals_seen:
                continue
            self._deals_seen.add(key)
            if self._first or float(getattr(d, "volume", 0) or 0) <= 0:
                continue
            sym = str(getattr(d, "symbol", ""))
            sym = _nm(sym if sym.endswith("m") else sym + "m")
            if int(getattr(d, "entry", -1)) == out_entry:
                pnl = float(getattr(d, "profit", 0) or 0)
                self.say("trade", f"Closed {sym} · {'+' if pnl >= 0 else '−'}${abs(pnl):.2f}",
                         ("Win: price moved its way far enough to bank the profit." if pnl > 0 else
                          "Loss: price moved against the trade and the stop-loss closed it to cap the damage.")
                         + " The memory scores the indicators that voted for it, so this result changes their weight.",
                         "good" if pnl > 0 else "bad")
            else:
                side = "Bought" if int(getattr(d, "type", 0)) == 0 else "Sold"
                self.say("trade", f"{side} {float(d.volume):.2f} {sym}",
                         f"Entered at ${float(getattr(d, 'price', 0)):,.2f}; stop-loss and take-profit set straight away.",
                         "info")
