"""Animated terminal dashboard for the live bot.

    scripts/mac/monitor.sh        (or: source ~/.exness/env && .venv/bin/python tools/monitor_tui.py)

Read-only: it never sends orders. Shows, live:
  * bot heartbeat / status, scrolling price ticker
  * braille price chart per symbol with the open position's entry / SL / TP
  * account, equity curve, daily-loss usage bar
  * each symbol's latest decision (models, blend, supervisor)
  * model memory heatmap: which model works in which market context
  * training progress, recent trades / risk blocks (new trades flash), server log
Ctrl-C to quit; the bot keeps running.
"""
from __future__ import annotations

import datetime as dt
import json
import math
import os
import re
import subprocess
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from rich import box
from rich.align import Align
from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import warroom as wr  # noqa: E402

LOG_DIR = os.path.join(ROOT, "logs")
AUDIT = os.path.join(LOG_DIR, "audit_events.jsonl")
SERVER_LOG = os.path.join(LOG_DIR, "server.log")
SERVER_OUT = os.path.join(LOG_DIR, "server_stdout.log")
CYCLE_LOG = os.path.join(LOG_DIR, "champion_cycle_run.log")
LSTM_LOG = os.path.join(LOG_DIR, "lstm_training.log")
PPO_LOG = os.path.join(LOG_DIR, "ppo_training.log")
LOCK = os.path.join(ROOT, ".tmp", "server_agi.lock")
REGISTRY = os.path.join(ROOT, "models", "registry", "active.json")
MEMORY = os.path.join(LOG_DIR, "memory", "model_memory.json")

SPIN = "⠋⠙⠹⠸⠼⠴⠦⠧⠇⠏"
HEART = ["♥", "♥", "♡", "♡"]
MODELS = ("ppo", "lstm", "dreamer", "blend")


# --------------------------------------------------------------------------- helpers
def _cfg() -> dict:
    try:
        import yaml

        with open(os.path.join(ROOT, "config.yaml"), "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}
    except Exception:
        return {}


def tail_lines(path: str, n: int, max_bytes: int = 400_000) -> list[str]:
    try:
        with open(path, "rb") as f:
            f.seek(0, os.SEEK_END)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", errors="replace")
        return data.splitlines()[-n:]
    except Exception:
        return []


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
        return True
    except Exception:
        return False


def age(ts) -> str:
    if not ts:
        return "-"
    try:
        t = dt.datetime.fromisoformat(str(ts))
        if t.tzinfo is None:
            t = t.replace(tzinfo=dt.timezone.utc)
        s = int((dt.datetime.now(dt.timezone.utc) - t).total_seconds())
        return f"{s}s" if s < 120 else (f"{s // 60}m" if s < 7200 else f"{s // 3600}h")
    except Exception:
        return str(ts)[:19]


def money(v, nd=2) -> str:
    try:
        return f"{float(v):,.{nd}f}"
    except Exception:
        return "-"


def signed(v, fmt="{:+.2f}") -> Text:
    try:
        f = float(v)
    except Exception:
        return Text("-", style="dim")
    return Text(fmt.format(f), style="green" if f > 0 else ("red" if f < 0 else "dim"))


def price_digits(v: float) -> int:
    return 2 if v >= 1000 else (3 if v >= 10 else 5)


# --------------------------------------------------------------------------- braille chart
def braille_chart(values: list[float], width: int, height: int, lines: dict[str, tuple[float, str]] | None = None,
                  color: str = "cyan") -> Text:
    """Line chart in braille (2x4 dots per cell). `lines` = {label: (price, style)} drawn as dotted rows."""
    vals = [v for v in values if v is not None and math.isfinite(v)]
    if len(vals) < 2 or width < 4 or height < 2:
        return Text("waiting for data…", style="dim")
    lines = lines or {}
    lo = min(vals + [p for p, _ in lines.values()])
    hi = max(vals + [p for p, _ in lines.values()])
    if hi - lo < 1e-12:
        hi, lo = hi + 1, lo - 1
    cols, rows = width * 2, height * 4
    # resample to `cols` points
    pts = [vals[int(i * (len(vals) - 1) / max(cols - 1, 1))] for i in range(cols)]
    ys = [int(round((p - lo) / (hi - lo) * (rows - 1))) for p in pts]
    grid = [[0] * width for _ in range(height)]
    dot = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]

    def plot(x, y):
        r = rows - 1 - y
        grid[r // 4][x // 2] |= dot[r % 4][x % 2]

    for x in range(cols):
        y0 = ys[x]
        y1 = ys[x + 1] if x + 1 < cols else y0
        for y in range(min(y0, y1), max(y0, y1) + 1):
            plot(x, y)
    line_rows = {}
    for label, (p, style) in lines.items():
        r = rows - 1 - int(round((p - lo) / (hi - lo) * (rows - 1)))
        line_rows.setdefault(r // 4, []).append((label, style))
    out = Text()
    for i, row in enumerate(grid):
        styled_line = i in line_rows
        for ch_bits in row:
            if ch_bits:
                out.append(chr(0x2800 + ch_bits), style=color)
            elif styled_line:
                out.append("┄", style=line_rows[i][0][1])
            else:
                out.append(" ")
        if styled_line:
            out.append(" " + "/".join(l for l, _ in line_rows[i]), style=line_rows[i][0][1])
        elif i == 0:
            out.append(f" {hi:,.{price_digits(hi)}f}", style="dim")
        elif i == height - 1:
            out.append(f" {lo:,.{price_digits(lo)}f}", style="dim")
        out.append("\n")
    return out


def tbar(frac: float, width: int, style: str = "cyan") -> Text:
    frac = max(0.0, min(1.0, float(frac or 0.0)))
    full = int(frac * width)
    return Text("█" * full + "░" * (width - full), style=style)


SPARK = "▁▂▃▄▅▆▇█"


def sparkline(values: list[float], width: int) -> str:
    vals = [v for v in values if v is not None]
    if len(vals) < 2:
        return ""
    vals = vals[-width:]
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-12:
        return SPARK[3] * len(vals)
    return "".join(SPARK[int((v - lo) / (hi - lo) * 7)] for v in vals)


# --------------------------------------------------------------------------- data
class Broker:
    """Read-only broker view polled in a background thread (never blocks drawing),
    throttled to stay well inside the API rate limits."""

    def __init__(self, symbols):
        from Python.mt5_compat import BROKER_BACKEND, MT5_AVAILABLE, mt5

        self.symbols = symbols
        self.backend = BROKER_BACKEND
        self.mt5 = mt5 if MT5_AVAILABLE else None
        self.ok, self.err = False, ""
        self.account, self.positions, self.pnl_today = None, [], None
        self.deals = []
        self.ticks: dict[str, object] = {}
        self.prev_bid: dict[str, float] = {}
        self.bars: dict[str, list[float]] = {}
        self.flash: dict[str, float] = {}
        self._t = {"acc": 0.0, "tick": 0.0, "deals": 0.0, "bars": 0.0}
        if self.mt5 is not None:
            try:
                self.ok = bool(self.mt5.initialize())
                if not self.ok:
                    self.err = str(self.mt5.last_error())
            except Exception as exc:
                self.err = str(exc)
        else:
            self.err = "broker backend unavailable"
        import threading

        threading.Thread(target=self._loop, daemon=True).start()

    def _loop(self):
        while True:
            self._poll()
            time.sleep(0.5)

    def refresh(self):
        pass  # polling happens in the background thread

    def _poll(self):
        if not self.ok:
            return
        now = time.time()
        try:
            if now - self._t["acc"] >= 4:
                self.account = self.mt5.account_info()
                self.positions = list(self.mt5.positions_get() or [])
                self._t["acc"] = now
            if now - self._t["tick"] >= 3:
                for s in self.symbols:
                    t = self.mt5.symbol_info_tick(s)
                    if t is not None:
                        old = self.ticks.get(s)
                        if old is not None and float(old.bid) != float(t.bid):
                            self.prev_bid[s] = float(old.bid)
                            self.flash[s] = time.time()
                        self.ticks[s] = t
                self._t["tick"] = now
            if now - self._t["bars"] >= 60:
                tf = getattr(self.mt5, "TIMEFRAME_M5", 5)
                for s in self.symbols:
                    r = self.mt5.copy_rates_from_pos(s, tf, 1, 150)
                    if r is not None and len(r):
                        self.bars[s] = [float(x) for x in r["close"]]
                self._t["bars"] = now
            if now - self._t["deals"] >= 60:
                start = dt.datetime.now(dt.timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
                deals = self.mt5.history_deals_get(start, dt.datetime.now(dt.timezone.utc)) or []
                self.deals = list(deals)
                out_entry = int(getattr(self.mt5, "DEAL_ENTRY_OUT", 1))
                self.pnl_today = sum(float(d.profit) + float(getattr(d, "commission", 0) or 0)
                                     + float(getattr(d, "swap", 0) or 0)
                                     for d in deals if int(getattr(d, "entry", -1)) == out_entry)
                self._t["deals"] = now
            self.err = ""
        except Exception as exc:
            self.err = str(exc)[:160]


class Logs:
    """Incremental reader for the audit log (decisions, trades, blocks, memory, snapshots)."""

    def __init__(self):
        self.last_signal, self.last_block, self.last_memory = {}, {}, {}
        self.last_council = {}
        self.sig_hist: dict[str, list] = {}
        self.pending_entry: dict[str, dict] = {}
        self.events: list[dict] = []
        self.equity: list[float] = []
        self.last_snapshot = None
        self._pos = 0
        self._t = 0.0
        try:
            size = os.path.getsize(AUDIT)
            self._pos = max(0, size - 3_000_000)
        except Exception:
            pass

    def refresh(self):
        if time.time() - self._t < 1.0:
            return
        self._t = time.time()
        try:
            with open(AUDIT, "rb") as f:
                f.seek(0, os.SEEK_END)
                size = f.tell()
                if size < self._pos:  # rotated
                    self._pos = 0
                f.seek(self._pos)
                chunk = f.read()
                cut = chunk.rfind(b"\n")
                if cut < 0:
                    return
                self._pos += cut + 1
                lines = chunk[:cut].decode("utf-8", errors="replace").splitlines()
        except Exception:
            return
        for line in lines:
            try:
                r = json.loads(line)
            except Exception:
                continue
            ev, p = r.get("event"), r.get("payload") or {}
            sym = p.get("symbol")
            if ev == "signal" and sym:
                self.last_signal[sym] = r
                prof = p.get("decision_profile") if isinstance(p.get("decision_profile"), dict) else {}
                h = self.sig_hist.setdefault(sym, [])
                h.append((r.get("ts"), float(p.get("raw_target", 0) or 0) * float(p.get("risk_scalar", 1) or 1),
                          float(p.get("exposure", 0) or 0), float(prof.get("min_trade_threshold", 0.15) or 0.15),
                          float(p.get("council", 0) or 0)))
                del h[:-240]
            elif ev == "model_memory" and sym:
                self.last_memory[sym] = r
            elif ev == "council" and sym:
                self.last_council[sym] = r
            elif ev == "risk_supervisor_block" and sym:
                self.last_block[sym] = r
                self.events.append(r)
            elif ev == "entry_planned" and sym:
                self.pending_entry[sym] = p
            elif ev == "entry_cancelled" and sym:
                self.pending_entry.pop(sym, None)
            elif ev == "trade_action":
                self.events.append(r)
                if sym:
                    self.pending_entry.pop(sym, None)
            elif ev == "snapshot":
                self.last_snapshot = r
                eq = p.get("equity")
                if eq is not None:
                    self.equity.append(float(eq))
        self.events = self.events[-40:]
        self.equity = self.equity[-400:]


def bot_status() -> tuple[bool, str]:
    try:
        pid = int(open(LOCK).read().strip() or 0)
    except Exception:
        return False, "STOPPED"
    return (True, f"RUNNING pid {pid}") if pid and pid_alive(pid) else (False, "STOPPED (crashed?)")


_train_cache = {"t": 0.0, "v": None}


def training_state() -> dict:
    if time.time() - _train_cache["t"] < 3 and _train_cache["v"] is not None:
        return _train_cache["v"]
    try:
        out = subprocess.run(["pgrep", "-fl", "champion_cycle|train_lstm|train_drl|train_dreamer"],
                             capture_output=True, text=True, timeout=3).stdout
    except Exception:
        out = ""
    running = [w for w in ("champion_cycle", "train_lstm", "train_drl", "train_dreamer") if w in out]
    st = {"running": running, "stage": "", "lstm": None, "ppo": None, "last": ""}
    for line in reversed(tail_lines(CYCLE_LOG, 60)):
        if " - " in line and "|" in line:
            st["last"] = line.split(" - ", 1)[-1][:150]
            break
    for line in reversed(tail_lines(LSTM_LOG, 80)):
        m = re.search(r"(\w+)\s*\|?\s*epoch\s+(\d+)\s*/\s*(\d+)(.*)", line)
        if m:
            st["lstm"] = {"symbol": m.group(1), "epoch": int(m.group(2)), "total": int(m.group(3)),
                          "detail": m.group(4).strip(" |")[:70]}
            break
    for line in reversed(tail_lines(PPO_LOG, 120)):
        m = re.search(r"step=(\d[\d,]*)/(\d[\d,]*)", line)
        if m:
            st["ppo"] = {"done": int(m.group(1).replace(",", "")), "total": int(m.group(2).replace(",", ""))}
            break
    _train_cache.update(t=time.time(), v=st)
    return st


def read_progress() -> dict:
    try:
        from Python.train_progress import read

        return read()
    except Exception:
        return {}


NAMES = {"BTCUSDm": "Bitcoin", "XAUUSDm": "gold"}


def _nm(sym) -> str:
    return NAMES.get(str(sym), str(sym))


def explain(prog: dict, tr: dict, reg: dict, symbols: list[str], alive: bool, logs) -> list[Text]:
    """Plain-English narration of what training and the bot are doing right now."""
    out: list[Text] = []

    def say(msg, style=""):
        out.append(Text(msg, style=style))

    live = bool(tr.get("running")) and prog and not prog.get("done") and time.time() - float(prog.get("ts", 0)) < 600
    last = (tr.get("last") or "").lower()

    # ---- training
    if live and prog.get("kind") == "lstm":
        sym, ep, eps = _nm(prog.get("symbol")), int(prog.get("epoch", 0)), int(prog.get("epochs", 1))
        say(f"📚 Teaching the LSTM to read {sym} charts.", "bold")
        say(f"Reads the last 60 bars × 150 indicators and predicts UP / DOWN / FLAT for the next bar. "
            f"Pass {ep} of up to {eps} over {int(prog.get('train_samples', 0)):,} past examples.")
        if prog.get("phase") == "validating" or int(prog.get("batch", 0)) >= int(prog.get("n_batches", 1)):
            say("🔍 Pass finished; now testing it on recent bars it was NOT trained on (the honest score).", "cyan")
        hist = prog.get("history") or []
        base = prog.get("majority_val_acc")
        if base is not None:
            cp = prog.get("val_class_pct") or {}
            say(f"Yardstick: always saying FLAT scores {base:.0f}% (test bars: flat {cp.get('hold', 0):.0f}% · "
                f"up {cp.get('up', 0):.0f}% · down {cp.get('down', 0):.0f}%).", "dim")
        if hist:
            h = hist[-1]
            va = float(h.get("val_acc", 0))
            say(f"Unseen bars right after last pass: {va:.1f}%.")
            if base is not None and va < base:
                say("Below the yardstick on purpose: it hunts the rarer UP/DOWN moves. Judge it by falling val_loss "
                    "and the trading exam.", "dim")
            if len(hist) >= 2:
                prev = hist[-2]
                tl, vl = float(h["loss"]) - float(prev["loss"]), float(h["val_loss"]) - float(prev["val_loss"])
                if tl < 0 and vl < 0:
                    say("✅ Error is falling on both practice and unseen data: it is genuinely learning.", "green")
                elif tl < 0 <= vl:
                    left = int(prog.get("patience", 4)) - sum(1 for x in reversed(hist) if not x.get("best"))
                    say(f"⚠️  Better on practice data but worse on unseen data: it is starting to memorise. "
                        f"Training stops early if this continues ({max(0, left)} more passes).", "yellow")
                else:
                    say("〰️  Error went up this pass; normal wobble. The best version so far is kept (★).", "dim")
            best = [x for x in hist if x.get("best")]
            if best:
                say(f"Best version so far: pass {best[-1]['epoch']} ({float(best[-1]['val_acc']):.1f}% on unseen bars).", "dim")
        else:
            say("First pass in progress: early numbers are noisy.", "dim")
        say("The LSTM is a helper; the PPO trader and the indicator council carry most of the decision.", "dim")
    elif live and prog.get("kind") == "ppo":
        st, tot = int(prog.get("step", 0)), max(1, int(prog.get("total", 1)))
        say(f"🎮 Training the PPO trader for {_nm(prog.get('symbol'))} in a simulator.", "bold")
        say(f"It replays past market data and practises going long, short or flat, paying realistic spread, "
            f"stop-loss and take-profit costs. {st:,} of {tot:,} practice steps done.")
        r = prog.get("ep_reward_mean")
        if r is not None:
            say(f"Average score per practice run: {r:+.3f} (higher = it made money in practice; "
                f"below zero = still losing to costs).", "green" if r > 0 else "yellow")
        say("Practice scores flatter it: the real test is the unseen test window that comes next.", "dim")
    elif "evaluate" in last or "backtest" in last:
        say("🧪 Exam time: replaying the trained model on the last ~15% of history it never saw.", "bold")
        say("To pass it must make money after costs, keep drawdown under 10%, have Sharpe ≥ 0.5, and beat "
            "both buy-and-hold and a simple moving-average strategy.")
    elif tr.get("running"):
        if "dukascopy" in last or "fetch" in last or "data" in last:
            say("⬇️  Downloading price history from Exness (cached after the first time).", "bold")
        else:
            say("⚙️  Preparing: loading price history and building the 150 indicators the models read.", "bold")
    else:
        say("💤 No training running. Type 'exness train' to start a new cycle.", "dim")

    # ---- what happened at the gate
    for line in tail_lines(CYCLE_LOG, 120):
        l = line.lower()
        if "demo exploration canary" in l:
            say("🧭 " + line.split(" - ", 1)[-1].split(":")[0] + ": failed the exam, so it trades at half size on demo "
                "to gather real results (never on a real account).", "magenta")
        elif "canary set for" in l:
            say("🏁 " + line.split(" - ", 1)[-1][:90] + ": passed the exam and is now trading.", "green")
        elif "candidate blocked" in l:
            say("🚫 " + line.split(" - ", 1)[-1][:90] + " (did not beat the benchmarks).", "yellow")

    if not alive:
        say("🔴 The bot is not running. Type 'exness' to start it.", "red")
    return out



# --------------------------------------------------------------------------- war-room widgets
SESSIONS = [("SYD", 21, 6), ("TYO", 0, 9), ("LDN", 7, 16), ("NYC", 12, 21)]


def _session_open(h0: int, h1: int, now: dt.datetime) -> bool:
    h = now.hour + now.minute / 60
    return (h0 <= h < h1) if h0 < h1 else (h >= h0 or h < h1)


def war_header(frame: int, alive: bool, engaged: bool, halted: bool, broker, symbols, logs) -> Group:
    now = dt.datetime.now(dt.timezone.utc)
    title = "◢◤ CAUTIOUS-GIGGLE  ·  TRADING WAR ROOM ◥◣"
    t = Text("  ")
    k = (frame // 1) % (len(title) + 12)
    for i, ch in enumerate(title):
        d = abs(i - k)
        t.append(ch, style="bold bright_white" if d == 0 else ("bold bright_cyan" if d < 3 else "bold cyan"))
    if not alive:
        badge, bstyle = " OFFLINE ", "bold white on red"
    elif halted:
        badge, bstyle = " HALTED ", "bold black on yellow"
    elif engaged:
        badge, bstyle = " ENGAGED ", ("bold black on bright_green" if (frame // 4) % 2 == 0 else "bold black on green")
    else:
        badge, bstyle = " ARMED · SCANNING ", "bold black on cyan"
    t.append("    ")
    t.append(badge, style=bstyle)
    t.append("   " + ("♥" if (frame // 2) % 2 == 0 and alive else "♡"), style="bold red")
    t.append(f"   UTC {now:%H:%M:%S}", style="bold")
    t.append(f"   LOCAL {dt.datetime.now():%H:%M:%S}", style="dim")
    weekend = now.weekday() == 5 or (now.weekday() == 6 and now.hour < 21) or (now.weekday() == 4 and now.hour >= 21)
    t.append("   SESSIONS ", style="dim")
    for name, h0, h1 in SESSIONS:
        op = _session_open(h0, h1, now) and not weekend
        t.append(f"{'●' if op else '○'}{name} ", style="bold green" if op else "grey42")

    tk = Text("  ")
    for s in symbols:
        tick = broker.ticks.get(s)
        if not tick:
            tk.append(f"{s} …   ", style="dim")
            continue
        prev = broker.prev_bid.get(s, tick.bid)
        arrow = "▲" if tick.bid > prev else ("▼" if tick.bid < prev else "◆")
        fresh = time.time() - broker.flash.get(s, 0) < 1.5
        col = "green" if tick.bid >= prev else "red"
        style = (f"bold black on {col}" if fresh and frame % 2 == 0 else f"bold {col}") if tick.bid != prev else "bold yellow"
        tk.append(f"{_nm(s).upper():<8}", style="bold white")
        tk.append(f" {tick.bid:,.{price_digits(tick.bid)}f} {arrow} ", style=style)
        tk.append(f" spr {(tick.ask - tick.bid):.{price_digits(tick.bid)}f}    ", style="dim")
    left = 300 - (time.time() % 300)
    w = 34
    full = int((1 - left / 300) * w)
    tk.append("NEXT DECISION ", style="dim")
    tk.append(f"{int(left // 60)}:{int(left % 60):02d} ", style="bold cyan" if left > 10 else "bold black on cyan")
    tk.append("▰" * full, style="cyan")
    tk.append("▱" * (w - full), style="grey35")
    return Group(t, tk)


def radar(frame: int, payload: dict, width: int = 30, height: int = 9) -> Text:
    """Braille radar: rings = timeframes (inner M5 -> outer H4), angle = indicator; blips glow
    when the sweep passes. Green = votes up, red = down, grey = ignored (no record here)."""
    from Python.indicator_council import STRATEGIES as IND, TIMEFRAMES

    cols, rows = width * 2, height * 4
    cx, cy = (cols - 1) / 2, (rows - 1) / 2
    R = min(cx, cy * 1.0)
    sweep = (frame * 0.14) % (2 * math.pi)
    grid = [[0] * width for _ in range(height)]
    style = [[None] * width for _ in range(height)]
    prio = [[0] * width for _ in range(height)]
    dot = [[0x01, 0x08], [0x02, 0x10], [0x04, 0x20], [0x40, 0x80]]

    def put(x, y, st, pr):
        xi, yi = int(round(x)), int(round(y))
        if 0 <= xi < cols and 0 <= yi < rows:
            r, c = yi // 4, xi // 2
            grid[r][c] |= dot[yi % 4][xi % 2]
            if pr >= prio[r][c]:
                style[r][c], prio[r][c] = st, pr

    for frac in (0.25, 0.5, 0.75, 1.0):
        n = int(2 * math.pi * R * frac)
        for i in range(0, max(8, n), 2):
            a = 2 * math.pi * i / max(8, n)
            put(cx + R * frac * math.cos(a), cy + R * frac * math.sin(a), "grey30", 1)
    for i in range(int(R) + 1):  # sweep beam + fading trail
        for lag, st in ((0.0, "bold bright_green"), (0.18, "green"), (0.36, "dark_green")):
            a = sweep - lag
            put(cx + i * math.cos(a), cy + i * math.sin(a), st, 2)
    votes_ = (payload or {}).get("votes") or {}
    assess = (payload or {}).get("assess") or {}
    names = list(IND)
    for ti, tf in enumerate(TIMEFRAMES):
        rr = R * (0.25 + 0.25 * ti) - 0.8
        for ii, name in enumerate(names):
            key = name if tf == "M5" else f"{name}@{tf}"
            v = votes_.get(key)
            if v is None or abs(v) <= 0.05:
                continue
            a = 2 * math.pi * ii / len(names)
            behind = (sweep - a) % (2 * math.pi)
            hot = behind < 0.9
            ignored = (assess.get(key) or {}).get("verdict") == "doesn't work"
            col = "grey50" if ignored else ("bright_green" if v > 0 else "bright_red")
            st = ("bold " + col) if hot else col.replace("bright_", "")
            x, y = cx + rr * math.cos(a), cy + rr * math.sin(a)
            for dx, dy in ((0, 0), (1, 0), (0, 1), (1, 1)):
                put(x + dx, y + dy, st, 3)
    out = Text()
    for r in range(height):
        for c in range(width):
            out.append(chr(0x2800 + grid[r][c]) if grid[r][c] else " ", style=style[r][c] or "")
        out.append("\n")
    return out


def gauge(value: float, threshold: float, width: int = 44) -> Text:
    """Decision needle on a -1..+1 scale with the trade thresholds marked."""
    v = max(-1.0, min(1.0, float(value or 0.0)))
    pos = int(round((v + 1) / 2 * (width - 1)))
    mid = (width - 1) // 2
    thr = [int(round((x + 1) / 2 * (width - 1))) for x in (-threshold, threshold)]
    out = Text()
    for i in range(width):
        if i == pos:
            out.append("▼" if True else "", style="bold bright_white")
        elif i in thr:
            out.append("┃", style="bold yellow")
        elif i == mid:
            out.append("┼", style="grey50")
        elif (mid < i < pos) or (pos < i < mid):
            strong = abs(v) >= threshold
            out.append("█", style=("bold green" if v > 0 else "bold red") if strong else ("green" if v > 0 else "red"))
        else:
            out.append("─", style="grey30")
    return out


def council_grid(payloads: dict, symbols: list[str]) -> Table:
    from Python.indicator_council import STRATEGIES as IND, TIMEFRAMES

    tfs = list(TIMEFRAMES)
    t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 0), show_edge=False)
    t.add_column("indicator", no_wrap=True, overflow="ellipsis", ratio=4)
    for s in symbols:
        for tf in tfs:
            t.add_column(f"{_nm(s)[:3].upper()}\n{tf}" if tf == "M5" else f"\n{tf}", justify="center", no_wrap=True, ratio=1)
    for name, desc in IND.items():
        row = [Text(desc[:26])]
        for s in symbols:
            p = payloads.get(s) or {}
            for tf in tfs:
                key = name if tf == "M5" else f"{name}@{tf}"
                v = (p.get("votes") or {}).get(key)
                a = (p.get("assess") or {}).get(key) or {}
                if v is None:
                    row.append(Text("··", style="grey30"))
                    continue
                verdict = a.get("verdict")
                if verdict == "doesn't work":
                    row.append(Text("╳╳", style="grey42"))
                elif abs(v) <= 0.05:
                    row.append(Text("──", style="grey35"))
                else:
                    strong = abs(v) > 0.5
                    col = "green" if v > 0 else "red"
                    st = f"bold bright_{col}" if (strong and verdict == "works") else (f"bold {col}" if strong else col)
                    row.append(Text("██" if strong else "▄▄", style=st))
        t.add_row(*row)
    tot = [Text("COUNCIL VOTE", style="bold")]
    for s in symbols:
        p = payloads.get(s) or {}
        tv = float(p.get("target", 0) or 0) if p else None
        cell = Text(("—" if tv is None else f"{'▲' if tv > 0.05 else ('▼' if tv < -0.05 else '◆')}{tv:+.2f}"),
                    style="bold black on green" if (tv or 0) > 0.3 else ("bold white on red" if (tv or 0) < -0.3 else "bold"))
        tot += [cell, Text(""), Text(""), Text("")]
    t.add_row(*tot)
    return t


def _threshold(cfg: dict, symbol: str, sg: dict) -> float:
    for k in ("min_trade_threshold", "trade_threshold", "threshold"):
        if sg.get(k) is not None:
            return float(sg[k])
    return {"BTC": 0.18, "XAU": 0.12}.get(symbol[:3], 0.15)


def _px(v):
    try:
        v = float(v)
        return f"{v:,.{price_digits(v)}f}"
    except Exception:
        return str(v)


def _default_meta() -> dict:
    try:
        with open(os.path.join(ROOT, "models", "lstm_agi_trained.meta.json"), "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def memory_rows(symbol: str):
    try:
        from Python.model_memory import ModelMemory

        mem = ModelMemory(MEMORY)
        return mem.table(symbol)
    except Exception:
        return []


_mem_cache = {"t": 0.0, "v": {}}


# --------------------------------------------------------------------------- render
def render(frame: int, broker: Broker, logs: Logs, cfg: dict, symbols: list[str], width: int) -> Layout:
    spin = SPIN[frame % len(SPIN)]
    alive, status = bot_status()
    risk = cfg.get("risk", {}) or {}

    # ---- header
    halted = False
    for x in symbols:
        blk, sgl = logs.last_block.get(x), logs.last_signal.get(x)
        if blk and str((blk.get("payload") or {}).get("reason", "")).startswith("halt") and \
                (not sgl or blk.get("ts", "") >= sgl.get("ts", "")):
            halted = True
    header = war_header(frame, alive, bool(broker.positions), halted, broker, symbols, logs)

    # ---- trade flash
    banner = Text(" ", style="dim")
    if logs.events:
        last = logs.events[-1]
        if last.get("event") == "trade_action":
            try:
                secs = (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(last["ts"])).total_seconds()
            except Exception:
                secs = 999
            if secs < 20:
                p = last.get("payload") or {}
                style = "bold black on green" if (frame // 4) % 2 == 0 else "bold green on black"
                banner = Text(f"  ⚡ TRADE {p.get('symbol')} {p.get('request_action')} {p.get('order_type')} "
                              f"{p.get('executed_lots', p.get('volume_lots'))} lots @ {_px(p.get('entry_price'))}  ",
                              style=style, justify="center")

    # ---- charts
    chart_w = max(20, int((width - 10) * 3 / (3 * len(symbols) + 2)) - 14)
    charts = Table.grid(expand=True)
    for _ in symbols:
        charts.add_column(ratio=3)
    charts.add_column(ratio=2)
    cells = []
    for s in symbols:
        closes = list(broker.bars.get(s, []))
        t = broker.ticks.get(s)
        if t is not None:
            closes.append(float(t.bid))
        mine = [p for p in broker.positions if str(getattr(p, "symbol", "")) in (s, s.rstrip("m"))]
        marks = {}
        title = Text(s, style="bold")
        if mine:
            p0 = mine[0]
            long_ = int(p0.type) == 0
            marks["entry"] = (float(p0.price_open), "yellow")
            if p0.sl:
                marks["SL"] = (float(p0.sl), "red")
            if p0.tp:
                marks["TP"] = (float(p0.tp), "green")
            pnl = sum(float(p.profit) for p in mine)
            title.append(f"  {'LONG' if long_ else 'SHORT'} {sum(float(p.volume) for p in mine):.2f} ",
                         style="bold green" if long_ else "bold red")
            title.append_text(signed(pnl))
        else:
            title.append("  flat", style="dim")
        color = "green" if len(closes) > 1 and closes[-1] >= closes[0] else "red"
        cells.append(Panel(braille_chart(closes, chart_w, 5, marks, color=color), title=title,
                           subtitle=wr.caption("5-min price · yellow entry · red stop · green target"), box=box.ROUNDED))
    sh_w = max(20, int((width - 10) * 2 / (3 * len(symbols) + 2)) - 12)
    cells.append(Panel(wr.signal_history(logs.sig_hist, symbols, braille_chart, sh_w),
                       title=Text("📈 SIGNAL vs TRADE LINE", style="bold bright_cyan"),
                       subtitle=wr.caption("blended signal each decision · trades when it crosses buy/sell"),
                       box=box.ROUNDED, border_style="bright_cyan"))
    charts.add_row(*cells)

    # ---- account
    acc = broker.account
    floating = sum(float(getattr(p, "profit", 0) or 0) for p in broker.positions)
    day_total = (broker.pnl_today or 0.0) + floating
    max_loss = float(risk.get("max_daily_loss", 0) or 0)
    acct = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    acct.add_column(style="dim")
    acct.add_column(justify="right")
    acct.add_row("Balance", money(getattr(acc, "balance", None)))
    acct.add_row("Equity", Text(money(getattr(acc, "equity", None)), style="bold"))
    acct.add_row("Free margin", money(getattr(acc, "margin_free", None)))
    acct.add_row("Floating", signed(floating))
    acct.add_row("Realized today", signed(broker.pnl_today))
    acct.add_row("Day total", signed(day_total))
    used = min(1.0, max(0.0, -day_total / max_loss)) if max_loss else 0.0
    loss_bar = tbar(used, 22, "red" if used > 0.6 else ("yellow" if used > 0.3 else "green"))
    eq_series = logs.equity or ([float(acc.equity)] if acc else [])
    eq_shown = wr.ease("equity", float(getattr(acc, "equity", 0) or 0), 0.3)
    acct_panel = Panel(Group(Text(f"${eq_shown:,.2f}", style="bold bright_white", justify="center"),
                             Text(f"day {day_total:+.2f}  ·  floating {floating:+.2f}", justify="center",
                                  style="green" if day_total >= 0 else "red"),
                             Text(f"loss limit used {used * 100:.0f}% of ${max_loss:,.0f}", style="dim"),
                             loss_bar, Text("equity " + sparkline(eq_series, 22), style="cyan")),
                       title=Text("💼 ACCOUNT", style="bold"), box=box.ROUNDED,
                       subtitle=wr.caption("demo · live from Exness"))

    # ---- decisions
    dec = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    for c in ("Symbol", "Spr", "Context", "Council", "PPO", "LSTM", "Blend", "Supervisor", "Age"):
        dec.add_column(c, justify="left" if c in ("Symbol", "Context", "Supervisor") else "right", no_wrap=True)
    for s in symbols:
        t = broker.ticks.get(s)
        spr = f"{(t.ask - t.bid) / ((t.ask + t.bid) / 2) * 1e4:.1f}" if t else "-"
        sg = (logs.last_signal.get(s) or {}).get("payload") or {}
        mm = (logs.last_memory.get(s) or {}).get("payload") or {}
        ctx = (mm.get("context") or {}).get("key", "-")
        blk = logs.last_block.get(s)
        if blk and (not logs.last_signal.get(s) or blk.get("ts", "") >= logs.last_signal[s].get("ts", "")):
            sup = Text("BLOCK " + str((blk.get("payload") or {}).get("reason", ""))[:22], style="yellow")
        else:
            sup = Text("ok", style="green") if sg else Text("waiting", style="dim")
        dec.add_row(s, spr, ctx, signed(sg.get("council")) if sg else Text("-"),
                    signed(sg.get("ppo_exposure")) if sg else Text("-"),
                    signed(sg.get("agi_bias")) if sg else Text("-"),
                    signed(sg.get("exposure")) if sg else Text("-"), sup,
                    age((logs.last_signal.get(s) or {}).get("ts")))

    gauges = Table.grid(expand=True, padding=(0, 1))
    gauges.add_column(no_wrap=True)
    gauges.add_column(no_wrap=True)
    gauges.add_column(no_wrap=True)
    for s in symbols:
        sg = (logs.last_signal.get(s) or {}).get("payload") or {}
        thr = _threshold(cfg, s, sg)
        val = float(sg.get("exposure", 0) or 0) if sg else 0.0
        rawv = float(sg.get("raw_target", 0) or 0) if sg else 0.0
        verdict = ("GO LONG" if val > 0 else "GO SHORT") if abs(val) >= thr and val != 0 else "HOLD"
        gauges.add_row(Text(f"{_nm(s).upper():<8}", style="bold"), gauge(rawv * 1.0, thr, 46),
                       Text(f" {rawv:+.2f} / need ±{thr:.2f}  {verdict}",
                            style="bold green" if verdict == "GO LONG" else ("bold red" if verdict == "GO SHORT" else "dim")))

    # ---- indicator council (multi-timeframe heat grid)
    cvs = {x: ((logs.last_council.get(x) or {}).get("payload") or {}) for x in symbols}
    council_panel = Panel(Group(council_grid(cvs, symbols),
                                Text("██ strong · ▄▄ weak · green up / red down · bright = proven here · ╳╳ ignored (poor record) · ── no view",
                                     style="dim", overflow="ellipsis", no_wrap=True)),
                          title="Indicator council · 12 indicators × 4 timeframes, weighted by track record",
                          box=box.ROUNDED, border_style="cyan")

    # ---- model memory heatmap
    if time.time() - _mem_cache["t"] > 10:
        _mem_cache.update(t=time.time(), v={s: memory_rows(s) for s in symbols})
    mem_tables = []
    for s in symbols:
        rows = _mem_cache["v"].get(s) or []
        cur = ((logs.last_memory.get(s) or {}).get("payload") or {}).get("context", {}).get("key")
        mt = Table(box=None, expand=True, padding=(0, 1), show_header=True, header_style="bold dim",
                   title=Text(s, style="bold"), title_justify="left")
        mt.add_column("context", no_wrap=True, min_width=20, ratio=3)
        for m in MODELS:
            mt.add_column(m, justify="center", no_wrap=True, min_width=10, ratio=2)
        rows = sorted(rows, key=lambda r: (r["ctx"] != cur, r["ctx"] == "*", -max(r[m]["n"] for m in MODELS)))[:5]
        for r in rows:
            label = "ALL" if r["ctx"] == "*" else r["ctx"]
            name = Text(("▶ " if r["ctx"] == cur else "  ") + label, style="bold yellow" if r["ctx"] == cur else "")
            vals = []
            for m in MODELS:
                a = r[m]
                if a["n"] < 1:
                    vals.append(Text("·", style="dim"))
                    continue
                mult = a["mult"]
                style = ("bold white on green" if mult >= 1.2 else "black on green3" if mult > 1.05 else
                         "bold white on red" if mult <= 0.6 else "black on orange3" if mult < 0.95 else "dim")
                hr = "" if a["hit_rate"] is None else f" {a['hit_rate'] * 100:.0f}%"
                vals.append(Text(f"{mult:.2f}×{hr}", style=style))
            mt.add_row(name, *vals)
        if not rows:
            mt.add_row(Text("learning… (starts when the bot runs)", style="dim"), *["·" for _ in MODELS])
        mem_tables.append(mt)
    mem_panel = Panel(Group(*mem_tables, Text("weight multiplier × · hit rate  │ green = works · red = doesn't",
                                              style="dim")), title="Model memory: what works where", box=box.ROUNDED)

    # ---- models / training
    reg = {}
    try:
        reg = json.load(open(REGISTRY))
    except Exception:
        pass
    tr = training_state()
    prog = read_progress()
    tw = 44
    parts = []
    for s in symbols:
        e = (reg.get("symbols") or {}).get(s) or {}
        ch, ca = e.get("champion"), e.get("canary")
        line = Text(f"{s:<9}", style="bold")
        if ch:
            line.append("champion " + os.path.basename(str(ch)), style="green")
        elif ca:
            expl = os.path.exists(os.path.join(str(ca), "exploration.json"))
            line.append(("exploring " if expl else "canary ") + os.path.basename(str(ca)), style="magenta" if expl else "yellow")
        else:
            safe = s.replace("/", "_")
            has_lstm = os.path.exists(os.path.join(ROOT, "models", "per_symbol", f"lstm_{safe}.pt")) or \
                str((_default_meta() or {}).get("symbol")) == s
            line.append("LSTM ready ✓" if has_lstm else "LSTM pending", style="green" if has_lstm else "grey50")
            line.append("  ·  PPO trader: in training (not approved yet)", style="yellow")
        parts.append(line)
    live_prog = (prog and not prog.get("done") and time.time() - float(prog.get("ts", 0)) < 600
                 and bool(tr["running"]))
    if live_prog and prog.get("kind") == "lstm":
        ep, eps = int(prog.get("epoch", 0)), int(prog.get("epochs", 1))
        b, nb = int(prog.get("batch", 0)), max(1, int(prog.get("n_batches", 1)))
        parts.append(Text(f"{spin} LSTM {prog.get('symbol')}  epoch {ep}/{eps}", style="bold cyan"))
        bar = Text("epoch  ")
        bar.append_text(tbar(b / nb, tw, "cyan"))
        bar.append(f" {b / nb * 100:5.1f}%", style="bold")
        parts.append(bar)
        tot = Text("total  ")
        tot.append_text(tbar(((ep - 1) + b / nb) / max(1, eps), tw, "blue"))
        parts.append(tot)
        parts.append(Text(f"batch {b}/{nb} · {prog.get('samples_per_s', 0):,.0f} samples/s · "
                          f"epoch ETA {int(prog.get('epoch_eta_s', 0))}s · loss {prog.get('run_loss', 0):.4f} · "
                          f"acc {prog.get('run_acc', 0):.1f}%", style="cyan"))
        parts.append(Text("live batch loss", style="dim"))
        parts.append(braille_chart(prog.get("recent") or [], tw + 8, 3, color="magenta"))
        hist = prog.get("history") or []
        if hist:
            ht = Table(box=None, padding=(0, 1), show_header=True, header_style="dim")
            for c in ("ep", "loss", "val_loss", "val_acc", ""):
                ht.add_column(c, justify="right")
            for h in hist[-4:]:
                ht.add_row(str(h.get("epoch")), f"{h.get('loss', 0):.4f}", f"{h.get('val_loss', 0):.4f}",
                           f"{h.get('val_acc', 0):.1f}%", Text("★ best" if h.get("best") else "", style="yellow"))
            parts.append(ht)
            parts.append(Text("val_loss " + sparkline([h.get("val_loss", 0) for h in hist], 40), style="green"))
    elif live_prog and prog.get("kind") == "ppo":
        st, tot_ = int(prog.get("step", 0)), max(1, int(prog.get("total", 1)))
        parts.append(Text(f"{spin} PPO {prog.get('symbol')}", style="bold cyan"))
        bar = Text("steps  ")
        bar.append_text(tbar(st / tot_, tw, "cyan"))
        bar.append(f" {st / tot_ * 100:5.1f}%", style="bold")
        parts.append(bar)
        parts.append(Text(f"{st:,}/{tot_:,} · {prog.get('fps', 0):,.0f} steps/s · ETA {int(prog.get('eta_s', 0)) // 60}m"
                          + (f" · ep reward {prog['ep_reward_mean']:.3f}" if prog.get("ep_reward_mean") is not None else ""),
                          style="cyan"))
        parts.append(Text("live step reward", style="dim"))
        parts.append(braille_chart(prog.get("recent") or [], tw + 8, 3, color="magenta"))
    elif tr["running"]:
        parts.append(Text(f"{spin} training: {', '.join(tr['running'])}", style="bold cyan"))
        if tr["lstm"]:
            l = tr["lstm"]
            parts.append(Text(f"LSTM {l['symbol']} epoch {l['epoch']}/{l['total']}", style="cyan"))
            parts.append(tbar(l["epoch"] / max(1, l["total"]), tw))
            parts.append(Text(l["detail"], style="dim", no_wrap=True, overflow="ellipsis"))
        k = (frame // 2) % 30
        parts.append(Text("░" * k + "███" + "░" * (30 - k), style="cyan"))
    else:
        parts.append(Text("training: idle", style="dim"))
    parts.append(Text(tr["last"], style="dim", overflow="ellipsis", no_wrap=True))
    models_panel = Panel(Group(*parts), title=Text("🧠 MODELS & TRAINING", style="bold"), box=box.ROUNDED,
                         subtitle=wr.caption("loss going down = learning · ★ = best version kept"))
    explain_panel = Panel(Group(*explain(prog, tr, reg, symbols, alive, logs)), title="What's happening (plain English)",
                          box=box.ROUNDED, border_style="bright_blue")

    # ---- events
    ev = Table(box=None, expand=True, show_header=False, padding=(0, 1))
    ev.add_column(no_wrap=True, style="dim")
    ev.add_column(no_wrap=True)
    ev.add_column(overflow="ellipsis", no_wrap=True)
    for r in reversed(logs.events[-7:]):
        p = r.get("payload") or {}
        if r.get("event") == "trade_action":
            ev.add_row(age(r.get("ts")), Text("TRADE", style="bold green"),
                       f"{p.get('symbol')} {p.get('request_action')} {p.get('order_type')} "
                       f"{p.get('executed_lots', p.get('volume_lots'))} @ {_px(p.get('entry_price'))} "
                       f"SL {p.get('sl_price')} TP {p.get('tp_price')}")
        else:
            ev.add_row(age(r.get("ts")), Text("BLOCK", style="yellow"),
                       f"{p.get('symbol')} {float(p.get('target_exposure', 0) or 0):+.2f}: {p.get('reason')}")
    if not logs.events:
        ev.add_row("-", "", Text("no trades or risk blocks yet", style="dim"))

    log_lines = [l for l in tail_lines(SERVER_LOG, 200)
                 if l.startswith(dt.date.today().isoformat()) and any(k in l for k in ("ACTION", "DECISION", "WARNING", "ERROR"))]
    if not log_lines:
        log_lines = tail_lines(SERVER_OUT, 5)
    logt = Text(no_wrap=True, overflow="ellipsis")
    for l in log_lines[-5:]:
        st = "red" if "ERROR" in l else ("yellow" if "WARNING" in l else ("bold green" if "ACTION" in l else "dim"))
        logt.append(re.sub(r"^\S+ (\S+)\.\d+ \S+ \| (\w+)\s+\| \S+ - ", r"\1 ", l)[: width - 6] + "\n", style=st)

    # ---- layout
    layout = Layout()
    pulse = ["cyan", "bright_cyan", "bright_white", "bright_cyan", "cyan", "dark_cyan"][frame % 6]
    brief = wr.briefing(frame, logs, cfg, symbols) or [Text("waiting for the first decision…", style="dim")]
    brief_panel = Panel(Group(*brief, *([] if alive else [Text("🔴 The bot is not running. Type 'exness' to start it.", style="red")])),
                        title=Text("🗣  BRIEFING · what the bot sees, why, and what's next", style="bold bright_cyan"),
                        box=box.ROUNDED, border_style="bright_cyan")
    layout.split_column(
        Layout(Panel(header, box=box.HEAVY, border_style=(pulse if alive else "red")), size=4),
        Layout(banner, size=1),
        Layout(charts, size=9),
        Layout(name="mid", size=2 * len(symbols) + 3),
        Layout(name="low"),
        Layout(wr.footer(frame, logs, width), size=2),
    )
    layout["mid"].split_row(Layout(acct_panel, size=32),
                            Layout(wr.pipeline(frame, logs, cfg, symbols, width), ratio=4))
    layout["low"].split_row(Layout(name="left", ratio=4), Layout(name="right", ratio=5))
    layout["left"].split_column(Layout(brief_panel, ratio=1), Layout(council_panel, size=19))
    train_panel = wr.training_panel(frame, prog, bool(tr.get("running")), tail_lines(CYCLE_LOG, 400), symbols,
                                    cfg, sparkline)
    layout["right"].split_column(Layout(train_panel, size=26),
                                 Layout(Panel(ev, title="Trades / risk blocks", box=box.ROUNDED,
                                              subtitle=wr.caption("every order and every time the risk guard said no"))),
                                 Layout(Panel(logt, title="Server log", box=box.ROUNDED,
                                              subtitle=wr.caption("raw engine output")), size=6))
    return layout


def main():
    cfg = _cfg()
    symbols = [str(s) for s in ((cfg.get("trading") or {}).get("symbols") or ["BTCUSDm", "XAUUSDm"])]
    console = Console()
    broker = Broker(symbols)
    logs = Logs()
    frame = 0
    with Live(console=console, screen=True, auto_refresh=False, transient=True,
              vertical_overflow="crop") as live:
        try:
            while True:
                t0 = time.time()
                logs.refresh()
                live.update(render(frame, broker, logs, cfg, symbols, console.size.width), refresh=True)
                frame += 1
                time.sleep(max(0.05, 0.25 - (time.time() - t0)))
        except KeyboardInterrupt:
            pass


if __name__ == "__main__":
    main()
