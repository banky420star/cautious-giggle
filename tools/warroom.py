"""War-room widgets for the monitor TUI: decision pipeline, symbol briefings, glossary ticker.

Everything here is pure rendering from the audit payloads the bot already writes; nothing talks
to the broker. Animation state (eased needles, typewriter text) lives in module-level dicts so
it survives between frames.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import time

from rich import box
from rich.console import Group
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ----------------------------------------------------------------------------- animation state
_EASE: dict[str, float] = {}
_TYPE: dict[str, tuple[str, int]] = {}


def ease(key: str, target: float, rate: float = 0.22) -> float:
    """Glide a displayed number toward its real value instead of jumping."""
    cur = _EASE.get(key, target)
    cur += (target - cur) * rate
    if abs(target - cur) < 1e-4:
        cur = target
    _EASE[key] = cur
    return cur


def typewriter(key: str, msg: str, frame: int, cps: int = 9) -> tuple[str, bool]:
    """Reveal new/changed text a few characters per frame. Returns (visible, still_typing)."""
    old = _TYPE.get(key)
    if not old or old[0] != msg:
        _TYPE[key] = (msg, frame)
        old = _TYPE[key]
    n = (frame - old[1]) * cps
    return (msg, False) if n >= len(msg) else (msg[: max(0, n)], True)


def _nm(s) -> str:
    s = str(s or "")
    return {"BTCUSDm": "Bitcoin", "XAUUSDm": "Gold"}.get(s, s)


def _age_s(ts: str | None) -> float:
    try:
        return (dt.datetime.now(dt.timezone.utc) - dt.datetime.fromisoformat(ts)).total_seconds()
    except Exception:
        return 1e9


# ----------------------------------------------------------------------------- decision pipeline
def _stage(label: str, value: str, style: str, note: str = "") -> Text:
    t = Text()
    t.append(f" {label} ", style="bold black on grey70")
    t.append(f" {value} ", style=style)
    if note:
        t.append(note, style="dim")
    return t


def _arrow(frame: int, idx: int, active: bool) -> Text:
    """Connector with a data packet that travels left -> right through the stages in turn."""
    w = 3
    t = Text(" ")
    pos = (frame % 25) - idx * 3  # packet visits each connector in sequence
    for i in range(w):
        if active and pos == i:
            t.append("●", style="bold bright_yellow")
        else:
            t.append("━", style="cyan" if active else "grey35")
    t.append("▶ ", style="bold cyan" if active else "grey35")
    return t


def _needle(value: float, thr: float, width: int) -> Text:
    v = max(-1.0, min(1.0, value))
    pos = int(round((v + 1) / 2 * (width - 1)))
    mid = (width - 1) // 2
    tl, tr = (int(round((x + 1) / 2 * (width - 1))) for x in (-thr, thr))
    out = Text()
    for i in range(width):
        if i == pos:
            out.append("◆", style="bold bright_white")
        elif i in (tl, tr):
            out.append("┃", style="bold yellow")
        elif i == mid:
            out.append("│", style="grey62")
        elif min(mid, pos) < i < max(mid, pos):
            strong = abs(v) >= thr
            col = "green" if v > 0 else "red"
            out.append("█" if strong else "▆", style=f"bold {col}" if strong else col)
        elif i < tl:
            out.append("·", style="red3")
        elif i > tr:
            out.append("·", style="green3")
        else:
            out.append("─", style="grey30")
    return out


def pipeline(frame: int, logs, cfg: dict, symbols: list[str], width: int) -> Panel:
    trading = (cfg.get("trading") or {})
    conf_min = float(trading.get("confidence_threshold", 0.85) or 0.85)
    rows = []
    for s in symbols:
        sig_ev = logs.last_signal.get(s) or {}
        sg = sig_ev.get("payload") or {}
        cv = (logs.last_council.get(s) or {}).get("payload") or {}
        prof = sg.get("decision_profile") if isinstance(sg.get("decision_profile"), dict) else {}
        thr = float(prof.get("min_trade_threshold", 0.15) or 0.15)
        cw = float(prof.get("council_weight", 0.35) or 0.0)
        pw = float(prof.get("ppo_weight", 0.6) or 0.0)
        fresh = _age_s(sig_ev.get("ts")) < 330
        head = Text(f"{_nm(s).upper():<8}", style="bold bright_white")
        if not sg:
            rows.append(Group(head + Text("  waiting for the first decision of this run…", style="dim italic")))
            continue

        up, down = int(cv.get("up", 0) or 0), int(cv.get("down", 0) or 0)
        n_vote = len(cv.get("votes") or {}) or 12
        n_ign = len(cv.get("ignored") or [])
        council = float(sg.get("council", 0) or 0)
        lstm = float(sg.get("agi_bias", 0) or 0)
        conf = float(sg.get("confidence", 0) or 0)
        ppo = float(sg.get("ppo_exposure", 0) or 0)
        raw = float(sg.get("raw_target", 0) or 0)
        risk = float(sg.get("risk_scalar", 1) or 1)
        final = float(sg.get("exposure", 0) or 0)
        lstm_used = conf >= conf_min and abs(lstm) > 0
        blk = logs.last_block.get(s)
        blocked = bool(blk and blk.get("ts", "") >= sig_ev.get("ts", ""))

        def col(v):
            return "bold green" if v > 0.05 else ("bold red" if v < -0.05 else "bold grey70")

        line = Text(no_wrap=True, overflow="ellipsis")
        line.append_text(head)
        if blocked:
            verdict, vstyle = "BLOCKED", "bold black on yellow"
        elif final > 0:
            verdict, vstyle = "▲ LONG", "bold black on green" if (frame // 3) % 2 == 0 else "bold green on black"
        elif final < 0:
            verdict, vstyle = "▼ SHORT", "bold white on red" if (frame // 3) % 2 == 0 else "bold red on black"
        else:
            verdict, vstyle = "■ HOLD", "bold black on grey62"
        line.append(f" {verdict:<8}", style=vstyle)
        line.append(" ◀ ", style="bold cyan" if fresh else "grey35")
        line.append_text(_stage("48 VOTES", f"{up}▲ {down}▼", "bold", f" {n_vote - n_ign} trusted"))
        line.append_text(_arrow(frame, 0, fresh))
        line.append_text(_stage("COUNCIL", f"{council:+.2f}", col(council), f" ×{cw:.2f}"))
        line.append_text(_arrow(frame, 1, fresh))
        line.append_text(_stage("LSTM", f"{'BUY' if lstm > 0 else ('SELL' if lstm < 0 else 'HOLD')} {conf * 100:.0f}%",
                                col(lstm) if lstm_used else "grey50",
                                "" if lstm_used else f" <{conf_min * 100:.0f}% off"))
        line.append_text(_arrow(frame, 2, fresh))
        line.append_text(_stage("PPO", f"{ppo:+.2f}" if ppo else "not trained", col(ppo) if ppo else "grey50",
                                f" ×{pw:.2f}" if ppo else ""))
        line.append_text(_arrow(frame, 3, fresh))
        line.append_text(_stage("Σ BLEND", f"{raw:+.2f}", col(raw)))

        shown = ease(f"needle:{s}", raw * risk)
        nl = Text(" " * 9, no_wrap=True, overflow="ellipsis")
        nl.append("SELL ", style="red")
        nl.append_text(_needle(shown, thr, min(60, max(30, width // 3))))
        nl.append(" BUY", style="green")
        gap = thr - abs(raw * risk)
        if blocked:
            why = f"  risk guard stopped it: {str((blk.get('payload') or {}).get('reason', ''))[:60]}"
        elif final != 0:
            why = f"  signal {raw * risk:+.2f} cleared the ±{thr:.2f} bar → position sized {abs(final):.2f} of max"
        else:
            why = f"  signal {raw * risk:+.2f}; needs ±{thr:.2f} to trade ({gap:.2f} short) → staying flat"
        nl.append(why, style="dim")
        rows.append(Group(line, nl))
    cap = Text("How a trade is made: indicators vote → council weighs them by track record → models add their view → "
               "if the blended signal crosses the yellow bars ┃ the bot trades. The ● shows data flowing each decision.",
               style="dim italic", overflow="ellipsis", no_wrap=True)
    return Panel(Group(*rows, cap), title=Text("◆ DECISION PIPELINE · live", style="bold magenta"),
                 box=box.ROUNDED, border_style="magenta")


# ----------------------------------------------------------------------------- per-symbol briefing
def briefing(frame: int, logs, cfg: dict, symbols: list[str]) -> list[Text]:
    """NOW / WHY / NEXT in plain English for each symbol, typed out when it changes."""
    from Python.indicator_council import describe

    conf_min = float(((cfg.get("trading") or {}).get("confidence_threshold", 0.85)) or 0.85)
    out: list[Text] = []
    for s in symbols:
        sg = (logs.last_signal.get(s) or {}).get("payload") or {}
        cv = (logs.last_council.get(s) or {}).get("payload") or {}
        mm = (logs.last_memory.get(s) or {}).get("payload") or {}
        if not sg:
            continue
        ctx = ((mm.get("context") or {}).get("key") or "").split("|")
        trend = {"up": "trending up", "down": "trending down", "range": "moving sideways"}.get(ctx[0] if ctx else "", "")
        vol = {"high": "choppy (high volatility)", "low": "quiet", "normal": "normal volatility"}.get(
            ctx[1] if len(ctx) > 1 else "", "")
        sess = {"asia": "Asian session", "europe": "London session", "us": "New York session",
                "late": "late session"}.get(ctx[2] if len(ctx) > 2 else "", "")
        final = float(sg.get("exposure", 0) or 0)
        raw = float(sg.get("raw_target", 0) or 0) * float(sg.get("risk_scalar", 1) or 1)
        prof = sg.get("decision_profile") if isinstance(sg.get("decision_profile"), dict) else {}
        thr = float(prof.get("min_trade_threshold", 0.15) or 0.15)

        votes = cv.get("votes") or {}
        assess = cv.get("assess") or {}
        strong = sorted(((k, v, (assess.get(k) or {}).get("mult", 1.0)) for k, v in votes.items()
                         if abs(v) > 0.3 and (assess.get(k) or {}).get("verdict") == "works"),
                        key=lambda x: -abs(x[1]) * x[2])
        pro = [describe(k) for k, v, _ in strong if (v > 0) == (raw >= 0)][:2]
        con = [describe(k) for k, v, _ in strong if (v > 0) != (raw >= 0)][:2]

        now = (f"{_nm(s)} is {trend}, {vol}, {sess}. "
               + (f"Holding a {'LONG' if final > 0 else 'SHORT'} position." if final else "No position."))
        why = (f"{cv.get('up', 0)} indicator signals point up and {cv.get('down', 0)} down across 5m–4h. ")
        if pro:
            why += f"Best-proven support: {', '.join(pro)}. "
        if con:
            why += f"Pushing the other way: {', '.join(con)}. "
        conf = float(sg.get("confidence", 0) or 0)
        if conf and conf < conf_min:
            why += f"The LSTM leans {sg.get('regime', '')} at {conf * 100:.0f}% confidence but needs {conf_min * 100:.0f}% to count."
        if final:
            nxt = "Stop-loss and take-profit are set; it re-checks every 5 minutes and exits if the signal flips or fades."
        elif abs(raw) >= thr * 0.6:
            nxt = f"Close to a trade: one more aligned timeframe would push it past ±{thr:.2f}."
        else:
            nxt = "Waiting: signals disagree, so trading now would mostly pay the spread."

        title = Text(f"▌{_nm(s).upper()}", style="bold bright_cyan")
        out.append(title)
        for tag, msg, st in (("NOW ", now, "bold"), ("WHY ", why, ""), ("NEXT", nxt, "cyan")):
            vis, typing = typewriter(f"{s}:{tag}", msg, frame)
            t = Text(f" {tag} ", style="bold black on cyan" if tag == "NOW " else "bold black on grey62")
            t.append(" " + vis, style=st)
            if typing:
                t.append("▌", style="bold bright_white" if frame % 2 == 0 else "")
            out.append(t)
    return out


# ----------------------------------------------------------------------------- glossary ticker
GLOSSARY = [
    ("Spread", "the gap between buy and sell price; every trade starts this far in the red."),
    ("Council", "12 classic indicators × 4 timeframes voting; each vote weighted by how well it has worked lately."),
    ("Timeframes", "5m reacts fast, 4h shows the big trend; agreement across them is what makes a signal strong."),
    ("Track record", "after every bar the memory scores who was right; proven voters get up to ×1.5, losers ×0.25 or ignored."),
    ("LSTM", "a neural net that reads 60 bars × 150 indicators and predicts up / down / flat for the next bar."),
    ("PPO", "a reinforcement-learning trader practised in a simulator with real costs; it decides the position size."),
    ("Threshold ┃", "the yellow bars: the blended signal must reach them before the bot risks money."),
    ("Stop-loss", "a pre-set exit that caps the loss on a trade; sized so one stop costs ≤ ~1.5% of the account."),
    ("Daily loss limit", "if the day's loss reaches the limit the bot halts new trades until tomorrow."),
    ("val_loss", "error on data the model never trained on; falling = genuinely learning, rising = memorising."),
    ("Demo exploration", "a model that failed its exam may trade half-size on demo only, to collect real evidence."),
    ("Context", "trend | volatility | session; the memory keeps a separate scorecard for each kind of market."),
]


def footer(frame: int, logs, width: int) -> Group:
    i = int(time.time() // 8) % len(GLOSSARY)
    term, desc = GLOSSARY[i]
    phase = time.time() % 8
    st = "grey50" if phase < 0.5 else ("" if phase < 7.5 else "grey50")
    g = Text(" ⓘ ", style="bold black on bright_cyan")
    g.append(f" {term}: ", style="bold bright_cyan")
    g.append(desc, style=st)
    g.append(f"   ({i + 1}/{len(GLOSSARY)})", style="grey42")

    items = []
    for ev in list(logs.events)[-8:]:
        p = ev.get("payload") or {}
        if ev.get("event") == "trade_action":
            items.append((f"TRADE {p.get('symbol')} {p.get('request_action')} {p.get('order_type', '')} "
                          f"{p.get('executed_lots', p.get('volume_lots'))} lots", "bold green"))
        elif ev.get("event") == "risk_supervisor_block":
            items.append((f"BLOCK {p.get('symbol')} {str(p.get('reason', ''))[:40]}", "yellow"))
    for s, ev in logs.last_signal.items():
        p = ev.get("payload") or {}
        items.append((f"{_nm(s).upper()} signal {float(p.get('raw_target', 0) or 0):+.2f} → "
                      f"{'TRADE' if p.get('exposure') else 'hold'}", "cyan"))
    tape = Text(no_wrap=True, overflow="crop")
    if items:
        seq = Text()
        for msg, st2 in items:
            seq.append(" ◆ ", style="magenta")
            seq.append(msg, style=st2)
        plain = seq.plain + "   "
        off = (frame // 1) % max(1, len(plain))
        rot = seq[off:] + Text("   ") + seq[:off]
        tape.append(" ⏵ LIVE ", style="bold white on red" if (frame // 4) % 2 == 0 else "bold red on black")
        tape.append_text(rot)
    return Group(g, tape)


def caption(text: str) -> Text:
    return Text(text, style="dim italic", overflow="ellipsis", no_wrap=True)


# ----------------------------------------------------------------------------- training mission control
MISSIONS = {
    "lstm": {
        "name": "LSTM pattern reader",
        "learn": "which patterns in the last 60 five-minute bars (150 indicators each) come just before an UP, DOWN "
                 "or FLAT next bar.",
        "goal": "make its error on bars it has never seen (val_loss) as low as possible, so its UP/DOWN calls are "
                "right more often than chance.",
        "test": "after every pass it is quizzed on the newest 15% of history it never trained on; the best pass is "
                "kept and training stops early if 4 passes in a row don't improve.",
        "role": "Live role: a helper vote. It only counts when ≥85% sure.",
    },
    "dreamer": {
        "name": "Dreamer world model",
        "learn": "a mini simulator of how this market moves (a 'world model'), then practises trading inside its "
                 "own imagined futures.",
        "goal": "predict upcoming bars well (world-model error falls) and earn a positive reward in its imagination.",
        "test": "no vote yet (weight 0). It is experimental and only gets a say after proving itself in the exam.",
        "role": "Live role: none yet. It's a research model.",
    },
    "ppo": {
        "name": "PPO trader",
        "learn": "when to be long, short or flat and how big, by trial and error over past data, paying real "
                 "spread, stop-loss and take-profit costs.",
        "goal": "the highest profit after costs with small drawdowns. Its practice score should climb above zero.",
        "test": "the exam below. It is replayed on the last 15% of history it never saw and must pass every gate.",
        "role": "Live role: the main decision-maker (60% of the vote) once it passes.",
    },
    "exam": {
        "name": "Exam (out-of-sample backtest)",
        "learn": "nothing. This is the test.",
        "goal": "prove the new PPO trader would have made money on unseen data, safely, and better than simple rules.",
        "test": "the gates below. Pass → it starts trading on demo (canary). Fail → it may trade half-size on demo "
                "only, to collect real evidence.",
        "role": "",
    },
}


STAGE_KEYS = ("Cycle start", " start | symbol=", " complete | symbol=", "failed", "Evaluate ", "Canary set for",
              "Candidate blocked for", "Skipping", "PPO inline eval", "Cycle complete", "Cycle done")


def stage_lines(path: str, max_bytes: int = 50_000_000) -> list[str]:
    """Only the stage-marker lines of the cycle log, from the whole file. Reading just the tail
    misses them once PPO starts printing its long stats tables."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            size = f.tell()
            f.seek(max(0, size - max_bytes))
            data = f.read().decode("utf-8", "replace")
    except OSError:
        return []
    return [l for l in data.splitlines() if any(k in l for k in STAGE_KEYS)]


def cycle_symbols(default: list[str]) -> list[str]:
    """Markets the running training cycle is working on (graduation trainees, else the defaults)."""
    try:
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        with open(os.path.join(root, "logs", "graduation.json"), encoding="utf-8") as f:
            st = json.load(f).get("symbols") or {}
        tr = [s for s, v in st.items() if v.get("status") == "training"]
        return tr or list(default)
    except Exception:
        return list(default)


def cycle_stages(lines: list[str], symbols: list[str]) -> list[dict]:
    """Ordered roadmap of the champion cycle with status from its log."""
    st = []
    for kind, label in (("lstm", "LSTM"), ("dreamer", "Dreamer"), ("ppo", "PPO")):
        for s in symbols:
            st.append({"kind": kind, "symbol": s, "label": f"{label} {s[:3]}", "status": "pending"})
    for s in symbols:
        st.append({"kind": "exam", "symbol": s, "label": f"Exam {s[:3]}", "status": "pending"})
    # only the most recent cycle
    start = 0
    for i, l in enumerate(lines):
        if "Cycle start" in l:
            start = i
    for l in lines[start:]:
        for row in st:
            s = row["symbol"]
            tag = {"lstm": "LSTM", "dreamer": "Dreamer", "ppo": "PPO"}.get(row["kind"])
            if tag and f"{tag} start | symbol={s}" in l:
                row["status"] = "running"
            elif tag and f"{tag} complete | symbol={s}" in l:
                row["status"] = "done"
            elif tag and ("failed" in l and f"symbol={s}" in l and tag.lower() in l.lower()):
                row["status"] = "failed"
            elif row["kind"] == "exam" and f"Evaluate {s} " in l:
                row["status"] = "running"
            elif row["kind"] == "exam" and f"Canary set for {s}" in l:
                row["status"] = "passed"
            elif row["kind"] == "exam" and (f"Candidate blocked for {s}" in l or f"Skipping evaluation for {s}" in l):
                row["status"] = "failed"
    # stages skipped because their model already exists count as done
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    for row in st:
        if row["status"] != "pending":
            continue
        f = {"lstm": os.path.join(root, "models", "per_symbol", f"lstm_{row['symbol']}.pt"),
             "dreamer": os.path.join(root, "models", "dreamer", f"dreamer_{row['symbol']}.pt")}.get(row["kind"])
        if f and os.path.exists(f) and any(("PPO start" in l or "Evaluate" in l) for l in lines[start:]):
            row["status"] = "done"
    return st


def _bar(frac: float, width: int, style: str = "bright_cyan", frame: int = 0) -> Text:
    frac = max(0.0, min(1.0, frac))
    full = int(frac * width)
    t = Text()
    for i in range(full):
        shimmer = (frame - i) % 24 == 0
        t.append("█", style="bold bright_white" if shimmer else style)
    if full < width:
        t.append("▌" if (frac * width - full) > 0.5 else " ", style=style)
        t.append("░" * (width - full - 1), style="grey30")
    return t


def training_panel(frame: int, prog: dict, running: bool, cycle_lines: list[str], symbols: list[str],
                   cfg: dict, spark) -> Panel:
    spin = "◐◓◑◒"[frame % 4]
    stages = cycle_stages(cycle_lines, symbols)
    road = Text(no_wrap=True, overflow="ellipsis")
    icon = {"done": ("✓", "green"), "passed": ("✓", "bold green"), "running": (spin, "bold bright_yellow"),
            "failed": ("✗", "red"), "pending": ("○", "grey42")}
    for i, row in enumerate(stages):
        ic, stl = icon[row["status"]]
        road.append(f"{ic} {row['label']}", style=stl if row["status"] != "running" else "bold black on bright_yellow")
        if i < len(stages) - 1:
            road.append(" › ", style="grey42")
    cur = next((r for r in stages if r["status"] == "running"), None)
    out = [road, Text("")]

    if not running and not cur:
        out.append(Text("No training running. Type 'exness train' to start a new learning cycle.", style="dim"))
    kind = (cur or {}).get("kind")
    if cur:
        m = MISSIONS[kind]
        out.append(Text(f"NOW TRAINING · {m['name']} · {_nm(cur['symbol'])}", style="bold bright_yellow"))
        for tag, key, st in (("LEARNING", "learn", ""), ("GOAL    ", "goal", "green"), ("TESTED  ", "test", "cyan")):
            t = Text(f" {tag} ", style="bold black on grey70")
            t.append(" " + m[key], style=st)
            out.append(t)
        if m["role"]:
            out.append(Text(" " + m["role"], style="dim italic"))
        out.append(Text(""))

    live = prog and not prog.get("done") and time.time() - float(prog.get("ts", 0) or 0) < 600
    if cur and live and prog.get("kind") == kind and prog.get("symbol") in (cur["symbol"], None) or \
            (cur and live and prog.get("kind") == kind):
        out.extend(_progress_lines(frame, prog, spark))
    elif cur and kind != "exam":
        started = _stage_start(cycle_lines, cur)
        out.append(Text(f" {spin} working… started {started}. Live numbers appear from the next run of this stage.",
                        style="dim"))
    out.append(Text(""))
    out.extend(_exam_lines(cfg, stages, symbols))
    return Panel(Group(*out), title=Text("🧠 TRAINING MISSION CONTROL", style="bold"),
                 subtitle=caption("what each model learns · what it aims for · how it's tested"),
                 box=box.ROUNDED, border_style="yellow" if cur else "grey50")


def _stage_start(lines, cur) -> str:
    tag = {"lstm": "LSTM", "dreamer": "Dreamer", "ppo": "PPO", "exam": "Evaluate"}[cur["kind"]]
    for l in reversed(lines):
        if tag in l and cur["symbol"] in l:
            try:
                t = dt.datetime.strptime(l[:19], "%Y-%m-%d %H:%M:%S")
                return f"{int((dt.datetime.now() - t).total_seconds() // 60)} min ago"
            except Exception:
                break
    return "recently"


def _progress_lines(frame: int, prog: dict, spark) -> list[Text]:
    out = []
    k = prog.get("kind")
    el = time.time() - float(prog.get("started", time.time()))
    if k == "lstm":
        ep, eps = int(prog.get("epoch", 0)), max(1, int(prog.get("epochs", 1)))
        b, nb = int(prog.get("batch", 0)), max(1, int(prog.get("n_batches", 1)))
        frac = ((ep - 1) + b / nb) / eps
        t = Text(f" pass {ep}/{eps} ")
        t.append_text(_bar(frac, 30, frame=frame))
        t.append(f" {frac * 100:.0f}%", style="bold")
        out.append(t)
        hist = prog.get("history") or []
        if hist:
            v0, vb = float(hist[0]["val_loss"]), min(float(h["val_loss"]) for h in hist)
            imp = (v0 - vb) / v0 * 100 if v0 else 0
            sc = Text(" SCORECARD ", style="bold black on cyan")
            sc.append(f"  unseen-data error {v0:.3f} → best {vb:.3f} ", style="bold")
            sc.append(f"({imp:.1f}% better)", style="green" if imp > 0 else "red")
            out.append(sc)
            out.append(Text(f"  error per pass (lower = better) {spark([float(h['val_loss']) for h in hist], 30)}",
                            style="cyan"))
            stale = 0
            for h in reversed(hist):
                if h.get("best"):
                    break
                stale += 1
            pat = int(prog.get("patience", 4))
            verdict = ("✅ still improving" if stale == 0 else f"⏳ no improvement for {stale} pass(es); "
                                                              f"stops after {pat}")
            out.append(Text("  " + verdict, style="green" if stale == 0 else "yellow"))
    elif k in ("ppo", "dreamer"):
        st, tot = int(prog.get("step", 0)), max(1, int(prog.get("total", 1)))
        frac = min(1.0, st / tot)
        eta = el / frac - el if frac > 0.01 else None
        t = Text(f" {'practice steps' if k == 'ppo' else 'imagination steps'} {st:,}/{tot:,} ")
        t.append_text(_bar(frac, 26, frame=frame))
        t.append(f" {frac * 100:.0f}%", style="bold")
        if eta:
            t.append(f"  ~{int(eta // 60)}m left", style="dim")
        out.append(t)
        r = prog.get("ep_reward_mean")
        if k == "dreamer":
            w0, w = prog.get("world_model_loss_start"), prog.get("world_model_loss")
            if w is not None and w0:
                sc = Text(" SCORECARD ", style="bold black on cyan")
                imp = (float(w0) - float(w)) / float(w0) * 100
                sc.append(f"  world-model error {float(w0):.3f} → {float(w):.3f} ", style="bold")
                sc.append(f"({imp:.0f}% better at predicting the market)", style="green" if imp > 0 else "red")
                out.append(sc)
        if r is not None:
            sc = Text(" PRACTICE SCORE ", style="bold black on cyan")
            sc.append(f"  {float(r):+.3f} per run ", style="bold green" if r > 0 else "bold yellow")
            sc.append("→ making money after costs in practice" if r > 0 else "→ still losing to costs; keeps learning",
                      style="dim")
            out.append(sc)
        rec = prog.get("recent") or []
        if rec:
            out.append(Text(f"  trend {spark(rec, 34)}", style="cyan"))
    return out


def _exam_lines(cfg: dict, stages: list[dict], symbols: list[str]) -> list[Text]:
    ev = (cfg.get("evaluation") or {})
    gates = [
        ("Profit after costs", f"> {float(ev.get('min_return', 0)) * 100:.1f}%"),
        ("Max drawdown", f"≤ {float(ev.get('max_drawdown', 0.10)) * 100:.0f}%"),
        ("Sharpe (return per risk)", f"≥ {float(ev.get('min_sharpe', 0.5)):.1f}"),
        ("Beats EMA-cross & buy-and-hold", "required" if ev.get("require_beat_baseline", True) else "optional"),
        ("Only unseen data", "last 15% of history"),
    ]
    t = Table(box=None, padding=(0, 1), expand=True, show_header=True, header_style="bold")
    t.add_column("EXAM GATES the PPO trader must pass", ratio=3)
    t.add_column("target", ratio=2)
    for s in symbols:
        t.add_column(_nm(s), justify="center", ratio=1)
    res = {r["symbol"]: r["status"] for r in stages if r["kind"] == "exam"}
    for name, target in gates:
        row = [Text(name), Text(target, style="cyan")]
        for s in symbols:
            r = res.get(s, "pending")
            row.append(Text("✓" if r == "passed" else ("✗" if r == "failed" else ("…" if r == "running" else "○")),
                            style={"passed": "bold green", "failed": "red", "running": "yellow"}.get(r, "grey42")))
        t.add_row(*row)
    return [t]


# ----------------------------------------------------------------------------- signal history
def signal_history(hist: dict, symbols: list[str], chart, width: int) -> Group:
    parts = []
    for s in symbols:
        h = hist.get(s) or []
        vals = [x[1] for x in h][-width * 2:]
        thr = h[-1][3] if h else 0.15
        trades = sum(1 for x in h[-60:] if x[2])
        last = vals[-1] if vals else 0.0
        head = Text(f"{_nm(s).upper():<8}", style="bold", no_wrap=True, overflow="ellipsis")
        head.append(f" now {last:+.2f} ", style="bold green" if last >= thr else ("bold red" if last <= -thr else "bold"))
        head.append(f"· needs ±{thr:.2f} · {trades} trade calls", style="dim")
        parts.append(head)
        if len(vals) >= 2:
            ct = chart(vals, width, 2, {"BUY": (thr, "green"), "SELL": (-thr, "red")}, color="bright_cyan")
            ct.rstrip()
            parts.append(ct)
        else:
            parts.append(Text("  collecting decisions…\n", style="dim"))
    return Group(*parts)


# rough minutes per stage on the M4, used to weight the overall bar by time instead of count
STAGE_MINUTES = {"lstm": 9, "dreamer": 25, "ppo": 45, "exam": 5}


def training_summary(prog: dict, stages: list[dict]) -> dict:
    """Current stage progress (own %, steps, ETA, phase) and a time-weighted overall %."""
    cur = next((r for r in stages if r["status"] == "running"), None)
    live = bool(prog and cur and not prog.get("done") and prog.get("kind") == cur["kind"]
                and time.time() - float(prog.get("ts", 0) or 0) < 600)
    stage_frac, step, total, eta, unit = None, None, None, None, None
    if live:
        el = max(1.0, float(prog.get("ts", time.time())) - float(prog.get("started", time.time())))
        if prog.get("total"):
            step, total, unit = int(prog.get("step", 0)), max(1, int(prog["total"])), "steps"
        elif prog.get("epochs"):
            ep, eps = int(prog.get("epoch", 0)), max(1, int(prog["epochs"]))
            b, nb = int(prog.get("batch", 0)), max(1, int(prog.get("n_batches", 1)))
            step, total, unit = ep, eps, "passes"
            stage_frac = ((ep - 1) + b / nb) / eps
        if stage_frac is None and total:
            stage_frac = step / total
        if stage_frac is not None and stage_frac >= 1.0:
            # PPO trains in whole 4,096-step batches, so it overshoots the target slightly,
            # then saves and evaluates: show 100% and "finishing up" instead of 102%
            stage_frac, step = 1.0, total
            if live:
                prog = dict(prog, phase="finishing")
        if stage_frac and 0.01 < stage_frac < 1.0:
            eta = el / stage_frac - el
        elif stage_frac == 1.0:
            eta = 0
    w = [STAGE_MINUTES.get(r["kind"], 10) for r in stages]
    done_w = sum(wi for wi, r in zip(w, stages) if r["status"] in ("done", "passed", "failed"))
    cur_w = STAGE_MINUTES.get(cur["kind"], 10) * (stage_frac or 0) if cur else 0
    overall = (done_w + cur_w) / max(1, sum(w))
    left_w = sum(wi for wi, r in zip(w, stages) if r["status"] == "pending")
    return {"stage_progress": stage_frac, "step": step, "total": total, "unit": unit,
            "stage_eta_s": eta, "phase": (prog or {}).get("phase") if live else None, "overall": overall,
            "cycle_eta_s": ((eta or 0) + left_w * 60) if cur else None}
