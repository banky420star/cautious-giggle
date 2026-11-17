"""War-room widgets for the monitor TUI: decision pipeline, symbol briefings, glossary ticker.

Everything here is pure rendering from the audit payloads the bot already writes; nothing talks
to the broker. Animation state (eased needles, typewriter text) lives in module-level dicts so
it survives between frames.
"""
from __future__ import annotations

import datetime as dt
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
