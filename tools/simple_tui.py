"""Simple dashboard: what is the bot doing, is it making money, is it getting better.

Run:  exness          (this view)
      exness pro      (the detailed war-room view, tools/monitor_tui.py)
"""
from __future__ import annotations

import datetime as dt
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from rich import box  # noqa: E402
from rich.align import Align  # noqa: E402
from rich.console import Console, Group  # noqa: E402
from rich.layout import Layout  # noqa: E402
from rich.live import Live  # noqa: E402
from rich.panel import Panel  # noqa: E402
from rich.table import Table  # noqa: E402
from rich.text import Text  # noqa: E402

import monitor_tui as m  # noqa: E402
import warroom as wr  # noqa: E402

NAMES = {"BTCUSDm": "BITCOIN", "XAUUSDm": "GOLD", "EURUSDm": "EUR/USD", "USDJPYm": "USD/JPY",
         "XAGUSDm": "SILVER", "ETHUSDm": "ETHER", "USOILm": "OIL"}
PLAY = {"fade": "fade the sweep", "breakout": "ride the breakout"}
PHASE = {"scheduled": ("scheduled", "grey62"), "prep": ("PREP · range marked, normal trades paused", "bold yellow"),
         "armed": ("ARMED · watching for the sweep", "bold yellow"), "in_trade": ("IN TRADE", "bold green"),
         "review": ("checking result", "cyan"), "done": ("done", "grey50")}


def _json(path, default):
    import json
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def news_panel(frame) -> Panel:
    st = _json(os.path.join(m.LOG_DIR, "news_state.json"), {})
    evs = [e for e in st.get("events", []) if e.get("phase") != "done"]
    body = []
    if not st.get("enabled"):
        body.append(Text("News trading is off.", style="dim"))
    elif not evs:
        body.append(Text("No big US release in the next 36 h.", style="dim"))
    groups = {}
    for e in evs:
        groups.setdefault((e["t0"], e["title"]), []).append(e)
    for (t0, title), rows in sorted(groups.items())[:2]:
        secs = t0 - time.time()
        when = (f"in {secs / 3600:.1f} h" if secs > 3600 else f"in {secs / 60:.0f} min") if secs > 0 else f"{-secs / 60:.0f} min ago"
        ph, sty = PHASE.get(rows[0]["phase"], (rows[0]["phase"], ""))
        live = rows[0]["phase"] in ("prep", "armed", "in_trade")
        hdr = Text("● " if live and frame % 4 < 2 else "  ", style="bold yellow")
        hdr.append(title, style="bold")
        hdr.append(f"  {when}  ", style="bright_cyan")
        hdr.append(ph, style=sty)
        body.append(hdr)
        for e in sorted(rows, key=lambda r: r["sym"]):
            pretty = {"BTCUSDm": "Bitcoin", "XAUUSDm": "Gold", "EURUSDm": "EUR/USD", "USDJPYm": "USD/JPY",
                      "XAGUSDm": "Silver", "ETHUSDm": "Ether", "USOILm": "Oil"}.get(e["sym"], e["sym"])
            ln = Text(f"   {pretty:9s}", style="bold")
            ln.append(f" {PLAY.get(e.get('plan'), 'stand aside'):18s}", style="green" if e.get("plan") else "dim")
            w = str(e.get("why") or "")
            half = "half size" in w or "no history" in w
            import re as _re
            mm = _re.search(r"averaged ([+-][0-9.]+R) per trade \(([0-9]+)% wins, ([0-9]+) trades", w)
            ln.append(f" {mm.group(1)} · {mm.group(2)}% wins · {mm.group(3)} past" if mm else " no track record", style="grey70")
            if half:
                ln.append("  half size", style="yellow")
            if e.get("result"):
                ln.append("  result: " + ", ".join(f"{k} {'—' if v is None else f'{v:+.2f}R'}" for k, v in e["result"].items()),
                          style="cyan")
            body.append(ln)
    body.append(Text("Before each release: mark the 30-min range (stops sit beyond it), pause normal trades,\n"
                     "then trade the play that worked before for that market.", style="grey50"))
    return Panel(Group(*body), title=Text(" 📰 NEWS TRADING ", style="bold"), title_align="left",
                 box=box.ROUNDED, border_style="yellow" if any(e["phase"] in ("prep", "armed", "in_trade") for e in evs) else "grey50")


def markets_panel(broker) -> Panel:
    g = Table.grid(expand=True, padding=(0, 1))
    rows = m.market_rows(broker)
    for _ in rows:
        g.add_column(ratio=1)
    cells = []
    for r in rows:
        c = Text(r["name"], style="bold")
        c.append(f"  {'' if r['bid'] is None else m._px(r['bid'])}\n", style="white")
        if r["chg_pct"] is not None:
            c.append(f"{'▲' if r['chg_pct'] >= 0 else '▼'} {r['chg_pct']:+.2f}% 24h", style="green" if r["chg_pct"] >= 0 else "red")
        c.append("\n")
        if r["lots"]:
            c.append(f"{'LONG' if r['lots'] > 0 else 'SHORT'} {abs(r['lots']):.2f} ", style="bold cyan")
            if r["pnl"] is not None:
                c.append(f"{'+' if r['pnl'] >= 0 else '−'}${abs(r['pnl']):.2f}", style="green" if r["pnl"] >= 0 else "red")
        else:
            g = r.get("grad")
            if g and r["mode"] == "news only":
                c.append(g["label"] + (f" ({g['attempts']}/3)" if g["status"] in ("training", "exam_failed") else ""),
                         style="bold magenta" if g["status"] == "training" else "magenta")
            else:
                c.append(r["mode"], style="grey50")
        c.append("\n")
        n = r.get("news")
        if n:
            secs = n["t0"] - time.time()
            when = f"{secs / 3600:.1f}h" if secs > 3600 else (f"{secs / 60:.0f}m" if secs > 0 else "now")
            live = n["phase"] in ("prep", "armed", "in_trade")
            c.append(f"news {when}: {n.get('plan') or 'aside'}", style="bold yellow" if live else "yellow")
        cells.append(c)
    g.add_row(*cells)
    return Panel(g, title=Text(" 🌍 ALL MARKETS ", style="bold"), title_align="left", box=box.ROUNDED, border_style="grey50")


def feed_panel(height: int) -> Panel:
    import json
    items = []
    try:
        with open(os.path.join(m.LOG_DIR, "narration.jsonl"), encoding="utf-8") as f:
            for line in f.readlines()[-40:]:
                try:
                    items.append(json.loads(line))
                except Exception:
                    pass
    except OSError:
        pass
    body = []
    tone = {"good": "green", "bad": "red", "info": "bright_cyan"}
    for it in reversed(items[-max(2, height // 3):]):
        age = time.time() - float(it.get("t", 0))
        ago = "just now" if age < 60 else (f"{age / 60:.0f} min ago" if age < 3600 else f"{age / 3600:.0f} h ago")
        h = Text("▍", style=tone.get(it.get("tone"), "white"))
        h.append(f"{it.get('title', '')}", style="bold")
        h.append(f"  {ago}", style="grey50")
        body.append(h)
        body.append(Text("  " + str(it.get("text", ""))[:230], style="grey82"))
    if not body:
        body.append(Text("Explanations appear here as training rounds finish, signals change and trades close.\n"
                         "(written by the web dashboard server: run 'exness web' if this stays empty)", style="dim"))
    return Panel(Group(*body), title=Text(" ✨ WHAT CHANGED & WHY ", style="bold"), title_align="left",
                 box=box.ROUNDED, border_style="grey50")


def nm(s):
    return NAMES.get(s, s)


def money(v, sign=False):
    if v is None:
        return "—"
    return (f"{v:+,.2f}" if sign else f"{v:,.2f}").replace("+", "+$").replace("-", "−$") if sign else f"${v:,.2f}"


def thr_for(sg: dict, symbol: str) -> float:
    prof = sg.get("decision_profile") if isinstance(sg.get("decision_profile"), dict) else {}
    return float(prof.get("min_trade_threshold", 0.15) or 0.15)


# ----------------------------------------------------------------------------- top bar
def top_bar(frame, alive, broker, logs, cfg) -> Panel:
    acc = broker.account
    eq = float(getattr(acc, "equity", 0) or 0)
    floating = sum(float(getattr(p, "profit", 0) or 0) for p in broker.positions)
    today = (broker.pnl_today or 0.0) + floating
    start_eq = eq - today if eq else 0
    pct = today / start_eq * 100 if start_eq else 0.0
    left = 300 - (time.time() % 300)

    t = Text()
    dot = "●" if (frame // 3) % 2 == 0 else "○"
    t.append(f" {dot} ", style="bold green" if alive else "bold red")
    t.append("BOT RUNNING" if alive else "BOT STOPPED", style="bold green" if alive else "bold red")
    t.append("   │   ", style="grey42")
    t.append("Demo account ", style="dim")
    t.append(f"${wr.ease('eq', eq, 0.3):,.2f}", style="bold bright_white")
    t.append("   │   ", style="grey42")
    t.append("Today ", style="dim")
    t.append(f"{'+' if today >= 0 else '−'}${abs(today):,.2f} ({pct:+.2f}%)", style="bold green" if today >= 0 else "bold red")
    t.append("   │   ", style="grey42")
    t.append("Next check in ", style="dim")
    t.append(f"{int(left // 60)}:{int(left % 60):02d}", style="bold cyan")
    learn = ((cfg.get("trading") or {}).get("demo_learning") or {}).get("enabled")
    if learn:
        t.append("   │   ", style="grey42")
        t.append(" LEARNING MODE ", style="bold black on magenta")
        t.append(" trades more often on demo to learn faster", style="dim")
    return Panel(t, box=box.HEAVY, border_style="green" if alive else "red", padding=(0, 1))


# ----------------------------------------------------------------------------- symbol card
def symbol_card(frame, s, broker, logs, width) -> Panel:
    tick = broker.ticks.get(s)
    sg = (logs.last_signal.get(s) or {}).get("payload") or {}
    cv = (logs.last_council.get(s) or {}).get("payload") or {}
    mine = [p for p in broker.positions if str(getattr(p, "symbol", "")) in (s, s.rstrip("m"))]

    title = Text(f" {nm(s)} ", style="bold black on bright_white")
    if tick:
        prev = broker.prev_bid.get(s, tick.bid)
        arrow = "▲" if tick.bid > prev else ("▼" if tick.bid < prev else "")
        col = "green" if tick.bid >= prev else "red"
        fresh = time.time() - broker.flash.get(s, 0) < 1.2
        title.append(f"  ${m._px(tick.bid)} {arrow} ", style=(f"bold black on {col}" if fresh else f"bold {col}"))

    rows = []
    # 1. status, big and plain
    st = Text()
    if mine:
        p0 = mine[0]
        long_ = int(p0.type) == 0
        pnl = sum(float(p.profit) for p in mine)
        st.append(f" {'▲ IN A BUY TRADE' if long_ else '▼ IN A SELL TRADE'} ",
                  style="bold black on green" if long_ else "bold white on red")
        st.append(f"  {sum(float(p.volume) for p in mine):.2f} lots   ", style="bold")
        st.append(f"{'+' if pnl >= 0 else '−'}${abs(pnl):,.2f} right now", style="bold green" if pnl >= 0 else "bold red")
    else:
        st.append(" ⏸  WAITING ", style="bold black on grey70")
        st.append("  no open trade", style="dim")
    rows.append(st)
    rows.append(Text(""))

    # 2. why, in one sentence
    if sg:
        raw = float(sg.get("raw_target", 0) or 0) * float(sg.get("risk_scalar", 1) or 1)
        thr = thr_for(sg, s)
        up, down = int(cv.get("up", 0) or 0), int(cv.get("down", 0) or 0)
        lean = "BUY" if raw > 0 else "SELL"
        if mine:
            why = "It's managing the trade: stop-loss and take-profit are set, and it exits if the signals turn."
        elif abs(raw) >= thr:
            why = f"Signals agree on {lean}. Placing a trade at the next check."
        elif ((raw > 0) and down > up) or ((raw < 0) and up > down):
            why = (f"Leaning {lean}: more signals say {'down' if raw > 0 else 'up'} ({up}↑ {down}↓), but the ones "
                   f"with the better track record say {'up' if raw > 0 else 'down'}.")
        elif abs(raw) >= thr * 0.6:
            why = f"Leaning {lean} ({up} signals up, {down} down). Almost strong enough to trade."
        else:
            why = f"Signals are mixed ({up} up, {down} down). Not worth paying the spread yet."
        rows.append(Text(why, style="bold"))
        rows.append(Text(""))
        # 3. meter
        shown = wr.ease(f"s:{s}", raw)
        w = max(24, min(56, width - 22))
        meter = Text("  SELL ", style="bold red")
        meter.append_text(wr._needle(shown, thr, w))
        meter.append(" BUY", style="bold green")
        rows.append(meter)
        lbl = Text(" " * 8)
        lbl.append(f"signal {raw:+.2f}", style="bold")
        lbl.append(f"   trades when it passes the yellow line (±{thr:.2f})", style="dim")
        rows.append(lbl)
    else:
        rows.append(Text("Waiting for its first check…", style="dim"))
    rows.append(Text(""))

    # 4. small price chart with trade levels
    closes = list(broker.bars.get(s, []))
    if tick:
        closes.append(float(tick.bid))
    marks = {}
    if mine:
        p0 = mine[0]
        marks["entry"] = (float(p0.price_open), "yellow")
        if p0.sl:
            marks["stop"] = (float(p0.sl), "red")
        if p0.tp:
            marks["target"] = (float(p0.tp), "green")
    color = "green" if len(closes) > 1 and closes[-1] >= closes[0] else "red"
    rows.append(m.braille_chart(closes, max(20, width - 18), 4, marks, color=color))
    rows.append(Text("last 12 hours", style="dim"))
    return Panel(Group(*rows), title=title, title_align="left", box=box.ROUNDED,
                 border_style="green" if mine else "grey50", padding=(0, 1))


# ----------------------------------------------------------------------------- self-improvement
def improvement_panel(frame, broker, logs, symbols, cfg) -> Panel:
    out_entry = int(getattr(broker.mt5, "DEAL_ENTRY_OUT", 1)) if broker.mt5 else 1
    closed = [d for d in broker.deals if int(getattr(d, "entry", -1)) == out_entry
              and float(getattr(d, "volume", 0) or 0) > 0]
    pnl = [float(d.profit) + float(getattr(d, "commission", 0) or 0) + float(getattr(d, "swap", 0) or 0) for d in closed]
    wins, losses = sum(1 for x in pnl if x > 0), sum(1 for x in pnl if x <= 0)

    t = Table.grid(padding=(0, 2), expand=True)
    t.add_column(ratio=1)
    t.add_column(ratio=1)
    t.add_column(ratio=1)

    def stat(label, value, style="bold bright_white", sub=""):
        g = Text(label + "\n", style="dim")
        g.append(value, style=style)
        if sub:
            g.append("\n" + sub, style="dim")
        return g

    wr_ = f"{wins / len(pnl) * 100:.0f}%" if pnl else "—"
    t.add_row(stat("TRADES CLOSED TODAY", f"{len(pnl)}", sub=f"{wins} won · {losses} lost"),
              stat("WIN RATE", wr_, "bold green" if pnl and wins >= losses else "bold"),
              stat("PROFIT FROM CLOSED TRADES", f"{'+' if sum(pnl) >= 0 else '−'}${abs(sum(pnl)):,.2f}",
                   "bold green" if sum(pnl) >= 0 else "bold red"))

    lines = [t, Text("")]
    # what it has learned (from the live track record)
    from Python.indicator_council import describe
    lines.append(Text("WHAT IT HAS LEARNED (updated after every 5-minute bar)", style="bold cyan"))
    for s in symbols:
        cv = (logs.last_council.get(s) or {}).get("payload") or {}
        a = cv.get("assess") or {}
        if not a:
            continue
        ranked = sorted(a.items(), key=lambda kv: -float((kv[1] or {}).get("mult", 1)))
        best = [describe(k) for k, v in ranked[:2] if (v or {}).get("verdict") == "works"]
        worst = [describe(k) for k, v in ranked[::-1][:2] if (v or {}).get("verdict") == "doesn't work"]
        line = Text(f" {nm(s):<8}", style="bold")
        if best:
            line.append(" trusts ", style="dim")
            line.append(", ".join(best), style="green")
        if worst:
            line.append("   ignores ", style="dim")
            line.append(", ".join(worst), style="red")
        lines.append(line)
    lines.append(Text(""))

    # model training, one line
    tr = m.training_state()
    prog = m.read_progress()
    stages = wr.cycle_stages(wr.stage_lines(m.CYCLE_LOG), wr.cycle_symbols(symbols))
    cur = next((r for r in stages if r["status"] == "running"), None)
    done = sum(1 for r in stages if r["status"] in ("done", "passed"))
    lines.append(Text("BRAIN TRAINING", style="bold cyan"))
    if cur:
        miss = wr.MISSIONS[cur["kind"]]
        summ = wr.training_summary(prog, stages)
        frac = summ["stage_progress"] if summ["stage_progress"] is not None else 0.0
        b = Text(" ")
        b.append_text(wr._bar(frac, 30, frame=frame))
        eta = summ["stage_eta_s"]
        b.append(f" {frac * 100:.0f}%" + (f" · ~{int(eta // 60)} min left" if eta else "") + "   ", style="bold")
        b.append(f"now: {miss['name']} for {nm(cur['symbol']).title()}", style="yellow")
        lines.append(b)
        lines.append(Text(f" It's learning {miss['learn']}", style=""))
        nxt = next((r for r in stages if r["status"] == "pending"), None)
        if nxt:
            lines.append(Text(f" Next up: {wr.MISSIONS[nxt['kind']]['name']} for {nm(nxt['symbol']).title()}. "
                              f"New models only trade after passing an exam on data they've never seen.", style="dim"))
    elif tr.get("running"):
        lines.append(Text(" Preparing data…", style="dim"))
    else:
        lines.append(Text(" No training running. Type 'exness train' to start a learning cycle.", style="dim"))
    return Panel(Group(*lines), title=Text(" 📈 IS IT GETTING BETTER? ", style="bold"), title_align="left",
                 box=box.ROUNDED, border_style="cyan", padding=(0, 1))


# ----------------------------------------------------------------------------- trades in plain English
def trades_panel(broker) -> Panel:
    t = Table(box=box.SIMPLE_HEAD, expand=True, padding=(0, 1))
    for c in ("Time", "What happened", "Result"):
        t.add_column(c, justify="right" if c == "Result" else "left", no_wrap=True)
    in_entry = 0
    deals = [d for d in broker.deals if float(getattr(d, "volume", 0) or 0) > 0 and str(getattr(d, "symbol", ""))]
    deals = sorted(deals, key=lambda d: getattr(d, "time", 0))[-10:]
    for d in reversed(deals):
        sym = nm(str(getattr(d, "symbol", "")) + ("" if str(getattr(d, "symbol", "")).endswith("m") else "m"))
        ts = dt.datetime.fromtimestamp(int(getattr(d, "time", 0) or 0)).strftime("%H:%M")
        buy = int(getattr(d, "type", 0)) == 0
        vol = float(getattr(d, "volume", 0) or 0)
        px = m._px(getattr(d, "price", 0))
        if int(getattr(d, "entry", 0)) == in_entry:
            what = f"{'Bought' if buy else 'Sold'} {vol:.2f} {sym.title()} at ${px}"
            res = Text("opened", style="dim")
        else:
            pnl = float(getattr(d, "profit", 0) or 0)
            what = f"Closed {sym.title()} at ${px}"
            res = Text(f"{'+' if pnl >= 0 else '−'}${abs(pnl):,.2f} {'✓ win' if pnl > 0 else '✗ loss'}",
                       style="bold green" if pnl > 0 else "bold red")
        t.add_row(ts, what, res)
    if not deals:
        t.add_row("", Text("No trades yet today.", style="dim"), "")
    return Panel(t, title=Text(" 🧾 TODAY'S TRADES ", style="bold"), title_align="left", box=box.ROUNDED,
                 border_style="grey50")


def render(frame, broker, logs, cfg, symbols, width):
    alive, _ = m.bot_status()
    lay = Layout()
    card_w = (width - 4) // max(1, len(symbols))
    cards = Table.grid(expand=True, padding=(0, 0))
    for _ in symbols:
        cards.add_column(ratio=1)
    cards.add_row(*[symbol_card(frame, s, broker, logs, card_w) for s in symbols])
    tip = wr.GLOSSARY[int(time.time() // 10) % len(wr.GLOSSARY)]
    foot = Text(" 💡 ", style="bold")
    foot.append(f"{tip[0]}: ", style="bold bright_cyan")
    foot.append(tip[1], style="")
    foot.append("      Ctrl+C to exit · type 'exness pro' for the detailed view", style="grey42")
    lay.split_column(
        Layout(top_bar(frame, alive, broker, logs, cfg), size=3),
        Layout(cards, size=15),
        Layout(markets_panel(broker), size=6),
        Layout(name="bottom"),
        Layout(foot, size=1),
    )
    lay["bottom"].split_row(Layout(improvement_panel(frame, broker, logs, symbols, cfg), ratio=2),
                            Layout(name="right", ratio=3))
    lay["right"].split_column(Layout(news_panel(frame), size=13), Layout(name="rb"))
    lay["rb"].split_row(Layout(feed_panel(30), ratio=1), Layout(trades_panel(broker), ratio=1))
    return lay


def main():
    cfg = m._cfg()
    symbols = [str(s) for s in ((cfg.get("trading") or {}).get("symbols") or ["BTCUSDm", "XAUUSDm"])]
    console = Console()
    broker = m.Broker(symbols, extra=[str(x) for x in (((cfg.get("trading") or {}).get("news_trading") or {}).get("symbols") or [])])
    logs = m.Logs()
    frame = 0
    with Live(console=console, screen=True, auto_refresh=False, transient=True, vertical_overflow="crop") as live:
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
