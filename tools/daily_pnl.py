"""End-of-day P&L report (user 2026-11-25): total, per market and per bot for one local calendar day.

Closed trades come from the broker's deal history (profit + commission + swap), so deposits and withdrawals never
count as profit or loss. Each run appends a row to logs/daily_pnl.csv, writes logs/daily_pnl_latest.txt and sends the
report to Telegram (the bot's telegram: config), or shows a Mac notification while Telegram isn't set up. launchd runs it at 00:05 for the day that just ended (scripts/mac/service_run.sh dailypnl).

  python tools/daily_pnl.py                    # yesterday, send once (skips a day already sent)
  python tools/daily_pnl.py --date 2026-11-25  # a given day
  python tools/daily_pnl.py --today --dry-run  # today so far, print only
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import html
import json
import os
import sys

import requests
import yaml

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
LOGS = os.path.join(ROOT, "logs")
OUT = os.path.join(LOGS, "daily_pnl.csv")
BOTS = {"gold_breakout": "Gold breakout", "meta_trader": "Meta-brain", "gemma_trader": "Gemma",
        "qwen_trader": "Qwen", "jev_trader": "Jev"}


def day_bounds(day: dt.date) -> tuple[dt.datetime, dt.datetime]:
    """Local midnight to midnight (the Mac's time zone), as aware datetimes."""
    start = dt.datetime.combine(day, dt.time()).astimezone()
    return start, (start + dt.timedelta(days=1)).astimezone()


def bot_by_ticket() -> dict[str, str]:
    """Position ticket -> bot name from the stand-alone traders' logs; everything else is the main bot."""
    out = {}
    for name, label in BOTS.items():
        try:
            with open(os.path.join(LOGS, f"{name}.jsonl"), encoding="utf-8") as f:
                for line in f:
                    try:
                        row = json.loads(line)
                    except ValueError:
                        continue
                    if row.get("event") == "open" and row.get("ticket"):
                        out[str(row["ticket"])] = label
        except OSError:
            continue
    return out


def summarize(deals, owners: dict[str, str]) -> dict:
    """Net P&L of the day's deals: closes carry the profit, every deal can carry commission / swap."""
    total, trades, wins, markets, bots = 0.0, 0, 0, {}, {}
    for d in deals or ():
        net = float(getattr(d, "profit", 0) or 0) + float(getattr(d, "commission", 0) or 0) + float(getattr(d, "swap", 0) or 0)
        sym = str(getattr(d, "symbol", "") or "")
        if not sym:
            continue                                     # deposit / withdrawal / balance entry: not trading P&L
        who = owners.get(str(getattr(d, "position_id", "") or ""), "Main bot")
        total += net
        markets[sym] = markets.get(sym, 0.0) + net
        bots[who] = bots.get(who, 0.0) + net
        if int(getattr(d, "entry", 0) or 0) == 1:      # DEAL_ENTRY_OUT: a closed trade
            trades += 1
            wins += float(getattr(d, "profit", 0) or 0) > 0
    return {"total": round(total, 2), "trades": trades, "wins": wins,
            "markets": {k: round(v, 2) for k, v in markets.items() if round(v, 2)},
            "bots": {k: round(v, 2) for k, v in bots.items() if round(v, 2)}}


def equity_range(start: dt.datetime, end: dt.datetime, account: str) -> tuple[float | None, float | None]:
    """First and last equity recorded during the day (logs/equity_history.csv: ts, equity, balance, account)."""
    first = last = None
    t0, t1 = start.timestamp(), end.timestamp()
    try:
        with open(os.path.join(LOGS, "equity_history.csv"), encoding="utf-8") as f:
            for row in csv.reader(f):
                try:
                    ts, eq = float(row[0]), float(row[1])
                except (ValueError, IndexError):
                    continue
                if t0 <= ts < t1 and (not account or len(row) < 4 or row[3] == account):
                    first = eq if first is None else first
                    last = eq
    except OSError:
        pass
    return first, last


def message(day: dt.date, s: dict, eq0, eq1, partial: bool) -> str:
    sign = lambda v: f"{'+' if v >= 0 else '-'}${abs(v):,.2f}"
    head = "🟢" if s["total"] > 0 else "🔴" if s["total"] < 0 else "⚪"
    lines = [f"{head} <b>Daily P&amp;L {day:%a %d %b %Y}{' (so far)' if partial else ''}: {sign(s['total'])}</b>"]
    if s["trades"]:
        lines.append(f"{s['trades']} closed trades, {s['wins']} won ({s['wins'] / s['trades'] * 100:.0f}%)")
    else:
        lines.append("No closed trades")
    if eq0 is not None and eq1 is not None:
        lines.append(f"Equity ${eq0:,.2f} → ${eq1:,.2f} (moves with open trades and withdrawals too)")
    if s["markets"]:
        lines.append("\n<b>By market</b>")
        lines += [f"{html.escape(k)}: {sign(v)}" for k, v in sorted(s["markets"].items(), key=lambda kv: -kv[1])]
    if s["bots"]:
        lines.append("\n<b>By bot</b>")
        lines += [f"{html.escape(k)}: {sign(v)}" for k, v in sorted(s["bots"].items(), key=lambda kv: -kv[1])]
    lines.append("\n(withdrawals and deposits not counted)")
    return "\n".join(lines)


def telegram_creds() -> tuple[str | None, str | None]:
    try:
        with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
            t = (yaml.safe_load(f) or {}).get("telegram") or {}
    except OSError:
        t = {}
    res = lambda v: os.environ.get(v[2:-1], "") if isinstance(v, str) and v.startswith("${") else v
    ok = lambda v: v if v and "YOUR_" not in str(v) else None      # config.yaml placeholders count as unset
    return (ok(os.environ.get("TELEGRAM_TOKEN") or res(t.get("token"))),
            ok(os.environ.get("TELEGRAM_CHAT_ID") or res(t.get("chat_id"))))


def notify_mac(text: str) -> bool:
    """Fallback while Telegram isn't set up: a macOS notification with the headline; the full report is in
    logs/daily_pnl_latest.txt."""
    import re
    import subprocess
    plain = [re.sub(r"<[^>]+>", "", ln).replace("&amp;", "&") for ln in text.splitlines() if ln.strip()]
    title = plain[0].lstrip("🟢🔴⚪ ") if plain else "Daily P&L"
    body = " · ".join(plain[1:3])
    esc = lambda x: x.replace("\\", "\\\\").replace('"', '\\"')
    r = subprocess.run(["osascript", "-e", f'display notification "{esc(body)}" with title "{esc(title)}" '
                        f'subtitle "Cautious Giggle" sound name "Glass"'], capture_output=True)
    return r.returncode == 0


def send(text: str) -> bool:
    try:
        import re
        with open(os.path.join(LOGS, "daily_pnl_latest.txt"), "w", encoding="utf-8") as f:
            f.write(re.sub(r"<[^>]+>", "", text).replace("&amp;", "&") + "\n")
    except OSError:
        pass
    token, chat = telegram_creds()
    if not token or not chat:
        print("telegram not configured (config.yaml telegram: token / chat_id): Mac notification instead")
        return notify_mac(text)
    r = requests.post(f"https://api.telegram.org/bot{token}/sendMessage",
                      json={"chat_id": chat, "text": text, "parse_mode": "HTML"}, timeout=15)
    if not r.ok:
        print(f"telegram send failed: HTTP {r.status_code}")
    return r.ok


def already_sent(day: dt.date, account: str) -> bool:
    try:
        with open(OUT, encoding="utf-8") as f:
            return any(r.get("date") == day.isoformat() and r.get("account") == account and r.get("sent") == "1"
                       for r in csv.DictReader(f))
    except OSError:
        return False


def record(day: dt.date, account: str, s: dict, eq0, eq1, sent: bool) -> None:
    new = not os.path.exists(OUT)
    with open(OUT, "a", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        if new:
            w.writerow(["date", "account", "pnl", "trades", "wins", "equity_start", "equity_end", "markets", "bots", "sent"])
        w.writerow([day.isoformat(), account, s["total"], s["trades"], s["wins"], eq0, eq1,
                    json.dumps(s["markets"]), json.dumps(s["bots"]), int(sent)])


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", help="YYYY-MM-DD (default: yesterday)")
    ap.add_argument("--today", action="store_true", help="today so far")
    ap.add_argument("--dry-run", action="store_true", help="print only: no Telegram, no log row")
    ap.add_argument("--force", action="store_true", help="send even if this day was already sent")
    a = ap.parse_args(argv)
    today = dt.date.today()
    day = dt.date.fromisoformat(a.date) if a.date else today if a.today else today - dt.timedelta(days=1)
    account = os.environ.get("EXNESS_ACCOUNT_ID", "")
    if not a.dry_run and not a.force and already_sent(day, account):
        print(f"{day} already sent")
        return 0

    from Python.exness_api import ExnessMT5, symbol_map_from_cfg
    mt5 = ExnessMT5(symbol_map=symbol_map_from_cfg())
    if not mt5.initialize():
        print("broker connection failed:", mt5.last_error())
        return 1
    start, end = day_bounds(day)
    deals = mt5.history_deals_get(start.astimezone(dt.timezone.utc), end.astimezone(dt.timezone.utc))
    if deals is None:
        print("deal history unavailable")
        return 1
    s = summarize(deals, bot_by_ticket())
    eq0, eq1 = equity_range(start, end, account)
    text = message(day, s, eq0, eq1, partial=day >= today)
    print(text)
    if a.dry_run:
        return 0
    ok = send(text)
    if day < today:                                      # a day in progress is never recorded as sent
        record(day, account, s, eq0, eq1, ok)
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
