"""Once-a-day jobs the running bot starts itself.

The market scout (tools/market_scout.py) was a launchd job, but macOS will not let a background launchd job open
files on the external AI_DRIVE: it never started ("spawn failed", exit 78; 2026-11-24). The bot runs from Terminal,
which can reach the drive, so it starts the scout in its own process once a day at 07:15 UTC. The scout keeps its
own "already ran today" check (logs/market_scout.json), and when it adds a market it stops the bot, which the
supervisor loop (scripts/mac/start_bot.sh) starts again with the new list.
"""
from __future__ import annotations

import datetime as dt
import json
import os
import subprocess
import sys

SCOUT_AT_UTC = (7, 15)


def _today(now: float) -> str:
    return dt.datetime.fromtimestamp(now, dt.timezone.utc).strftime("%Y-%m-%d")


def scout_due(now: float, last_run_at: str | None, started_on: str | None, at=SCOUT_AT_UTC) -> bool:
    """True once it is past `at` (UTC) today, the scout has not finished a run today, and the bot has not
    already started one today (a run that crashed is not retried until tomorrow)."""
    t = dt.datetime.fromtimestamp(now, dt.timezone.utc)
    day = t.strftime("%Y-%m-%d")
    if (t.hour, t.minute) < tuple(at):
        return False
    return not str(last_run_at or "").startswith(day) and started_on != day


def maybe_start_scout(root: str, now: float, logger=None):
    """Start tools/market_scout.py in the background if it is due. Returns the process or None."""
    logs, tmp = os.path.join(root, "logs"), os.path.join(root, ".tmp")
    try:
        with open(os.path.join(logs, "market_scout.json"), encoding="utf-8") as f:
            last = ((json.load(f).get("last_run") or {}).get("at"))
    except (OSError, ValueError):
        last = None
    marker = os.path.join(tmp, "scout_started")
    try:
        with open(marker, encoding="utf-8") as f:
            started = f.read().strip()
    except OSError:
        started = None
    if not scout_due(now, last, started):
        return None
    os.makedirs(tmp, exist_ok=True)
    with open(marker, "w", encoding="utf-8") as f:
        f.write(_today(now))
    log = open(os.path.join(logs, "market_scout.log"), "a", encoding="utf-8")
    log.write(f"{dt.datetime.now():%F %T} bot: daily market scout\n")
    log.flush()
    proc = subprocess.Popen([sys.executable, os.path.join("tools", "market_scout.py")], cwd=root, stdout=log,
                            stderr=subprocess.STDOUT, start_new_session=True,
                            env={**os.environ, "PYTHONPATH": root, "PYTHONUNBUFFERED": "1"})
    if logger:
        logger.info(f"MARKET SCOUT started (pid {proc.pid}): testing the next new market, log logs/market_scout.log")
    return proc
