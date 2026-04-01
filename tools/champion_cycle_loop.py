import argparse
import json
import os
import signal
import sys
import time
from datetime import datetime, timezone

from loguru import logger

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from tools import champion_cycle

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
_LOCK_PATH = os.path.join(ROOT, ".tmp", "champion_cycle.lock")
_HEARTBEAT_PATH = os.path.join(ROOT, ".tmp", "champion_loop.heartbeat")
_METRICS_PATH = os.path.join(ROOT, "logs", "cycle_metrics.jsonl")
_LOG_PATH = os.path.join(ROOT, "logs", "champion_loop.log")

# ---------------------------------------------------------------------------
# Shutdown flag – set by signal handlers
# ---------------------------------------------------------------------------
_shutdown_requested = False


def _handle_signal(signum, _frame):
    global _shutdown_requested
    _shutdown_requested = True
    logger.warning("Received signal {} – will shut down after current sleep", signum)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ensure_dirs():
    """Create output directories if they do not exist."""
    os.makedirs(os.path.join(ROOT, ".tmp"), exist_ok=True)
    os.makedirs(os.path.join(ROOT, "logs"), exist_ok=True)


def _write_heartbeat(status, last_cycle, consecutive_failures):
    """Atomically write the heartbeat file."""
    payload = {
        "pid": os.getpid(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "status": status,
        "last_cycle": last_cycle,
        "consecutive_failures": consecutive_failures,
    }
    tmp = _HEARTBEAT_PATH + ".tmp"
    try:
        with open(tmp, "w") as fh:
            json.dump(payload, fh)
        # os.replace is atomic on POSIX; best-effort on Windows
        os.replace(tmp, _HEARTBEAT_PATH)
    except OSError as exc:
        logger.warning("Failed to write heartbeat: {}", exc)


def _write_metric(ts, status, duration_sec, error, consecutive_failures, next_retry_sec):
    """Append one JSON line to the cycle-metrics log."""
    record = {
        "ts": ts,
        "status": status,
        "duration_sec": round(duration_sec, 3),
        "error": error,
        "consecutive_failures": consecutive_failures,
        "next_retry_sec": next_retry_sec,
    }
    try:
        with open(_METRICS_PATH, "a") as fh:
            fh.write(json.dumps(record) + "\n")
    except OSError as exc:
        logger.warning("Failed to write cycle metric: {}", exc)


def _is_pid_alive(pid):
    """Return True if *pid* appears to be a running process."""
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    except SystemError:
        return False
    return True


def _cleanup_stale_lock():
    """Remove the champion-cycle lock file if the owning process is dead."""
    if not os.path.exists(_LOCK_PATH):
        return
    try:
        with open(_LOCK_PATH, "r") as fh:
            content = fh.read().strip()
        pid = int(content)
    except (OSError, ValueError):
        # Unreadable or unparseable – remove it
        logger.warning("Lock file unreadable or corrupt; removing {}", _LOCK_PATH)
        try:
            os.remove(_LOCK_PATH)
        except OSError:
            pass
        return

    if not _is_pid_alive(pid):
        logger.warning(
            "Stale lock detected – PID {} is dead; removing {}",
            pid,
            _LOCK_PATH,
        )
        try:
            os.remove(_LOCK_PATH)
        except OSError as exc:
            logger.error("Could not remove stale lock: {}", exc)


def _is_oom(exc):
    """Return True when *exc* looks like an out-of-memory error."""
    if isinstance(exc, MemoryError):
        return True
    msg = str(exc).lower()
    return "memoryerror" in msg or "out of memory" in msg


def _interruptible_sleep(seconds):
    """Sleep in 1-second increments so we can honour the shutdown flag."""
    remaining = seconds
    while remaining > 0 and not _shutdown_requested:
        chunk = min(remaining, 1.0)
        time.sleep(chunk)
        remaining -= chunk


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args():
    parser = argparse.ArgumentParser(description="Run champion cycle continuously.")
    parser.add_argument(
        "--interval-minutes",
        type=int,
        default=30,
        help="Minutes to wait between cycle starts (used on success).",
    )
    parser.add_argument(
        "--max-backoff-minutes",
        type=int,
        default=30,
        help="Maximum backoff delay in minutes after repeated failures.",
    )
    parser.add_argument(
        "--max-failures",
        type=int,
        default=10,
        help="Consecutive failures before engaging the circuit breaker (1-hour wait).",
    )
    return parser.parse_args()


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

_BASE_BACKOFF_SEC = 60
_CIRCUIT_BREAKER_SEC = 3600  # 1 hour


def main():
    args = _parse_args()
    interval_sec = max(1, args.interval_minutes * 60)
    max_backoff_sec = max(1, args.max_backoff_minutes * 60)
    max_failures = max(1, args.max_failures)

    _ensure_dirs()

    # Log rotation: 10 MB per file, keep 5
    logger.add(
        _LOG_PATH,
        rotation="10 MB",
        retention=5,
        enqueue=True,
        backtrace=True,
        diagnose=True,
    )

    # Graceful shutdown on SIGINT / SIGTERM
    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    consecutive_failures = 0
    last_cycle = "none"

    logger.info(
        "Champion loop starting | pid={} | interval_min={} | max_backoff_min={} | max_failures={}",
        os.getpid(),
        args.interval_minutes,
        args.max_backoff_minutes,
        max_failures,
    )

    while not _shutdown_requested:
        # --- pre-cycle bookkeeping ---
        _cleanup_stale_lock()
        _write_heartbeat("running", last_cycle, consecutive_failures)

        ts_start = datetime.now(timezone.utc)
        error_text = None
        oom = False

        try:
            logger.info(
                "Champion loop tick start | utc={} | interval_min={}",
                ts_start.isoformat(),
                args.interval_minutes,
            )
            champion_cycle.main()
            logger.info("Champion loop tick success")
            last_cycle = "success"
            consecutive_failures = 0
        except Exception as exc:
            last_cycle = "failure"
            consecutive_failures += 1
            error_text = str(exc)
            oom = _is_oom(exc)

            if oom:
                logger.critical(
                    "OOM detected during champion cycle (consecutive_failures={}): {}",
                    consecutive_failures,
                    exc,
                )
            else:
                logger.exception(
                    "Champion loop tick failed (consecutive_failures={}): {}",
                    consecutive_failures,
                    exc,
                )

        # --- compute next wait ---
        duration_sec = (datetime.now(timezone.utc) - ts_start).total_seconds()

        if last_cycle == "success":
            next_retry_sec = interval_sec
        else:
            # Exponential backoff: 60, 120, 240, ... capped at max_backoff_sec
            backoff = min(
                _BASE_BACKOFF_SEC * (2 ** (consecutive_failures - 1)),
                max_backoff_sec,
            )
            if oom:
                backoff = min(backoff * 2, max_backoff_sec)

            # Circuit breaker
            if consecutive_failures >= max_failures:
                logger.critical(
                    "Circuit breaker engaged – {} consecutive failures reached; "
                    "waiting {} seconds before next attempt",
                    consecutive_failures,
                    _CIRCUIT_BREAKER_SEC,
                )
                backoff = _CIRCUIT_BREAKER_SEC

            next_retry_sec = backoff

        # --- metrics + heartbeat ---
        _write_metric(
            ts=ts_start.isoformat(),
            status=last_cycle,
            duration_sec=duration_sec,
            error=error_text,
            consecutive_failures=consecutive_failures,
            next_retry_sec=next_retry_sec,
        )
        _write_heartbeat("sleeping", last_cycle, consecutive_failures)

        logger.info("Sleeping {} seconds before next cycle", next_retry_sec)
        _interruptible_sleep(next_retry_sec)

    # --- clean exit ---
    logger.info("Shutdown complete – exiting cleanly")
    # Remove heartbeat so monitors know the loop is gone
    try:
        os.remove(_HEARTBEAT_PATH)
    except OSError:
        pass


if __name__ == "__main__":
    main()
