#!/usr/bin/env python
"""Process watchdog daemon.

Monitors critical processes and auto-restarts them on failure with
exponential backoff.  Writes a JSON heartbeat file so external tools
can verify the watchdog itself is alive.

Usage:
    python tools/watchdog.py
    python tools/watchdog.py --check-interval 10 --max-backoff 600
"""

import argparse
import json
import os
import signal
import subprocess
import sys
import time
import urllib.request
import urllib.error
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.config_utils import load_project_config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
_STABLE_THRESHOLD_SEC = 600  # reset backoff after 10 min of uptime
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
HFT_CONFIG_PATH = os.path.join(PROJECT_ROOT, "config_hft.yaml")
SERVER_LOCK_PATH = os.path.join(PROJECT_ROOT, ".tmp", "server_agi.lock")
HFT_LOCK_PATH = os.path.join(PROJECT_ROOT, ".tmp", "server_agi_hft.lock")


# ---------------------------------------------------------------------------
# Process definitions
# ---------------------------------------------------------------------------

class ProcessSpec:
    """Immutable description of a managed process."""

    def __init__(
        self,
        name: str,
        detection_tokens: List[str],
        cmd: List[str],
        cwd: str,
        env_extra: Optional[Dict[str, str]] = None,
        health_url: Optional[str] = None,
        lock_file: Optional[str] = None,
    ):
        self.name = name
        self.detection_tokens = detection_tokens
        self.cmd = cmd
        self.cwd = cwd
        self.env_extra = env_extra or {}
        self.health_url = health_url
        self.lock_file = lock_file


def _load_env_file(path: str) -> None:
    if not os.path.exists(path):
        return
    with open(path, "r", encoding="utf-8") as fh:
        for raw_line in fh:
            line = raw_line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            key = key.strip()
            if key and key not in os.environ:
                os.environ[key] = value.strip()


def _validate_live_env() -> None:
    _load_env_file(ENV_PATH)
    if not os.environ.get("AGI_TOKEN", "").strip():
        raise RuntimeError("AGI_TOKEN is not set in the environment or .env")
    prev_cfg = os.environ.pop("AGI_CONFIG", None)
    try:
        load_project_config(PROJECT_ROOT, live_mode=True)
        if os.path.exists(HFT_CONFIG_PATH):
            os.environ["AGI_CONFIG"] = HFT_CONFIG_PATH
            load_project_config(PROJECT_ROOT, live_mode=True)
    finally:
        if prev_cfg is None:
            os.environ.pop("AGI_CONFIG", None)
        else:
            os.environ["AGI_CONFIG"] = prev_cfg


def _build_specs() -> List[ProcessSpec]:
    """Return the list of processes to manage."""
    return [
        ProcessSpec(
            name="server_agi",
            detection_tokens=[],
            cmd=[sys.executable, "-m", "Python.Server_AGI", "--live"],
            cwd=PROJECT_ROOT,
            lock_file=SERVER_LOCK_PATH,
        ),
        ProcessSpec(
            name="hft_server",
            detection_tokens=["start_hft"],
            cmd=[sys.executable, "-m", "Python.Server_AGI", "--live"],
            cwd=PROJECT_ROOT,
            env_extra={
                "AGI_CONFIG": "config_hft.yaml",
                "AGI_MODE_TAG": "hft",
                "AGI_LOOP_SEC": "5",
                "AGI_HEARTBEAT_SEC": "300",
                "AGI_SYMBOL_CARD_SEC": "60",
                "AGI_TRADE_LEARN_SEC": "300",
            },
            lock_file=HFT_LOCK_PATH,
        ),
        ProcessSpec(
            name="dashboard_ui",
            detection_tokens=["project_status_ui"],
            cmd=[sys.executable, "-u", "tools/project_status_ui.py"],
            cwd=PROJECT_ROOT,
            env_extra={
                "AGI_UI_HOST": "127.0.0.1",
                "AGI_UI_PORT": "8088",
            },
            health_url="http://127.0.0.1:8088/",
        ),
        ProcessSpec(
            name="champion_cycle",
            detection_tokens=["champion_cycle_loop"],
            cmd=[
                sys.executable, "-u",
                "tools/champion_cycle_loop.py",
                "--interval-minutes", "30",
            ],
            cwd=PROJECT_ROOT,
        ),
    ]


# ---------------------------------------------------------------------------
# Runtime state per process
# ---------------------------------------------------------------------------

class ProcessState:
    """Mutable runtime bookkeeping for one managed process."""

    def __init__(self, spec: ProcessSpec):
        self.spec = spec
        self.popen: Optional[subprocess.Popen] = None  # only set when WE started it
        self.externally_running = False
        self.consecutive_failures = 0
        self.last_start_ts: Optional[float] = None
        self.status = "unknown"  # running | stopped | backoff

    # -- backoff helpers ----------------------------------------------------

    def backoff_seconds(self, max_backoff: int) -> float:
        """Exponential backoff capped at *max_backoff*."""
        delay = min(2 ** self.consecutive_failures, max_backoff)
        return float(delay)

    def record_failure(self) -> None:
        self.consecutive_failures += 1

    def maybe_reset_backoff(self) -> None:
        """Reset the failure counter once the process has been stable."""
        if (
            self.last_start_ts is not None
            and (time.monotonic() - self.last_start_ts) >= _STABLE_THRESHOLD_SEC
        ):
            if self.consecutive_failures > 0:
                logger.info(
                    "{} stable for {}s -- resetting backoff counter",
                    self.spec.name,
                    _STABLE_THRESHOLD_SEC,
                )
            self.consecutive_failures = 0


# ---------------------------------------------------------------------------
# Detection: is a process already running?
# ---------------------------------------------------------------------------

def _query_running_commandlines() -> str:
    """Return the combined output of ``wmic process get CommandLine``."""
    try:
        result = subprocess.run(
            ["wmic", "process", "get", "CommandLine"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=NO_WINDOW,
        )
        return result.stdout
    except Exception as exc:
        logger.warning("wmic query failed: {}", exc)
        return ""


def _pid_alive(pid: int) -> bool:
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            import ctypes

            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        os.kill(pid, 0)
        return True
    except (OSError, ProcessLookupError):
        return False


def _pid_from_lock_file(path: Optional[str]) -> int:
    if not path or not os.path.exists(path):
        return 0
    try:
        return int((Path(path).read_text(encoding="utf-8") or "").strip() or "0")
    except Exception:
        return 0


def _process_detected_externally(
    spec: ProcessSpec, wmic_output: str, own_popen: Optional[subprocess.Popen]
) -> bool:
    """Return True if *any* matching process exists that we did NOT spawn."""
    lock_pid = _pid_from_lock_file(spec.lock_file)
    if lock_pid:
        if own_popen is not None and own_popen.pid == lock_pid:
            return False
        if _pid_alive(lock_pid):
            return True

    # Cheap check: do the detection tokens appear in the wmic output at all?
    lines = wmic_output.splitlines()
    for line in lines:
        if any(tok in line for tok in spec.detection_tokens):
            # If we spawned a process, exclude lines that match our own PID.
            if own_popen is not None and str(own_popen.pid) in line:
                continue
            return True
    return False


# ---------------------------------------------------------------------------
# Health checks
# ---------------------------------------------------------------------------

def _check_popen_alive(popen: Optional[subprocess.Popen]) -> bool:
    """Return True if the Popen handle is alive (poll() is None)."""
    if popen is None:
        return False
    return popen.poll() is None


def _http_health_check(url: str, timeout: float = 5.0) -> bool:
    """Return True if *url* responds with HTTP 200."""
    try:
        req = urllib.request.Request(url, method="GET")
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return resp.status == 200
    except Exception:
        return False


def is_healthy(state: ProcessState) -> bool:
    """Composite health check for a process."""
    spec = state.spec

    # If we spawned it, the handle must be alive.
    if state.popen is not None and not _check_popen_alive(state.popen):
        return False

    # HTTP endpoint check (Dashboard UI).
    if spec.health_url is not None:
        if not _http_health_check(spec.health_url):
            return False

    return True


# ---------------------------------------------------------------------------
# Starting / stopping processes
# ---------------------------------------------------------------------------

def start_process(state: ProcessState) -> None:
    """Launch the process described by *state.spec*."""
    spec = state.spec
    env = os.environ.copy()
    env.update(spec.env_extra)

    logger.info("Starting {} -- cmd={}", spec.name, spec.cmd)

    state.popen = subprocess.Popen(
        spec.cmd,
        cwd=spec.cwd,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    state.last_start_ts = time.monotonic()
    state.externally_running = False
    state.status = "running"
    logger.info("{} started with PID {}", spec.name, state.popen.pid)


def stop_process(state: ProcessState) -> None:
    """Terminate a process we spawned.  No-op for externally-started ones."""
    if state.popen is None:
        return
    if state.popen.poll() is not None:
        state.popen = None
        return

    pid = state.popen.pid
    logger.info("Terminating {} (PID {})", state.spec.name, pid)
    try:
        state.popen.terminate()
        state.popen.wait(timeout=10)
    except subprocess.TimeoutExpired:
        logger.warning("Killing {} (PID {}) after timeout", state.spec.name, pid)
        state.popen.kill()
        state.popen.wait(timeout=5)
    except Exception as exc:
        logger.error("Error stopping {} (PID {}): {}", state.spec.name, pid, exc)
    finally:
        state.popen = None
        state.status = "stopped"


# ---------------------------------------------------------------------------
# Heartbeat file
# ---------------------------------------------------------------------------

def _write_heartbeat(heartbeat_path: str, states: List[ProcessState]) -> None:
    """Atomically write the heartbeat JSON file."""
    payload = {
        "pid": os.getpid(),
        "ts": datetime.now(timezone.utc).isoformat(),
        "processes": {s.spec.name: s.status for s in states},
    }
    tmp_path = heartbeat_path + ".tmp"
    try:
        os.makedirs(os.path.dirname(heartbeat_path), exist_ok=True)
        with open(tmp_path, "w", encoding="utf-8") as fh:
            json.dump(payload, fh, indent=2)
        # Atomic-ish rename (Windows replaces if dst exists on Py 3.x).
        if os.path.exists(heartbeat_path):
            os.replace(tmp_path, heartbeat_path)
        else:
            os.rename(tmp_path, heartbeat_path)
    except Exception as exc:
        logger.warning("Failed to write heartbeat: {}", exc)


# ---------------------------------------------------------------------------
# Main loop
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Process watchdog daemon for AGI platform services.",
    )
    parser.add_argument(
        "--check-interval",
        type=int,
        default=15,
        help="Seconds between health-check cycles (default: 15).",
    )
    parser.add_argument(
        "--max-backoff",
        type=int,
        default=300,
        help="Maximum backoff delay in seconds (default: 300).",
    )
    parser.add_argument(
        "--log-file",
        type=str,
        default=os.path.join(PROJECT_ROOT, "logs", "watchdog.log"),
        help="Path to the log file.",
    )
    return parser.parse_args()


def main() -> None:  # noqa: C901  (complexity acceptable for a daemon main loop)
    args = _parse_args()

    # -- Logging setup ------------------------------------------------------
    logger.remove()  # drop default stderr handler
    logger.add(
        sys.stderr,
        level="INFO",
        format="<green>{time:YYYY-MM-DD HH:mm:ss}</green> | <level>{level:<8}</level> | {message}",
    )
    os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
    logger.add(
        args.log_file,
        rotation="10 MB",
        retention="7 days",
        compression="zip",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )

    _validate_live_env()

    heartbeat_path = os.path.join(PROJECT_ROOT, ".tmp", "watchdog.heartbeat")

    specs = _build_specs()
    states: List[ProcessState] = [ProcessState(s) for s in specs]

    # Map from process name to the monotonic timestamp at which we may next
    # attempt a restart (implements the backoff wait).
    next_restart_eligible: Dict[str, float] = {s.spec.name: 0.0 for s in states}

    # -- Graceful-shutdown handling -----------------------------------------
    shutdown_requested = False

    def _handle_signal(signum: int, _frame) -> None:
        nonlocal shutdown_requested
        sig_name = signal.Signals(signum).name if hasattr(signal, "Signals") else str(signum)
        logger.info("Received {} -- initiating graceful shutdown", sig_name)
        shutdown_requested = True

    signal.signal(signal.SIGINT, _handle_signal)
    signal.signal(signal.SIGTERM, _handle_signal)

    logger.info(
        "Watchdog starting  (interval={}s, max_backoff={}s, log={})",
        args.check_interval,
        args.max_backoff,
        args.log_file,
    )

    # -- Main loop ----------------------------------------------------------
    try:
        while not shutdown_requested:
            wmic_output = _query_running_commandlines()

            for state in states:
                if shutdown_requested:
                    break

                spec = state.spec

                # 1. Reset backoff if process has been stable long enough.
                state.maybe_reset_backoff()

                # 2. Check if someone else already runs this process.
                external = _process_detected_externally(spec, wmic_output, state.popen)

                # 3. If we spawned it, check our handle.
                we_spawned_alive = _check_popen_alive(state.popen)

                if external:
                    # Externally managed -- hands off.
                    state.externally_running = True
                    state.status = "running"
                    if we_spawned_alive:
                        # Edge case: both ours AND an external copy are alive.
                        # Keep ours alive; don't duplicate.
                        pass
                    logger.debug("{} detected externally -- skipping", spec.name)
                    continue

                state.externally_running = False

                if we_spawned_alive:
                    # Our child is running.  Perform health check.
                    if is_healthy(state):
                        state.status = "running"
                        logger.debug("{} healthy (PID {})", spec.name, state.popen.pid)
                        continue
                    else:
                        logger.warning(
                            "{} failed health check -- stopping and scheduling restart",
                            spec.name,
                        )
                        stop_process(state)
                        state.record_failure()
                        delay = state.backoff_seconds(args.max_backoff)
                        next_restart_eligible[spec.name] = time.monotonic() + delay
                        state.status = "backoff"
                        logger.info(
                            "{} backoff: next restart in {:.0f}s (failures={})",
                            spec.name,
                            delay,
                            state.consecutive_failures,
                        )
                        continue

                # Process is not running (neither ours nor external).
                # Has the popen handle exited? Clean it up.
                if state.popen is not None and not we_spawned_alive:
                    rc = state.popen.poll()
                    logger.warning(
                        "{} exited with code {} -- scheduling restart",
                        spec.name,
                        rc,
                    )
                    state.popen = None
                    state.record_failure()
                    delay = state.backoff_seconds(args.max_backoff)
                    next_restart_eligible[spec.name] = time.monotonic() + delay
                    state.status = "backoff"
                    logger.info(
                        "{} backoff: next restart in {:.0f}s (failures={})",
                        spec.name,
                        delay,
                        state.consecutive_failures,
                    )
                    continue

                # Ready to (re)start?
                now = time.monotonic()
                eligible_at = next_restart_eligible.get(spec.name, 0.0)
                if now < eligible_at:
                    remaining = eligible_at - now
                    state.status = "backoff"
                    logger.debug(
                        "{} in backoff -- {:.0f}s remaining", spec.name, remaining
                    )
                    continue

                # Start it.
                try:
                    start_process(state)
                    next_restart_eligible[spec.name] = 0.0
                except Exception as exc:
                    logger.error("Failed to start {}: {}", spec.name, exc)
                    state.record_failure()
                    delay = state.backoff_seconds(args.max_backoff)
                    next_restart_eligible[spec.name] = time.monotonic() + delay
                    state.status = "backoff"

            # Write heartbeat at the end of each cycle.
            _write_heartbeat(heartbeat_path, states)

            # Sleep in small increments so we notice signals quickly.
            deadline = time.monotonic() + args.check_interval
            while time.monotonic() < deadline and not shutdown_requested:
                time.sleep(0.5)

    finally:
        # -- Graceful shutdown: terminate all children we spawned -----------
        logger.info("Shutting down -- terminating child processes")
        for state in states:
            stop_process(state)

        # Final heartbeat reflecting stopped status.
        for state in states:
            if state.popen is None and not state.externally_running:
                state.status = "stopped"
        _write_heartbeat(heartbeat_path, states)
        logger.info("Watchdog exited cleanly")


if __name__ == "__main__":
    main()
