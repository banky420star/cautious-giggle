"""
Cautious Giggle -- Production Orchestrator
Unified startup, lifecycle management, and graceful shutdown for all services.

Replaces ad-hoc process launching with a deterministic startup sequence,
health checks between each step, PID tracking, and watchdog hand-off.

Usage:
    python tools/production_orchestrator.py            # start all services
    python tools/production_orchestrator.py --status   # print service status table
    python tools/production_orchestrator.py --shutdown  # graceful shutdown of all managed services
"""
import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime, timezone

from loguru import logger

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Python.config_utils import load_project_config

NO_WINDOW = getattr(subprocess, "CREATE_NO_WINDOW", 0)
HOST = "127.0.0.1"
PORT = 8088

PID_DIR = os.path.join(PROJECT_ROOT, ".tmp")
PID_FILE = os.path.join(PID_DIR, "orchestrator.json")
LOG_DIR = os.path.join(PROJECT_ROOT, "logs")
LOG_FILE = os.path.join(LOG_DIR, "orchestrator.log")
CONFIG_PATH = os.path.join(PROJECT_ROOT, "config.yaml")
MODELS_DIR = os.path.join(PROJECT_ROOT, "models")
HFT_CONFIG = os.path.join(PROJECT_ROOT, "config_hft.yaml")
ENV_PATH = os.path.join(PROJECT_ROOT, ".env")

# Service names used as keys throughout the orchestrator
SVC_UI = "ui"
SVC_SERVER_AGI = "server_agi"
SVC_HFT = "hft_server"
SVC_CHAMPION = "champion_cycle"
SVC_WATCHDOG = "watchdog"

SERVICE_ORDER = [SVC_UI, SVC_SERVER_AGI, SVC_HFT, SVC_CHAMPION, SVC_WATCHDOG]

SERVICE_LABELS = {
    SVC_UI: "Dashboard UI",
    SVC_SERVER_AGI: "Server_AGI (M5 standard)",
    SVC_HFT: "HFT Server (M1)",
    SVC_CHAMPION: "Champion Cycle Loop",
    SVC_WATCHDOG: "Watchdog",
}

# Fragments used by wmic to detect already-running processes
PROCESS_SIGNATURES = {
    SVC_UI: "project_status_ui",
    SVC_SERVER_AGI: "Server_AGI",
    SVC_HFT: "AGI_MODE_TAG=hft",
    SVC_CHAMPION: "champion_cycle_loop",
    SVC_WATCHDOG: "watchdog.py",
}

SERVICE_LOCK_FILES = {
    SVC_SERVER_AGI: os.path.join(PID_DIR, "server_agi.lock"),
    SVC_HFT: os.path.join(PID_DIR, "server_agi_hft.lock"),
    SVC_CHAMPION: os.path.join(PID_DIR, "champion_cycle.lock"),
}

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------

def _configure_logging() -> None:
    """Set up loguru to write to the orchestrator log file."""
    logger.remove()
    logger.add(sys.stderr, level="INFO", format="<green>{time:HH:mm:ss}</green> | <level>{level:<8}</level> | {message}")
    os.makedirs(LOG_DIR, exist_ok=True)
    logger.add(
        LOG_FILE,
        rotation="10 MB",
        retention="7 days",
        level="DEBUG",
        format="{time:YYYY-MM-DD HH:mm:ss.SSS} | {level:<8} | {message}",
    )


# ---------------------------------------------------------------------------
# Utility helpers
# ---------------------------------------------------------------------------

def _port_open(host: str, port: int, timeout: float = 1.0) -> bool:
    """Return True if a TCP connection to *host*:*port* succeeds."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        sock.settimeout(timeout)
        return sock.connect_ex((host, port)) == 0


def _http_200(host: str, port: int, path: str = "/", timeout: float = 2.0) -> bool:
    """Return True if an HTTP GET to *host*:*port*/*path* returns status 200."""
    try:
        import http.client
        conn = http.client.HTTPConnection(host, port, timeout=timeout)
        conn.request("GET", path)
        resp = conn.getresponse()
        conn.close()
        return resp.status == 200
    except Exception:
        return False


def _process_running_wmic(name_fragment: str) -> bool:
    """Check if a process whose command line contains *name_fragment* is alive (Windows)."""
    try:
        out = subprocess.check_output(
            ["wmic", "process", "get", "CommandLine"],
            creationflags=NO_WINDOW,
            stderr=subprocess.DEVNULL,
            text=True,
        )
        return name_fragment in out
    except Exception:
        return False


def _read_pid_from_file(path: str) -> int:
    try:
        with open(path, "r", encoding="utf-8") as fh:
            return int((fh.read() or "").strip() or "0")
    except Exception:
        return 0


def _pid_file_alive(path: str) -> bool:
    if not os.path.exists(path):
        return False
    pid = _read_pid_from_file(path)
    return _pid_alive(pid)


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


def _validate_live_config(config_path: str) -> tuple[bool, str | None]:
    prev_cfg = os.environ.get("AGI_CONFIG")
    try:
        os.environ["AGI_CONFIG"] = config_path
        load_project_config(PROJECT_ROOT, live_mode=True)
        return True, None
    except Exception as exc:
        return False, str(exc)
    finally:
        if prev_cfg is None:
            os.environ.pop("AGI_CONFIG", None)
        else:
            os.environ["AGI_CONFIG"] = prev_cfg


def _is_service_running(svc: str) -> bool:
    """Determine whether *svc* is already running using the best available heuristic."""
    if svc == SVC_UI:
        return _port_open(HOST, PORT)
    lock_path = SERVICE_LOCK_FILES.get(svc)
    if lock_path and _pid_file_alive(lock_path):
        return True
    sig = PROCESS_SIGNATURES.get(svc)
    if sig:
        return _process_running_wmic(sig)
    return False


def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a live process."""
    if pid <= 0:
        return False
    try:
        if sys.platform == "win32":
            # On Windows, os.kill with signal 0 raises OSError if process is gone
            import ctypes
            kernel32 = ctypes.windll.kernel32
            handle = kernel32.OpenProcess(0x100000, False, pid)  # SYNCHRONIZE
            if handle:
                kernel32.CloseHandle(handle)
                return True
            return False
        else:
            os.kill(pid, 0)
            return True
    except (OSError, ProcessLookupError):
        return False


def _make_env(**extras: str) -> dict[str, str]:
    """Return a copy of the current environment with *extras* merged in."""
    env = os.environ.copy()
    env.update(extras)
    return env


def _terminate_pid(pid: int, label: str = "") -> None:
    """Send SIGTERM to *pid*, wait 5 s, then SIGKILL if still alive."""
    tag = f" ({label})" if label else ""
    try:
        if sys.platform == "win32":
            # On Windows, terminate via subprocess.Popen handle if possible;
            # fall back to taskkill.
            subprocess.run(
                ["taskkill", "/PID", str(pid), "/F"],
                creationflags=NO_WINDOW,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Sent taskkill /F to PID {}{}.", pid, tag)
        else:
            os.kill(pid, signal.SIGTERM)
            logger.info("Sent SIGTERM to PID {}{}.", pid, tag)
            deadline = time.monotonic() + 5.0
            while time.monotonic() < deadline:
                if not _pid_alive(pid):
                    return
                time.sleep(0.25)
            os.kill(pid, signal.SIGKILL)
            logger.info("Sent SIGKILL to PID {}{}.", pid, tag)
    except Exception as exc:
        logger.debug("Could not terminate PID {}{}: {}", pid, tag, exc)


# ---------------------------------------------------------------------------
# PID file management
# ---------------------------------------------------------------------------

def _load_pid_file() -> dict:
    """Load the orchestrator PID file, returning an empty structure on failure."""
    try:
        with open(PID_FILE, "r", encoding="utf-8") as fh:
            return json.load(fh)
    except Exception:
        return {}


def _save_pid_file(data: dict) -> None:
    """Persist *data* to the orchestrator PID file."""
    os.makedirs(PID_DIR, exist_ok=True)
    tmp = PID_FILE + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    # Atomic rename (best-effort on Windows)
    try:
        os.replace(tmp, PID_FILE)
    except OSError:
        # Fallback: remove then rename
        try:
            os.remove(PID_FILE)
        except FileNotFoundError:
            pass
        os.rename(tmp, PID_FILE)


def _update_service_pid(svc: str, pid: int, status: str = "running") -> None:
    """Set the PID and status for a single service in the PID file."""
    data = _load_pid_file()
    if "services" not in data:
        data["services"] = {}
    data["services"][svc] = {"pid": pid, "status": status}
    _save_pid_file(data)


def _init_pid_file() -> dict:
    """Create a fresh PID file structure with an ISO-8601 start timestamp."""
    data = {
        "started": datetime.now(timezone.utc).isoformat(),
        "services": {},
    }
    _save_pid_file(data)
    return data


# ---------------------------------------------------------------------------
# Status table
# ---------------------------------------------------------------------------

_STATUS_ICONS = {
    "running": "[OK]",
    "skipped": "[SKIP]",
    "failed": "[FAIL]",
    "stopped": "[--]",
    "unknown": "[??]",
}


def _resolve_service_status(svc: str, info: dict) -> str:
    """Return the live status string for a service based on PID + heuristic."""
    pid = info.get("pid", 0)
    recorded = info.get("status", "unknown")
    if recorded in ("skipped",):
        return "skipped"
    if _is_service_running(svc):
        return "running"
    if pid and _pid_alive(pid):
        return "running"
    if recorded == "running":
        return "stopped"
    return recorded


def _print_status_table(data: dict) -> None:
    """Print a formatted table of all services and their states."""
    started = data.get("started", "N/A")
    services = data.get("services", {})

    print()
    print("=" * 62)
    print("  Cautious Giggle — Production Orchestrator")
    print(f"  Started: {started}")
    print("=" * 62)
    print(f"  {'Service':<30} {'PID':>7}  {'Status':<10}")
    print("-" * 62)

    for svc in SERVICE_ORDER:
        label = SERVICE_LABELS.get(svc, svc)
        info = services.get(svc, {})
        pid = info.get("pid", "")
        status = _resolve_service_status(svc, info)
        icon = _STATUS_ICONS.get(status, "[??]")
        pid_str = str(pid) if pid else "-"
        print(f"  {label:<30} {pid_str:>7}  {icon} {status}")

    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Pre-flight checks
# ---------------------------------------------------------------------------

def _preflight() -> bool:
    """Run pre-flight checks.  Return True if all critical checks pass."""
    ok = True
    _load_env_file(ENV_PATH)

    if os.environ.get("AGI_TOKEN", "").strip():
        logger.info("Pre-flight: AGI_TOKEN available via environment/.env.")
        _print_step(SVC_SERVER_AGI, "AGI_TOKEN", "configured", ok=True, pre=True)
    else:
        logger.error("Pre-flight FAIL: AGI_TOKEN is not set in the environment or .env.")
        _print_step(SVC_SERVER_AGI, "AGI_TOKEN", "MISSING", ok=False, pre=True)
        ok = False

    # 1. config.yaml
    if os.path.isfile(CONFIG_PATH):
        logger.info("Pre-flight: config.yaml found.")
        _print_step(SVC_UI, "config.yaml", "found", ok=True, pre=True)
        cfg_ok, cfg_error = _validate_live_config(CONFIG_PATH)
        if cfg_ok:
            logger.info("Pre-flight: live config validation OK.")
            _print_step(SVC_SERVER_AGI, "live config", "validated", ok=True, pre=True)
        else:
            logger.error("Pre-flight FAIL: live config validation failed: {}", cfg_error)
            _print_step(SVC_SERVER_AGI, "live config", f"INVALID ({cfg_error})", ok=False, pre=True)
            ok = False
    else:
        logger.error("Pre-flight FAIL: config.yaml not found at {}", CONFIG_PATH)
        _print_step(SVC_UI, "config.yaml", "MISSING", ok=False, pre=True)
        ok = False

    if os.path.isfile(HFT_CONFIG):
        logger.info("Pre-flight: config_hft.yaml found.")
        _print_step(SVC_HFT, "config_hft.yaml", "found", ok=True, pre=True)
        hft_ok, hft_error = _validate_live_config(HFT_CONFIG)
        if hft_ok:
            logger.info("Pre-flight: HFT config validation OK.")
            _print_step(SVC_HFT, "HFT live config", "validated", ok=True, pre=True)
        else:
            logger.error("Pre-flight FAIL: HFT live config validation failed: {}", hft_error)
            _print_step(SVC_HFT, "HFT live config", f"INVALID ({hft_error})", ok=False, pre=True)
            ok = False
    else:
        logger.error("Pre-flight FAIL: config_hft.yaml not found at {}", HFT_CONFIG)
        _print_step(SVC_HFT, "config_hft.yaml", "MISSING", ok=False, pre=True)
        ok = False

    # 2. models directory
    if os.path.isdir(MODELS_DIR):
        logger.info("Pre-flight: models/ directory exists.")
        _print_step(SVC_UI, "models/", "found", ok=True, pre=True)
    else:
        logger.error("Pre-flight FAIL: models/ directory not found at {}", MODELS_DIR)
        _print_step(SVC_UI, "models/", "MISSING", ok=False, pre=True)
        ok = False

    # 3. MT5 connection (optional — warn only)
    try:
        import MetaTrader5 as mt5
        if mt5.initialize():
            logger.info("Pre-flight: MT5 connection OK.")
            _print_step(SVC_UI, "MT5 connection", "OK", ok=True, pre=True)
            mt5.shutdown()
        else:
            logger.warning("Pre-flight: MT5 initialize() returned False — continuing without MT5.")
            _print_step(SVC_UI, "MT5 connection", "WARN (not connected)", ok=True, pre=True)
    except ImportError:
        logger.warning("Pre-flight: MetaTrader5 package not installed — continuing.")
        _print_step(SVC_UI, "MT5 connection", "WARN (package missing)", ok=True, pre=True)
    except Exception as exc:
        logger.warning("Pre-flight: MT5 check failed ({}) — continuing.", exc)
        _print_step(SVC_UI, "MT5 connection", f"WARN ({exc})", ok=True, pre=True)

    return ok


def _print_step(svc: str, description: str, result: str, *, ok: bool, pre: bool = False) -> None:
    """Print a single step line in the startup table."""
    tag = "PRE" if pre else SERVICE_LABELS.get(svc, svc)
    marker = "[OK]" if ok else "[FAIL]"
    print(f"  {marker}  {tag:<30} {description}: {result}")


# ---------------------------------------------------------------------------
# Service launchers
# ---------------------------------------------------------------------------

def _launch_ui() -> subprocess.Popen | None:
    """Start Dashboard UI server and wait for HTTP 200 (up to 30 s)."""
    if _port_open(HOST, PORT):
        logger.info("Dashboard UI already listening on {}:{}; skipping launch.", HOST, PORT)
        return None

    env = _make_env(AGI_UI_HOST=HOST, AGI_UI_PORT=str(PORT))
    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(PROJECT_ROOT, "tools", "project_status_ui.py")],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    logger.info("Dashboard UI launched (PID {}).  Waiting for HTTP 200 on {}:{} ...", proc.pid, HOST, PORT)

    deadline = time.monotonic() + 30.0
    while time.monotonic() < deadline:
        if proc.poll() is not None:
            logger.error("Dashboard UI process exited prematurely (code {}).", proc.returncode)
            return proc
        if _http_200(HOST, PORT):
            logger.info("Dashboard UI healthy — HTTP 200 received.")
            return proc
        time.sleep(0.5)

    # Timed out — check if at least the port is open
    if _port_open(HOST, PORT):
        logger.warning("Dashboard UI port open but HTTP 200 not confirmed within 30 s.  Continuing.")
    else:
        logger.error("Dashboard UI did not start within 30 s timeout.")
    return proc


def _launch_server_agi() -> subprocess.Popen | None:
    """Start Server_AGI (M5 standard) and confirm it stays alive for 5 s."""
    if _is_service_running(SVC_SERVER_AGI):
        logger.info("Server_AGI already running; skipping launch.")
        return None

    proc = subprocess.Popen(
        [sys.executable, "-m", "Python.Server_AGI", "--live"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    logger.info("Server_AGI launched (PID {}).  Waiting 5 s to confirm stability...", proc.pid)
    time.sleep(5)

    if proc.poll() is not None:
        logger.error("Server_AGI crashed within 5 s (exit code {}).", proc.returncode)
    else:
        logger.info("Server_AGI stable after 5 s.")
    return proc


def _launch_hft() -> subprocess.Popen | None:
    """Start HFT Server (M1) and confirm it stays alive for 5 s."""
    if _is_service_running(SVC_HFT):
        logger.info("HFT Server already running; skipping launch.")
        return None

    env = _make_env(
        AGI_CONFIG=HFT_CONFIG,
        AGI_MODE_TAG="hft",
        AGI_LOOP_SEC="5",
        AGI_HEARTBEAT_SEC="300",
        AGI_SYMBOL_CARD_SEC="60",
        AGI_TRADE_LEARN_SEC="300",
    )
    proc = subprocess.Popen(
        [sys.executable, "-m", "Python.Server_AGI", "--live"],
        cwd=PROJECT_ROOT,
        env=env,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    logger.info("HFT Server launched (PID {}).  Waiting 5 s to confirm stability...", proc.pid)
    time.sleep(5)

    if proc.poll() is not None:
        logger.error("HFT Server crashed within 5 s (exit code {}).", proc.returncode)
    else:
        logger.info("HFT Server stable after 5 s.")
    return proc


def _launch_champion() -> subprocess.Popen | None:
    """Start Champion Cycle Loop and verify it is alive."""
    if _process_running_wmic("champion_cycle_loop"):
        logger.info("Champion Cycle Loop already running; skipping launch.")
        return None

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(PROJECT_ROOT, "tools", "champion_cycle_loop.py"),
         "--interval-minutes", "30"],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    logger.info("Champion Cycle Loop launched (PID {}).", proc.pid)
    time.sleep(1)

    if proc.poll() is not None:
        logger.error("Champion Cycle Loop exited immediately (code {}).", proc.returncode)
    else:
        logger.info("Champion Cycle Loop alive.")
    return proc


def _launch_watchdog() -> subprocess.Popen | None:
    """Start Watchdog and verify it is alive."""
    if _process_running_wmic("watchdog.py"):
        logger.info("Watchdog already running; skipping launch.")
        return None

    proc = subprocess.Popen(
        [sys.executable, "-u", os.path.join(PROJECT_ROOT, "tools", "watchdog.py")],
        cwd=PROJECT_ROOT,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
        creationflags=NO_WINDOW,
    )
    logger.info("Watchdog launched (PID {}).", proc.pid)
    time.sleep(1)

    if proc.poll() is not None:
        logger.error("Watchdog exited immediately (code {}).", proc.returncode)
    else:
        logger.info("Watchdog alive.")
    return proc


# Mapping from service key to launcher function
_LAUNCHERS = {
    SVC_UI: _launch_ui,
    SVC_SERVER_AGI: _launch_server_agi,
    SVC_HFT: _launch_hft,
    SVC_CHAMPION: _launch_champion,
    SVC_WATCHDOG: _launch_watchdog,
}

# ---------------------------------------------------------------------------
# Startup orchestration
# ---------------------------------------------------------------------------

def _run_startup() -> None:
    """Execute the full startup sequence with pre-flight checks, ordered launches, and PID tracking."""
    print()
    print("=" * 62)
    print("  Cautious Giggle — Production Orchestrator")
    print(f"  {datetime.now(timezone.utc).isoformat()}")
    print("=" * 62)
    print()
    print("  Pre-flight checks")
    print("-" * 62)

    if not _preflight():
        logger.error("Pre-flight checks failed.  Aborting startup.")
        print()
        print("  [FAIL] Pre-flight checks failed — aborting.")
        print()
        sys.exit(1)

    print()
    print("  Service startup sequence")
    print("-" * 62)

    pid_data = _init_pid_file()
    services = pid_data["services"]
    procs: dict[str, subprocess.Popen | None] = {}

    for svc in SERVICE_ORDER:
        label = SERVICE_LABELS[svc]
        launcher = _LAUNCHERS[svc]

        already = _is_service_running(svc)
        if already:
            print(f"  [SKIP] {label:<35} already running")
            logger.info("{} already running — skipped.", label)
            services[svc] = {"pid": 0, "status": "skipped"}
            procs[svc] = None
            _save_pid_file(pid_data)
            continue

        proc = launcher()
        procs[svc] = proc

        if proc is None:
            # Launcher detected it was already running internally
            print(f"  [SKIP] {label:<35} already running")
            services[svc] = {"pid": 0, "status": "skipped"}
        elif proc.poll() is not None:
            print(f"  [FAIL] {label:<35} exited (code {proc.returncode})")
            services[svc] = {"pid": proc.pid, "status": "failed"}
        else:
            print(f"  [OK]   {label:<35} PID {proc.pid}")
            services[svc] = {"pid": proc.pid, "status": "running"}

        _save_pid_file(pid_data)

    print()
    print("=" * 62)

    # Final summary
    running = sum(1 for s in services.values() if s["status"] == "running")
    skipped = sum(1 for s in services.values() if s["status"] == "skipped")
    failed = sum(1 for s in services.values() if s["status"] == "failed")

    summary = f"  Done: {running} started, {skipped} skipped, {failed} failed"
    print(summary)
    logger.info("Startup complete: {} started, {} skipped, {} failed.", running, skipped, failed)

    if failed:
        print("  WARNING: Some services failed to start.  Check logs/orchestrator.log")
        logger.warning("Some services failed to start.  Review logs for details.")

    print(f"  PID file: {PID_FILE}")
    print(f"  Log file: {LOG_FILE}")
    print("=" * 62)
    print()


# ---------------------------------------------------------------------------
# Status mode
# ---------------------------------------------------------------------------

def _run_status() -> None:
    """Print the current status of all managed services."""
    data = _load_pid_file()
    if not data:
        print()
        print("  No orchestrator PID file found.  Has the orchestrator been started?")
        print(f"  Expected: {PID_FILE}")
        print()
        return
    _print_status_table(data)


# ---------------------------------------------------------------------------
# Shutdown mode
# ---------------------------------------------------------------------------

def _run_shutdown() -> None:
    """Gracefully shut down all managed services (SIGTERM, wait 5 s, SIGKILL)."""
    data = _load_pid_file()
    if not data:
        print()
        print("  No orchestrator PID file found.  Nothing to shut down.")
        print()
        return

    services = data.get("services", {})
    print()
    print("=" * 62)
    print("  Cautious Giggle — Shutting down managed services")
    print("=" * 62)

    # Shut down in reverse order for clean teardown
    for svc in reversed(SERVICE_ORDER):
        label = SERVICE_LABELS.get(svc, svc)
        info = services.get(svc, {})
        pid = info.get("pid", 0)
        status = info.get("status", "unknown")

        if status == "skipped":
            print(f"  [SKIP] {label:<35} was not managed by orchestrator")
            logger.info("Shutdown: {} was skipped at startup, not terminating.", label)
            continue

        if not pid or pid <= 0:
            print(f"  [--]   {label:<35} no PID recorded")
            continue

        if not _pid_alive(pid):
            print(f"  [--]   {label:<35} PID {pid} already dead")
            logger.info("Shutdown: {} (PID {}) already dead.", label, pid)
            info["status"] = "stopped"
            continue

        logger.info("Shutdown: terminating {} (PID {})...", label, pid)
        _terminate_pid(pid, label)

        # Verify termination
        time.sleep(0.5)
        if _pid_alive(pid):
            print(f"  [WARN] {label:<35} PID {pid} still alive after kill")
            logger.warning("Shutdown: {} (PID {}) still alive after termination attempt.", label, pid)
            info["status"] = "running"
        else:
            print(f"  [OK]   {label:<35} PID {pid} terminated")
            info["status"] = "stopped"

    _save_pid_file(data)

    print("=" * 62)
    print("  Shutdown complete.")
    print("=" * 62)
    print()
    logger.info("Shutdown sequence finished.")


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Cautious Giggle — Production Orchestrator.  Start, monitor, or stop all services.",
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--status",
        action="store_true",
        help="Print the current state of all managed services and exit.",
    )
    group.add_argument(
        "--shutdown",
        action="store_true",
        help="Gracefully terminate all managed services and exit.",
    )
    return parser.parse_args()


def main() -> None:
    _configure_logging()
    args = _parse_args()

    # Ensure cwd is project root so relative paths inside sub-processes work
    os.chdir(PROJECT_ROOT)

    if args.status:
        _run_status()
    elif args.shutdown:
        logger.info("Shutdown requested via CLI.")
        _run_shutdown()
    else:
        logger.info("Startup requested via CLI.")
        _run_startup()


if __name__ == "__main__":
    main()
