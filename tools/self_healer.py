"""Self-healing module: detects and fixes common failure modes.

Can be imported by other tools (watchdog, champion_cycle_loop) or run
standalone as a periodic cleanup daemon.

    python -m tools.self_healer --interval 300
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from loguru import logger

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# ---------------------------------------------------------------------------
# Optional psutil import with graceful fallback
# ---------------------------------------------------------------------------
try:
    import psutil  # type: ignore[import-untyped]

    PSUTIL_AVAILABLE = True
except ImportError:
    psutil = None  # type: ignore[assignment]
    PSUTIL_AVAILABLE = False

# ---------------------------------------------------------------------------
# Known lock files that this healer is aware of
# ---------------------------------------------------------------------------
KNOWN_LOCKS = ("champion_cycle.lock", "server_agi.lock", "hft.lock")

# Thresholds
LOG_MAX_BYTES = 50 * 1024 * 1024  # 50 MB
DISK_WARNING_GB = 1.0
MEMORY_WARNING_PCT = 90.0
MEMORY_CRITICAL_PCT = 95.0
HEARTBEAT_STALE_SECONDS = 5 * 60  # 5 minutes


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _pid_alive(pid: int) -> bool:
    """Return True if *pid* refers to a running process."""
    if pid <= 0:
        return False
    try:
        os.kill(pid, 0)
        return True
    except OSError:
        return False


def _now_utc() -> datetime:
    return datetime.now(timezone.utc)


def _ts_tag() -> str:
    """Return a timestamp string suitable for file suffixes."""
    return _now_utc().strftime("%Y%m%d_%H%M%S")


def _age_days(path: str) -> float:
    """Return the age of *path* in fractional days based on mtime."""
    try:
        mtime = os.path.getmtime(path)
    except OSError:
        return 0.0
    return (time.time() - mtime) / 86400.0


def _get_free_disk_gb(path: str) -> float:
    """Return free disk space in GB for the volume containing *path*."""
    try:
        usage = shutil.disk_usage(path)
        return usage.free / (1024 ** 3)
    except Exception:
        return float("inf")


def _read_memory_windows_wmic() -> float | None:
    """Fallback memory reader for Windows via wmic."""
    try:
        result = subprocess.run(
            ["wmic", "OS", "get", "FreePhysicalMemory,TotalVisibleMemorySize", "/VALUE"],
            capture_output=True,
            text=True,
            timeout=10,
        )
        lines = result.stdout.strip().splitlines()
        values: dict[str, int] = {}
        for line in lines:
            line = line.strip()
            if "=" in line:
                key, val = line.split("=", 1)
                try:
                    values[key.strip()] = int(val.strip())
                except ValueError:
                    pass
        total_kb = values.get("TotalVisibleMemorySize", 0)
        free_kb = values.get("FreePhysicalMemory", 0)
        if total_kb > 0:
            used_pct = (1.0 - free_kb / total_kb) * 100.0
            return round(used_pct, 2)
    except Exception:
        pass
    return None


def _read_memory_proc() -> float | None:
    """Fallback memory reader for Linux via /proc/meminfo."""
    try:
        with open("/proc/meminfo", "r", encoding="utf-8") as fh:
            info: dict[str, int] = {}
            for line in fh:
                parts = line.split()
                if len(parts) >= 2:
                    key = parts[0].rstrip(":")
                    try:
                        info[key] = int(parts[1])
                    except ValueError:
                        pass
            total = info.get("MemTotal", 0)
            available = info.get("MemAvailable", 0)
            if total > 0:
                used_pct = (1.0 - available / total) * 100.0
                return round(used_pct, 2)
    except Exception:
        pass
    return None


# ---------------------------------------------------------------------------
# SelfHealer
# ---------------------------------------------------------------------------

class SelfHealer:
    """Detects and fixes common failure modes for the trading system."""

    def __init__(self, project_root: str):
        self.project_root = os.path.abspath(project_root)
        self.tmp_dir = os.path.join(self.project_root, ".tmp")
        self.logs_dir = os.path.join(self.project_root, "logs")
        self.candidates_dir = os.path.join(
            self.project_root, "models", "registry", "candidates"
        )

    # ------------------------------------------------------------------
    # 1. Stale lock cleanup
    # ------------------------------------------------------------------
    def clean_stale_locks(self) -> list[str]:
        """Remove stale lock files. Returns list of cleaned lock names."""
        cleaned: list[str] = []
        if not os.path.isdir(self.tmp_dir):
            return cleaned

        try:
            entries = os.listdir(self.tmp_dir)
        except OSError as exc:
            logger.error("Cannot list tmp dir {}: {}", self.tmp_dir, exc)
            return cleaned

        lock_files = [e for e in entries if e.endswith(".lock")]
        for name in lock_files:
            lock_path = os.path.join(self.tmp_dir, name)
            pid: int | None = None
            try:
                with open(lock_path, "r", encoding="utf-8") as fh:
                    raw = fh.read().strip()
                pid = int(raw) if raw else None
            except (ValueError, OSError):
                pid = None

            # If we cannot read a valid PID, treat the lock as stale.
            if pid is not None and _pid_alive(pid):
                logger.debug(
                    "Lock {} held by live pid={}, skipping", name, pid
                )
                continue

            # PID is dead (or unreadable) -- remove the lock
            try:
                os.remove(lock_path)
                logger.info(
                    "Removed stale lock {} (pid={})", name, pid
                )
                cleaned.append(name)
            except OSError as exc:
                logger.error("Failed to remove stale lock {}: {}", name, exc)

        return cleaned

    # ------------------------------------------------------------------
    # 2. Log rotation check
    # ------------------------------------------------------------------
    def rotate_large_logs(self) -> list[str]:
        """Rotate oversized .log files. Returns list of rotated files."""
        rotated: list[str] = []
        if not os.path.isdir(self.logs_dir):
            return rotated

        try:
            entries = os.listdir(self.logs_dir)
        except OSError as exc:
            logger.error("Cannot list logs dir {}: {}", self.logs_dir, exc)
            return rotated

        for name in entries:
            # Only rotate .log files, skip .jsonl data files
            if not name.endswith(".log"):
                continue

            full_path = os.path.join(self.logs_dir, name)
            if not os.path.isfile(full_path):
                continue

            try:
                size = os.path.getsize(full_path)
            except OSError:
                continue

            if size <= LOG_MAX_BYTES:
                continue

            tag = _ts_tag()
            bak_name = f"{name}.{tag}.bak"
            bak_path = os.path.join(self.logs_dir, bak_name)

            try:
                os.rename(full_path, bak_path)
                # Create empty replacement
                with open(full_path, "w", encoding="utf-8") as fh:
                    pass
                logger.info(
                    "Rotated {} ({:.1f} MB) -> {}",
                    name,
                    size / (1024 * 1024),
                    bak_name,
                )
                rotated.append(name)
            except OSError as exc:
                logger.error("Failed to rotate {}: {}", name, exc)

        return rotated

    # ------------------------------------------------------------------
    # 3. Disk space check
    # ------------------------------------------------------------------
    def check_disk_space(self) -> dict:
        """Check disk space. Returns {"free_gb": float, "warning": bool, "critical": bool}."""
        free_gb = _get_free_disk_gb(self.project_root)
        warning = free_gb < DISK_WARNING_GB
        critical = warning  # < 1 GB is both warning and critical here

        result: dict = {
            "free_gb": round(free_gb, 3),
            "warning": warning,
            "critical": critical,
        }

        if critical:
            logger.critical(
                "Disk space critically low: {:.2f} GB free", free_gb
            )
            # Emergency cleanup: delete old .bak logs
            bak_cleaned = self._delete_old_bak_logs(max_age_days=7)
            if bak_cleaned:
                logger.info("Emergency cleanup: deleted {} old .bak log files", len(bak_cleaned))
            result["bak_cleaned"] = bak_cleaned

            # Emergency cleanup: prune old candidates
            cand_cleaned = self.cleanup_old_candidates(max_age_days=30, keep_newest=5)
            if cand_cleaned:
                logger.info(
                    "Emergency cleanup: deleted {} old candidate dirs",
                    len(cand_cleaned),
                )
            result["candidates_cleaned"] = cand_cleaned

            # Re-check after cleanup
            result["free_gb_after"] = round(
                _get_free_disk_gb(self.project_root), 3
            )
        elif warning:
            logger.warning("Disk space low: {:.2f} GB free", free_gb)

        return result

    def _delete_old_bak_logs(self, max_age_days: int = 7) -> list[str]:
        """Delete logs/*.bak files older than *max_age_days*."""
        deleted: list[str] = []
        if not os.path.isdir(self.logs_dir):
            return deleted

        try:
            entries = os.listdir(self.logs_dir)
        except OSError:
            return deleted

        for name in entries:
            if not name.endswith(".bak"):
                continue
            full = os.path.join(self.logs_dir, name)
            if not os.path.isfile(full):
                continue
            if _age_days(full) > max_age_days:
                try:
                    os.remove(full)
                    logger.info("Deleted old bak log: {}", name)
                    deleted.append(name)
                except OSError as exc:
                    logger.error("Failed to delete {}: {}", name, exc)

        return deleted

    # ------------------------------------------------------------------
    # 4. MT5 connection recovery
    # ------------------------------------------------------------------
    def check_mt5(self) -> bool:
        """Check and recover MT5 connection. Returns True if healthy/recovered."""
        try:
            import MetaTrader5 as mt5  # type: ignore[import-untyped]
        except ImportError:
            logger.debug("MetaTrader5 not importable; skipping MT5 check")
            return False

        try:
            if mt5.initialize() and mt5.terminal_info() is not None:
                logger.debug("MT5 connection healthy")
                return True
        except Exception:
            pass

        # Connection is dead -- attempt recovery
        logger.warning("MT5 connection dead, attempting recovery")
        try:
            mt5.shutdown()
        except Exception:
            pass

        try:
            if mt5.initialize() and mt5.terminal_info() is not None:
                logger.info("MT5 connection recovered successfully")
                return True
        except Exception as exc:
            logger.error("MT5 recovery failed: {}", exc)

        logger.error("MT5 connection could not be recovered")
        return False

    # ------------------------------------------------------------------
    # 5. Memory pressure detection
    # ------------------------------------------------------------------
    def check_memory(self) -> dict:
        """Check memory usage. Returns {"used_pct": float, "warning": bool, "critical": bool}."""
        used_pct: float | None = None

        # Strategy 1: psutil
        if PSUTIL_AVAILABLE and psutil is not None:
            try:
                mem = psutil.virtual_memory()
                used_pct = round(mem.percent, 2)
            except Exception:
                used_pct = None

        # Strategy 2: platform-specific fallback
        if used_pct is None:
            if platform.system() == "Windows":
                used_pct = _read_memory_windows_wmic()
            else:
                used_pct = _read_memory_proc()

        if used_pct is None:
            logger.warning("Could not determine memory usage")
            return {"used_pct": -1.0, "warning": False, "critical": False}

        warning = used_pct >= MEMORY_WARNING_PCT
        critical = used_pct >= MEMORY_CRITICAL_PCT

        if critical:
            logger.critical("Memory usage critical: {:.1f}%", used_pct)
        elif warning:
            logger.warning("Memory usage high: {:.1f}%", used_pct)
        else:
            logger.debug("Memory usage OK: {:.1f}%", used_pct)

        return {
            "used_pct": used_pct,
            "warning": warning,
            "critical": critical,
        }

    # ------------------------------------------------------------------
    # 6. Heartbeat staleness check
    # ------------------------------------------------------------------
    def check_heartbeats(self) -> list[dict]:
        """Check heartbeat staleness. Returns list of stale entries."""
        stale: list[dict] = []
        if not os.path.isdir(self.tmp_dir):
            return stale

        try:
            entries = os.listdir(self.tmp_dir)
        except OSError as exc:
            logger.error("Cannot list tmp dir {}: {}", self.tmp_dir, exc)
            return stale

        now = time.time()
        for name in entries:
            if not name.endswith(".heartbeat"):
                continue

            full = os.path.join(self.tmp_dir, name)
            if not os.path.isfile(full):
                continue

            try:
                mtime = os.path.getmtime(full)
            except OSError:
                continue

            age_sec = now - mtime
            if age_sec > HEARTBEAT_STALE_SECONDS:
                logger.warning(
                    "Stale heartbeat: {} (age {:.0f}s, threshold {}s)",
                    name,
                    age_sec,
                    HEARTBEAT_STALE_SECONDS,
                )
                stale.append(
                    {
                        "file": name,
                        "age_seconds": round(age_sec, 1),
                        "last_beat": datetime.fromtimestamp(
                            mtime, tz=timezone.utc
                        ).isoformat(),
                    }
                )

        return stale

    # ------------------------------------------------------------------
    # 7. Cleanup old candidate model directories
    # ------------------------------------------------------------------
    def cleanup_old_candidates(
        self, max_age_days: int = 30, keep_newest: int = 5
    ) -> list[str]:
        """Remove old candidate model directories. Returns list of removed dir names."""
        removed: list[str] = []
        if not os.path.isdir(self.candidates_dir):
            return removed

        try:
            entries = os.listdir(self.candidates_dir)
        except OSError as exc:
            logger.error(
                "Cannot list candidates dir {}: {}", self.candidates_dir, exc
            )
            return removed

        # Collect candidate directories with their mtime
        dirs_with_mtime: list[tuple[str, float]] = []
        for name in entries:
            full = os.path.join(self.candidates_dir, name)
            if os.path.isdir(full):
                try:
                    mtime = os.path.getmtime(full)
                except OSError:
                    mtime = 0.0
                dirs_with_mtime.append((name, mtime))

        # Sort newest first
        dirs_with_mtime.sort(key=lambda t: t[1], reverse=True)

        # Always keep the newest *keep_newest* regardless of age
        protected = {name for name, _ in dirs_with_mtime[:keep_newest]}

        for name, _ in dirs_with_mtime:
            if name in protected:
                continue
            full = os.path.join(self.candidates_dir, name)
            age = _age_days(full)
            if age > max_age_days:
                try:
                    shutil.rmtree(full)
                    logger.info(
                        "Deleted old candidate dir: {} (age {:.0f} days)",
                        name,
                        age,
                    )
                    removed.append(name)
                except OSError as exc:
                    logger.error(
                        "Failed to delete candidate dir {}: {}", name, exc
                    )

        return removed

    # ------------------------------------------------------------------
    # Run all checks
    # ------------------------------------------------------------------
    def run_all_checks(self) -> dict:
        """Run all checks. Returns summary dict."""
        ts = _now_utc().isoformat()
        logger.info("Self-healer: starting full check sweep")

        locks = self._safe("clean_stale_locks", self.clean_stale_locks)
        disk = self._safe("check_disk_space", self.check_disk_space)
        memory = self._safe("check_memory", self.check_memory)
        rotated = self._safe("rotate_large_logs", self.rotate_large_logs)
        mt5_ok = self._safe("check_mt5", self.check_mt5)
        heartbeats = self._safe("check_heartbeats", self.check_heartbeats)
        candidates = self._safe(
            "cleanup_old_candidates", self.cleanup_old_candidates
        )

        summary = {
            "timestamp": ts,
            "stale_locks_cleaned": locks,
            "disk": disk,
            "memory": memory,
            "logs_rotated": rotated,
            "mt5_ok": mt5_ok,
            "stale_heartbeats": heartbeats,
            "candidates_cleaned": candidates,
        }

        logger.info("Self-healer: check sweep complete")
        return summary

    @staticmethod
    def _safe(label: str, fn):
        """Run *fn* and return its result; on error, log and return the error string."""
        try:
            return fn()
        except Exception as exc:
            logger.exception("Self-healer check '{}' failed: {}", label, exc)
            return {"error": str(exc)}


# ---------------------------------------------------------------------------
# Standalone daemon mode
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Self-healing daemon: periodic cleanup and health checks."
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=300,
        help="Seconds between check sweeps (default: 300).",
    )
    parser.add_argument(
        "--once",
        action="store_true",
        help="Run a single sweep and exit.",
    )
    return parser.parse_args()


def _append_jsonl(path: str, record: dict) -> None:
    """Append a JSON record to a .jsonl file."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "a", encoding="utf-8") as fh:
        fh.write(json.dumps(record, default=str) + "\n")


def main() -> None:
    args = _parse_args()
    healer = SelfHealer(PROJECT_ROOT)
    output_path = os.path.join(healer.logs_dir, "self_healer.jsonl")

    logger.info(
        "Self-healer daemon starting | interval={}s | output={}",
        args.interval,
        output_path,
    )

    while True:
        try:
            result = healer.run_all_checks()
            _append_jsonl(output_path, result)
        except Exception as exc:
            logger.exception("Self-healer sweep error: {}", exc)
            _append_jsonl(
                output_path,
                {
                    "timestamp": _now_utc().isoformat(),
                    "error": str(exc),
                },
            )

        if args.once:
            break
        time.sleep(max(1, args.interval))


if __name__ == "__main__":
    main()
