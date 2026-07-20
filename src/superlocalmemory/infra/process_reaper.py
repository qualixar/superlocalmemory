# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 -- Process Reaper.

Detects and kills orphaned SLM MCP processes whose parent IDE/Claude session
has died. Prevents RAM exhaustion from zombie embedding workers (1.5-2 GB each).

Discovery: March 30, 2026 -- Saturday session's SLM MCP (PID 15493) still alive
Monday night, consuming 1.8 GB.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from superlocalmemory.infra.pid_manager import PidManager

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Local config (not modifying core/config.py per instructions)
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ReaperConfig:
    """Configuration for process health & stale reaper (Phase H0).

    Ships enabled by default. Prevents zombie SLM processes from
    exhausting RAM (1.5-2 GB per orphaned embedding worker).

    Advisory ranges: heartbeat 10-300s, orphan_age 0.5-48h,
    graceful_timeout 1-30s.
    """

    enabled: bool = True
    heartbeat_interval_seconds: int = 60
    orphan_age_threshold_hours: float = 1.0  # v3.4.57: reduced from 4.0 — confirmed-dead parent = safe to kill after 1h
    pid_file_path: str = ""  # Empty = default (~/.superlocalmemory/slm.pids)
    graceful_timeout_seconds: float = 5.0


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

KNOWN_LAUNCHER_NAMES: frozenset[str] = frozenset({
    "node", "claude", "code", "cursor", "windsurf", "zed",
    "vim", "nvim", "emacs", "idea", "pycharm", "webstorm",
    "python", "python3",
})


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class SlmProcessInfo:
    """Information about a running SLM process."""

    pid: int
    ppid: int
    start_time: float
    command: str
    is_orphan: bool
    parent_name: str
    age_hours: float


def is_mcp_server_process(process: SlmProcessInfo) -> bool:
    """Return whether *process* is an MCP stdio server, not an SLM daemon.

    The startup reaper is deliberately aggressive (zero age threshold), but
    only an orphaned MCP process is safe to reap from ``slm mcp``.  A unified
    daemon is intentionally detached from the shell that launched it, so on
    some hosts it has PPID 1 and looks like an orphan.  Reaping that daemon at
    MCP startup drops in-flight CLI and dashboard requests.

    Keep this predicate narrow and based on the command line recorded by the
    OS: every supported MCP entry point includes the standalone ``mcp``
    argument or imports the MCP server module.  The daemon's command does
    neither.
    """
    command = process.command.lower()
    return (
        "/slm mcp" in command
        or " slm mcp" in command
        or "superlocalmemory.cli.main mcp" in command
        or "superlocalmemory.mcp.server" in command
    )


# ---------------------------------------------------------------------------
# Windows no-op stubs (AUDIT FIX H0-HIGH-02)
# ---------------------------------------------------------------------------

if sys.platform == "win32":  # pragma: no cover
    logger.info(
        "Process reaper not supported on Windows, "
        "all functions return no-op results"
    )

    def find_slm_processes() -> list[SlmProcessInfo]:
        return []

    def _check_parent(ppid: int) -> tuple[bool, str]:
        return (False, "")

    def find_orphans(config: ReaperConfig) -> list[SlmProcessInfo]:
        return []

    def kill_orphan(
        pid: int, graceful_timeout_seconds: float = 5.0
    ) -> dict[str, object]:
        return {
            "pid": pid, "killed": False,
            "method": "unsupported",
            "error": "Windows not supported",
        }

    def reap_stale_on_startup(
        config: ReaperConfig, pid_manager: PidManager
    ) -> dict[str, object]:
        return {
            "orphans_found": 0, "orphans_killed": 0,
            "errors": [], "registered_pid": 0,
        }

    def cleanup_all_orphans(
        config: ReaperConfig, dry_run: bool = False, force: bool = False
    ) -> dict[str, object]:
        return {
            "total_found": 0, "orphans_found": 0, "killed": 0,
            "skipped": 0, "errors": [], "processes": [],
        }

else:
    # -----------------------------------------------------------------------
    # Full POSIX implementation
    # -----------------------------------------------------------------------

    def _check_parent(ppid: int) -> tuple[bool, str]:
        """Check if a parent process is alive and is a known launcher.

        Returns (is_orphan, parent_name).
        Conservative: never marks a living process as orphan.
        """
        if ppid <= 1:
            return (True, "init")

        try:
            os.kill(ppid, 0)
        except ProcessLookupError:
            return (True, "")
        except PermissionError:
            return (False, "unknown")

        # Parent is alive -- try to read its name
        parent_name = ""
        try:
            result = subprocess.run(
                ["ps", "-p", str(ppid), "-o", "comm="],
                capture_output=True, text=True, timeout=5,
            )
            if result.returncode == 0:
                parent_name = os.path.basename(result.stdout.strip())
        except (subprocess.SubprocessError, OSError):
            pass

        return (False, parent_name)

    def find_slm_processes() -> list[SlmProcessInfo]:
        """Find all running SLM-related processes on the system.

        Uses POSIX `ps` command. Returns empty list on failure.
        """
        my_pid = os.getpid()
        results: list[SlmProcessInfo] = []

        try:
            proc = subprocess.run(
                ["ps", "-eo", "pid,ppid,lstart,command"],
                capture_output=True, text=True, timeout=10,
            )
            if proc.returncode != 0:
                logger.warning(
                    "ps command failed with code %d", proc.returncode
                )
                return []
        except (subprocess.SubprocessError, FileNotFoundError) as exc:
            logger.warning("Cannot run ps command: %s", exc)
            return []

        for line in proc.stdout.strip().split("\n")[1:]:
            line = line.strip()
            if not line:
                continue

            # v3.5.8: Expand detection to include entry-point launched `slm mcp`
            # processes. Previously only `python -m superlocalmemory.*` was caught;
            # `/opt/homebrew/bin/slm mcp` processes were invisible to the reaper,
            # causing zombie accumulation (55 processes / ~5 GB RAM observed).
            _slm_identifiers = (
                "superlocalmemory",
                "/slm ",
                "slm mcp",
                "slm serve",
                "slm daemon",
            )
            if not any(kw in line for kw in _slm_identifiers):
                continue

            try:
                parts = line.split()
                pid = int(parts[0])
                ppid = int(parts[1])

                if pid == my_pid:
                    continue

                # lstart is 5 fields: "Mon Mar 30 14:25:01 2026"
                lstart_str = " ".join(parts[2:7])
                command = " ".join(parts[7:])

                if not any(
                    kw in command.lower() for kw in ("mcp", "server", "slm")
                ):
                    continue

                try:
                    start_epoch = time.mktime(
                        time.strptime(lstart_str, "%a %b %d %H:%M:%S %Y")
                    )
                except ValueError:
                    try:
                        start_epoch = time.mktime(
                            time.strptime(lstart_str, "%c")
                        )
                    except ValueError:
                        logger.debug(
                            "Cannot parse lstart '%s', skipping PID %d",
                            lstart_str, pid,
                        )
                        continue

                age_hours = (time.time() - start_epoch) / 3600.0
                is_orphan, parent_name = _check_parent(ppid)

                results.append(SlmProcessInfo(
                    pid=pid,
                    ppid=ppid,
                    start_time=start_epoch,
                    command=command,
                    is_orphan=is_orphan,
                    parent_name=parent_name,
                    age_hours=age_hours,
                ))

            except (ValueError, IndexError) as exc:
                logger.debug("Cannot parse ps line '%s': %s", line, exc)
                continue

        results.sort(key=lambda p: p.pid)
        return results

    def find_orphans(config: ReaperConfig) -> list[SlmProcessInfo]:
        """Filter SLM processes to only confirmed orphans safe to kill.

        Safety invariants:
        - Parent PID is confirmed dead or reparented to PID 1
        - Process has been running longer than the age threshold
        - Process is not self and not own parent
        """
        all_procs = find_slm_processes()
        orphans: list[SlmProcessInfo] = []

        for p in all_procs:
            if not p.is_orphan:
                continue

            if p.age_hours < config.orphan_age_threshold_hours:
                logger.warning(
                    "Young orphan PID %d (age=%.1fh < threshold=%.1fh), "
                    "skipping",
                    p.pid, p.age_hours,
                    config.orphan_age_threshold_hours,
                )
                continue

            # Triple safety: HR-09
            if p.pid == os.getpid():
                continue
            if p.pid == os.getppid():
                continue

            orphans.append(p)

        return orphans

    def kill_orphan(
        pid: int,
        graceful_timeout_seconds: float = 5.0,
    ) -> dict[str, object]:
        """Kill a single orphaned process safely.

        Triple safety check (HR-02):
        1. Refuse PID <= 1 (init/launchd)
        2. Refuse self-kill
        3. Refuse parent-kill

        Two-phase kill (HR-04):
        1. SIGTERM (graceful)
        2. SIGKILL (forced, after timeout)
        """
        # --- Safety checks (HR-02) ---
        if pid <= 1:
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": (
                    f"Safety check failed: refusing to kill PID {pid} (<= 1)"
                ),
            }
        if pid == os.getpid():
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": "Safety check failed: refusing to kill self",
            }
        if pid == os.getppid():
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": "Safety check failed: refusing to kill own parent",
            }

        # --- Verify target still exists ---
        try:
            os.kill(pid, 0)
        except ProcessLookupError:
            return {
                "pid": pid, "killed": False,
                "method": "already_dead", "error": None,
            }
        except PermissionError:
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": f"Permission denied checking PID {pid}",
            }

        # --- Phase 1: SIGTERM (graceful) ---
        try:
            os.kill(pid, signal.SIGTERM)
        except ProcessLookupError:
            return {
                "pid": pid, "killed": False,
                "method": "already_dead", "error": None,
            }
        except PermissionError:
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": f"Permission denied killing PID {pid}",
            }

        # --- Wait for graceful death ---
        deadline = time.monotonic() + graceful_timeout_seconds
        while time.monotonic() < deadline:
            time.sleep(0.5)
            try:
                os.kill(pid, 0)
            except ProcessLookupError:
                logger.info(
                    "PID %d terminated gracefully (SIGTERM)", pid
                )
                return {
                    "pid": pid, "killed": True,
                    "method": "sigterm", "error": None,
                }

        # --- Phase 2: SIGKILL (forced) ---
        try:
            os.kill(pid, signal.SIGKILL)
            time.sleep(0.5)
            logger.warning(
                "PID %d force-killed (SIGKILL after %ds timeout)",
                pid, graceful_timeout_seconds,
            )
            return {
                "pid": pid, "killed": True,
                "method": "sigkill", "error": None,
            }
        except ProcessLookupError:
            return {
                "pid": pid, "killed": True,
                "method": "sigterm", "error": None,
            }
        except PermissionError:
            return {
                "pid": pid, "killed": False, "method": "refused",
                "error": f"Permission denied force-killing PID {pid}",
            }

    def reap_stale_on_startup(
        config: ReaperConfig, pid_manager: PidManager
    ) -> dict[str, object]:
        """Run on MCP server startup to clean orphans and register self.

        Cleans dead PIDs from the PID file, kills confirmed orphans,
        then registers the current process.
        """
        result: dict[str, object] = {
            "orphans_found": 0,
            "orphans_killed": 0,
            "errors": [],
            "registered_pid": os.getpid(),
        }

        if not config.enabled:
            logger.info("Reaper disabled, skipping orphan scan")
            pid_manager.register(os.getpid(), os.getppid())
            return result

        # Phase 1: Clean dead PIDs from PID file
        records = pid_manager.read_all()
        for record in records:
            try:
                os.kill(record.pid, 0)
            except ProcessLookupError:
                pid_manager.remove(record.pid)
                continue
            except PermissionError:
                continue

            # Process alive -- check if orphaned
            is_orphan, _parent_name = _check_parent(record.ppid)
            if is_orphan:
                started = record.started_at
                # Estimate age from started_at
                try:
                    from datetime import datetime, UTC
                    start_dt = datetime.fromisoformat(started)
                    age_hours = (
                        datetime.now(UTC) - start_dt
                    ).total_seconds() / 3600.0
                except (ValueError, TypeError):
                    age_hours = 0.0

                if age_hours > config.orphan_age_threshold_hours:
                    result["orphans_found"] = (
                        int(result["orphans_found"]) + 1
                    )
                    kill_result = kill_orphan(record.pid)
                    if kill_result["killed"]:
                        result["orphans_killed"] = (
                            int(result["orphans_killed"]) + 1
                        )
                        pid_manager.remove(record.pid)
                    elif kill_result["error"]:
                        errors = list(result["errors"])  # type: ignore[arg-type]
                        errors.append(str(kill_result["error"]))
                        result["errors"] = errors

        # Phase 2: Also scan for untracked orphans
        tracked_pids = {r.pid for r in pid_manager.read_all()}
        try:
            untracked_orphans = [
                o for o in find_orphans(config)
                if o.pid not in tracked_pids
            ]
            for orphan in untracked_orphans:
                logger.warning(
                    "Found untracked orphan PID %d", orphan.pid
                )
                result["orphans_found"] = (
                    int(result["orphans_found"]) + 1
                )
                kill_result = kill_orphan(orphan.pid)
                if kill_result["killed"]:
                    result["orphans_killed"] = (
                        int(result["orphans_killed"]) + 1
                    )
                elif kill_result["error"]:
                    errors = list(result["errors"])  # type: ignore[arg-type]
                    errors.append(str(kill_result["error"]))
                    result["errors"] = errors
        except Exception as exc:
            logger.warning("Untracked orphan scan failed: %s", exc)

        # Phase 3: Register self
        pid_manager.register(os.getpid(), os.getppid())

        logger.info(
            "Startup reaper: found=%d, killed=%d, errors=%d",
            result["orphans_found"],
            result["orphans_killed"],
            len(result["errors"]),  # type: ignore[arg-type]
        )

        return result

    def cleanup_all_orphans(
        config: ReaperConfig,
        dry_run: bool = False,
        force: bool = False,
    ) -> dict[str, object]:
        """CLI-facing function for `slm cleanup`.

        Finds and optionally kills all orphans.
        With --force, kills ALL SLM processes except current.
        With --dry-run, reports but does not kill.
        """
        result: dict[str, object] = {
            "total_found": 0,
            "orphans_found": 0,
            "killed": 0,
            "skipped": 0,
            "errors": [],
            "processes": [],
        }

        all_procs = find_slm_processes()
        result["total_found"] = len(all_procs)
        processes: list[dict[str, object]] = []

        if force:
            for p in all_procs:
                if p.pid == os.getpid():
                    continue

                proc_info: dict[str, object] = {
                    "pid": p.pid, "ppid": p.ppid,
                    "age_hours": p.age_hours,
                    "parent_name": p.parent_name,
                    "command": p.command,
                }

                if dry_run:
                    proc_info["status"] = "would_kill"
                    processes.append(proc_info)
                else:
                    kill_result = kill_orphan(p.pid)
                    if kill_result["killed"]:
                        result["killed"] = int(result["killed"]) + 1
                        proc_info["status"] = "killed"
                    else:
                        proc_info["status"] = "error"
                        proc_info["error"] = kill_result["error"]
                        if kill_result["error"]:
                            errors = list(result["errors"])  # type: ignore[arg-type]
                            errors.append(str(kill_result["error"]))
                            result["errors"] = errors
                    processes.append(proc_info)

        else:
            orphans = find_orphans(config)
            orphan_pids = {o.pid for o in orphans}
            result["orphans_found"] = len(orphans)

            for orphan in orphans:
                proc_info = {
                    "pid": orphan.pid, "ppid": orphan.ppid,
                    "age_hours": orphan.age_hours,
                    "parent_name": orphan.parent_name,
                    "command": orphan.command,
                }

                if dry_run:
                    proc_info["status"] = "would_kill"
                    processes.append(proc_info)
                else:
                    kill_result = kill_orphan(orphan.pid)
                    if kill_result["killed"]:
                        result["killed"] = int(result["killed"]) + 1
                        proc_info["status"] = "killed"
                    else:
                        proc_info["status"] = "error"
                        proc_info["error"] = kill_result["error"]
                        if kill_result["error"]:
                            errors = list(result["errors"])  # type: ignore[arg-type]
                            errors.append(str(kill_result["error"]))
                            result["errors"] = errors
                    processes.append(proc_info)

            # Report non-orphan processes as skipped
            for p in all_procs:
                if p.pid not in orphan_pids:
                    processes.append({
                        "pid": p.pid, "ppid": p.ppid,
                        "age_hours": p.age_hours,
                        "parent_name": p.parent_name,
                        "command": p.command,
                        "status": "active",
                    })
                    result["skipped"] = int(result["skipped"]) + 1

        result["processes"] = processes
        return result
