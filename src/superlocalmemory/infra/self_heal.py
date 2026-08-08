# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""Boot self-heal: idempotently remove provably-dead SLM lock/PID artifacts.

H1 from the No-Deadlock Hardening Plan:
  - Called on every daemon boot BEFORE the writer claim.
  - Removes ONLY artifacts whose owner PID is provably dead.
  - Never removes an artifact whose owner is a verified-live SLM process.
  - Never kills any process — file removal only.
  - PID-reuse safety: verifies process create_time or command-line when a PID
    is numerically alive to avoid treating an unrelated process as ours.

H4 from the plan (mesh TTL expiry):
  - expire_stale_mesh_locks() purges expired mesh_locks rows on boot.
"""

from __future__ import annotations

import json
import logging
import os
import socket
from pathlib import Path

logger = logging.getLogger(__name__)

# Plain-text single-PID files to check.
_PLAIN_PID_NAMES: tuple[str, ...] = (
    ".reranker-worker.pid",
    ".embedding.lock",
    ".config.json.lock",
    "daemon.pid",
)

# Unix socket artifact names.
_SOCKET_NAMES: tuple[str, ...] = ("hook_daemon.sock",)

# SLM process fingerprints for command-line checks.
# Covers the main daemon, packaged CLI entry points, and all named workers.
_SLM_EXECUTABLE_NAMES: tuple[str, ...] = (
    "slm",
    "superlocalmemory",
    "unified_daemon.py",
    "remember_runtime.py",
    "reranker_worker.py",
    "embedding_worker.py",
    "recall_worker.py",
)


# ---------------------------------------------------------------------------
# Low-level PID helpers (additive; reuse CLI helpers where possible)
# ---------------------------------------------------------------------------

def _is_pid_alive(pid: int) -> bool:
    """Return True iff *pid* exists in the kernel process table.

    G-09: PermissionError means the process EXISTS but we cannot signal it
    (e.g. different user).  We treat it as ALIVE to avoid wrongly removing
    a live process's artifact.
    """
    try:
        import psutil
        return psutil.pid_exists(pid)
    except ImportError:
        try:
            os.kill(pid, 0)
            return True
        except ProcessLookupError:
            return False
        except PermissionError:
            # Process exists but we lack permission to signal it — treat as alive.
            return True


def _pid_create_time(pid: int) -> float | None:
    """Return process create_time (seconds since epoch) or None if unavailable."""
    try:
        import psutil
        return psutil.Process(pid).create_time()
    except Exception:
        return None


def _pid_cmdline(pid: int) -> str:
    """Return space-joined cmdline for *pid*, empty string on error."""
    try:
        import psutil
        return " ".join(psutil.Process(pid).cmdline())
    except Exception:
        return ""


def _pid_cmdline_parts(pid: int) -> list[str]:
    """Return argv components for *pid*, or an empty list on failure."""
    try:
        import psutil
        return [str(part) for part in psutil.Process(pid).cmdline()]
    except Exception:
        return []


def _pid_is_slm(pid: int) -> bool:
    """Return True iff *pid* is alive AND its command line looks like an SLM process."""
    if not _is_pid_alive(pid):
        return False
    argv = _pid_cmdline_parts(pid)
    if not argv:
        return False
    executable = Path(argv[0]).name.lower()
    if executable in _SLM_EXECUTABLE_NAMES:
        return True
    if not (executable.startswith("python") or executable.startswith("pypy")):
        return False

    # Parse interpreter arguments in order.  Only a leading ``-m`` module or
    # the first script operand can establish identity; anything after ``-c``
    # or a script operand belongs to the executed program, not the interpreter.
    index = 1
    options_with_values = {"-W", "-X", "--check-hash-based-pycs"}
    while index < len(argv):
        part = argv[index]
        if part == "--":
            if index + 1 >= len(argv):
                return False
            return Path(argv[index + 1]).name.lower() in _SLM_EXECUTABLE_NAMES
        if part == "-":
            return False
        if part == "-m":
            if index + 1 >= len(argv):
                return False
            module = argv[index + 1].lower()
            return module == "superlocalmemory" or module.startswith(
                "superlocalmemory."
            )
        if part.startswith("-m") and len(part) > 2:
            module = part[2:].lower()
            return module == "superlocalmemory" or module.startswith(
                "superlocalmemory."
            )
        if part.startswith("-c"):
            return False
        if part in options_with_values:
            index += 2
            continue
        if part.startswith("-W") or part.startswith("-X"):
            index += 1
            continue
        if part.startswith("-"):
            index += 1
            continue
        return Path(part).name.lower() in _SLM_EXECUTABLE_NAMES
    return False


def _pid_matches_claimed_at(pid: int, claimed_at_ms: int) -> bool:
    """Return True iff process create_time matches *claimed_at_ms* within 300 s.

    Converts the millisecond timestamp from the writer-lock JSON to seconds
    and compares against the kernel's process-start epoch.

    G-03: window widened from 10 s → 300 s.  This is a SECONDARY signal only —
    create_time mismatch alone never removes a verified-live SLM process.
    It is only checked to detect PID reuse when the PID is alive but NOT
    identified as SLM by _pid_is_slm().
    """
    actual = _pid_create_time(pid)
    if actual is None:
        # psutil unavailable — cannot verify, assume matches (safe default).
        return True
    claimed_s = claimed_at_ms / 1000.0
    return abs(actual - claimed_s) <= 300.0


# ---------------------------------------------------------------------------
# Per-artifact-type PID extraction
# ---------------------------------------------------------------------------

def _read_json_pid(path: Path) -> tuple[int | None, int | None]:
    """Read ``(pid, claimed_at_ms)`` from a JSON writer-lock metadata file.

    The ``*.writer.lock`` files written by WriteCoordinator contain::

        {"pid": N, "owner_id": "...", "claimed_at_ms": N, "database": "..."}

    Returns ``(None, None)`` when the file is absent, empty, or malformed.
    """
    try:
        raw = path.read_bytes()
        if not raw.strip():
            return None, None
        data = json.loads(raw)
        pid = int(data["pid"])
        claimed_at_ms_raw = data.get("claimed_at_ms")
        claimed_at_ms = int(claimed_at_ms_raw) if claimed_at_ms_raw else None
        return pid, claimed_at_ms
    except Exception:
        return None, None


def _read_plain_pid(path: Path) -> int | None:
    """Read a plain-text (integer) PID file. Returns ``None`` on any error."""
    try:
        return int(path.read_text(encoding="utf-8").strip())
    except Exception:
        return None


def _socket_has_listener(sock_path: Path) -> bool:
    """Return True iff a Unix-domain socket file has an active listener.

    G-06: retry 3× with short backoff (total < 1.5 s) before returning False.
    The daemon may be mid-bind when we check at boot; a single probe can give
    a false-negative that causes a valid socket to be deleted.

    Backoff schedule: attempt 0 (immediate), attempt 1 (+0.2 s), attempt 2 (+0.4 s).
    Per-attempt connect timeout: 0.3 s.  Total wall time ≤ 0.9 s + 0.6 s = 1.5 s.
    """
    import time

    _DELAYS = (0.0, 0.2, 0.4)  # pre-attempt sleep seconds
    for delay in _DELAYS:
        if delay > 0:
            time.sleep(delay)
        try:
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect(str(sock_path))
            s.close()
            return True
        except (ConnectionRefusedError, FileNotFoundError, OSError):
            continue
        except Exception:
            continue
    return False


# ---------------------------------------------------------------------------
# Safe-remove helper
# ---------------------------------------------------------------------------

def _safe_unlink(path: Path, report: dict, reason: str) -> None:
    """Remove *path* and record the result in *report*. Never raises."""
    try:
        path.unlink(missing_ok=True)
        report["removed"].append({"path": str(path), "reason": reason})
        logger.info("self_heal: removed stale artifact %s (%s)", path.name, reason)
    except OSError as exc:
        report["errors"].append({"path": str(path), "error": str(exc)})
        logger.warning("self_heal: could not remove %s: %s", path, exc)


# ---------------------------------------------------------------------------
# Public API — H1
# ---------------------------------------------------------------------------

def reap_stale_artifacts(data_dir: Path) -> dict:
    """Idempotently remove SLM artifacts whose owner PID is provably dead.

    Safety invariants (never violated):
    - A live, verified SLM process's artifacts are NEVER removed.
    - No process is ever signalled or killed; only file paths are unlinked.
    - PID-reuse is detected and treated as a dead owner (removes the stale
      artifact, never the live unrelated process).

    Returns a report dict::

        {
            "removed": [{"path": str, "reason": str}, ...],
            "kept":    [str, ...],       # paths left because owner is alive
            "errors":  [{"path": str, "error": str}, ...],
        }
    """
    report: dict = {"removed": [], "kept": [], "errors": []}
    data_dir = Path(data_dir)
    if not data_dir.is_dir():
        return report

    # --- JSON writer-lock metadata files (*.writer.lock) --------------------
    # The portalocker flock is already auto-released when the holder dies;
    # these JSON bodies are purely informational.  Removing them when the
    # recorded PID is dead or reused clears up stale metadata without
    # affecting the actual OS advisory lock.
    for lock_file in data_dir.glob("*.writer.lock"):
        pid, claimed_at_ms = _read_json_pid(lock_file)
        if pid is None:
            # Empty or unreadable — safe to remove.
            _safe_unlink(lock_file, report, "unreadable_metadata")
            continue
        if not _is_pid_alive(pid):
            _safe_unlink(lock_file, report, "dead_owner_pid")
        elif _pid_is_slm(pid):
            # G-03: live AND verified SLM owner → KEEP unconditionally.
            # create_time check is skipped — a live SLM process always wins.
            report["kept"].append(str(lock_file))
        elif claimed_at_ms is not None and not _pid_matches_claimed_at(pid, claimed_at_ms):
            # Alive but NOT SLM and create_time mismatch → PID was reused.
            _safe_unlink(lock_file, report, "pid_reused")
        else:
            # Alive but cmdline is not SLM → reused by an unrelated process.
            _safe_unlink(lock_file, report, "pid_reused_non_slm")

    # --- Plain-text PID files -----------------------------------------------
    for name in _PLAIN_PID_NAMES:
        path = data_dir / name
        if not path.exists():
            continue
        pid = _read_plain_pid(path)
        if pid is None:
            _safe_unlink(path, report, "unreadable_pid_file")
            continue
        if not _is_pid_alive(pid):
            _safe_unlink(path, report, "dead_owner_pid")
        elif not _pid_is_slm(pid):
            # PID alive but command is not SLM → reused by an unrelated process.
            _safe_unlink(path, report, "pid_reused_non_slm")
        else:
            report["kept"].append(str(path))

    # --- Unix socket artifacts -----------------------------------------------
    for name in _SOCKET_NAMES:
        path = data_dir / name
        if not path.exists():
            continue
        if not _socket_has_listener(path):
            _safe_unlink(path, report, "no_listener")
        else:
            report["kept"].append(str(path))

    return report


# ---------------------------------------------------------------------------
# Public API — H4 (team-mode mesh-lock expiry)
# ---------------------------------------------------------------------------

_MESH_NEVER_EXPIRES = "9999-12-31T23:59:59Z"


def expire_stale_mesh_locks(db_path: Path) -> int:
    """Delete expired TTL rows from ``mesh_locks`` on boot.

    A row is stale iff:
    - ``expires_at`` is not NULL,
    - ``expires_at`` is not the legacy ``_NEVER_EXPIRES`` sentinel, and
    - ``expires_at <= now_iso`` (i.e. the lease has elapsed).

    Keeps fencing intact: only the TTL expiry gate is touched.  Any row whose
    lease is still valid is left untouched.  Fail-soft — returns 0 on error
    (e.g. if the mesh_locks table does not exist yet).

    Returns the count of deleted rows.
    """
    import sqlite3
    from datetime import datetime, timezone

    now = datetime.now(timezone.utc).isoformat()
    try:
        conn = sqlite3.connect(str(db_path), timeout=2.0)
        try:
            # G-07: BEGIN IMMEDIATE prevents a concurrent write from seeing our
            # DELETE mid-flight.  On OperationalError (DB locked by another
            # writer) we log a warning and return 0 — fail-soft, non-blocking.
            try:
                conn.execute("BEGIN IMMEDIATE")
            except sqlite3.OperationalError as _lock_exc:
                logger.warning(
                    "expire_stale_mesh_locks: DB locked, skipping this boot: %s",
                    _lock_exc,
                )
                return 0
            cur = conn.execute(
                "DELETE FROM mesh_locks "
                "WHERE expires_at IS NOT NULL "
                "  AND expires_at != ? "
                "  AND expires_at <= ?",
                (_MESH_NEVER_EXPIRES, now),
            )
            conn.commit()
            deleted = cur.rowcount or 0
            if deleted:
                logger.info(
                    "self_heal: expired %d stale mesh_lock row(s) from %s",
                    deleted, db_path.name,
                )
            return deleted
        finally:
            conn.close()
    except Exception as exc:
        logger.debug(
            "expire_stale_mesh_locks: %s (table may not exist yet)", exc,
        )
        return 0


__all__ = [
    "reap_stale_artifacts",
    "expire_stale_mesh_locks",
]
