# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-05 §2, §4.4, §9.3

"""Cross-platform adapter base — shared Protocol and atomic write primitive.

LLD reference: ``.backup/active-brain/lld/LLD-05-cross-platform-adapters.md``
Sections: 2 (component catalog), 4.4 (atomic write), 9.3 (sync log contract).

Every adapter (Cursor, Antigravity, Copilot) implements the ``Adapter``
Protocol. The shared ``_atomic_write`` primitive enforces the hard-rule
matrix (A1–A3, A7): ``safe_resolve`` + ``O_NOFOLLOW`` + tempfile +
``os.replace`` + durable content-hash skip via ``cross_platform_sync_log``.

Hard rules enforced here:
  - A1: every write goes through ``safe_resolve``; POSIX uses ``O_NOFOLLOW``.
  - A2: atomic replace via tempfile → ``os.replace``.
  - A3: durable content-hash skip — previous ``content_sha256`` comes from
    the DB, so this survives daemon restarts.
  - A7: sync log records ``target_path_sha256`` (full 64-hex, no truncation)
    + ``target_basename``; never the raw absolute path string.
"""

from __future__ import annotations

import hashlib
import os
import sqlite3
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Protocol, runtime_checkable


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

HARD_BYTES_CAP = 4096
COPILOT_SOFT_BYTES = 2048
TRUNCATION_MARKER = b"\n<!-- truncated -->"


# ---------------------------------------------------------------------------
# Protocol
# ---------------------------------------------------------------------------


@runtime_checkable
class Adapter(Protocol):
    """Contract every cross-platform adapter implements."""

    name: str

    @property
    def target_path(self) -> Path: ...

    def is_active(self) -> bool: ...

    def sync(self) -> bool: ...

    def disable(self) -> None: ...


# ---------------------------------------------------------------------------
# Sync-log helpers (LLD-07 M004 canonical columns)
# ---------------------------------------------------------------------------


def path_sha256(path: Path) -> str:
    """SHA-256 of the absolute path string, full 64-hex (never truncated)."""
    return hashlib.sha256(str(path.resolve() if path.exists()
                              else path).encode("utf-8")).hexdigest()


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _ensure_memory_log(db_path: Path) -> None:
    """Lazily create ``cross_platform_sync_log`` if a test-mode memory.db is
    fresh. Production code goes through the migration runner, but tests can
    hand us an empty DB; this keeps adapters usable without pre-running
    migrations."""
    conn = sqlite3.connect(str(db_path))
    try:
        conn.executescript(
            "CREATE TABLE IF NOT EXISTS cross_platform_sync_log ("
            " adapter_name TEXT NOT NULL,"
            " profile_id TEXT NOT NULL,"
            " target_path_sha256 TEXT NOT NULL,"
            " target_basename TEXT NOT NULL,"
            " last_sync_at TEXT NOT NULL,"
            " bytes_written INTEGER NOT NULL,"
            " content_sha256 TEXT NOT NULL,"
            " success INTEGER NOT NULL,"
            " error_msg TEXT,"
            " PRIMARY KEY (adapter_name, target_path_sha256));"
        )
        conn.commit()
    finally:
        conn.close()


def sync_log_last_content_sha256(
    db_path: Path, adapter_name: str, target_path_sha256: str,
) -> str | None:
    """Read last successful ``content_sha256`` for this (adapter, path)."""
    if not db_path.exists():
        return None
    _ensure_memory_log(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        row = conn.execute(
            "SELECT content_sha256, success FROM cross_platform_sync_log "
            "WHERE adapter_name = ? AND target_path_sha256 = ?",
            (adapter_name, target_path_sha256),
        ).fetchone()
    except sqlite3.Error:
        return None
    finally:
        conn.close()
    if row is None:
        return None
    content_hash, success = row
    if not success:
        return None
    return content_hash


def sync_log_record(
    db_path: Path,
    *,
    adapter_name: str,
    profile_id: str,
    target_path_sha256: str,
    target_basename: str,
    bytes_written: int,
    content_sha256: str,
    success: bool,
    error_msg: str | None = None,
) -> None:
    """Upsert the canonical M004 row. Rule A7: hash is full 64 hex, no raw path.

    Raises ``ValueError`` if caller attempts to sneak a raw absolute path into
    ``target_path_sha256`` (defence in depth for the CI grep guard).
    """
    if len(target_path_sha256) != 64:
        raise ValueError(
            f"target_path_sha256 must be 64 hex chars, got {len(target_path_sha256)}"
        )
    if os.sep in target_path_sha256 or "/" in target_path_sha256:
        raise ValueError("target_path_sha256 must be a hash, not a raw path")
    _ensure_memory_log(db_path)
    conn = sqlite3.connect(str(db_path))
    try:
        conn.execute(
            "INSERT INTO cross_platform_sync_log ("
            "adapter_name, profile_id, target_path_sha256, target_basename, "
            "last_sync_at, bytes_written, content_sha256, success, error_msg"
            ") VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(adapter_name, target_path_sha256) DO UPDATE SET "
            " profile_id = excluded.profile_id,"
            " target_basename = excluded.target_basename,"
            " last_sync_at = excluded.last_sync_at,"
            " bytes_written = excluded.bytes_written,"
            " content_sha256 = excluded.content_sha256,"
            " success = excluded.success,"
            " error_msg = excluded.error_msg",
            (
                adapter_name, profile_id, target_path_sha256, target_basename,
                _now_iso(), bytes_written, content_sha256,
                1 if success else 0, error_msg,
            ),
        )
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Atomic write with durable content-hash skip (A1, A2, A3, A7)
# ---------------------------------------------------------------------------


def _is_posix() -> bool:
    return os.name == "posix"


@dataclass(frozen=True, slots=True)
class WriteResult:
    """Outcome of ``atomic_write``."""
    wrote: bool
    bytes_written: int
    content_sha256: str


def atomic_write(
    resolved_path: Path,
    content: bytes,
    *,
    adapter_name: str,
    profile_id: str,
    sync_log_db: Path,
    posix_mode: int = 0o600,
    windows_mode: int = 0o644,
) -> WriteResult:
    """Atomic write with durable content-hash skip.

    ``resolved_path`` MUST already be a ``safe_resolve`` output. Passing a
    raw user-controlled path bypasses the hard-rule A1 guarantee — adapters
    are responsible for calling ``safe_resolve`` first (enforced by tests).
    """
    new_hash = hashlib.sha256(content).hexdigest()
    target_sha = path_sha256(resolved_path)
    prev = sync_log_last_content_sha256(sync_log_db, adapter_name, target_sha)

    if prev == new_hash and resolved_path.exists():
        # Durable skip — no write, no new sync-log row (the prior row still
        # reflects on-disk truth).
        return WriteResult(wrote=False, bytes_written=0, content_sha256=new_hash)

    resolved_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = resolved_path.with_suffix(resolved_path.suffix + ".slm-tmp")

    flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
    if hasattr(os, "O_NOFOLLOW") and _is_posix():
        flags |= os.O_NOFOLLOW  # SEC — POSIX refuses symlinks

    mode = posix_mode if _is_posix() else windows_mode
    fd = os.open(str(tmp), flags, mode)
    try:
        os.write(fd, content)
        try:
            os.fsync(fd)
        except OSError:  # pragma: no cover — exotic FS
            pass
    finally:
        os.close(fd)
    os.replace(tmp, resolved_path)

    if _is_posix():
        try:
            os.chmod(resolved_path, posix_mode)
        except OSError:  # pragma: no cover
            pass

    sync_log_record(
        sync_log_db,
        adapter_name=adapter_name,
        profile_id=profile_id,
        target_path_sha256=target_sha,
        target_basename=resolved_path.name,
        bytes_written=len(content),
        content_sha256=new_hash,
        success=True,
        error_msg=None,
    )
    return WriteResult(wrote=True, bytes_written=len(content),
                       content_sha256=new_hash)


def record_disable(
    resolved_path: Path,
    *,
    adapter_name: str,
    profile_id: str,
    sync_log_db: Path,
) -> None:
    """Log a disable row (LLD-05 §9.4). File deletion is the caller's job."""
    target_sha = path_sha256(resolved_path)
    sync_log_record(
        sync_log_db,
        adapter_name=adapter_name,
        profile_id=profile_id,
        target_path_sha256=target_sha,
        target_basename=resolved_path.name,
        bytes_written=0,
        content_sha256="0" * 64,
        success=True,
        error_msg="disabled_by_user",
    )


def truncate_to_cap(content: bytes, *, cap: int = HARD_BYTES_CAP) -> bytes:
    """Truncate ``content`` to ``cap`` bytes, appending the marker.

    Adapters call this AFTER domain-level truncation (trim sections) and
    treat the marker as a last-resort safety net — not the primary cap
    mechanism. Kept in the base module because all three adapters need it.
    """
    if len(content) <= cap:
        return content
    marker = TRUNCATION_MARKER
    # Leave room for the marker in the cap.
    head = content[: max(0, cap - len(marker))]
    return head + marker


__all__ = (
    "Adapter",
    "HARD_BYTES_CAP",
    "COPILOT_SOFT_BYTES",
    "TRUNCATION_MARKER",
    "WriteResult",
    "atomic_write",
    "path_sha256",
    "record_disable",
    "sync_log_last_content_sha256",
    "sync_log_record",
    "truncate_to_cap",
)


# Internal platform flag used by adapters to choose permissions.
IS_POSIX = _is_posix()
