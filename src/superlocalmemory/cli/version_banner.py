# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Post-upgrade version banner.

Writes ``$SLM_DATA_DIR/.version`` (default ``~/.superlocalmemory/.version``)
and prints a short factual banner the first time the CLI or daemon is
invoked after a ``pip install -U`` / ``npm install -g`` that changes the
installed version. Every subsequent invocation is a no-op.

The banner is best-effort and must never raise — a disk error or a
corrupt marker makes the banner silent, not the CLI broken.
"""
from __future__ import annotations

import os
import re
import sys
from pathlib import Path

# Strict allowlist: semver-ish only. Anything else in the marker is
# treated as "unknown" so a tampered or binary marker can't leak into
# banner strings or downstream consumers.
_VERSION_PATTERN = re.compile(r"^[0-9A-Za-z.\-+_]{1,32}$")

_MAX_MARKER_BYTES = 64  # a semver string is ≤ 32 chars; 64 is plenty

_RELEASE_NOTES: dict[str, tuple[str, ...]] = {
    "3.8.1": (
        "Large existing databases no longer delay daemon readiness",
        "Failed enrichment retries stop after ten automatic attempts",
        "Dashboard panes stay mounted between navigation changes",
        "Local model workers stay warm for a 30-minute working session",
    ),
}


def _data_dir() -> Path:
    from superlocalmemory.infra.data_root import canonical_data_root
    return canonical_data_root()


def _marker_path() -> Path:
    return _data_dir() / ".version"


def read_marker_version() -> str | None:
    """Return the marker string or None if missing / unreadable / invalid."""
    target = _marker_path()
    # Refuse to follow symlinks on the marker file — a co-tenant
    # attacker who can drop a symlink into our data dir must not be
    # able to redirect us to an arbitrary file.
    try:
        st = target.lstat()
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    import stat as _stat
    if _stat.S_ISLNK(st.st_mode) or not _stat.S_ISREG(st.st_mode):
        return None
    try:
        with open(target, "rb") as f:
            raw = f.read(_MAX_MARKER_BYTES + 1)
    except (FileNotFoundError, NotADirectoryError, PermissionError, OSError):
        return None
    if len(raw) > _MAX_MARKER_BYTES:
        return None
    try:
        text = raw.decode("ascii").strip()
    except UnicodeDecodeError:
        return None
    if not _VERSION_PATTERN.fullmatch(text):
        return None
    return text


def write_marker_version(version: str) -> bool:
    """Persist the current version to the marker with 0600 perms.

    Uses a PID-scoped tmp file + ``os.replace`` for atomicity so
    concurrent CLI / daemon / npm-postinstall invocations can't corrupt
    each other's write. The final file is mode 0600 so the upgrade
    timestamp is not a side-channel for co-tenant attackers.
    """
    # Validate input — never write garbage, even on a programmer bug.
    if not _VERSION_PATTERN.fullmatch(version):
        return False
    target = _marker_path()
    try:
        target.parent.mkdir(parents=True, exist_ok=True)
        # Parent dir 0700 on POSIX (mirrors safe_fs._ensure_parent_0700)
        if sys.platform != "win32":
            try:
                os.chmod(target.parent, 0o700)
            except OSError:
                pass
        tmp = target.parent / f".version.tmp.{os.getpid()}"
        # Open with 0600 so the tmp is never world-readable either.
        flags = os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        if hasattr(os, "O_NOFOLLOW"):
            flags |= os.O_NOFOLLOW
        fd = os.open(tmp, flags, 0o600)
        try:
            os.write(fd, (version + "\n").encode("ascii"))
        finally:
            os.close(fd)
        os.replace(tmp, target)
        if sys.platform != "win32":
            try:
                os.chmod(target, 0o600)
            except OSError:
                pass
        return True
    except OSError:
        return False


def _has_existing_db() -> bool:
    """True if a memory.db already exists — signals a pre-v3.4.26 user
    upgrading in-place (the marker only began shipping in v3.4.26)."""
    return (_data_dir() / "memory.db").exists()


def _banner(prior: str | None, current: str) -> str:
    header = (f"SuperLocalMemory upgraded from {prior} to {current}"
              if prior else
              f"SuperLocalMemory upgraded to {current} (from an earlier version)")
    notes = _RELEASE_NOTES.get(
        current,
        (
            "The installed version has changed; existing memory remains in place",
            "See the release notes for version-specific changes",
        ),
    )
    return "\n".join([
        header,
        *(f"  - {note}" for note in notes),
        "Run `slm doctor` to verify your setup.",
        "",
    ])


def check_and_emit_upgrade_banner(current: str) -> bool:
    """Print the banner once per upgrade boundary. Idempotent.

    Returns True if the banner was emitted on this call, else False.
    Never raises — swallows I/O errors so a broken data directory cannot
    take down the CLI.

    Concurrent CLI + daemon + npm-postinstall callers serialize through
    an O_CREAT|O_EXCL lock file so at most one emits the banner.
    """
    try:
        prior = read_marker_version()

        if prior == current:
            return False

        # Fresh install: no marker, no DB. Stay quiet — the setup wizard
        # handles the welcome. Still write the marker so subsequent
        # invocations are no-ops.
        if prior is None and not _has_existing_db():
            write_marker_version(current)
            return False

        # Serialize concurrent emitters via O_CREAT|O_EXCL lock file.
        lock = _data_dir() / ".version-banner.lock"
        lock_fd = None
        try:
            lock.parent.mkdir(parents=True, exist_ok=True)
            lock_fd = os.open(
                str(lock), os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600,
            )
        except FileExistsError:
            return False
        except OSError:
            pass

        try:
            # Re-check after winning the lock — the loser may have
            # already written the marker while we raced.
            if read_marker_version() == current:
                return False

            # v3.6.13 (CRITICAL): write to STDERR, never stdout. On the `slm mcp`
            # stdio transport, ANY stdout byte that isn't JSON-RPC corrupts the
            # stream and the MCP client (e.g. Claude Desktop) rejects every
            # message as "not valid JSON". The banner is a human notice → stderr.
            sys.stderr.write(_banner(prior, current))
            sys.stderr.flush()
            write_marker_version(current)
            return True
        finally:
            if lock_fd is not None:
                os.close(lock_fd)
            try:
                lock.unlink(missing_ok=True)
            except OSError:
                pass
    except Exception:
        # Banner is advisory. A failure here must never propagate.
        return False
