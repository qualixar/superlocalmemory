# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The hook event spool: one durable queue between the hooks and the daemon.

WHY

Hooks used to write ``tool_events`` and reward signals straight into
memory.db with a 50-500 ms busy timeout and swallowed the failure. Under a
long memory write that lost a third to two thirds of the events, silently.
Now a hook never opens a write connection: it hands its events to the daemon,
which appends them here and acks, and the daemon's single writer thread moves
them into the database. When the daemon is down or slow, the hook appends here
itself. Either way the event is on disk before anyone says "done".

LAYOUT (``<data root>/hook_spool``)

* ``<profile>.jsonl`` -- the active file. Producers append one JSON line per
  event with a single ``O_APPEND`` write while holding ``.lock``.
* ``<profile>.<ns>.draining`` -- a file the daemon claimed (renamed under the
  same lock). Nobody appends to it again; the daemon drains it, then deletes it.
* ``stats.json`` -- events refused (spool full, oversized) and unreadable
  lines found while draining, per profile, for ``slm doctor``.

Stdlib only: this module runs inside every hook process.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
import time
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterator, Mapping, Sequence

SPOOL_DIR_NAME = "hook_spool"
#: Per-profile cap on spooled bytes (active + claimed files). About 10,000
#: ordinary tool events: several days of heavy use with the daemon stopped.
DEFAULT_MAX_BYTES = 16 * 1024 * 1024
#: One event line. Hook summaries are capped at 500 characters, so a real
#: event is far below this; anything above it is refused and counted.
MAX_LINE_BYTES = 16 * 1024
#: How long a producer waits for the spool lock. Holders keep it for one
#: append or one rename (microseconds), so this is only reached when a
#: process died holding it -- flock releases on exit, so that self-heals too.
LOCK_WAIT_S = 1.0

_ACTIVE_SUFFIX = ".jsonl"
_DRAINING_SUFFIX = ".draining"
_STATS_NAME = "stats.json"
_LOCK_NAME = ".lock"
_SAFE_PROFILE = re.compile(r"^[A-Za-z0-9_-]{1,64}$")  # no dot: files group by name.split(".")[0]


@dataclass(frozen=True)
class AppendResult:
    """What one append did: events written, and events refused (counted)."""

    written: int
    dropped: int


@dataclass(frozen=True)
class SpoolStats:
    """Read-only spool summary for ``slm doctor``."""

    pending_events: int
    pending_bytes: int
    dropped: int
    invalid: int
    last_drop_at: str


def _max_bytes_default() -> int:
    try:
        return max(4096, int(os.environ.get("SLM_HOOK_SPOOL_MAX_BYTES", DEFAULT_MAX_BYTES)))
    except (TypeError, ValueError):
        return DEFAULT_MAX_BYTES


def spool_root(root: str | Path | None = None) -> Path:
    """``<data root>/hook_spool``, or ``root`` when a caller pins one."""
    if root is not None:
        return Path(root)
    from superlocalmemory.infra.data_root import canonical_data_root

    return canonical_data_root() / SPOOL_DIR_NAME


def profile_stem(profile: str) -> str:
    """A filename-safe stem for ``profile``; unsafe names map to a hash."""
    name = str(profile or "default")
    if _SAFE_PROFILE.match(name):
        return name
    return "p-" + hashlib.sha256(name.encode("utf-8")).hexdigest()[:16]


def _ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    if os.name == "posix":
        try:
            os.chmod(path, 0o700)
        except OSError:
            pass
    return path


# ---------------------------------------------------------------------------
# Cross-process lock (flock on POSIX, a one-byte msvcrt lock on Windows)
# ---------------------------------------------------------------------------

def _try_lock(fd: int) -> bool:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        if os.fstat(fd).st_size == 0:
            os.write(fd, b"\0")
        os.lseek(fd, 0, os.SEEK_SET)
        try:
            msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            return True
        except OSError:
            return False
    try:
        fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        return True
    except BlockingIOError:
        return False


def _unlock(fd: int) -> None:
    try:
        import fcntl
    except ImportError:  # pragma: no cover - Windows
        import msvcrt

        os.lseek(fd, 0, os.SEEK_SET)
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        return
    fcntl.flock(fd, fcntl.LOCK_UN)


@contextmanager
def spool_lock(root: Path, wait_s: float = LOCK_WAIT_S) -> Iterator[None]:
    """Hold the spool lock; raise ``TimeoutError`` after ``wait_s``."""
    fd = os.open(str(_ensure_dir(root) / _LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + max(0.0, wait_s)
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise TimeoutError("hook spool lock is busy")
            time.sleep(0.002)
        try:
            yield
        finally:
            _unlock(fd)
    finally:
        os.close(fd)


# ---------------------------------------------------------------------------
# Producer side
# ---------------------------------------------------------------------------

def _profile_files(root: Path, stem: str) -> list[Path]:
    return [p for p in root.glob(f"{stem}.*")
            if p.suffix in (_ACTIVE_SUFFIX, _DRAINING_SUFFIX)
            and p.name.split(".", 1)[0] == stem]


def _spooled_bytes(root: Path, stem: str) -> int:
    total = 0
    for path in _profile_files(root, stem):
        try:
            total += path.stat().st_size
        except OSError:
            continue
    return total


def _encode(event: Mapping) -> bytes | None:
    try:
        line = json.dumps(dict(event), separators=(",", ":"), ensure_ascii=True,
                          default=str).encode("ascii") + b"\n"
    except (TypeError, ValueError):
        return None
    return line if len(line) <= MAX_LINE_BYTES else None


def append_events(
    events: Sequence[Mapping],
    *,
    profile: str,
    root: str | Path | None = None,
    max_bytes: int | None = None,
) -> AppendResult:
    """Append ``events`` to ``profile``'s spool; refuse (and count) past the cap.

    Raises ``TimeoutError``/``OSError`` only when the spool itself cannot be
    used; the caller reports that, it is never swallowed here.
    """
    base = spool_root(root)
    cap = max_bytes if max_bytes is not None else _max_bytes_default()
    stem = profile_stem(profile)
    with spool_lock(base):
        size = _spooled_bytes(base, stem)
        accepted: list[bytes] = []
        dropped = 0
        for event in events:
            line = _encode(event)
            if line is None or size + len(line) > cap:
                dropped += 1
                continue
            accepted.append(line)
            size += len(line)
        if accepted:
            path = base / f"{stem}{_ACTIVE_SUFFIX}"
            fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
            try:
                os.write(fd, b"".join(accepted))
            finally:
                os.close(fd)
        if dropped:
            _bump_counts_locked(base, stem, dropped=dropped)
    return AppendResult(written=len(accepted), dropped=dropped)


# ---------------------------------------------------------------------------
# Consumer side (the daemon)
# ---------------------------------------------------------------------------

def claim_ready(root: str | Path | None = None) -> list[Path]:
    """Claim every active file and return all files waiting to be drained.

    Files claimed earlier (a drain that stopped part way) come first, oldest
    first, so events keep their order within a profile.
    """
    base = spool_root(root)
    if not base.is_dir():
        return []
    with spool_lock(base):
        for path in sorted(base.glob(f"*{_ACTIVE_SUFFIX}")):
            stem = path.name[: -len(_ACTIVE_SUFFIX)]
            target = base / f"{stem}.{time.time_ns():020d}{_DRAINING_SUFFIX}"
            os.replace(path, target)
    return sorted(base.glob(f"*{_DRAINING_SUFFIX}"),
                  key=lambda p: (p.name.split(".", 1)[0], p.name))


def erase_profile(root: str | Path | None, profile: str) -> int:
    """Delete every spooled event of ``profile`` (GDPR erase). Returns how many."""
    base = spool_root(root)
    if not base.is_dir():
        return 0
    stem = profile_stem(profile)
    removed = 0
    with spool_lock(base):
        for path in _profile_files(base, stem):
            removed += _count_lines(path)
            path.unlink(missing_ok=True)
        data = _read_stats_file(base)
        profiles = dict(data.get("profiles") or {})
        if profiles.pop(stem, None) is not None:
            tmp = base / f".{_STATS_NAME}.{os.getpid()}.tmp"
            tmp.write_text(json.dumps({"profiles": profiles}, sort_keys=True), encoding="utf-8")
            os.replace(tmp, base / _STATS_NAME)
    return removed


def record_counts(root: str | Path | None, profile: str, *, dropped: int = 0,
                  invalid: int = 0) -> None:
    """Add to ``profile``'s refused/unreadable counters."""
    base = spool_root(root)
    with spool_lock(base):
        _bump_counts_locked(base, profile_stem(profile), dropped=dropped, invalid=invalid)


def _read_stats_file(base: Path) -> dict:
    try:
        data = json.loads((base / _STATS_NAME).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return data if isinstance(data, dict) else {}


def _bump_counts_locked(base: Path, stem: str, *, dropped: int = 0, invalid: int = 0) -> None:
    data = _read_stats_file(base)
    profiles = dict(data.get("profiles") or {})
    entry = dict(profiles.get(stem) or {})
    entry["dropped"] = int(entry.get("dropped", 0)) + int(dropped)
    entry["invalid"] = int(entry.get("invalid", 0)) + int(invalid)
    if dropped:
        entry["last_drop_at"] = datetime.now(timezone.utc).isoformat()
    profiles[stem] = entry
    tmp = base / f".{_STATS_NAME}.{os.getpid()}.tmp"
    tmp.write_text(json.dumps({"profiles": profiles}, sort_keys=True), encoding="utf-8")
    os.replace(tmp, base / _STATS_NAME)


def _count_lines(path: Path) -> int:
    try:
        with path.open("rb") as handle:
            return sum(chunk.count(b"\n") for chunk in iter(lambda: handle.read(1 << 16), b""))
    except OSError:
        return 0


def read_stats(root: str | Path | None = None) -> SpoolStats:
    """Read-only summary. Creates nothing. Pending counts lines not yet
    deleted, so a file the daemon is draining right now still counts whole."""
    base = spool_root(root)
    if not base.is_dir():
        return SpoolStats(0, 0, 0, 0, "")
    files = [p for p in base.iterdir() if p.suffix in (_ACTIVE_SUFFIX, _DRAINING_SUFFIX)]
    pending = sum(_count_lines(p) for p in files)
    size = 0
    for path in files:
        try:
            size += path.stat().st_size
        except OSError:
            continue
    profiles = (_read_stats_file(base).get("profiles") or {}).values()
    dropped = sum(int(e.get("dropped", 0)) for e in profiles if isinstance(e, dict))
    invalid = sum(int(e.get("invalid", 0)) for e in profiles if isinstance(e, dict))
    last = max((str(e.get("last_drop_at", "")) for e in profiles if isinstance(e, dict)),
               default="")
    return SpoolStats(pending, size, dropped, invalid, last)


__all__ = [
    "AppendResult", "DEFAULT_MAX_BYTES", "MAX_LINE_BYTES", "SPOOL_DIR_NAME",
    "SpoolStats", "append_events", "claim_ready", "erase_profile", "profile_stem", "read_stats",
    "record_counts", "spool_lock", "spool_root",
]
