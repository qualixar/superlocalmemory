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
* ``<profile>.<ns>-<pid>-<rand>.draining`` -- a batch a producer wrote beside
  the lock because the lock stayed busy past its wait. Written in full as
  ``.<profile>.<ns>-<pid>-<rand>.part`` and renamed, so the daemon never sees
  half of it; drained like any claimed file. A busy lock never costs an event.
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
#: How long a caller waits for the spool lock. Holders keep it for one append
#: or one claim pass: a directory scan, an open and a write or a rename, which
#: is milliseconds on a loaded machine with a slow disk, not microseconds.
#: Many producers at once can queue past this wait; a producer that does then
#: writes its batch beside the lock (see the module docstring) instead of failing.
LOCK_WAIT_S = 1.0

_ACTIVE_SUFFIX = ".jsonl"
_DRAINING_SUFFIX = ".draining"
_PART_SUFFIX = ".part"
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


class SpoolLockBusy(TimeoutError):
    """The spool lock stayed held past the caller's wait."""


@contextmanager
def spool_lock(root: Path, wait_s: float = LOCK_WAIT_S) -> Iterator[None]:
    """Hold the spool lock; raise ``SpoolLockBusy`` (a ``TimeoutError``) after ``wait_s``."""
    fd = os.open(str(_ensure_dir(root) / _LOCK_NAME), os.O_RDWR | os.O_CREAT, 0o600)
    try:
        deadline = time.monotonic() + max(0.0, wait_s)
        while not _try_lock(fd):
            if time.monotonic() >= deadline:
                raise SpoolLockBusy("hook spool lock is busy")
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


#: The free-text fields of a tool event: what a tool was given and what it printed.
_TOOL_TEXT = ("input_summary", "output_summary")


def _screened(event: Mapping) -> Mapping | None:
    """The event with credentials in a tool's text replaced by typed placeholders.

    Only while ``trust.secret_screen`` is on (WS-5): a command's output can print a key,
    and this file is written before anything else sees the text. ``None`` when the screen
    is on and cannot run: the event is counted as dropped, never spooled in clear.
    """
    if event.get("kind") != "tool_event":
        return event
    try:
        from superlocalmemory.core.feature_flags import FlagState
        from superlocalmemory.trust import secret_screen

        acting = secret_screen.state() is FlagState.ON
    except Exception:  # noqa: BLE001 - the flag cannot be read: the screen is off, as before
        return event
    if not acting:
        return event
    try:
        out = dict(event)
        for name in _TOOL_TEXT:
            if isinstance(out.get(name), str):
                out[name] = secret_screen.screen(out[name]).text
        return out
    except Exception:  # noqa: BLE001 - on and unable to screen: refuse the event, keep no clear text
        return None


def _encode(event: Mapping) -> bytes | None:
    event = _screened(event)
    if event is None:
        return None
    try:
        line = json.dumps(dict(event), separators=(",", ":"), ensure_ascii=True,
                          default=str).encode("ascii") + b"\n"
    except (TypeError, ValueError):
        return None
    return line if len(line) <= MAX_LINE_BYTES else None


def _admit(lines: Sequence[bytes | None], size: int, cap: int) -> tuple[list[bytes], int]:
    """The lines that fit under ``cap`` given ``size`` already spooled, and how
    many are refused (unencodable, oversized, or past the cap)."""
    accepted: list[bytes] = []
    dropped = 0
    for line in lines:
        if line is None or size + len(line) > cap:
            dropped += 1
            continue
        accepted.append(line)
        size += len(line)
    return accepted, dropped


def _write_all(fd: int, data: bytes) -> None:
    view = memoryview(data)
    while view:
        view = view[os.write(fd, view):]


def append_events(
    events: Sequence[Mapping],
    *,
    profile: str,
    root: str | Path | None = None,
    max_bytes: int | None = None,
    wait_s: float = LOCK_WAIT_S,
) -> AppendResult:
    """Append ``events`` to ``profile``'s spool; refuse (and count) past the cap.

    ``wait_s`` bounds the wait for the lock. Past it the batch is written
    beside the lock instead (:func:`_append_beside`), so a busy lock delays
    an event by at most ``wait_s`` and never loses it.

    Raises ``OSError`` only when the spool itself cannot be used; the caller
    reports that, it is never swallowed here.
    """
    base = spool_root(root)
    cap = max_bytes if max_bytes is not None else _max_bytes_default()
    stem = profile_stem(profile)
    # Encode (and screen) before taking the lock: the screen reads the flag file and
    # runs its patterns, and every other hook process waits while the lock is held.
    lines = [_encode(event) for event in events]
    try:
        with spool_lock(base, wait_s):
            accepted, dropped = _admit(lines, _spooled_bytes(base, stem), cap)
            if accepted:
                path = base / f"{stem}{_ACTIVE_SUFFIX}"
                fd = os.open(str(path), os.O_WRONLY | os.O_APPEND | os.O_CREAT, 0o600)
                try:
                    _write_all(fd, b"".join(accepted))
                finally:
                    os.close(fd)
            if dropped:
                _bump_counts_locked(base, stem, dropped=dropped)
        return AppendResult(written=len(accepted), dropped=dropped)
    except SpoolLockBusy:
        pass  # raised only while waiting: nothing above was written
    return _append_beside(base, stem, lines, cap)


def _append_beside(base: Path, stem: str, lines: Sequence[bytes | None],
                   cap: int) -> AppendResult:
    """Write a batch to a file of its own when the lock stayed busy.

    The file appears under its ``.draining`` name only once it is complete
    (written as ``.part``, then renamed), so the daemon drains all of it or
    none of it. The cap is checked without the lock, so concurrent producers
    on this path can pass it by at most their own batches. Refused events are
    still counted, under the lock with the ordinary wait; if even that stays
    busy the ``TimeoutError`` reaches the caller, which reports it.
    """
    accepted, dropped = _admit(lines, _spooled_bytes(base, stem), cap)
    if accepted:
        token = f"{time.time_ns():020d}-{os.getpid()}-{os.urandom(4).hex()}"
        part = base / f".{stem}.{token}{_PART_SUFFIX}"
        fd = os.open(str(part), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        try:
            try:
                _write_all(fd, b"".join(accepted))
            finally:
                os.close(fd)
            os.replace(part, base / f"{stem}.{token}{_DRAINING_SUFFIX}")
        except BaseException:
            part.unlink(missing_ok=True)
            raise
    if dropped:
        with spool_lock(base):
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
    _sweep_stale_parts(base)
    with spool_lock(base):
        for path in sorted(base.glob(f"*{_ACTIVE_SUFFIX}")):
            stem = path.name[: -len(_ACTIVE_SUFFIX)]
            target = base / f"{stem}.{time.time_ns():020d}{_DRAINING_SUFFIX}"
            os.replace(path, target)
    return sorted(base.glob(f"*{_DRAINING_SUFFIX}"),
                  key=lambda p: (p.name.split(".", 1)[0], p.name))


#: A ``.part`` file this old belongs to a producer that died before its rename
#: (a live one renames within milliseconds); its batch was never acknowledged.
_STALE_PART_S = 3600.0


def _sweep_stale_parts(base: Path) -> None:
    cutoff = time.time() - _STALE_PART_S
    for path in base.glob(f".*{_PART_SUFFIX}"):
        try:
            if path.stat().st_mtime < cutoff:
                path.unlink(missing_ok=True)
        except OSError:
            continue


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
        # A producer that died between writing beside the lock and renaming.
        for path in base.glob(f".{stem}.*{_PART_SUFFIX}"):
            if path.name.split(".", 2)[1] == stem:
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
    "AppendResult", "DEFAULT_MAX_BYTES", "LOCK_WAIT_S", "MAX_LINE_BYTES", "SPOOL_DIR_NAME",
    "SpoolLockBusy", "SpoolStats", "append_events", "claim_ready", "erase_profile", "profile_stem", "read_stats",
    "record_counts", "spool_lock", "spool_root",
]
