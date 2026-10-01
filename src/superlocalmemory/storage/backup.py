# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Pre-migration database backup utilities.

Provides a consistent, WAL-safe snapshot of both managed databases before
any schema migration runs. Uses the SQLite backup API rather than a
filesystem copy so that in-flight writers on a live WAL database cannot
produce a torn snapshot.

Snapshots are written as flat files directly in ``snapshots_root``
(default: ``canonical_data_root() / "pre-migration-snapshots"``).
This directory is separate from the ``backups/`` directory managed by
``BackupManager``, so pre-migration snapshots are never subject to
``BackupManager._enforce_retention()`` regardless of how many ordinary
backups accumulate.  The separation is structural — not timing-dependent.

Public API intended for use by the migration runner:
  - _backup_via_sqlite_api(src, dest)
  - _pre_migration_backup(learning_db, memory_db, *, backups_root) -> Path
  - _gc_old_backups(backups_root, keep=2) -> None
  - InsufficientDiskSpaceError

Restoring a pre-migration snapshot
-----------------------------------
Use ``restore_pre_migration_snapshot()`` in this module::

    from pathlib import Path
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot
    from superlocalmemory.infra.data_root import canonical_data_root

    root = canonical_data_root()
    snap = root / "pre-migration-snapshots" / "memory-20260819-120000-pre-migration.db"
    restore_pre_migration_snapshot(snap, root / "memory.db")

It verifies the snapshot is a readable database with content and refuses before
touching the live store if it is not, copies the current live database aside
into ``pre-restore/`` first, and only then writes the snapshot into place.

**Do not use ``BackupManager.restore_backup()`` for these snapshots.** It checks
that the source exists, then takes its own pre-restore backup, which runs
retention across the same directory. Retention can unlink the file being
restored; ``sqlite3.connect`` then recreates that path as an EMPTY database, and
the empty database is copied over the live store — and the call returns ``True``.
The snapshot is left as a zero-byte file under its original name, so a second
attempt also appears to succeed. Reproduced: a 500-fact store became 0 tables
while the call reported success.


Why snapshots live outside ``backups/``
----------------------------------------
``BackupManager._enforce_retention()`` globs only its own ``backup_dir``
(``canonical_data_root() / "backups"``).  Because pre-migration snapshots
are in ``pre-migration-snapshots/`` — a completely different directory — no
retention policy can delete them, regardless of how many ordinary backups
accumulate.  An ordinary ``BackupManager()`` call (without ``backup_dir``
override) will not list or touch these files.  That is intentional.
"""

from __future__ import annotations

import logging
import re
import shutil
import os
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.infra.data_root import canonical_data_root

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Filename parsing helpers for snapshot ordering and generation grouping
# ---------------------------------------------------------------------------

# Timestamps in snapshot filenames are YYYYMMDD-HHmmss-ffffff (microseconds).
# _free_name appends a collision suffix -N (integer ≥ 1) when a file already
# exists.  Both forms are matched by this pattern.
_SNAPSHOT_TIMESTAMP_RE = re.compile(r"^(\d{8}-\d{6}-\d{6})(?:-(\d+))?$")

# Pattern used to extract a timestamp (and optional collision suffix) from the
# RIGHT side of a stripped filename.  Anchoring at the END means the stem may
# contain hyphens without confusing the parser: only the rightmost field that
# looks like a full YYYYMMDD-HHmmss-ffffff[-N] is extracted.
_SNAPSHOT_TIMESTAMP_TAIL_RE = re.compile(
    r"(\d{8}-\d{6}-\d{6})(?:-(\d+))?$"
)


def _extract_snapshot_stamp(name_without_suffix: str) -> re.Match | None:
    """Return a regex match for the timestamp[-N] tail of a stripped filename.

    ``name_without_suffix`` is the filename after removing the
    ``-pre-migration.db`` suffix (i.e. ``{stem}-{timestamp}[-{N}]``).

    Using ``re.search`` anchored at ``$`` instead of ``split('-', 1)[-1]``
    means a stem that contains hyphens (e.g. ``pre-migration``) does not
    shift the extracted timestamp to the right, which would cause the file
    to sort as if it were newer than any validly-named snapshot.
    """
    return _SNAPSHOT_TIMESTAMP_TAIL_RE.search(name_without_suffix)


def _parse_generation_stamp(raw_stamp: str) -> str:
    """Return the base YYYYMMDD-HHmmss-ffffff portion of a raw snapshot stamp.

    Strips any trailing -N collision suffix added by ``_free_name``, so that::

        memory-20260819-120000-123456-pre-migration.db
        memory-20260819-120000-123456-1-pre-migration.db

    are recognised as the same logical generation.

    If the stamp does not match the expected pattern (e.g. an externally
    created file), the raw stamp is returned unchanged so GC does not crash.
    """
    m = _SNAPSHOT_TIMESTAMP_RE.match(raw_stamp)
    return m.group(1) if m else raw_stamp


def _snapshot_sort_key(path: Path) -> tuple[str, int]:
    """Stable sort key for a ``*-pre-migration.db`` file.

    Returns ``(base_timestamp, collision_suffix_int)`` so that:

    * Files from earlier migrations sort before later ones.
    * Among files sharing a base timestamp (collision duplicates), the
      unsuffixed file sorts first (-0) and each higher suffix follows.

    Sorting ascending and indexing ``[-1]`` yields the newest snapshot,
    regardless of filesystem mtime granularity.

    The timestamp is found by searching from the right of the stripped name
    rather than by splitting on the first hyphen, so stems that contain
    hyphens do not corrupt the extracted timestamp.
    """
    name = path.name
    stripped = name.rsplit("-pre-migration.db", 1)[0]
    m = _extract_snapshot_stamp(stripped)
    if m:
        base = m.group(1)
        suffix = int(m.group(2)) if m.group(2) is not None else 0
        return (base, suffix)
    # Unrecognised filename format; use the full stripped name as a fallback
    # key so GC never crashes, but log a warning — a file that lands here
    # may sort unexpectedly relative to validly-named snapshots.
    logger.warning(
        "[SLM] Cannot parse timestamp from snapshot filename %r; "
        "it will sort by raw name and may be kept or deleted out of order",
        name,
    )
    return (stripped, 0)


class InsufficientDiskSpaceError(Exception):
    """Raised when the filesystem cannot hold the pre-migration backup.

    Attributes:
        needed_bytes: How many bytes would be required.
        free_bytes: How many bytes are currently available.
    """

    def __init__(self, needed_bytes: int, free_bytes: int) -> None:
        self.needed_bytes = needed_bytes
        self.free_bytes = free_bytes
        super().__init__(
            f"Insufficient disk space for pre-migration backup: "
            f"need {needed_bytes:,} bytes, have {free_bytes:,} bytes free"
        )


def _backup_via_sqlite_api(src: Path, dest: Path) -> None:
    """Copy a live SQLite database to dest using the SQLite backup API.

    Unlike a filesystem copy, sqlite3.Connection.backup() acquires page-level
    read locks one batch at a time, allowing concurrent writers to proceed
    between batches. The resulting snapshot reflects only committed pages —
    uncommitted WAL frames are never included.

    dest.parent is created if it does not exist.
    Both connections are closed in a finally block even if an error occurs.
    """
    dest.parent.mkdir(parents=True, exist_ok=True)

    # Write to a temporary sibling and rename into place. A copy interrupted by
    # a full disk or a crash would otherwise leave a truncated file at the final
    # name — a snapshot that looks present and restores nothing. rename() within
    # one directory is atomic, so the final name only ever appears complete.
    staging = dest.with_name(dest.name + ".partial")
    src_conn = sqlite3.connect(str(src), check_same_thread=False)
    dst_conn = sqlite3.connect(str(staging))
    try:
        # pages=-1 copies all pages in a single pass (fastest; no yielding to
        # other writers between batches, which is acceptable here because the
        # backup happens before the migration run begins — no other migration
        # writer is active at this point).
        src_conn.backup(dst_conn, pages=-1)
        dst_conn.commit()
    finally:
        src_conn.close()
        dst_conn.close()

    # Verify and durably flush BEFORE the rename, so the final name never
    # appears over incomplete or corrupt content.
    try:
        # Windows durable flush requires write access. This is our temporary
        # snapshot, not the source database; do not suppress a failed flush.
        fd = os.open(str(staging), os.O_RDWR | getattr(os, "O_BINARY", 0))
        try:
            os.fsync(fd)
        finally:
            os.close(fd)
        verify = sqlite3.connect(f"file:{staging}?mode=ro", uri=True)
        try:
            if verify.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise SnapshotUnusableError(
                    f"snapshot failed its integrity check immediately after copy: {dest}")
        finally:
            verify.close()
    except Exception:
        staging.unlink(missing_ok=True)   # never leave a partial file behind
        raise

    # Verifying the copy read-only makes SQLite materialise a -shm beside the
    # staging file, and a read-only connection cannot remove it on close. The
    # rename below moves only the main file, so the companion is left behind
    # under the staging name — observed: every snapshot left a stray
    # `.partial-shm` and `.partial-wal` in the snapshot directory. Removing them
    # is safe because the copy was checkpointed when its read-write connection
    # closed, so whatever exists now came from verification and holds nothing.
    # That is checked rather than trusted: content in the log would mean the copy
    # was not fully checkpointed, and renaming it would strand those pages.
    for suffix in ("-wal", "-shm"):
        companion = staging.with_name(staging.name + suffix)
        if not companion.exists():
            continue
        if suffix == "-wal":
            leftover = companion.stat().st_size
            if leftover > 0:
                # Read the size BEFORE unlinking. Reading it inside the message
                # after the unlink raised FileNotFoundError instead of this
                # error, so the caller saw a generic crash — and the staging
                # file was already gone, taking the evidence with it.
                staging.unlink(missing_ok=True)
                companion.unlink(missing_ok=True)
                raise SnapshotUnusableError(
                    f"copy left {leftover} bytes in its write-ahead log; "
                    f"renaming it would strand those pages: {dest}"
                )
        companion.unlink(missing_ok=True)

    staging.replace(dest)


class SnapshotUnusableError(RuntimeError):
    """Raised when a snapshot cannot be verified, BEFORE the live store is touched."""


def restore_pre_migration_snapshot(snapshot: Path, target: Path) -> Path:
    """Restore ``snapshot`` over ``target``, verifying before it destroys anything.

    Do NOT restore these snapshots with ``BackupManager.restore_backup()``. That
    method checks the source exists, then takes its own "pre-restore" backup,
    which runs retention over the same directory. Retention can unlink the very
    file being restored; ``sqlite3.connect`` then RECREATES that path as an empty
    database, and the empty database is copied over the live store. It returns
    True. The snapshot is left as a zero-byte file with its original name, so a
    second attempt appears to succeed as well. Reproduced: a 500-fact store
    restored to 0 tables while the call reported success.

    This function instead:
      1. verifies the snapshot is a readable database with content, and refuses
         before touching ``target`` if it is not,
      2. copies the CURRENT ``target`` aside first, outside the snapshot
         directory so no retention policy can reclaim it,
      3. copies the snapshot into place through the SQLite backup API.

    Returns the path of the safety copy of the pre-restore state.
    """
    if not snapshot.is_file() or snapshot.stat().st_size == 0:
        raise SnapshotUnusableError(f"snapshot is missing or empty: {snapshot}")
    try:
        conn = sqlite3.connect(f"file:{snapshot}?mode=ro", uri=True)
        try:
            tables = [r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")]
            if not tables:
                raise SnapshotUnusableError(
                    f"snapshot contains no tables, refusing to restore it over "
                    f"{target.name}: {snapshot}")
            if conn.execute("PRAGMA quick_check").fetchone()[0] != "ok":
                raise SnapshotUnusableError(f"snapshot failed integrity check: {snapshot}")
        finally:
            conn.close()
    except sqlite3.Error as exc:
        raise SnapshotUnusableError(f"snapshot is not a readable database: {snapshot}") from exc

    # Safety copy of what we are about to overwrite, deliberately NOT in the
    # snapshot directory — nothing prunes this location.
    safety_dir = target.parent / "pre-restore"
    safety_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    safety = safety_dir / f"{target.stem}-{stamp}-before-restore{target.suffix}"
    if target.exists():
        _backup_via_sqlite_api(target, safety)

    _backup_via_sqlite_api(snapshot, target)
    logger.info("[SLM] Restored %s from %s (previous state saved to %s)",
                target.name, snapshot.name, safety)
    return safety


def _find_existing_ancestor(path: Path) -> Path:
    """Return the nearest ancestor of path that exists on the filesystem."""
    p = path
    while not p.exists():
        if p.parent == p:
            # Reached the root without finding an existing dir; use cwd.
            return Path.cwd()
        p = p.parent
    return p


def _pre_migration_backup(
    learning_db: Path,
    memory_db: Path,
    *,
    backups_root: Path | None = None,
) -> Path:
    """Snapshot both databases as flat files before migration.

    Creates ``{db_stem}-{YYYYMMDD-HHmmss}-pre-migration.db`` files directly
    in ``backups_root`` (no subdirectory). Files are named with a
    ``-pre-migration`` suffix so that the GC glob ``*-pre-migration.db``
    identifies them unambiguously without matching any file produced by
    ``BackupManager``.

    Only databases that exist on disk are copied; a missing database is
    silently skipped (first-install scenario where learning.db may not
    exist yet).

    The ``backups_root`` directory defaults to
    ``canonical_data_root() / "pre-migration-snapshots"`` — a directory
    separate from ``BackupManager``'s ``backups/`` directory.  This
    separation means ``BackupManager._enforce_retention()`` can never
    reach these files regardless of how many routine backups accumulate.

    Args:
        learning_db: Path to the learning-plane database.
        memory_db: Path to the memory database.
        backups_root: Override the canonical snapshots root.  Tests pass a
            tmp_path here to avoid writing to the user's data directory.

    Returns:
        The Path of ``backups_root`` (the directory holding the new flat
        snapshot files).

    Raises:
        InsufficientDiskSpaceError: When the free space on the target
            filesystem is less than 110% of the combined source database sizes.
    """
    if backups_root is None:
        backups_root = canonical_data_root() / "pre-migration-snapshots"

    # Second granularity is not enough. Two migrations inside the same second —
    # a daemon restart loop, or the second apply_all() during startup — produced
    # identical filenames, and the atomic rename then replaced the FIRST
    # snapshot cleanly. The first is the valuable one: it holds the state before
    # anything was touched. Microseconds make a collision practically
    # impossible, and the loop below refuses to overwrite regardless.
    timestamp = datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")

    # Measure combined size of databases that actually exist, INCLUDING their
    # write-ahead log and shared-memory files. On a busy store the -wal file can
    # hold a large fraction of the data not yet checkpointed into the main file,
    # and the snapshot materialises all of it. Sizing against the main file
    # alone under-counts the requirement and lets a migration start with too
    # little room, which is the situation the check exists to prevent.
    total_bytes = 0
    for db_path in (memory_db, learning_db):
        for companion in (db_path, Path(f"{db_path}-wal"), Path(f"{db_path}-shm")):
            if companion.exists():
                total_bytes += companion.stat().st_size

    # Check that the target filesystem has enough room.  We check against an
    # existing ancestor because the snapshot directory itself may not exist yet.
    check_path = _find_existing_ancestor(backups_root)
    free_bytes = shutil.disk_usage(str(check_path)).free
    # Peak disk requirement — sequential writes, staging-then-rename:
    #
    #   Step 1: write memory.staging  (up to memory_size bytes)
    #   Step 2: rename memory.staging → memory.final  (0 extra bytes; atomic)
    #   Step 3: write learning.staging (up to learning_size bytes)
    #   Step 4: rename learning.staging → learning.final (0 extra bytes)
    #
    #   Peak between steps 3-4: memory.final + learning.staging
    #                         = memory_size + learning_size = total_bytes
    #
    # The 10 % buffer (0.1 × total_bytes) covers:
    #   - filesystem metadata: directory entries and inode table entries for
    #     2 new files are at most a few KiB — negligible for MB-sized stores.
    #   - WAL pages materialised during the copy: _backup_via_sqlite_api uses
    #     sqlite3.Connection.backup(pages=-1), a single-pass read lock.  The
    #     daemon is expected to be idle while this script runs (the caller
    #     checks _writer_lock_held() before calling this function), so WAL
    #     growth during the copy is near zero.  If the caller has not stopped
    #     the daemon, a concurrent checkpoint could transiently inflate the
    #     source WAL, but that WAL growth is bounded by the daemon's own write
    #     rate and is already counted in total_bytes (we measure the -wal file
    #     size before the copy).  10 % headroom covers reasonable variance.
    #
    # The previous formula used total_bytes * 2.1.  That was derived for a
    # SINGLE database (1.0 × staging + 1.0 × final + 0.1 × headroom = 2.1 ×)
    # but was mistakenly applied to the SUM of two databases, over-reserving
    # by ~2 × and blocking migrations on machines with limited disk space.
    needed_bytes = int(total_bytes * 1.1)
    if free_bytes < needed_bytes:
        raise InsufficientDiskSpaceError(needed_bytes, free_bytes)

    backups_root.mkdir(parents=True, exist_ok=True)

    # Perform the backup.  Each db gets a flat file with a -pre-migration suffix
    # so the GC glob *-pre-migration.db identifies our files precisely.
    t0 = time.monotonic()

    def _free_name(stem: str) -> Path:
        """Never overwrite an existing snapshot; the older one may be the only
        copy of the pre-migration state."""
        candidate = backups_root / f"{stem}-{timestamp}-pre-migration.db"
        suffix = 1
        while candidate.exists():
            candidate = backups_root / f"{stem}-{timestamp}-{suffix}-pre-migration.db"
            suffix += 1
        return candidate

    # Keep each snapshot paired with the database it came from. Emitting a
    # single restore command built from written[0] named the LEARNING snapshot
    # (it sorts first) against memory.db as the target — a command that would
    # restore the wrong database over the user's memories.
    pairs: list[tuple[Path, Path]] = []
    for stem, db_path in (("memory", memory_db), ("learning", learning_db)):
        if db_path.exists():
            dest = _free_name(stem)
            _backup_via_sqlite_api(db_path, dest)
            pairs.append((dest, db_path))

    elapsed = time.monotonic() - t0

    written = [snap for snap, _ in pairs]
    size_bytes = sum(f.stat().st_size for f in written)
    size_mb = size_bytes / (1024 * 1024)
    filenames = "\n  ".join(f.name for f in written)

    logger.info(
        "[SLM] Pre-migration snapshot written (%.0f MB in %.1fs):\n"
        "  Location : %s\n"
        "  Files    :\n  %s\n"
        "  To restore if migration fails — run the line for the database you\n"
        "  need; each snapshot restores only its own database:\n"
        "    from pathlib import Path\n"
        "    from superlocalmemory.storage.backup import restore_pre_migration_snapshot\n"
        "%s",
        size_mb,
        elapsed,
        str(backups_root),
        filenames,
        "\n".join(
            f"    restore_pre_migration_snapshot(Path({str(snap)!r}), Path({str(db)!r}))"
            for snap, db in pairs
        ) or "    (no snapshot was written — nothing to restore)",
    )

    return backups_root


def _gc_old_backups(backups_root: Path, keep: int = 2) -> None:
    """Remove old pre-migration snapshot GENERATIONS, retaining ``keep`` newest.

    A generation is one migration's snapshots — ``memory-<ts>-pre-migration.db``
    and ``learning-<ts>-pre-migration.db`` share a timestamp and are only useful
    together. Counting files instead of generations kept ``keep`` FILES: with
    ``keep=2`` that is a single generation, and where mtimes interleave it could
    retain a ``memory`` snapshot whose matching ``learning`` snapshot had been
    deleted — a half set that cannot restore a consistent store.

    Only files directly under ``backups_root`` matching ``*-pre-migration.db``
    are eligible. Every deletion uses an explicit full path; no glob is ever
    passed to the deletion call.
    """
    if not backups_root.exists():
        return

    generations: dict[str, list[Path]] = {}
    for candidate in backups_root.glob("*-pre-migration.db"):
        if not candidate.is_file() or candidate.parent != backups_root:
            continue
        # Extract the raw stamp and normalise away any collision suffix so that
        # "memory-20260819-120000-123456-pre-migration.db" and
        # "memory-20260819-120000-123456-1-pre-migration.db" land in the same
        # generation bucket.  Without normalisation a collision suffix makes GC
        # count one migration's files as two separate generations and can delete
        # one file from a paired set, leaving a snapshot that cannot be used for
        # a consistent restore.
        #
        # Search from the RIGHT side of the stripped name so a stem that
        # contains hyphens does not shift the extracted timestamp — the same
        # fix applied to _snapshot_sort_key.
        stripped = candidate.name.rsplit("-pre-migration.db", 1)[0]
        m = _extract_snapshot_stamp(stripped)
        raw_stamp = m.group(1) if m else stripped
        stamp = _parse_generation_stamp(raw_stamp)
        generations.setdefault(stamp, []).append(candidate)

    if len(generations) <= keep:
        return

    # Sort by the base timestamp string.  YYYYMMDD-HHmmss-ffffff is lexically
    # monotonic, so alphabetical order is chronological order.  Using st_mtime
    # here was nondeterministic when two snapshots landed in the same filesystem
    # timestamp second (FAT, relatime ext4).
    ordered = sorted(
        generations.items(),
        key=lambda kv: kv[0],
    )
    for _stamp, files in ordered[: len(generations) - keep]:
        for target in sorted(files):
            if target.parent != backups_root or not target.is_file():
                continue
            logger.info("[SLM] Removing old pre-migration snapshot: %s", target)
            target.unlink()
