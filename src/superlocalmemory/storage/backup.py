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
  - _gc_old_backups(backups_root, keep=2, protect=frozenset()) -> None
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
import os
import re
import shutil
import sqlite3
import time
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.infra.data_root import canonical_data_root

logger = logging.getLogger(__name__)

#: The databases a safety copy can hold, by the stem their copies are named with.
SNAPSHOT_DATABASES = frozenset({"memory", "learning"})
#: Copy speed used for the "about N seconds" estimate, verification included.
#: Measured on a warm store (2026-10, Apple Silicon SSD): ~8.4 s per GB.
_COPY_BYTES_PER_S = 110 * 1024 * 1024
#: An estimate at or above this is announced as a warning, so it is seen.
_ANNOUNCE_WARN_S = 5.0

# ---------------------------------------------------------------------------
# Filename parsing helpers for snapshot ordering and generation grouping
# ---------------------------------------------------------------------------

# Timestamps in snapshot filenames are YYYYMMDD-HHmmss-ffffff (microseconds).
# _free_name appends a collision suffix -N (integer ≥ 1) when a file already
# exists. Retention parses every naming itself (_snapshot_retention).
#
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


def _create_private_file(path: Path) -> None:
    """Create (or truncate) ``path`` readable and writable by its owner only."""
    fd = os.open(str(path), os.O_WRONLY | os.O_CREAT | os.O_TRUNC
                 | getattr(os, "O_BINARY", 0), 0o600)
    os.close(fd)
    if os.name != "nt":
        os.chmod(path, 0o600)        # an existing file keeps its old mode otherwise


def _make_private_dir(path: Path) -> None:
    """Create ``path`` (and parents) and keep it owner-only on POSIX."""
    path.mkdir(parents=True, exist_ok=True, mode=0o700)
    if os.name != "nt":
        os.chmod(path, 0o700)


def _quick_check(path: Path) -> list[str]:
    """``PRAGMA quick_check`` findings for ``path``; empty when it is ok."""
    verify = sqlite3.connect(f"file:{path}?mode=ro", uri=True)
    try:
        rows = [str(r[0]) for r in verify.execute("PRAGMA quick_check").fetchall()]
    finally:
        verify.close()
    return [] if rows == ["ok"] else rows


def _rebuild_damaged_keyword_indexes(staging: Path, findings: list[str]) -> bool:
    """Rebuild the copy's keyword indexes when they are the ONLY damage (#204).

    A keyword index is derived from the stored facts, so a live store with a
    damaged one still copies every memory faithfully -- and the copy reproduces
    the damage, which failed the snapshot and with it every migration and the
    daemon start. When each finding names a rebuildable index, the copy's
    indexes are rebuilt from its own facts. Any other finding (a damaged page,
    a broken B-tree) returns False and the snapshot is refused as before. The
    live store is untouched; ``slm db repair`` still fixes it.
    """
    from superlocalmemory.storage.fts_residue import (
        rebuild_keyword_index,
        recreate_keyword_index,
    )
    from superlocalmemory.storage.integrity_diagnosis import damaged_indexes_in

    if not all(damaged_indexes_in([f]) for f in findings):
        return False
    conn = sqlite3.connect(str(staging))
    try:
        for table in damaged_indexes_in(findings):
            try:
                rebuild_keyword_index(conn, table)
            except sqlite3.DatabaseError:
                recreate_keyword_index(conn, table)
        conn.commit()
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()
    logger.warning(
        "The store's keyword index is damaged (%s); the safety copy was taken "
        "with it rebuilt. Fix the store itself with: slm db repair --apply",
        ", ".join(damaged_indexes_in(findings)),
    )
    return True


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
    # A snapshot holds every memory, so it is created owner-only (0600) before
    # a single byte is copied -- the live store is 0600 and a copy beside it
    # must not be readable by other accounts on the machine. SQLite opens an
    # existing file as-is and gives its journal the same mode.
    _create_private_file(staging)
    src_conn = sqlite3.connect(str(src), check_same_thread=False)
    dst_conn = sqlite3.connect(str(staging))
    try:
        # pages=-1 copies all pages in a single pass (fastest; no yielding to
        # other writers between batches, which is acceptable here because the
        # backup happens before the migration run begins — no other migration
        # writer is active at this point).
        src_conn.backup(dst_conn, pages=-1)
        dst_conn.commit()
    except Exception:
        # A damaged source fails here; never leave an empty staging file (#153).
        dst_conn.close()
        for leftover in (staging, staging.with_name(staging.name + "-wal"),
                         staging.with_name(staging.name + "-shm")):
            leftover.unlink(missing_ok=True)
        raise
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
        findings = _quick_check(staging)
        if findings and _rebuild_damaged_keyword_indexes(staging, findings):
            findings = _quick_check(staging)
        if findings:
            raise SnapshotUnusableError(
                f"snapshot failed its integrity check immediately after copy: {dest} "
                f"({'; '.join(findings[:3])})")
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


class LiveStoreWriteError(RuntimeError):
    """Raised when a source cannot be written into a live store; the store is untouched."""


# How long a restore waits for another connection to release a live store's
# write lock before giving up, and how long each SQLite busy wait lasts within
# that window. Without a bound the backup API retries a held lock forever.
RESTORE_LOCK_WAIT_SECONDS = 30.0
_LOCK_POLL_SECONDS = 1.0

#: Pages written per backup step when restoring into a live store. -1 is one
#: step. However many steps, the destination is one transaction: a restore cut
#: off part-way (power loss, a killed process) leaves the target as it was.
_RESTORE_PAGES_PER_STEP = -1
#: Optional ``(remaining, total)`` observer called after every restore step.
_restore_progress = None


def _write_into_live_db(
    source: Path,
    target: Path,
    *,
    lock_wait_seconds: float = RESTORE_LOCK_WAIT_SECONDS,
    pages: int = -1,
    progress=None,
) -> None:
    """Make ``target`` hold exactly ``source``'s content, written by SQLite.

    Use this to overwrite a database that may be live; use
    ``_backup_via_sqlite_api`` only to create a NEW file. The pages are written
    into ``target`` on its own connection, as one transaction, so its ``-wal``
    and ``-shm`` stay paired with the file they describe. Renaming a copy over
    the live file instead left the old ``-wal`` beside a new database, and
    SQLite read (and later checkpointed) the old store's committed frames over
    the restored pages: reproduced as a restore that reported success and
    changed nothing, and as "database disk image is malformed" when the live
    store had moved on since the backup. A write that raises is rolled back by
    SQLite, so the target is either fully written or untouched.

    ``source`` must be a verified, quiescent file. It is opened immutable, which
    reads it without creating a ``-wal`` or ``-shm`` beside it — and without
    reading one, so a source whose ``-wal`` still holds frames is refused rather
    than restored without them.

    Raises:
        LiveStoreWriteError: if ``source``'s ``-wal`` is not empty, if another
            connection holds ``target``'s write lock for longer than
            ``lock_wait_seconds``, or if ``target`` is a WAL database whose
            page size differs from ``source``'s, which the backup API cannot
            write.
    """
    source_wal = Path(f"{source}-wal")
    if source_wal.exists() and source_wal.stat().st_size > 0:
        raise LiveStoreWriteError(
            f"{source.name} has {source_wal.stat().st_size} bytes in its "
            f"write-ahead log; restoring it would drop those pages")

    src_conn = sqlite3.connect(
        f"{source.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        dst_conn = sqlite3.connect(str(target), timeout=_LOCK_POLL_SECONDS)
        try:
            src_page = src_conn.execute("PRAGMA page_size").fetchall()[0][0]
            dst_page = dst_conn.execute("PRAGMA page_size").fetchall()[0][0]
            dst_mode = dst_conn.execute("PRAGMA journal_mode").fetchall()[0][0]
            if dst_mode.lower() == "wal" and src_page != dst_page:
                raise LiveStoreWriteError(
                    f"{target.name} is a WAL database with page size "
                    f"{dst_page}, the source has {src_page}; SQLite cannot "
                    f"restore across page sizes into a WAL database")

            deadline = time.monotonic() + lock_wait_seconds

            def _bounded_wait(status: int, remaining: int, total: int) -> None:
                if (status in (sqlite3.SQLITE_BUSY, sqlite3.SQLITE_LOCKED)
                        and time.monotonic() > deadline):
                    raise LiveStoreWriteError(
                        f"{target.name} stayed locked by another connection "
                        f"for over {lock_wait_seconds:g}s")
                if progress is not None:
                    progress(remaining, total)

            try:
                src_conn.backup(dst_conn, pages=pages, progress=_bounded_wait)
            finally:
                # Success or rollback, the write went through the target's
                # -wal, which then stays as large as the store until something
                # truncates it -- nothing else ever does on a live store.
                _shrink_wal(dst_conn, target)
        finally:
            dst_conn.close()
    finally:
        src_conn.close()


# How long the log truncation may wait for readers still inside an older
# snapshot. They are never interrupted: past this the log is left at its size
# and the next checkpoint that finds it unused reuses it.
_WAL_TRUNCATE_WAIT_MS = 2000


def _shrink_wal(conn: sqlite3.Connection, target: Path) -> None:
    """Checkpoint ``target`` and truncate its -wal to zero bytes. Never raises.

    ``wal_checkpoint(TRUNCATE)`` copies the log into the database and then
    waits, through the busy handler, until every reader has moved off the log
    before truncating it. A reader mid-query therefore keeps its snapshot and
    is never disturbed; if one outlasts the wait, SQLite reports busy, nothing
    is truncated, and the store is exactly as correct as before.
    """
    try:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() != "wal":
            return
        conn.execute(f"PRAGMA busy_timeout={_WAL_TRUNCATE_WAIT_MS}")
        busy, _log, _done = conn.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchone()
    except sqlite3.Error as exc:
        logger.info("[SLM] Could not shrink %s-wal after the restore write: %s",
                    target.name, exc)
        return
    if busy:
        logger.info(
            "[SLM] %s-wal was not shrunk after the restore write: a reader was "
            "still using it. It is reused at the next checkpoint.", target.name)


def restore_pre_migration_snapshot(
    snapshot: Path, target: Path, expected_sha256: str | None = None,
) -> Path:
    """Restore ``snapshot`` over ``target``, verifying before it destroys anything.

    ``expected_sha256`` (from the generation's manifest) makes it refuse a copy
    whose bytes changed since it was taken, before anything is touched.

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
      3. writes the snapshot's pages into ``target`` on ``target``'s own
         connection, so a live WAL store's ``-wal`` stays paired with it.

    Returns the path of the safety copy of the pre-restore state.

    Raises:
        SnapshotUnusableError: the snapshot failed verification; nothing was
            touched.
        LiveStoreWriteError: the snapshot could not be written into ``target``;
            ``target`` is unchanged and the safety copy exists.
    """
    if not snapshot.is_file() or snapshot.stat().st_size == 0:
        raise SnapshotUnusableError(f"snapshot is missing or empty: {snapshot}")
    if expected_sha256:
        verify_snapshot(snapshot, expected_sha256)
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
    _make_private_dir(safety_dir)
    # Microseconds, and never over an existing copy: a restore re-run within the
    # same second (a crash after the write, before the intent was retired) would
    # otherwise replace the first safety copy -- the only one holding the state
    # from before the first restore -- with a copy of the restored store.
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S-%f")
    safety = safety_dir / f"{target.stem}-{stamp}-before-restore{target.suffix}"
    n = 1
    while safety.exists():
        safety = safety_dir / f"{target.stem}-{stamp}-{n}-before-restore{target.suffix}"
        n += 1
    if target.exists():
        _backup_via_sqlite_api(target, safety)

    # Not _backup_via_sqlite_api: its rename over a live WAL store leaves the
    # old -wal beside the restored file, and SQLite reads those frames instead.
    _write_into_live_db(snapshot, target, pages=_RESTORE_PAGES_PER_STEP,
                        progress=_restore_progress)
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
    reason: str = "migration",
    databases: frozenset[str] = SNAPSHOT_DATABASES,
) -> Path:
    """Snapshot the databases about to change as flat files before migration.

    Creates ``{db_stem}-{YYYYMMDD-HHmmss}-pre-migration.db`` files directly
    in ``backups_root`` (no subdirectory). Files are named with a
    ``-pre-migration`` suffix so that the GC glob ``*-pre-migration.db``
    identifies them unambiguously without matching any file produced by
    ``BackupManager``.

    Only the databases named in ``databases`` (``"memory"``, ``"learning"``)
    are copied: an update that changes learning.db alone has no reason to copy
    a memory store of several GB first. Only databases that exist on disk are
    copied; a missing database is silently skipped (first-install scenario
    where learning.db may not exist yet). Before anything is copied, one line
    says what is being copied, how large it is and about how long it takes.

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
    unknown = set(databases) - SNAPSHOT_DATABASES
    if unknown:
        raise ValueError(f"unknown databases to copy: {sorted(unknown)}")
    chosen = [(stem, db_path) for stem, db_path in (("memory", memory_db),
                                                     ("learning", learning_db))
              if stem in databases]
    total_bytes = 0
    for _stem, db_path in chosen:
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

    _make_private_dir(backups_root)
    # A copy a crash interrupted keeps its staging name, the size of the store.
    _cleanup_stale_partials(backups_root)
    facts_before = _live_fact_count(memory_db) if "memory" in databases else None
    _announce_copy([db for _stem, db in chosen if db.exists()], total_bytes, reason)

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
    for stem, db_path in chosen:
        if db_path.exists():
            dest = _free_name(stem)
            _backup_via_sqlite_api(db_path, dest)
            pairs.append((dest, db_path))

    _prove_and_describe(backups_root, pairs, memory_db, facts_before, reason=reason)
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


def _announce_copy(dbs: list[Path], total_bytes: int, reason: str) -> None:
    """Say, before it starts, what is copied, how large it is and about how long."""
    if not dbs:
        return
    seconds = total_bytes / _COPY_BYTES_PER_S
    size = (f"{total_bytes / 1024 ** 3:.1f} GB" if total_bytes >= 1024 ** 3
            else f"{max(total_bytes, 1) / 1024 ** 2:.0f} MB")
    about = "under a second" if seconds < 1 else f"about {seconds:.0f} seconds"
    why = "before going back to an older version" if reason == "pre-downgrade" \
        else "before updating it"
    logger.log(
        logging.WARNING if seconds >= _ANNOUNCE_WARN_S else logging.INFO,
        "[SLM] Taking a safety copy of %s %s (%s, %s). SuperLocalMemory carries "
        "on as soon as it is done; please do not stop it meanwhile.",
        " and ".join(db.name for db in dbs), why, size, about)


def _gc_old_backups(
    backups_root: Path, keep: int = 2, protect: frozenset[str] = frozenset(),
) -> None:
    """Remove old pre-migration snapshot GENERATIONS, retaining ``keep`` newest.

    A generation is one migration's snapshots — ``memory-<ts>-pre-migration.db``
    and ``learning-<ts>-pre-migration.db`` share a timestamp and are only useful
    together. Counting files instead of generations kept ``keep`` FILES: with
    ``keep=2`` that is a single generation, and where mtimes interleave it could
    retain a ``memory`` snapshot whose matching ``learning`` snapshot had been
    deleted — a half set that cannot restore a consistent store.

    Generations are ordered by the time in their NAME, never by mtime, which
    was nondeterministic when two copies landed in one filesystem second. A
    ``-N`` collision suffix stays in its generation. Copies named for the
    version they precede (``memory-<local time>-pre-4.1.0.db``) and the first
    release's second-granularity names are generations too, and every copy's
    ``-wal`` / ``-shm`` goes with it; ``_snapshot_retention`` has the rules.

    Only regular files directly under ``backups_root`` are eligible, never
    through a symlink. Every deletion uses an explicit full path; no glob is
    ever passed to the deletion call.

    ``protect`` names the files of the copy just taken: that generation is
    kept whatever time is in its name (a clock running behind would otherwise
    have it pruned on the spot). Staging files a crash left behind go too.
    """
    from superlocalmemory.storage import _snapshot_manifest
    from superlocalmemory.storage._snapshot_retention import prune

    prune(backups_root, keep, protect)
    # A manifest goes with the generation it describes (explicit paths only).
    _snapshot_manifest.prune_orphan_manifests(backups_root)


def _live_fact_count(memory_db: Path) -> int | None:
    """How many facts the live store holds, read-only; None when unreadable."""
    if not memory_db.exists():
        return None
    try:
        conn = sqlite3.connect(f"{memory_db.absolute().as_uri()}?mode=ro", uri=True)
        try:
            return int(conn.execute("SELECT COUNT(*) FROM atomic_facts").fetchone()[0])
        finally:
            conn.close()
    except sqlite3.Error:
        return None


def _prove_and_describe(
    backups_root: Path,
    pairs: list[tuple[Path, Path]],
    memory_db: Path,
    facts_before: int | None,
    *,
    reason: str,
) -> Path | None:
    """Prove the copies hold the store, then write their manifest.

    Runs after both copies are renamed into place and before the caller lets
    any migration touch the store. A memory copy whose fact count the store
    cannot explain -- an empty or truncated database that still passes SQLite's
    own check -- raises ``SnapshotUnusableError``, so the update does not run.
    The unproven copies are removed (explicit paths): left under a final name,
    retention would count them as a generation and prune a good one instead.
    """
    from superlocalmemory.storage import _snapshot_manifest

    if not pairs:
        return None
    try:
        counts = None
        for snap, db in pairs:
            if Path(db) == Path(memory_db):
                counts = _snapshot_manifest.check_copy_holds_the_store(
                    snap, facts_before, _live_fact_count(memory_db))
        stamp = _snapshot_manifest.stamp_of(pairs[0][0]) or pairs[0][0].stem
        return _snapshot_manifest.write_generation_manifest(
            backups_root, stamp, pairs, reason=reason, counts=counts)
    except BaseException:
        for snap, _db in pairs:
            Path(snap).unlink(missing_ok=True)
        raise


def _write_generation_manifest(
    backups_root: Path, stamp: str, pairs: list[tuple[Path, Path]], *, reason: str,
) -> Path:
    """manifest-<stamp>-pre-migration.json for copies already renamed into place."""
    from superlocalmemory.storage import _snapshot_manifest

    return _snapshot_manifest.write_generation_manifest(
        backups_root, stamp, pairs, reason=reason,
        counts=_snapshot_manifest.store_counts(pairs[0][0]) if pairs else None)


def _cleanup_stale_partials(backups_root: Path) -> list[Path]:
    """Unlink '*-pre-migration.db.partial' (+ -wal/-shm) older than 1 h."""
    from superlocalmemory.storage import _snapshot_manifest

    return _snapshot_manifest.cleanup_stale_partials(backups_root)


def verify_snapshot(snapshot: Path, expected_sha256: str | None) -> None:
    """Raise ``SnapshotUnusableError`` unless ``snapshot`` is intact and readable."""
    from superlocalmemory.storage import _snapshot_manifest

    _snapshot_manifest.verify_snapshot(snapshot, expected_sha256)
