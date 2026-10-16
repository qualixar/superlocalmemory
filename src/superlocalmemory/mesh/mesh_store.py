# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""mesh.db: the mesh's own SQLite file, next to memory.db.

Up to 4.2.7 the mesh tables lived in memory.db, so every heartbeat and
registration opened its own connection and raced the canonical memory writer
("database is locked", 382 times on 13-14 Oct). From 4.2.8 they live in
``mesh.db`` in the same data directory, and a mesh write never waits on a
memory write.

The mesh is live coordination, not memory. On first open this module builds
mesh.db in a side file, copies the *configuration* rows of the old tables
(owner controls on peer profiles, web-app peers, node identity, the replay
window of nonces) and only the last ``retention_days`` of *activity*, writes a
marker, and renames the file into place. Only a verified mesh.db lets the old
tables be dropped from memory.db. Every step is idempotent: a crash before the
rename leaves no mesh.db and the next start builds again; a crash after it
leaves both, and the next start merges the configuration rows once more
(INSERT OR IGNORE) and drops.

A damaged mesh.db is quarantined (``mesh.db.quarantine-<ms>``) and rebuilt; a
lost one is rebuilt empty. ``mesh.db.migrated`` (an empty file) records that a
mesh.db existed here, so a later loss is logged as one instead of a first start.
"""

from __future__ import annotations

import logging
import os
import sqlite3
import threading
import time
from collections.abc import Sequence
from contextlib import contextmanager
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Iterator

from . import broker_cleanup
from .broker_security import apply_security_schema

logger = logging.getLogger("superlocalmemory.mesh")

MESH_DB_NAME = "mesh.db"
BUILD_SUFFIX = ".building"
DEFAULT_RETENTION_DAYS = broker_cleanup.DEFAULT_RETENTION_DAYS
BUSY_TIMEOUT_MS = 2000
_LOCK_WAIT_S = 30.0

#: Settings, never aged out: copied whole. Web-app peers (in mesh_peers) are
#: configuration too and are selected by kind below.
CONFIG_TABLES: tuple[str, ...] = ("mesh_peer_profiles", "mesh_node_identity", "mesh_nonces")

#: Activity copied only from the retention window: table -> time filter.
_RECENT = "{col} >= :cutoff"
_ACTIVITY_FILTERS: dict[str, str] = {
    "mesh_peers": ("last_heartbeat >= :cutoff OR peer_id IN "
                   "(SELECT peer_id FROM legacy.mesh_peer_profiles WHERE kind='web')"),
    "mesh_messages": _RECENT.format(col="created_at"),
    "mesh_events": _RECENT.format(col="created_at"),
    "mesh_state": _RECENT.format(col="updated_at"),
    "mesh_locks": "locked_at >= :cutoff AND expires_at >= :now",
    "mesh_sent_ops": _RECENT.format(col="created_at"),
    "mesh_outbox_remote": "created_at >= :cutoff_epoch",
}
#: Rows that belong to a message travel with it.
_MESSAGE_CHILDREN: tuple[str, ...] = (
    "mesh_reads", "mesh_message_envelopes", "mesh_web_deliveries")

MESH_TABLES: tuple[str, ...] = (
    "mesh_peers", "mesh_messages", "mesh_state", "mesh_locks", "mesh_events",
    "mesh_reads", "mesh_sent_ops", "mesh_nonces", "mesh_peer_profiles",
    "mesh_message_envelopes", "mesh_web_deliveries", "mesh_node_identity",
    "mesh_outbox_remote",
)

_META_DDL = "CREATE TABLE IF NOT EXISTS mesh_meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)"
_SEQUENCED = ("mesh_messages", "mesh_events", "mesh_outbox_remote")
_warned: set[str] = set()
_warned_lock = threading.Lock()


class MeshStoreError(RuntimeError):
    """mesh.db could not be built or verified."""


def mesh_db_path(memory_db: str | Path) -> Path:
    """mesh.db sits next to memory.db, so it follows SLM_DATA_DIR and the store root."""
    return Path(memory_db).parent / MESH_DB_NAME


def _warn_once(key: str, message: str, *args: object) -> None:
    with _warned_lock:
        if key in _warned:
            return
        _warned.add(key)
    logger.warning(message, *args)


def connect(path: str | Path) -> sqlite3.Connection:
    """A mesh.db connection with the broker's pragmas."""
    conn = sqlite3.connect(str(path), timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        conn.execute(f"PRAGMA busy_timeout={BUSY_TIMEOUT_MS}")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.row_factory = sqlite3.Row
    except BaseException:
        conn.close()
        raise
    return conn


def create_schema(conn: sqlite3.Connection) -> None:
    """Every mesh table at its current shape. Idempotent."""
    from superlocalmemory.storage import schema_v343 as v343
    from superlocalmemory.storage.migrations import M023_mesh_profile_isolation as m023
    from superlocalmemory.storage.migrations import M025_perf_indexes as m025

    from .node_identity import _CREATE_SQL as node_ddl
    from .outbox_remote import _CREATE_INDEX_SQL, _CREATE_TABLE_SQL

    conn.executescript(v343._MESH_DDL)
    for sql in v343._MESH_V346_ALTERS:
        _add_column(conn, sql)
    conn.executescript(v343._MESH_V346_DDL)
    m023.apply(conn)
    m025.apply(conn)
    apply_security_schema(conn)
    conn.executescript(node_ddl + ";" + _CREATE_TABLE_SQL + _CREATE_INDEX_SQL)
    conn.execute(_META_DDL)
    conn.commit()


def _add_column(conn: sqlite3.Connection, sql: str) -> None:
    try:
        conn.execute(sql)
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def inspect(path: str | Path) -> dict:
    """Health of a mesh.db: ``status`` is ok, missing, unmigrated or corrupt."""
    p = Path(path)
    if not p.exists():
        return {"status": "missing", "path": str(p)}
    try:
        # mode=rw, not ro: a WAL file with no live -shm cannot be opened read-only.
        conn = sqlite3.connect(f"{p.absolute().as_uri()}?mode=rw", uri=True, timeout=2)
        try:
            check = conn.execute("PRAGMA quick_check").fetchone()[0]
            if check != "ok":
                return {"status": "corrupt", "path": str(p), "detail": check}
            row = conn.execute(
                "SELECT value FROM mesh_meta WHERE key='migrated_at'").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError as exc:
        status = "unmigrated" if "no such table" in str(exc).lower() else "corrupt"
        return {"status": status, "path": str(p), "detail": str(exc)}
    if row is None:
        return {"status": "unmigrated", "path": str(p)}
    return {"status": "ok", "path": str(p), "migrated_at": row[0],
            "size_bytes": p.stat().st_size}


@contextmanager
def _migration_lock(target: Path) -> Iterator[None]:
    """One process at a time builds or drops; waits for the other to finish."""
    from superlocalmemory.core.file_lock import LockHeldError, exclusive_lock

    lock_path = target.with_name(target.name + ".lock")
    deadline = time.monotonic() + _LOCK_WAIT_S
    while True:
        try:
            with exclusive_lock(lock_path):
                yield
                return
        except LockHeldError:
            if time.monotonic() > deadline:
                raise MeshStoreError(f"{lock_path} stayed locked for {_LOCK_WAIT_S:.0f}s")
            time.sleep(0.05)


def _siblings(path: Path) -> list[Path]:
    return [Path(str(path) + s) for s in ("", "-wal", "-shm", "-journal")]


def _quarantine(target: Path, reason: str) -> None:
    stamp = int(time.time() * 1000)
    for part in _siblings(target):
        if part.exists():
            part.rename(part.with_name(f"{part.name}.quarantine-{stamp}"))
    _warn_once(f"quarantine:{target}", "mesh.db at %s was %s; quarantined and rebuilt",
               target, reason)


def _legacy_tables(memory_db: Path) -> set[str]:
    if not memory_db.exists():
        return set()
    conn = sqlite3.connect(f"{memory_db.absolute().as_uri()}?mode=rw", uri=True,
                           timeout=BUSY_TIMEOUT_MS / 1000)
    try:
        return {r[0] for r in conn.execute(
            "SELECT name FROM sqlite_master WHERE type='table'")} & set(MESH_TABLES)
    finally:
        conn.close()


def _columns(conn: sqlite3.Connection, schema: str, table: str) -> list[str]:
    return [r[1] for r in conn.execute(f'PRAGMA {schema}.table_info("{table}")')]


def _copy(conn: sqlite3.Connection, table: str, where: str, params: dict,
          verb: str = "INSERT") -> int:
    """Copy one legacy table's rows (shared columns only), verified by count."""
    cols = [c for c in _columns(conn, "legacy", table) if c in _columns(conn, "main", table)]
    if not cols:
        return 0
    names = ", ".join(f'"{c}"' for c in cols)
    expected = conn.execute(
        f'SELECT COUNT(*) FROM legacy."{table}" WHERE {where}', params).fetchone()[0]
    cur = conn.execute(f'{verb} INTO main."{table}" ({names}) '
                       f'SELECT {names} FROM legacy."{table}" WHERE {where}', params)
    if verb == "INSERT" and cur.rowcount != expected:
        raise MeshStoreError(f"{table}: copied {cur.rowcount} of {expected} rows")
    return cur.rowcount


def _window(retention_days: int) -> dict:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=retention_days)
    return {"cutoff": cutoff.isoformat(), "now": now.isoformat(),
            "cutoff_epoch": cutoff.timestamp()}


def _copy_activity(conn: sqlite3.Connection, present: set[str], window: dict) -> dict:
    counts: dict[str, int] = {}
    for table in CONFIG_TABLES:
        if table in present:
            counts[table] = _copy(conn, table, "1=1", window)
    for table, where in _ACTIVITY_FILTERS.items():
        if table not in present:
            continue
        if table == "mesh_peers" and "mesh_peer_profiles" not in present:
            where = "last_heartbeat >= :cutoff"
        counts[table] = _copy(conn, table, where, window)
    for table in _MESSAGE_CHILDREN:
        if table in present:
            counts[table] = _copy(
                conn, table, "message_id IN (SELECT id FROM main.mesh_messages)", window)
    _carry_sequences(conn, present)
    return counts


def _carry_sequences(conn: sqlite3.Connection, present: set[str]) -> None:
    """An AUTOINCREMENT id deleted in memory.db is never handed out again."""
    if not conn.execute("SELECT 1 FROM legacy.sqlite_master "
                        "WHERE name='sqlite_sequence'").fetchone():
        return
    for table in _SEQUENCED:
        row = conn.execute("SELECT seq FROM legacy.sqlite_sequence WHERE name=?",
                           (table,)).fetchone()
        if table not in present or row is None:
            continue
        updated = conn.execute(
            "UPDATE main.sqlite_sequence SET seq=MAX(seq, ?) WHERE name=?", (row[0], table))
        if updated.rowcount == 0:
            conn.execute("INSERT INTO main.sqlite_sequence (name, seq) VALUES (?, ?)",
                         (table, row[0]))


def _attach_legacy(conn: sqlite3.Connection, memory_db: Path) -> None:
    """memory.db attached for reading only (nothing here writes to it). A plain
    path, not a read-only URI: a WAL file with no live -shm cannot open read-only."""
    conn.execute("ATTACH DATABASE ? AS legacy", (str(memory_db.absolute()),))


def _build(target: Path, memory_db: Path, retention_days: int) -> dict:
    """Build mesh.db in a side file, then rename it into place."""
    building = target.with_name(target.name + BUILD_SUFFIX)
    for part in _siblings(building):
        part.unlink(missing_ok=True)
    present = _legacy_tables(memory_db)
    conn = sqlite3.connect(f"{building.absolute().as_uri()}?mode=rwc", uri=True)
    try:
        create_schema(conn)
        counts: dict[str, int] = {}
        if present:
            _attach_legacy(conn, memory_db)  # ATTACH cannot run inside a transaction
        conn.execute("BEGIN")
        if present:
            counts = _copy_activity(conn, present, _window(retention_days))
        source = "memory.db" if present else "fresh"
        _write_marker(conn, source)
        conn.commit()
        if present:
            conn.execute("DETACH DATABASE legacy")
    finally:
        conn.close()
    _fsync(building)
    for part in _siblings(target)[1:]:
        part.unlink(missing_ok=True)  # a stale WAL must never replay onto the new file
    os.replace(building, target)
    _enable_wal(target)
    logger.info("mesh.db ready at %s (from %s): %s", target, source, counts)
    return counts


def _write_marker(conn: sqlite3.Connection, source: str) -> None:
    stamp = datetime.now(timezone.utc).isoformat()
    conn.executemany("INSERT OR REPLACE INTO mesh_meta (key, value) VALUES (?, ?)",
                     (("migrated_at", stamp), ("source", source), ("schema", "1")))


def _fsync(path: Path) -> None:
    fd = os.open(str(path), os.O_RDONLY)
    try:
        os.fsync(fd)
    finally:
        os.close(fd)


def _enable_wal(target: Path) -> None:
    conn = connect(target)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
    finally:
        conn.close()


def _drop_tables(conn: sqlite3.Connection, names: Sequence[str]) -> None:
    """Drop the old mesh tables in one transaction: all of them or none."""
    conn.execute("BEGIN IMMEDIATE")
    try:
        for name in names:
            conn.execute(f'DROP TABLE IF EXISTS "{name}"')
        conn.execute("COMMIT")
    except BaseException:
        conn.execute("ROLLBACK")
        raise


def _merge_config_and_drop(target: Path, memory_db: Path) -> None:
    """With mesh.db verified, keep any configuration still only in memory.db, then drop."""
    present = _legacy_tables(memory_db)
    if not present:
        return
    conn = connect(target)
    try:
        _attach_legacy(conn, memory_db)
        conn.execute("BEGIN")
        for table in CONFIG_TABLES:
            if table in present:
                _copy(conn, table, "1=1", {}, verb="INSERT OR IGNORE")
        if {"mesh_peers", "mesh_peer_profiles"} <= present:
            _copy(conn, "mesh_peers", "peer_id IN (SELECT peer_id FROM "
                  "legacy.mesh_peer_profiles WHERE kind='web')", {}, verb="INSERT OR IGNORE")
        conn.commit()
        conn.execute("DETACH DATABASE legacy")
    finally:
        conn.close()
    try:
        _drop_legacy(memory_db, sorted(present))
    except sqlite3.OperationalError as exc:
        # The memory writer held memory.db past the wait. mesh.db is already
        # verified and in use; the old tables are dropped at the next start.
        _warn_once(f"drop:{memory_db}", "mesh tables not yet dropped from %s (%s); "
                   "retrying at next start", memory_db.name, exc)


def _drop_legacy(memory_db: Path, names: list[str]) -> None:
    from superlocalmemory.storage.write_lock import get_write_lock

    with get_write_lock(memory_db):  # the in-process memory writers wait, not race
        conn = sqlite3.connect(f"{memory_db.absolute().as_uri()}?mode=rw", uri=True,
                               timeout=30, isolation_level=None)
        try:
            conn.execute("PRAGMA busy_timeout=30000")
            _drop_tables(conn, names)
        finally:
            conn.close()
    logger.info("mesh tables dropped from %s: %s", memory_db.name, ", ".join(names))


def apply_retention(target: Path, retention_days: int) -> None:
    """The start-of-day sweep; the broker's loop repeats it every five minutes."""
    conn = connect(target)
    try:
        broker_cleanup.run_cleanup(conn, retention_days=retention_days)
    finally:
        conn.close()


def prepare_mesh_db(memory_db: str | Path, *, retention_days: int = DEFAULT_RETENTION_DAYS,
                    drop_legacy: bool = True) -> Path:
    """Return a verified mesh.db next to ``memory_db``, building or rebuilding it if needed."""
    memory_db = Path(memory_db)
    target = mesh_db_path(memory_db)
    target.parent.mkdir(parents=True, exist_ok=True)
    with _migration_lock(target):
        state = inspect(target)
        if state["status"] != "ok":
            if state["status"] != "missing":
                _quarantine(target, state["status"])
            elif not _legacy_tables(memory_db) and target.with_name(
                    target.name + ".migrated").exists():
                _warn_once(f"lost:{target}", "mesh.db at %s was missing; recreated empty",
                           target)
            _build(target, memory_db, retention_days)
            target.with_name(target.name + ".migrated").touch()
            if inspect(target)["status"] != "ok":
                raise MeshStoreError(f"{target} failed verification after build")
        if drop_legacy:
            _merge_config_and_drop(target, memory_db)
        apply_retention(target, retention_days)
    return target


__all__ = [
    "BUILD_SUFFIX", "CONFIG_TABLES", "DEFAULT_RETENTION_DAYS", "MESH_DB_NAME", "MESH_TABLES",
    "MeshStoreError", "apply_retention", "connect", "create_schema", "inspect",
    "mesh_db_path", "prepare_mesh_db",
]
