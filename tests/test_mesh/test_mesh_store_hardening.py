# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""mesh.db pre-merge hardening (PR #271 review).

Settings are kept forever: shared state (mesh_state) is configuration, a
damaged mesh.db gives its settings to the one rebuilt in its place, a busy
mesh.db is never mistaken for a damaged one, and the rename that publishes a
new mesh.db is made durable in its directory.
"""

from __future__ import annotations

import os
import shutil
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from superlocalmemory.mesh import broker_cleanup, mesh_store
from superlocalmemory.mesh.node_identity import get_node_id

NOW = datetime.now(timezone.utc)
MONTH_AGO = (NOW - timedelta(days=30)).isoformat()
WEEKS_AGO = (NOW - timedelta(days=20)).isoformat()


def _col(path: Path, sql: str) -> list:
    conn = sqlite3.connect(path)
    try:
        return [r[0] for r in conn.execute(sql)]
    finally:
        conn.close()


def _exec(path: Path, *statements: tuple[str, tuple]) -> None:
    conn = sqlite3.connect(path)
    try:
        for sql, params in statements:
            conn.execute(sql, params)
        conn.commit()
    finally:
        conn.close()


def _legacy_with_old_setting(path: Path) -> Path:
    """A 4.2.7 memory.db whose shared state was set a month ago."""
    from tests.test_mesh.test_mesh_store import _legacy_store

    _legacy_store(path)
    _exec(path, ("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                 " VALUES ('feature.flag', 'on', 'p', ?)", (MONTH_AGO,)))
    return path


def _settings(mesh: Path) -> tuple[Path, str]:
    """A built mesh.db (legacy tables already dropped) holding every kind of setting."""
    _exec(mesh,
          ("INSERT INTO mesh_peer_profiles (peer_id, kind, updated_at, muted, display_name)"
           " VALUES ('web-app', 'web', ?, 1, 'Phone')", (WEEKS_AGO,)),
          ("INSERT INTO mesh_peers (peer_id, session_id, registered_at, last_heartbeat,"
           " peer_key) VALUES ('web-app', 'web-app', ?, ?, 'k-web')", (WEEKS_AGO, WEEKS_AGO)),
          ("INSERT INTO mesh_state (key, value, set_by, updated_at)"
           " VALUES ('task.owner', 'agent-7', 'p', ?)", (MONTH_AGO,)),
          ("INSERT INTO mesh_messages (from_peer, to_peer, content, created_at)"
           " VALUES ('a', 'b', 'activity', ?)", (NOW.isoformat(),)),
          ("INSERT INTO mesh_nonces VALUES ('n-keep', 1.0)", ()))
    return mesh, get_node_id(str(mesh))


def _checkpoint(path: Path) -> None:
    conn = sqlite3.connect(path)
    try:
        conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
    finally:
        conn.close()


def _break_index_page(mesh: Path, index: str = "idx_mesh_events_type") -> None:
    """Garbage over one index page: quick_check fails, the tables stay readable."""
    _exec(mesh, *[("INSERT INTO mesh_events (event_type, emitted_by, created_at)"
                   " VALUES (?, 'p', ?)", (f"e{i}", NOW.isoformat())) for i in range(50)])
    _checkpoint(mesh)
    page_size = _col(mesh, "PRAGMA page_size")[0]
    root = _col(mesh, f"SELECT rootpage FROM sqlite_master WHERE name='{index}'")[0]
    with open(mesh, "r+b") as fh:
        fh.seek((root - 1) * page_size)
        fh.write(b"\xa5" * page_size)
    assert mesh_store.inspect(mesh)["status"] == "corrupt"


# -- 1: mesh_state is configuration --------------------------------------------

def test_month_old_shared_state_survives_first_start(tmp_path: Path) -> None:
    memory = _legacy_with_old_setting(tmp_path / "memory.db")
    mesh = mesh_store.prepare_mesh_db(memory, retention_days=7)
    assert _col(mesh, "SELECT value FROM mesh_state WHERE key='feature.flag'") == ["on"]


def test_retention_never_ages_out_shared_state(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _exec(mesh, ("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                 " VALUES ('old', 'v', 'p', ?)", (MONTH_AGO,)))
    conn = sqlite3.connect(mesh)
    broker_cleanup.run_cleanup(conn, retention_days=1)
    conn.close()
    assert _col(mesh, "SELECT key FROM mesh_state") == ["old"]
    assert "mesh_state" in mesh_store.CONFIG_TABLES


def test_shared_state_left_in_memory_db_is_merged(tmp_path: Path) -> None:
    memory = _legacy_with_old_setting(tmp_path / "memory.db")
    mesh = mesh_store.prepare_mesh_db(memory, drop_legacy=False)
    _exec(memory, ("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                   " VALUES ('late', 'v', 'p', ?)", (MONTH_AGO,)))
    mesh_store.prepare_mesh_db(memory)
    assert sorted(_col(mesh, "SELECT key FROM mesh_state")) == ["feature.flag", "late"]


# -- 2: a rebuilt mesh.db keeps the settings of the damaged one -----------------

def _assert_settings_kept(mesh: Path, node_id: str) -> None:
    assert _col(mesh, "SELECT display_name FROM mesh_peer_profiles"
                      " WHERE peer_id='web-app'") == ["Phone"]
    assert _col(mesh, "SELECT peer_key FROM mesh_peers WHERE peer_id='web-app'") == ["k-web"]
    assert _col(mesh, "SELECT value FROM mesh_state WHERE key='task.owner'") == ["agent-7"]
    assert "n-keep" in _col(mesh, "SELECT nonce FROM mesh_nonces")
    assert get_node_id(str(mesh)) == node_id
    # Activity is not configuration: the damaged file's messages stay in quarantine.
    assert _col(mesh, "SELECT COUNT(*) FROM mesh_messages") == [0]


def test_corrupt_mesh_db_rebuild_salvages_settings(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    mesh, node_id = _settings(mesh)
    _break_index_page(mesh)
    damaged = mesh.read_bytes()
    assert mesh_store.prepare_mesh_db(tmp_path / "memory.db") == mesh
    assert mesh_store.inspect(mesh)["status"] == "ok"
    _assert_settings_kept(mesh, node_id)
    quarantined = [p for p in mesh.parent.glob("mesh.db.quarantine-*")]
    assert len(quarantined) == 1
    assert quarantined[0].read_bytes() == damaged  # the evidence is never modified
    assert not list(mesh.parent.glob("*salvage*"))


def test_settings_only_in_the_wal_are_salvaged(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _checkpoint(mesh)
    snapshot = tmp_path / "snap"
    snapshot.mkdir()
    keeper = sqlite3.connect(mesh)  # stays open: the WAL is not checkpointed away
    keeper.execute("PRAGMA wal_autocheckpoint=0")
    keeper.execute("INSERT INTO mesh_peer_profiles (peer_id, kind, updated_at, display_name)"
                   " VALUES ('wal-only', 'local', ?, 'W')", (NOW.isoformat(),))
    keeper.execute("DELETE FROM mesh_meta")  # an unmigrated file: rebuilt
    keeper.commit()
    for suffix in ("", "-wal"):
        shutil.copy2(f"{mesh}{suffix}", snapshot / f"mesh.db{suffix}")
    keeper.close()
    for suffix in ("", "-wal", "-shm"):
        Path(f"{mesh}{suffix}").unlink(missing_ok=True)
    for suffix in ("", "-wal"):
        shutil.copy2(snapshot / f"mesh.db{suffix}", f"{mesh}{suffix}")
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    assert _col(mesh, "SELECT display_name FROM mesh_peer_profiles"
                      " WHERE peer_id='wal-only'") == ["W"]


def test_unreadable_mesh_db_still_rebuilds(tmp_path: Path) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    mesh.write_bytes(b"not a database at all" * 100)
    assert mesh_store.prepare_mesh_db(tmp_path / "memory.db") == mesh
    assert mesh_store.inspect(mesh)["status"] == "ok"


def test_salvage_keeps_rows_of_a_table_damaged_part_way(tmp_path: Path, monkeypatch) -> None:
    """A read that fails mid-table keeps every row read before the failure."""
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _settings(mesh)
    _exec(mesh, ("INSERT INTO mesh_nonces VALUES ('n-second', 2.0)", ()))
    _break_index_page(mesh)
    real_rows = mesh_store._salvage_rows

    def failing_mid_nonces(conn, table, select):
        rows = real_rows(conn, table, select)
        if table != "mesh_nonces":
            yield from rows
            return
        try:
            yield next(rows)
            raise sqlite3.DatabaseError("database disk image is malformed")
        finally:
            rows.close()

    monkeypatch.setattr(mesh_store, "_salvage_rows", failing_mid_nonces)
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    assert len(_col(mesh, "SELECT nonce FROM mesh_nonces")) == 1
    assert _col(mesh, "SELECT value FROM mesh_state WHERE key='task.owner'") == ["agent-7"]


# -- 3: busy is not damaged -----------------------------------------------------

def _hold_exclusive(mesh: Path) -> sqlite3.Connection:
    holder = sqlite3.connect(mesh, isolation_level=None, check_same_thread=False)
    holder.execute("PRAGMA locking_mode=EXCLUSIVE")
    holder.execute("BEGIN EXCLUSIVE")
    holder.execute("INSERT INTO mesh_nonces VALUES ('held', 1.0)")
    return holder


def test_locked_mesh_db_is_unavailable_not_corrupt(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mesh_store, "_INSPECT_TIMEOUT_S", 0.1)
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    holder = _hold_exclusive(mesh)
    try:
        state = mesh_store.inspect(mesh)
    finally:
        holder.close()
    assert state["status"] == "unavailable"
    assert "locked" in state["detail"]


def test_prepare_raises_on_a_lock_that_never_clears_and_quarantines_nothing(
        tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mesh_store, "_INSPECT_TIMEOUT_S", 0.1)
    monkeypatch.setattr(mesh_store, "_INSPECT_BACKOFF_S", 0.01)
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    _exec(mesh, ("INSERT INTO mesh_state (key, value, set_by, updated_at)"
                 " VALUES ('keep', 'v', 'p', ?)", (NOW.isoformat(),)))
    holder = _hold_exclusive(mesh)
    try:
        with pytest.raises(mesh_store.MeshStoreError, match="locked"):
            mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    finally:
        holder.rollback()
        holder.close()
    assert not list(mesh.parent.glob("mesh.db*.quarantine-*"))
    assert _col(mesh, "SELECT key FROM mesh_state") == ["keep"]


def test_prepare_waits_out_a_lock_that_clears(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mesh_store, "_INSPECT_TIMEOUT_S", 0.1)
    monkeypatch.setattr(mesh_store, "_INSPECT_BACKOFF_S", 0.2)
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    holder = _hold_exclusive(mesh)
    released = threading.Timer(0.3, lambda: (holder.rollback(), holder.close()))
    released.start()
    try:
        assert mesh_store.prepare_mesh_db(tmp_path / "memory.db") == mesh
    finally:
        released.join()
    assert not list(mesh.parent.glob("mesh.db*.quarantine-*"))
    assert "held" not in _col(mesh, "SELECT nonce FROM mesh_nonces")


def test_open_error_is_unavailable(tmp_path: Path, monkeypatch) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")

    def refuse(*_a, **_k):
        raise sqlite3.OperationalError("unable to open database file")

    monkeypatch.setattr(mesh_store.sqlite3, "connect", refuse)
    assert mesh_store.inspect(mesh)["status"] == "unavailable"


@pytest.mark.parametrize("message", ["database disk image is malformed",
                                     "file is not a database"])
def test_damage_errors_are_corrupt(tmp_path: Path, monkeypatch, message: str) -> None:
    mesh = mesh_store.prepare_mesh_db(tmp_path / "memory.db")

    def damaged(*_a, **_k):
        raise sqlite3.DatabaseError(message)

    monkeypatch.setattr(mesh_store.sqlite3, "connect", damaged)
    assert mesh_store.inspect(mesh)["status"] == "corrupt"


def test_broker_does_not_rebuild_a_busy_store(tmp_path: Path, monkeypatch) -> None:
    from superlocalmemory.mesh.broker import MeshBroker

    broker = MeshBroker.for_store(tmp_path / "memory.db")
    monkeypatch.setattr(mesh_store, "inspect",
                        lambda _p: {"status": "unavailable", "detail": "database is locked"})
    rebuilt: list[int] = []
    monkeypatch.setattr(mesh_store, "prepare_mesh_db", lambda *a, **k: rebuilt.append(1))
    assert broker._recover_store() is False
    assert rebuilt == []


# -- 6: the rename is durable ---------------------------------------------------

@pytest.mark.skipif(os.name != "posix", reason="directory fsync is POSIX only")
def test_parent_directory_is_fsynced_after_the_rename(tmp_path: Path, monkeypatch) -> None:
    events: list[str] = []
    real_replace, real_fsync, real_open = os.replace, os.fsync, os.open
    opened: dict[int, str] = {}

    def replace(src, dst):
        real_replace(src, dst)
        events.append(f"replace:{Path(dst).name}")

    def tracking_open(path, flags, *args):
        fd = real_open(path, flags, *args)
        opened[fd] = str(path)
        return fd

    def fsync(fd):
        events.append(f"fsync:{opened.get(fd, '?')}")
        return real_fsync(fd)

    monkeypatch.setattr(mesh_store.os, "replace", replace)
    monkeypatch.setattr(mesh_store.os, "open", tracking_open)
    monkeypatch.setattr(mesh_store.os, "fsync", fsync)
    mesh_store.prepare_mesh_db(tmp_path / "memory.db")
    at = events.index("replace:mesh.db")
    assert f"fsync:{tmp_path}" in events[at + 1:]


def test_directory_fsync_is_skipped_off_posix(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(mesh_store.os, "name", "nt")
    fsynced: list[int] = []
    monkeypatch.setattr(mesh_store.os, "fsync", fsynced.append)
    mesh_store._fsync_dir(tmp_path)
    assert fsynced == []
