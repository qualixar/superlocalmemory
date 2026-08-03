# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# Integration tests: SLM V4 sub-wave 3c — LWW remote state convergence (3c-1)
#
# Serial-safe (all tests operate on isolated temp SQLite DBs; no shared state).
# Run: pytest tests/integration/test_mesh_state_sync_p6c.py -p no:cacheprovider -x -v

from __future__ import annotations

import sqlite3
from datetime import datetime, timezone

import pytest

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_mesh_db(tmp_path, name: str = "state_test.db"):
    """Create a mesh DB with the full schema, including revision + origin_node."""
    db_path = tmp_path / name
    conn = sqlite3.connect(str(db_path))

    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL,
        _MESH_V346_ALTERS,
        _MESH_V346_DDL,
    )

    conn.executescript(_MESH_DDL)
    for alter_sql in _MESH_V346_ALTERS:
        try:
            conn.execute(alter_sql)
        except sqlite3.OperationalError:
            pass
    conn.executescript(_MESH_V346_DDL)

    # Add revision column (normally added by apply_security_schema in broker init)
    from superlocalmemory.mesh.broker_security import apply_security_schema

    apply_security_schema(conn)

    # Add origin_node column (added by StateSyncer._ensure_origin_node_column on first
    # instantiation).  Including it here means _write_row can always reference it
    # without requiring a prior StateSyncer call in the test setup.
    try:
        conn.execute(
            "ALTER TABLE mesh_state ADD COLUMN origin_node TEXT NOT NULL DEFAULT ''"
        )
    except sqlite3.OperationalError:
        pass  # Already present

    conn.commit()
    conn.close()
    return db_path


class _FakeBroker:
    """Minimal broker-like object: exposes only _db_path and set_state.

    Mirrors what the real MeshBroker exposes for the StateSyncer contract.
    """

    def __init__(self, db_path) -> None:
        self._db_path = str(db_path)
        self._shared_secret = None

    def set_state(
        self,
        key: str,
        value: str,
        set_by: str,
        profile_id: str = "default",
        expected_revision: int | None = None,
    ) -> dict:
        """Write a state row mimicking broker.set_state (auto-increments revision)."""
        conn = sqlite3.connect(self._db_path, timeout=5.0)
        conn.row_factory = sqlite3.Row
        try:
            now = datetime.now(timezone.utc).isoformat()
            conn.execute(
                "INSERT INTO mesh_state"
                " (profile_id, key, value, set_by, updated_at, revision)"
                " VALUES (?, ?, ?, ?, ?, 1)"
                " ON CONFLICT(profile_id, key) DO UPDATE SET"
                " value=excluded.value, set_by=excluded.set_by,"
                " updated_at=excluded.updated_at,"
                " revision=COALESCE(mesh_state.revision, 0) + 1",
                (profile_id, key, value, set_by, now),
            )
            conn.commit()
            return {"ok": True}
        except sqlite3.Error as exc:
            return {"ok": False, "error": str(exc)}
        finally:
            conn.close()


def _write_row(
    db_path,
    profile_id: str,
    key: str,
    value: str,
    set_by: str,
    revision: int,
    origin_node: str = "",
) -> None:
    """Insert or replace a mesh_state row with explicit revision + origin_node."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    try:
        now = datetime.now(timezone.utc).isoformat()
        conn.execute(
            "INSERT INTO mesh_state"
            " (profile_id, key, value, set_by, updated_at, revision, origin_node)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)"
            " ON CONFLICT(profile_id, key) DO UPDATE SET"
            " value=excluded.value, set_by=excluded.set_by,"
            " updated_at=excluded.updated_at,"
            " revision=excluded.revision,"
            " origin_node=excluded.origin_node",
            (profile_id, key, value, set_by, now, revision, origin_node),
        )
        conn.commit()
    finally:
        conn.close()


def _read_row(db_path, profile_id: str, key: str) -> dict | None:
    """Fetch a single mesh_state row as a plain dict, or None."""
    conn = sqlite3.connect(str(db_path), timeout=5.0)
    conn.row_factory = sqlite3.Row
    try:
        row = conn.execute(
            "SELECT value, set_by, revision, COALESCE(origin_node,'') AS origin_node"
            " FROM mesh_state WHERE profile_id=? AND key=?",
            (profile_id, key),
        ).fetchone()
        return dict(row) if row else None
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestMergeRemote:
    """Core LWW merge convergence cases."""

    def test_remote_higher_revision_wins(self, tmp_path):
        """Remote rev=2 beats local rev=1; remote value is applied, rev preserved."""
        db = _make_mesh_db(tmp_path, "remote_higher.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "mode", "idle", "agent_a", revision=1, origin_node="aaa")

        remote = [
            {
                "key": "mode",
                "value": "active",
                "set_by": "agent_b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 2,
                "node_id": "bbb",
            }
        ]
        result = syncer.merge_remote("default", remote)

        assert result == {"applied": 1, "skipped": 0}
        row = _read_row(db, "default", "mode")
        assert row is not None
        assert row["value"] == "active"
        assert row["revision"] == 2  # preserved, NOT incremented to 3
        assert row["origin_node"] == "bbb"

    def test_local_higher_revision_wins(self, tmp_path):
        """Local rev=3 beats remote rev=1; no change to local row."""
        db = _make_mesh_db(tmp_path, "local_higher.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "status", "leader", "agent_a", revision=3, origin_node="aaa")

        remote = [
            {
                "key": "status",
                "value": "follower",
                "set_by": "agent_b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 1,
                "node_id": "bbb",
            }
        ]
        result = syncer.merge_remote("default", remote)

        assert result == {"applied": 0, "skipped": 1}
        row = _read_row(db, "default", "status")
        assert row["value"] == "leader"
        assert row["revision"] == 3

    def test_equal_revision_higher_node_id_wins(self, tmp_path):
        """Same revision, remote node_id > local node_id → remote wins."""
        db = _make_mesh_db(tmp_path, "tie_remote_wins.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # local: (rev=5, node="aaa") — lower node_id
        _write_row(db, "default", "coord", "v_local", "agent_a", revision=5, origin_node="aaa")

        remote = [
            {
                "key": "coord",
                "value": "v_remote",
                "set_by": "agent_b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 5,   # same revision
                "node_id": "zzz",  # zzz > aaa
            }
        ]
        result = syncer.merge_remote("default", remote)

        assert result["applied"] == 1
        row = _read_row(db, "default", "coord")
        assert row["value"] == "v_remote"
        assert row["revision"] == 5  # preserved
        assert row["origin_node"] == "zzz"

    def test_equal_revision_lower_node_id_loses(self, tmp_path):
        """Same revision, remote node_id < local node_id → local wins."""
        db = _make_mesh_db(tmp_path, "tie_local_wins.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # local: (rev=5, node="zzz") — higher node_id
        _write_row(db, "default", "coord", "v_local", "agent_a", revision=5, origin_node="zzz")

        remote = [
            {
                "key": "coord",
                "value": "v_remote",
                "set_by": "agent_b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 5,   # same revision
                "node_id": "aaa",  # aaa < zzz
            }
        ]
        result = syncer.merge_remote("default", remote)

        assert result["applied"] == 0
        row = _read_row(db, "default", "coord")
        assert row["value"] == "v_local"
        assert row["origin_node"] == "zzz"

    def test_idempotent_merge_twice_noop(self, tmp_path):
        """Merging the same delta twice: second merge is a no-op; revision unchanged."""
        db = _make_mesh_db(tmp_path, "idempotent.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # Local starts lower
        _write_row(db, "default", "phase", "init", "a", revision=1, origin_node="aaa")

        remote = [
            {
                "key": "phase",
                "value": "running",
                "set_by": "b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 3,
                "node_id": "bbb",
            }
        ]

        r1 = syncer.merge_remote("default", remote)
        assert r1 == {"applied": 1, "skipped": 0}

        r2 = syncer.merge_remote("default", remote)
        assert r2 == {"applied": 0, "skipped": 1}, "second merge must be a no-op"

        row = _read_row(db, "default", "phase")
        assert row["value"] == "running"
        assert row["revision"] == 3  # must stay at 3, not become 4

    def test_two_way_convergence(self, tmp_path):
        """Node A and B with same key at same rev converge on the higher node_id.

        Both nodes exchange deltas → both end with the same (value, node_id).
        """
        db_a = _make_mesh_db(tmp_path, "a.db")
        db_b = _make_mesh_db(tmp_path, "b.db")

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer_a = StateSyncer(_FakeBroker(db_a))
        syncer_b = StateSyncer(_FakeBroker(db_b))

        # Override node_ids for determinism ("bbb" > "aaa" lexicographically)
        syncer_a._node_id = "aaa"
        syncer_b._node_id = "bbb"

        # Both nodes write the same key at the same revision concurrently
        _write_row(db_a, "default", "leader", "node_a_val", "s_a", revision=2, origin_node="aaa")
        _write_row(db_b, "default", "leader", "node_b_val", "s_b", revision=2, origin_node="bbb")

        # Exchange deltas
        delta_a = syncer_a.local_delta("default", since_revision=0)
        delta_b = syncer_b.local_delta("default", since_revision=0)

        syncer_a.merge_remote("default", delta_b)  # A receives B's delta
        syncer_b.merge_remote("default", delta_a)  # B receives A's delta

        row_a = _read_row(db_a, "default", "leader")
        row_b = _read_row(db_b, "default", "leader")

        # "bbb" > "aaa" → B's value wins on both sides
        assert row_a["value"] == "node_b_val", "A must converge to B's value"
        assert row_b["value"] == "node_b_val", "B must keep its own value"
        assert row_a["origin_node"] == "bbb"
        assert row_b["origin_node"] == "bbb"

    def test_merge_new_key_no_local_row(self, tmp_path):
        """Remote key that doesn't exist locally is inserted."""
        db = _make_mesh_db(tmp_path, "new_key.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # No local row for "heartbeat"
        remote = [
            {
                "key": "heartbeat",
                "value": "2026-01-01T00:00:00",
                "set_by": "remote_agent",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 1,
                "node_id": "remote_node",
            }
        ]
        result = syncer.merge_remote("default", remote)

        assert result == {"applied": 1, "skipped": 0}
        row = _read_row(db, "default", "heartbeat")
        assert row is not None
        assert row["value"] == "2026-01-01T00:00:00"

    def test_merge_empty_remote_entries_noop(self, tmp_path):
        """merge_remote with an empty list returns applied=0, skipped=0."""
        db = _make_mesh_db(tmp_path, "empty.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        result = syncer.merge_remote("default", [])
        assert result == {"applied": 0, "skipped": 0}

    def test_merge_returns_correct_applied_skipped_counts(self, tmp_path):
        """Mixed entries: some win, some lose — counts are accurate."""
        db = _make_mesh_db(tmp_path, "mixed.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "k1", "v1", "a", revision=5, origin_node="zzz")  # local wins
        _write_row(db, "default", "k2", "v2", "a", revision=1, origin_node="aaa")  # remote wins

        remote = [
            # k1: remote rev=3 < local rev=5 → skip
            {
                "key": "k1",
                "value": "v1_remote",
                "set_by": "b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 3,
                "node_id": "bbb",
            },
            # k2: remote rev=2 > local rev=1 → apply
            {
                "key": "k2",
                "value": "v2_remote",
                "set_by": "b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 2,
                "node_id": "bbb",
            },
        ]
        result = syncer.merge_remote("default", remote)
        assert result == {"applied": 1, "skipped": 1}

    def test_revision_preserved_not_incremented(self, tmp_path):
        """After remote wins, the row's revision equals the remote revision, not +1."""
        db = _make_mesh_db(tmp_path, "rev_preserved.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "token", "old", "a", revision=1, origin_node="aaa")

        remote = [
            {
                "key": "token",
                "value": "new",
                "set_by": "b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 7,
                "node_id": "bbb",
            }
        ]
        syncer.merge_remote("default", remote)
        row = _read_row(db, "default", "token")
        assert row["revision"] == 7, "revision must be 7, not 8"


class TestLocalDelta:
    """local_delta serialization and filtering."""

    def test_local_delta_returns_all_rows_since_zero(self, tmp_path):
        """since_revision=0 returns all rows in the table."""
        db = _make_mesh_db(tmp_path, "delta_all.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "a", "v_a", "x", revision=1, origin_node="nA")
        _write_row(db, "default", "b", "v_b", "x", revision=3, origin_node="nB")

        delta = syncer.local_delta("default", since_revision=0)
        keys = {e["key"] for e in delta}
        assert {"a", "b"} == keys

    def test_local_delta_since_revision_filtering(self, tmp_path):
        """since_revision=2 excludes rows with revision <= 2."""
        db = _make_mesh_db(tmp_path, "delta_since.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "old", "x", "a", revision=1)
        _write_row(db, "default", "newer", "y", "a", revision=3)

        delta = syncer.local_delta("default", since_revision=2)
        assert len(delta) == 1
        assert delta[0]["key"] == "newer"
        assert delta[0]["revision"] == 3

    def test_local_delta_node_id_resolution_empty_origin(self, tmp_path):
        """BC rows with origin_node='' resolve to the local node_id in the delta."""
        db = _make_mesh_db(tmp_path, "node_id_resolve.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # Simulate a broker-written row: origin_node stays ''
        broker.set_state("flag", "1", "agent", profile_id="default")
        # StateSyncer was instantiated first — origin_node column exists now;
        # but broker.set_state uses its own SQL that doesn't set origin_node,
        # so the column default '' applies.

        delta = syncer.local_delta("default", since_revision=0)
        assert len(delta) == 1
        entry = delta[0]
        # node_id must resolve to this node's local id, not the empty string
        assert entry["node_id"] == syncer._node_id
        assert entry["node_id"] != ""

    def test_local_delta_explicit_origin_node_passed_through(self, tmp_path):
        """Rows with a non-empty origin_node return that node_id verbatim."""
        db = _make_mesh_db(tmp_path, "origin_explicit.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "x", "v", "a", revision=2, origin_node="remote_node_xyz")

        delta = syncer.local_delta("default", since_revision=0)
        assert len(delta) == 1
        assert delta[0]["node_id"] == "remote_node_xyz"


class TestBackwardCompatibility:
    """Additive column and BC row handling."""

    def test_bc_column_noop_on_db_that_already_has_origin_node(self, tmp_path):
        """Creating a second StateSyncer on a DB that already has origin_node doesn't error."""
        db = _make_mesh_db(tmp_path, "bc_noop.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        # First instantiation adds the column
        s1 = StateSyncer(broker)
        # Second instantiation must silently no-op the duplicate column
        s2 = StateSyncer(broker)

        # Both should function correctly
        _write_row(db, "default", "alive", "yes", "x", revision=1, origin_node="n1")
        assert s1.local_delta("default") != []
        assert s2.local_delta("default") != []

    def test_bc_existing_broker_rows_with_empty_origin_merge_correctly(self, tmp_path):
        """Rows written by broker (origin_node='') participate in LWW correctly."""
        db = _make_mesh_db(tmp_path, "bc_broker_rows.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        # Write via broker — origin_node will be '' (default)
        broker.set_state("task", "pending", "local_agent", profile_id="default")

        # Local row: (rev=1, origin_node='') → effective node = syncer._node_id
        # Force a known local node_id so the comparison is deterministic
        syncer._node_id = "local_node_aaa"

        # Remote entry with higher revision → should win regardless
        remote = [
            {
                "key": "task",
                "value": "complete",
                "set_by": "remote_agent",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 5,
                "node_id": "remote_node_zzz",
            }
        ]
        result = syncer.merge_remote("default", remote)
        assert result["applied"] == 1
        row = _read_row(db, "default", "task")
        assert row["value"] == "complete"
        assert row["revision"] == 5

    def test_bc_local_wins_over_empty_origin_remote(self, tmp_path):
        """Higher local revision wins over a remote entry, even when BC rows are involved."""
        db = _make_mesh_db(tmp_path, "bc_local_wins.db")
        broker = _FakeBroker(db)

        from superlocalmemory.mesh.state_sync import StateSyncer

        syncer = StateSyncer(broker)
        _write_row(db, "default", "cfg", "local_cfg", "a", revision=10, origin_node="local_n")

        remote = [
            {
                "key": "cfg",
                "value": "remote_cfg",
                "set_by": "b",
                "updated_at": datetime.now(timezone.utc).isoformat(),
                "revision": 2,
                "node_id": "remote_n",
            }
        ]
        result = syncer.merge_remote("default", remote)
        assert result["applied"] == 0
        row = _read_row(db, "default", "cfg")
        assert row["value"] == "local_cfg"


class TestRouteDelta:
    """Smoke-test the /mesh/state/delta FastAPI route via TestClient."""

    def test_route_returns_entries_and_node_id(self, tmp_path):
        """GET /mesh/state/delta returns {entries, node_id} shaped response."""
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from superlocalmemory.server.routes.mesh_state import router

        db = _make_mesh_db(tmp_path, "route_test.db")
        broker = _FakeBroker(db)

        app = FastAPI()
        app.state.mesh_broker = broker
        app.include_router(router)

        _write_row(db, "default", "route_key", "route_val", "a", revision=1, origin_node="n1")

        with (
            patch(
                "superlocalmemory.server.routes.mesh_state._get_broker",
                return_value=broker,
            ),
            patch(
                "superlocalmemory.server.routes.mesh_state._active_profile",
                return_value="default",
            ),
        ):
            client = TestClient(app)
            resp = client.get("/mesh/state/delta")

        assert resp.status_code == 200
        data = resp.json()
        assert "entries" in data
        assert "node_id" in data
        assert isinstance(data["entries"], list)
        assert len(data["entries"]) == 1
        assert data["entries"][0]["key"] == "route_key"
        assert data["entries"][0]["revision"] == 1

    def test_route_since_param_filters_results(self, tmp_path):
        """GET /mesh/state/delta?since=1 excludes rev=1 rows."""
        from unittest.mock import patch

        from fastapi import FastAPI
        from fastapi.testclient import TestClient

        from superlocalmemory.server.routes.mesh_state import router

        db = _make_mesh_db(tmp_path, "route_since.db")
        broker = _FakeBroker(db)

        app = FastAPI()
        app.include_router(router)

        _write_row(db, "default", "k_old", "v_old", "a", revision=1, origin_node="n1")
        _write_row(db, "default", "k_new", "v_new", "a", revision=2, origin_node="n1")

        with (
            patch(
                "superlocalmemory.server.routes.mesh_state._get_broker",
                return_value=broker,
            ),
            patch(
                "superlocalmemory.server.routes.mesh_state._active_profile",
                return_value="default",
            ),
        ):
            client = TestClient(app)
            resp = client.get("/mesh/state/delta?since=1")

        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entries"]) == 1
        assert data["entries"][0]["key"] == "k_new"
