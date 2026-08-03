# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# Integration tests for SLM V4 sub-wave 3c-2: leaderless cross-node lock
# coordination (LockCoordinator — fencing-token total-order resolution).
#
# xdist-UNSAFE: share a SQLite file per test via tmp_path; run SERIALLY.
# Run with:
#   pytest tests/integration/test_mesh_lock_protocol_p6c.py -p no:cacheprovider -x
#
# Coverage targets (≥12 cases):
#   1.  remote-higher-token wins → local row YIELDED (deleted)
#   2.  local-higher-token wins → local row KEPT
#   3.  token tie, remote node_id > local → remote wins → YIELDED
#   4.  token tie, local node_id > remote → local wins → KEPT
#   5.  yield deletes ONLY the contested file_path; other locks untouched
#   6.  after yield, validate_lock_fence rejects the yielded token
#   7.  resolve is idempotent (twice = same DB state, no extra effect)
#   8.  resolve no-ops on empty remote_locks
#   9.  resolve no-ops on expired remote claims
#   10. local_lock_delta excludes expired locks
#   11. local_lock_delta excludes _NEVER_EXPIRES sentinel locks
#   12. local_lock_delta includes live locks with correct node_id
#   13. resolve no-ops when no local row exists for the remote's file_path
#   14. fencing_token string vs int — "10" vs 9: must be int comparison

from __future__ import annotations

import sqlite3
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Generator
from types import SimpleNamespace

import pytest

from superlocalmemory.mesh.broker import MeshBroker, _NEVER_EXPIRES
from superlocalmemory.mesh.lock_protocol import LockCoordinator
from superlocalmemory.mesh.node_identity import get_node_id


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_HOUR = 3600  # seconds
_PROFILE = "default"


def _future_iso(seconds: int = _HOUR) -> str:
    """Return an ISO timestamp `seconds` from now (UTC)."""
    return (datetime.now(timezone.utc) + timedelta(seconds=seconds)).isoformat()


def _past_iso(seconds: int = _HOUR) -> str:
    """Return an ISO timestamp `seconds` in the past (UTC)."""
    return (datetime.now(timezone.utc) - timedelta(seconds=seconds)).isoformat()


def _make_lock_db(tmp_path: Path) -> tuple[MeshBroker, str]:
    """Create a minimal mesh DB and return (MeshBroker, db_path_str).

    Pattern mirrors test_mesh_lock_expiry.py: apply full DDL first so the
    MeshBroker constructor can apply incremental security schema without
    hitting a missing-table error.
    """
    db_path = tmp_path / "mesh_lock_p6c.db"
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
    conn.commit()
    conn.close()
    broker = MeshBroker(str(db_path))
    return broker, str(db_path)


def _insert_lock_raw(
    db_path: str,
    file_path: str,
    locked_by: str,
    expires_at: str,
    fencing_token: int,
    profile_id: str = _PROFILE,
) -> None:
    """Directly insert a mesh_locks row, bypassing broker (for setup only)."""
    locked_at = datetime.now(timezone.utc).isoformat()
    conn = sqlite3.connect(db_path)
    conn.execute(
        "INSERT INTO mesh_locks"
        " (profile_id, file_path, locked_by, locked_at, expires_at, fencing_token)"
        " VALUES (?, ?, ?, ?, ?, ?)"
        " ON CONFLICT(profile_id, file_path) DO UPDATE SET"
        "  locked_by=excluded.locked_by, locked_at=excluded.locked_at,"
        "  expires_at=excluded.expires_at, fencing_token=excluded.fencing_token",
        (profile_id, file_path, locked_by, locked_at, expires_at, fencing_token),
    )
    conn.commit()
    conn.close()


def _row_exists(db_path: str, file_path: str, profile_id: str = _PROFILE) -> bool:
    """Return True if a mesh_locks row exists for the given file_path."""
    conn = sqlite3.connect(db_path)
    try:
        row = conn.execute(
            "SELECT 1 FROM mesh_locks WHERE profile_id=? AND file_path=?",
            (profile_id, file_path),
        ).fetchone()
        return row is not None
    finally:
        conn.close()


def _make_remote_lock(
    file_path: str,
    fencing_token: int,
    node_id: str,
    expires_at: str | None = None,
) -> dict:
    """Build a remote lock dict as returned by local_lock_delta()."""
    return {
        "file_path": file_path,
        "locked_by": f"remote-peer-{node_id[:8]}",
        "locked_at": datetime.now(timezone.utc).isoformat(),
        "expires_at": expires_at or _future_iso(),
        "fencing_token": fencing_token,
        "node_id": node_id,
    }


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> Generator[tuple[MeshBroker, str, str], None, None]:
    """Yield (broker, db_path, local_node_id)."""
    broker, db_path = _make_lock_db(tmp_path)
    local_node_id = get_node_id(db_path)
    yield broker, db_path, local_node_id
    broker.stop()


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestLocalLockDelta:
    """local_lock_delta: returns only live locks with node_id attached."""

    def test_includes_live_locks(self, db: tuple) -> None:
        """Live lock appears in delta with correct structure and node_id."""
        broker, db_path, node_id = db
        _insert_lock_raw(db_path, "/a/b.py", "peerA", _future_iso(), 42)
        coord = LockCoordinator(broker)

        result = coord.local_lock_delta()

        assert len(result) == 1
        lock = result[0]
        assert lock["file_path"] == "/a/b.py"
        assert lock["fencing_token"] == 42
        assert lock["node_id"] == node_id

    def test_excludes_expired_locks(self, db: tuple) -> None:
        """Expired lock must NOT appear in delta."""
        broker, db_path, _ = db
        _insert_lock_raw(db_path, "/a/b.py", "peerA", _past_iso(), 10)
        coord = LockCoordinator(broker)

        result = coord.local_lock_delta()

        assert result == []

    def test_excludes_never_expires_sentinel(self, db: tuple) -> None:
        """_NEVER_EXPIRES sentinel lock is treated as stale, excluded."""
        broker, db_path, _ = db
        _insert_lock_raw(db_path, "/a/b.py", "peerA", _NEVER_EXPIRES, 5)
        coord = LockCoordinator(broker)

        result = coord.local_lock_delta()

        assert result == []

    def test_mixed_live_and_expired(self, db: tuple) -> None:
        """Only the live lock among several is returned."""
        broker, db_path, node_id = db
        _insert_lock_raw(db_path, "/live.py", "peerA", _future_iso(), 100)
        _insert_lock_raw(db_path, "/expired.py", "peerB", _past_iso(), 99)
        _insert_lock_raw(db_path, "/sentinel.py", "peerC", _NEVER_EXPIRES, 98)
        coord = LockCoordinator(broker)

        result = coord.local_lock_delta()

        assert len(result) == 1
        assert result[0]["file_path"] == "/live.py"
        assert result[0]["node_id"] == node_id


class TestResolveRemoteHigher:
    """resolve(): remote-higher-token cases."""

    def test_remote_higher_token_yields_local(self, db: tuple) -> None:
        """Remote has higher fencing_token → local row deleted."""
        broker, db_path, local_node_id = db
        _insert_lock_raw(db_path, "/f.py", "peerLocal", _future_iso(), 5)
        coord = LockCoordinator(broker)
        remote_node_id = "aaaa" + local_node_id[4:]  # different, lower than local
        remote_locks = [_make_remote_lock("/f.py", fencing_token=10, node_id=remote_node_id)]

        result = coord.resolve(_PROFILE, remote_locks)

        assert "/f.py" in result["yielded"]
        assert result["kept"] == 0
        assert not _row_exists(db_path, "/f.py")

    def test_local_higher_token_keeps_local(self, db: tuple) -> None:
        """Local has higher fencing_token → local row untouched."""
        broker, db_path, local_node_id = db
        _insert_lock_raw(db_path, "/f.py", "peerLocal", _future_iso(), 20)
        coord = LockCoordinator(broker)
        remote_locks = [_make_remote_lock("/f.py", fencing_token=5, node_id="remote-node")]

        result = coord.resolve(_PROFILE, remote_locks)

        assert result["yielded"] == []
        assert result["kept"] == 1
        assert _row_exists(db_path, "/f.py")


class TestResolveTokenTie:
    """resolve(): tie-breaking via node_id when tokens are equal."""

    def test_tie_remote_node_id_greater_remote_wins(self, db: tuple) -> None:
        """Equal tokens, remote node_id lexicographically greater → remote wins."""
        broker, db_path, _ = db
        local_node_id = get_node_id(db_path)
        # Choose a remote node_id that is guaranteed lexicographically larger.
        remote_node_id = "z" + local_node_id[1:]
        assert remote_node_id > local_node_id, "test setup: remote must be > local"

        _insert_lock_raw(db_path, "/tie.py", "peerLocal", _future_iso(), 7)
        coord = LockCoordinator(broker)
        remote_locks = [_make_remote_lock("/tie.py", fencing_token=7, node_id=remote_node_id)]

        result = coord.resolve(_PROFILE, remote_locks)

        assert "/tie.py" in result["yielded"]
        assert not _row_exists(db_path, "/tie.py")

    def test_tie_local_node_id_greater_local_wins(self, db: tuple) -> None:
        """Equal tokens, local node_id lexicographically greater → local wins."""
        broker, db_path, _ = db
        local_node_id = get_node_id(db_path)
        # Choose a remote node_id that is guaranteed lexicographically smaller.
        remote_node_id = "0" + local_node_id[1:]
        # Ensure local_node_id does not also start with '0'; if it does, use 'a' vs 'Z'
        if local_node_id[0] == "0":
            remote_node_id = "0" + "a" * (len(local_node_id) - 1)
            # patch the local node so it's clearly higher
            coord = LockCoordinator(broker)
            coord._node_id = "z" * len(local_node_id)
        else:
            coord = LockCoordinator(broker)
        # Confirm ordering assumption holds
        assert coord._node_id > remote_node_id, "test setup: local must be > remote"

        _insert_lock_raw(db_path, "/tie2.py", "peerLocal", _future_iso(), 7)
        remote_locks = [_make_remote_lock("/tie2.py", fencing_token=7, node_id=remote_node_id)]

        result = coord.resolve(_PROFILE, remote_locks)

        assert result["yielded"] == []
        assert result["kept"] == 1
        assert _row_exists(db_path, "/tie2.py")


class TestResolveScope:
    """resolve(): only contested paths are touched."""

    def test_yield_only_contested_path(self, db: tuple) -> None:
        """Yielding one path leaves other local locks untouched."""
        broker, db_path, local_node_id = db
        remote_node = "aaaa" + local_node_id[4:]
        _insert_lock_raw(db_path, "/contested.py", "peerLocal", _future_iso(), 5)
        _insert_lock_raw(db_path, "/uncontested.py", "peerLocal", _future_iso(), 99)
        coord = LockCoordinator(broker)
        remote_locks = [
            _make_remote_lock("/contested.py", fencing_token=10, node_id=remote_node)
            # /uncontested.py is NOT in remote_locks
        ]

        result = coord.resolve(_PROFILE, remote_locks)

        assert "/contested.py" in result["yielded"]
        assert _row_exists(db_path, "/uncontested.py"), "uncontested lock must survive"
        assert not _row_exists(db_path, "/contested.py")

    def test_no_local_row_noop(self, db: tuple) -> None:
        """Remote claims a file_path with no local row → no row to yield, no error."""
        broker, db_path, _ = db
        coord = LockCoordinator(broker)
        remote_locks = [_make_remote_lock("/ghost.py", fencing_token=100, node_id="remote")]

        result = coord.resolve(_PROFILE, remote_locks)

        assert result["yielded"] == []
        assert result["kept"] == 0


class TestFenceSafety:
    """Prove the fence rejects the yielded node's stale token."""

    def test_fence_rejects_stale_token_after_yield(self, db: tuple) -> None:
        """After resolve() yields our lock, validate_lock_fence rejects old token."""
        broker, db_path, local_node_id = db
        # Local acquires a lock and remembers its token.
        acquire_result = broker.lock_action("/guarded.py", "peerLocal", "acquire")
        assert acquire_result.get("ok"), f"acquire failed: {acquire_result}"
        old_token: int = acquire_result["fencing_token"]

        # Remote peer has a higher token — wins the resolve.
        remote_node = "z" * len(local_node_id)  # guaranteed > any uuid4 hex
        remote_locks = [
            _make_remote_lock("/guarded.py", fencing_token=old_token + 1000, node_id=remote_node)
        ]
        coord = LockCoordinator(broker)
        resolve_result = coord.resolve(_PROFILE, remote_locks)
        assert "/guarded.py" in resolve_result["yielded"], "lock must have been yielded"

        # The fence must now reject the old token.
        fence_result = broker.validate_lock_fence("/guarded.py", old_token)
        assert fence_result["ok"] is False, (
            "validate_lock_fence must reject the stale token of the yielded node; "
            f"got: {fence_result}"
        )

    def test_fence_rejects_stale_token_after_reacquire(self, db: tuple) -> None:
        """After yield + remote reacquire, fence rejects the original token."""
        broker, db_path, local_node_id = db
        acquire = broker.lock_action("/h.py", "peerLocal", "acquire")
        old_token = acquire["fencing_token"]

        # Simulate yield: remove local row (as resolve() would).
        conn = sqlite3.connect(db_path)
        conn.execute(
            "DELETE FROM mesh_locks WHERE profile_id=? AND file_path=?",
            (_PROFILE, "/h.py"),
        )
        conn.commit()
        conn.close()

        # Remote reacquires with a new higher token.
        new_acquire = broker.lock_action("/h.py", "peerRemote", "acquire")
        assert new_acquire.get("ok")
        new_token = new_acquire["fencing_token"]
        assert new_token > old_token

        # Old token must be rejected.
        fence = broker.validate_lock_fence("/h.py", old_token)
        assert fence["ok"] is False


class TestResolveIdempotency:
    """resolve() called twice produces the same DB state."""

    def test_resolve_idempotent(self, db: tuple) -> None:
        """Second resolve call has no further effect."""
        broker, db_path, local_node_id = db
        remote_node = "z" * len(local_node_id)
        _insert_lock_raw(db_path, "/idem.py", "peerLocal", _future_iso(), 3)
        coord = LockCoordinator(broker)
        remote_locks = [_make_remote_lock("/idem.py", fencing_token=99, node_id=remote_node)]

        r1 = coord.resolve(_PROFILE, remote_locks)
        assert "/idem.py" in r1["yielded"]

        # Second call — row is already gone.
        r2 = coord.resolve(_PROFILE, remote_locks)
        assert r2["yielded"] == []
        assert r2["kept"] == 0
        # DB must still have no row for /idem.py.
        assert not _row_exists(db_path, "/idem.py")


class TestResolveNoOp:
    """resolve() no-ops on empty / non-live remote claims."""

    def test_empty_remote_locks_noop(self, db: tuple) -> None:
        """Empty remote_locks → no local rows touched."""
        broker, db_path, _ = db
        _insert_lock_raw(db_path, "/safe.py", "peerLocal", _future_iso(), 5)
        coord = LockCoordinator(broker)

        result = coord.resolve(_PROFILE, [])

        assert result == {"yielded": [], "kept": 0}
        assert _row_exists(db_path, "/safe.py"), "local lock must be untouched"

    def test_expired_remote_claim_noop(self, db: tuple) -> None:
        """Expired remote lock is not live → local row untouched."""
        broker, db_path, _ = db
        _insert_lock_raw(db_path, "/safe2.py", "peerLocal", _future_iso(), 5)
        coord = LockCoordinator(broker)
        remote_locks = [
            _make_remote_lock("/safe2.py", fencing_token=999, node_id="remote",
                              expires_at=_past_iso())
        ]

        result = coord.resolve(_PROFILE, remote_locks)

        assert result["yielded"] == []
        assert _row_exists(db_path, "/safe2.py"), "local lock must survive expired remote claim"

    def test_sentinel_remote_claim_noop(self, db: tuple) -> None:
        """Remote lock with _NEVER_EXPIRES is treated as not-live → local kept."""
        broker, db_path, _ = db
        _insert_lock_raw(db_path, "/safe3.py", "peerLocal", _future_iso(), 5)
        coord = LockCoordinator(broker)
        remote_locks = [
            _make_remote_lock("/safe3.py", fencing_token=999, node_id="remote",
                              expires_at=_NEVER_EXPIRES)
        ]

        result = coord.resolve(_PROFILE, remote_locks)

        assert result["yielded"] == []
        assert _row_exists(db_path, "/safe3.py")


class TestTokenAsInt:
    """fencing_token comparison must be numeric, not lexicographic."""

    def test_string_token_10_beats_int_token_9(self, db: tuple) -> None:
        """Remote token '10' (string from JSON) must beat local token 9 (int).

        String comparison: '10' < '9' (WRONG, would let local keep the lock).
        Int comparison:     10  >  9  (CORRECT, remote wins → local yielded).
        """
        broker, db_path, local_node_id = db
        _insert_lock_raw(db_path, "/num.py", "peerLocal", _future_iso(), 9)
        coord = LockCoordinator(broker)
        remote_node = "aaaa" + local_node_id[4:]  # lower node_id; token must decide
        remote_lock = _make_remote_lock("/num.py", fencing_token=10, node_id=remote_node)
        # Simulate JSON string serialization of the token.
        remote_lock["fencing_token"] = "10"

        result = coord.resolve(_PROFILE, [remote_lock])

        # Int comparison: 10 > 9 → remote wins → yielded.
        assert "/num.py" in result["yielded"], (
            "fencing_token '10' (string) must be coerced to int before comparison; "
            "string '10' < '9' but int 10 > 9"
        )
        assert not _row_exists(db_path, "/num.py")


# ---------------------------------------------------------------------------
# Route smoke test (TestClient, no network)
# ---------------------------------------------------------------------------

class TestLockDeltaRoute:
    """GET /mesh/lock/delta returns correct structure."""

    def test_route_returns_locks_and_node_id(self, db: tuple, tmp_path: Path) -> None:
        """GET /mesh/lock/delta returns live locks + node_id for this node."""
        fastapi_mod = pytest.importorskip("fastapi", reason="fastapi not installed")
        from fastapi import FastAPI
        from fastapi.testclient import TestClient
        from superlocalmemory.server.routes import mesh_lock as mesh_lock_routes

        broker, db_path, node_id = db
        _insert_lock_raw(db_path, "/route-test.py", "peerA", _future_iso(), 77)

        app = FastAPI()
        # Mirror test_mesh_http.py: wire broker and disable shared-secret auth.
        app.state.mesh_broker = broker
        app.state.config = None
        app.state.daemon_descriptor = SimpleNamespace(
            capability="mesh-capability",
            instance_id="mesh-instance",
            capability_fingerprint="mesh-fp",
        )
        broker._shared_secret = None  # loopback auth path (no secret)
        app.include_router(mesh_lock_routes.router)

        client = TestClient(app)
        resp = client.get(
            "/mesh/lock/delta",
            headers={
                "X-SLM-Daemon-Capability": "mesh-capability",
                "X-SLM-Target-Instance": "mesh-instance",
            },
        )
        assert resp.status_code == 200
        body = resp.json()
        assert body["node_id"] == node_id
        locks = body["locks"]
        assert any(lk["file_path"] == "/route-test.py" for lk in locks)
