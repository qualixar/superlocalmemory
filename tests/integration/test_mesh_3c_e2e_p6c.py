# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
#
# SLM V4 sub-wave 3c — TWO-NODE (loopback) END-TO-END convergence.
# Delivery-lead integration test: exercises the REAL StateSyncer.merge_remote,
# LockCoordinator.resolve, local_delta/local_lock_delta, broker.set_state,
# broker.lock_action, and broker.validate_lock_fence across TWO independent
# mesh DBs (two "nodes") — no central server. Proves:
#   * deterministic LWW state convergence,
#   * fencing-token distributed-lock resolution to a single effective holder,
#   * the fence rejects the yielded node's stale token (single-writer safety),
#   * idempotent / stable under repeated sync rounds.
#
# xdist-UNSAFE (SQLite files) — run SERIALLY.

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.mesh.broker import MeshBroker
from superlocalmemory.mesh.lock_protocol import LockCoordinator
from superlocalmemory.mesh.node_identity import get_node_id
from superlocalmemory.mesh.state_sync import StateSyncer

_PROFILE = "default"


def _make_node(tmp_path: Path, name: str) -> MeshBroker:
    """Build an independent mesh node (fresh DB + broker) — a separate machine."""
    db_path = tmp_path / f"{name}.db"
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
    # MeshBroker.__init__ applies apply_security_schema → adds origin_node etc.
    return MeshBroker(str(db_path))


def _pull_round(src: MeshBroker, dst: MeshBroker) -> None:
    """dst pulls src's state + lock deltas and converges (one direction).

    This is exactly what RemoteSyncClient._sync_protocol_from_remote does over
    HTTP, minus the transport — the convergence math is identical.
    """
    StateSyncer(dst).merge_remote(_PROFILE, StateSyncer(src).local_delta(_PROFILE))
    LockCoordinator(dst).resolve(
        _PROFILE, LockCoordinator(src).local_lock_delta(_PROFILE)
    )


def _converge(a: MeshBroker, b: MeshBroker) -> None:
    """One full bidirectional sync round."""
    _pull_round(a, b)
    _pull_round(b, a)


# ---------------------------------------------------------------------------
# STATE — deterministic LWW convergence
# ---------------------------------------------------------------------------


def test_state_converges_on_concurrent_writes(tmp_path):
    """Two nodes write the same key concurrently → both converge on ONE value."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    a.set_state("leader", "from_a", "agentA", _PROFILE)
    b.set_state("leader", "from_b", "agentB", _PROFILE)

    _converge(a, b)

    va = a.get_state_key("leader", _PROFILE)
    vb = b.get_state_key("leader", _PROFILE)
    assert va is not None and vb is not None
    assert va["value"] == vb["value"], "state did not converge"
    assert va["revision"] == vb["revision"]

    # Deterministic winner: higher node_id wins the equal-revision tie.
    nid_a = get_node_id(a._db_path)
    nid_b = get_node_id(b._db_path)
    expected = "from_a" if nid_a > nid_b else "from_b"
    assert va["value"] == expected


def test_state_convergence_is_idempotent(tmp_path):
    """Re-running sync after convergence changes nothing (stable)."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    a.set_state("k", "va", "agentA", _PROFILE)
    b.set_state("k", "vb", "agentB", _PROFILE)
    _converge(a, b)
    settled = a.get_state_key("k", _PROFILE)["value"]
    _converge(a, b)
    _converge(a, b)
    assert a.get_state_key("k", _PROFILE)["value"] == settled
    assert b.get_state_key("k", _PROFILE)["value"] == settled


def test_state_local_write_after_merge_still_converges(tmp_path):
    """Audit P0-1 regression: a LOCAL write after a remote merge must reset
    provenance to local (origin_node='') so the two nodes do not deadlock at
    the same (revision, node_id) with different values."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    a.set_state("k", "a1", "agentA", _PROFILE)
    b.set_state("k", "b1", "agentB", _PROFILE)
    _converge(a, b)  # one value wins on both

    # Now BOTH nodes write locally again (revision bumps to 2 on each).
    a.set_state("k", "a2", "agentA", _PROFILE)
    b.set_state("k", "b2", "agentB", _PROFILE)
    _converge(a, b)

    va = a.get_state_key("k", _PROFILE)
    vb = b.get_state_key("k", _PROFILE)
    assert va["value"] == vb["value"], "sticky-origin divergence (P0-1) regressed"
    assert va["revision"] == vb["revision"]


# ---------------------------------------------------------------------------
# LOCK — fencing-token resolution to a single holder + fence safety
# ---------------------------------------------------------------------------


def _holds(broker: MeshBroker, path: str) -> bool:
    return bool(broker.lock_action(path, "probe", "query", _PROFILE).get("locked"))


def test_lock_resolves_to_single_holder_and_fence_rejects_loser(tmp_path):
    """Both nodes acquire the same path; after sync exactly ONE holds it, and
    the loser's stale token is rejected by its own fence."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    tok_a = a.lock_action("f.py", "agentA", "acquire", _PROFILE)["fencing_token"]
    tok_b = b.lock_action("f.py", "agentB", "acquire", _PROFILE)["fencing_token"]

    _converge(a, b)

    a_holds = _holds(a, "f.py")
    b_holds = _holds(b, "f.py")
    assert a_holds != b_holds, "exactly one node must hold the lock after sync"

    nid_a = get_node_id(a._db_path)
    nid_b = get_node_id(b._db_path)
    # equal tokens (each node's first acquire) → higher node_id wins
    winner_is_a = (tok_a, nid_a) > (tok_b, nid_b)
    assert a_holds is winner_is_a

    # Fence: the LOSER's old token is rejected on the loser's node (row gone).
    loser, loser_tok = (b, tok_b) if winner_is_a else (a, tok_a)
    fence = loser.validate_lock_fence("f.py", loser_tok, _PROFILE)
    assert fence["ok"] is False, "yielded node's stale token must be rejected by the fence"


def test_lock_higher_token_wins_regardless_of_node(tmp_path):
    """A node that reacquires (higher token) wins even if its node_id is lower."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    b.lock_action("f.py", "agentB", "acquire", _PROFILE)          # token 1
    a.lock_action("f.py", "agentA", "acquire", _PROFILE)          # token 1
    a.lock_action("f.py", "agentA", "release", _PROFILE)
    tok_a2 = a.lock_action("f.py", "agentA", "acquire", _PROFILE)["fencing_token"]  # token 2
    assert tok_a2 == 2

    _converge(a, b)
    assert _holds(a, "f.py") is True, "higher token must win"
    assert _holds(b, "f.py") is False, "lower-token node must yield"


def test_lock_convergence_idempotent(tmp_path):
    """Repeated sync after lock resolution is stable (no oscillation)."""
    a = _make_node(tmp_path, "a")
    b = _make_node(tmp_path, "b")
    a.lock_action("f.py", "agentA", "acquire", _PROFILE)
    b.lock_action("f.py", "agentB", "acquire", _PROFILE)
    _converge(a, b)
    a1, b1 = _holds(a, "f.py"), _holds(b, "f.py")
    _converge(a, b)
    _converge(a, b)
    assert (_holds(a, "f.py"), _holds(b, "f.py")) == (a1, b1)


# ---------------------------------------------------------------------------
# Audit-fix regressions (delivery-lead)
# ---------------------------------------------------------------------------


def test_lock_expired_local_yields_to_live_remote(tmp_path):
    """Audit P1 (Opus lock): an EXPIRED local lock — even with a HIGHER token —
    must yield to a LIVE remote claim with a lower token, and the stale row is
    removed so its high token can't pass the fence."""
    from datetime import datetime, timedelta, timezone

    a = _make_node(tmp_path, "a")
    past = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    # Insert an EXPIRED local lock with a high token directly.
    conn = sqlite3.connect(a._db_path)
    conn.execute(
        "INSERT INTO mesh_locks (profile_id, file_path, locked_by, locked_at,"
        " expires_at, fencing_token) VALUES (?,?,?,?,?,?)",
        (_PROFILE, "f.py", "ghost", past, past, 100),
    )
    conn.commit()
    conn.close()

    # A LIVE remote claim with a LOWER token.
    future = (datetime.now(timezone.utc) + timedelta(hours=1)).isoformat()
    remote = [{
        "file_path": "f.py", "locked_by": "agentR", "locked_at": future,
        "expires_at": future, "fencing_token": 5, "node_id": "remote-node",
    }]
    result = LockCoordinator(a).resolve(_PROFILE, remote)
    assert "f.py" in result["yielded"], "expired local (token 100) must yield to live remote (token 5)"
    assert _holds(a, "f.py") is False
    assert a.validate_lock_fence("f.py", 100, _PROFILE)["ok"] is False


def test_state_merge_ignores_non_dict_entries(tmp_path):
    """Audit P0/P2: non-dict remote entries must be skipped, never crash merge."""
    a = _make_node(tmp_path, "a")
    valid = {
        "key": "k", "value": "v", "set_by": "peer", "updated_at": "t",
        "revision": 3, "node_id": "remote-node",
    }
    summary = StateSyncer(a).merge_remote(_PROFILE, [None, "not-a-dict", 42, valid])
    assert summary["applied"] == 1
    assert a.get_state_key("k", _PROFILE)["value"] == "v"


def test_lock_resolve_ignores_non_dict_entries(tmp_path):
    """Audit P2 (Grok lock): non-dict remote lock entries must not crash resolve."""
    a = _make_node(tmp_path, "a")
    a.lock_action("f.py", "agentA", "acquire", _PROFILE)
    result = LockCoordinator(a).resolve(_PROFILE, [None, "x", {"no_file_path": 1}])
    assert result == {"yielded": [], "kept": 0}
    assert _holds(a, "f.py") is True  # untouched
