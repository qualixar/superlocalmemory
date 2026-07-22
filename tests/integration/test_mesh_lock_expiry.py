# M-02 (3.7.9): mesh file locks must auto-expire so a crashed session cannot
# deadlock a path forever. (Deliberately NOT marked slow — it's a fast unit-level
# check that should run in the default suite.)
import sqlite3

import pytest

from superlocalmemory.mesh.broker import MeshBroker, _NEVER_EXPIRES


@pytest.fixture
def broker(tmp_path):
    db_path = tmp_path / "mesh_lock.db"
    conn = sqlite3.connect(str(db_path))
    from superlocalmemory.storage.schema_v343 import (
        _MESH_DDL, _MESH_V346_ALTERS, _MESH_V346_DDL,
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
    return MeshBroker(str(db_path)), str(db_path)


def test_acquire_sets_real_expiry(broker):
    b, _ = broker
    r = b.lock_action("f.py", "peerA", "acquire")
    assert r.get("ok") and r["action"] == "acquired"
    assert r.get("expires_at") and r["expires_at"] != _NEVER_EXPIRES


def test_live_lock_blocks_other_peer(broker):
    b, _ = broker
    b.lock_action("f.py", "peerA", "acquire")
    r = b.lock_action("f.py", "peerB", "acquire")
    assert r.get("locked") is True and r.get("by") == "peerA"


def test_expired_lock_is_free(broker):
    b, db = broker
    b.lock_action("f.py", "peerA", "acquire")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE mesh_locks SET expires_at=? WHERE file_path=?",
                 ("2000-01-01T00:00:00+00:00", "f.py"))
    conn.commit(); conn.close()
    r = b.lock_action("f.py", "peerB", "acquire")
    assert r.get("ok") and r["action"] == "acquired"


def test_legacy_never_expires_lock_does_not_deadlock(broker):
    b, db = broker
    b.lock_action("g.py", "peerA", "acquire")
    conn = sqlite3.connect(db)
    conn.execute("UPDATE mesh_locks SET expires_at=? WHERE file_path=?",
                 (_NEVER_EXPIRES, "g.py"))
    conn.commit(); conn.close()
    r = b.lock_action("g.py", "peerB", "acquire")
    assert r.get("ok") and r["action"] == "acquired"
