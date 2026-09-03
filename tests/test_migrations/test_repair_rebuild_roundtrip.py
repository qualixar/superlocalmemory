# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Crash-resume round trips for table-rebuild migrations (4.1.14 #133).

Rebuilds (RENAME → CREATE → copy → DROP) are the one repair class where
a naive re-run destroys data. Each test drives the full cycle the
reporter lived: old shape → apply → sabotage → repair → verify, with
row survival asserted — plus the interrupted-rebuild resume where the
only copy of the data sits in the ``_old`` table.
"""
from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import (
    M021_ingestion_log_profile,
    M023_mesh_profile_isolation,
    M026_rbac_memberships_fk,
    M027_transferable_patterns_profile,
)


def _conn() -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:")
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


# --- M021: legacy rebuild + interrupted resume + leftover cleanup --------

_LEGACY_LOG = """
CREATE TABLE ingestion_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_type TEXT NOT NULL,
    dedup_key TEXT NOT NULL,
    fact_ids TEXT DEFAULT '[]',
    metadata TEXT DEFAULT '{}',
    status TEXT DEFAULT 'ingested',
    ingested_at TEXT NOT NULL
);
INSERT INTO ingestion_log
    (source_type, dedup_key, fact_ids, metadata, status, ingested_at)
VALUES
    ('cli', 'k1', '["f1"]', '{}', 'ingested', '2026-01-01'),
    ('cli', 'k2', '["f2"]', '{}', 'ingested', '2026-01-02');
"""


def test_m021_legacy_rebuild_backfills_default() -> None:
    conn = _conn()
    conn.executescript(_LEGACY_LOG)

    assert not M021_ingestion_log_profile.verify(conn)
    M021_ingestion_log_profile.apply(conn)
    assert M021_ingestion_log_profile.verify(conn)

    rows = conn.execute(
        "SELECT profile_id, COUNT(*) FROM ingestion_log GROUP BY profile_id"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("default", 2)]
    M021_ingestion_log_profile.repair(conn)  # idempotent re-run
    assert M021_ingestion_log_profile.verify(conn)
    conn.close()


def test_m021_interrupted_rebuild_resumes_from_old() -> None:
    conn = _conn()
    conn.executescript(_LEGACY_LOG)
    # Crash between RENAME and DROP: canonical table gone, only copy in _old.
    conn.execute("ALTER TABLE ingestion_log RENAME TO _ingestion_log_old")

    assert not M021_ingestion_log_profile.verify(conn)
    M021_ingestion_log_profile.repair(conn)
    assert M021_ingestion_log_profile.verify(conn)

    count = conn.execute("SELECT COUNT(*) FROM ingestion_log").fetchone()[0]
    assert int(count) == 2
    leftovers = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='_ingestion_log_old'"
    ).fetchall()
    assert leftovers == []
    conn.close()


def test_m021_duplicate_old_table_cleaned_when_safe() -> None:
    conn = _conn()
    conn.executescript(_LEGACY_LOG)
    M021_ingestion_log_profile.apply(conn)
    conn.execute(
        "CREATE TABLE _ingestion_log_old AS SELECT * FROM ingestion_log"
    )

    M021_ingestion_log_profile.repair(conn)
    assert M021_ingestion_log_profile.verify(conn)
    leftovers = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='_ingestion_log_old'"
    ).fetchall()
    assert leftovers == []
    conn.close()


def test_m021_mid_copy_crash_resumes_missing_rows() -> None:
    """4.1.14 audit (blocker): RENAME+CREATE done, copy interrupted — the
    canonical table exists WITH profile_id but is empty while _old holds
    the only copy. verify must be False and repair must restore the rows,
    never record success over the stranded ledger."""
    conn = _conn()
    conn.executescript(_LEGACY_LOG)
    M021_ingestion_log_profile.apply(conn)
    # Simulate the crash: empty new-shape table, full _old table.
    conn.execute("DELETE FROM ingestion_log")
    conn.execute(
        "CREATE TABLE _ingestion_log_old ("
        "id INTEGER PRIMARY KEY AUTOINCREMENT, source_type TEXT NOT NULL, "
        "dedup_key TEXT NOT NULL, fact_ids TEXT DEFAULT '[]', "
        "metadata TEXT DEFAULT '{}', status TEXT DEFAULT 'ingested', "
        "ingested_at TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO _ingestion_log_old "
        "(source_type, dedup_key, fact_ids, metadata, status, ingested_at) "
        "VALUES ('cli', 'k1', '[\"f1\"]', '{}', 'ingested', '2026-01-01'), "
        "('cli', 'k2', '[\"f2\"]', '{}', 'ingested', '2026-01-02')"
    )
    conn.commit()

    assert not M021_ingestion_log_profile.verify(conn)
    M021_ingestion_log_profile.repair(conn)
    assert M021_ingestion_log_profile.verify(conn)

    rows = conn.execute(
        "SELECT profile_id, COUNT(*) FROM ingestion_log GROUP BY profile_id"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("default", 2)]
    leftovers = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' "
        "AND name='_ingestion_log_old'"
    ).fetchall()
    assert leftovers == []
    conn.close()


# --- M026: old shape without FK ------------------------------------------

_OLD_MEMBERSHIPS = """
CREATE TABLE rbac_memberships (
    profile_id TEXT NOT NULL,
    user_id TEXT NOT NULL,
    role TEXT NOT NULL,
    added_at TEXT NOT NULL,
    PRIMARY KEY (profile_id, user_id)
);
INSERT INTO rbac_memberships (profile_id, user_id, role, added_at)
VALUES ('default', 'u1', 'owner', '2026-01-01');
"""


def test_m026_old_shape_gains_fk_preserving_grants() -> None:
    conn = _conn()
    conn.execute("CREATE TABLE profiles (profile_id TEXT PRIMARY KEY)")
    conn.execute("INSERT INTO profiles VALUES ('default')")
    conn.executescript(_OLD_MEMBERSHIPS)

    assert not M026_rbac_memberships_fk.verify(conn)
    M026_rbac_memberships_fk.apply(conn)
    assert M026_rbac_memberships_fk.verify(conn)

    rows = conn.execute(
        "SELECT profile_id, user_id, role FROM rbac_memberships"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [("default", "u1", "owner")]
    M026_rbac_memberships_fk.repair(conn)  # idempotent re-run
    assert M026_rbac_memberships_fk.verify(conn)
    conn.close()


# --- M023: single-table column drop ---------------------------------------

def test_m023_dropped_column_returns_with_backfill() -> None:
    conn = _conn()
    conn.execute(
        "CREATE TABLE mesh_peers (peer_id TEXT PRIMARY KEY, status TEXT)"
    )
    conn.execute("INSERT INTO mesh_peers VALUES ('p1', 'active')")
    M023_mesh_profile_isolation.apply(conn)
    assert M023_mesh_profile_isolation.verify(conn)

    # Partial restore dropped the additive column (the framework's own
    # documented drift case): rebuild the table without profile_id.
    # (SQLite cannot DROP an indexed column, so the restore path — a
    # table-level rebuild — is simulated directly.)
    conn.execute("ALTER TABLE mesh_peers RENAME TO _mesh_peers_drifted")
    conn.execute(
        "CREATE TABLE mesh_peers (peer_id TEXT PRIMARY KEY, status TEXT)"
    )
    conn.execute(
        "INSERT INTO mesh_peers (peer_id, status) "
        "SELECT peer_id, status FROM _mesh_peers_drifted"
    )
    conn.execute("DROP TABLE _mesh_peers_drifted")
    assert not M023_mesh_profile_isolation.verify(conn)
    M023_mesh_profile_isolation.repair(conn)
    assert M023_mesh_profile_isolation.verify(conn)

    rows = conn.execute(
        "SELECT peer_id, profile_id FROM mesh_peers"
    ).fetchall()
    assert [(r[0], r[1]) for r in rows] == [("p1", "default")]
    conn.close()


# --- M027: old shape without profile_id ------------------------------------

_OLD_PATTERNS = """
CREATE TABLE transferable_patterns (
    pattern_type TEXT NOT NULL,
    key TEXT NOT NULL,
    value TEXT NOT NULL,
    confidence REAL NOT NULL DEFAULT 0.5,
    evidence_count INTEGER NOT NULL DEFAULT 1,
    profiles_seen TEXT NOT NULL DEFAULT '[]',
    decay_factor REAL NOT NULL DEFAULT 1.0,
    contradictions INTEGER NOT NULL DEFAULT 0,
    first_seen TEXT NOT NULL,
    last_seen TEXT NOT NULL
);
INSERT INTO transferable_patterns
    (pattern_type, key, value, confidence, evidence_count, profiles_seen,
     decay_factor, contradictions, first_seen, last_seen)
VALUES
    ('prompt', 'k1', 'v1', 0.9, 3, '[]', 1.0, 0, '2026-01-01', '2026-02-01');
"""


def test_m027_old_shape_rebuilds_preserving_rows() -> None:
    conn = _conn()
    conn.executescript(_OLD_PATTERNS)

    assert not M027_transferable_patterns_profile.verify(conn)
    M027_transferable_patterns_profile.apply(conn)
    assert M027_transferable_patterns_profile.verify(conn)

    rows = conn.execute(
        "SELECT profile_id, pattern_type, key FROM transferable_patterns"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [("default", "prompt", "k1")]
    M027_transferable_patterns_profile.repair(conn)  # idempotent re-run
    assert M027_transferable_patterns_profile.verify(conn)
    conn.close()


# --- M046: old CHECK value converts, rows survive -----------------------------

def test_m046_old_value_converts_and_rows_survive() -> None:
    from superlocalmemory.storage.migrations import (
        M046_prospective_memory_has_its_own_name,
    )

    conn = _conn()
    conn.execute(
        "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT, "
        "fact_type TEXT CHECK(fact_type IN ('episodic', 'temporal')))"
    )
    conn.execute(
        "INSERT INTO atomic_facts VALUES ('f1', 'Launch next week', 'temporal')"
    )
    conn.commit()  # close the fixture transaction: modules manage their own

    assert not M046_prospective_memory_has_its_own_name.verify(conn)
    M046_prospective_memory_has_its_own_name.repair(conn)
    assert M046_prospective_memory_has_its_own_name.verify(conn)

    rows = conn.execute(
        "SELECT fact_id, fact_type, content FROM atomic_facts"
    ).fetchall()
    assert [(r[0], r[1], r[2]) for r in rows] == [
        ("f1", "prospective", "Launch next week")
    ]
    M046_prospective_memory_has_its_own_name.repair(conn)
    assert M046_prospective_memory_has_its_own_name.verify(conn)
    conn.close()


# --- M047: text Fisher vectors convert to buffers ----------------------------

def test_m047_text_vectors_convert_to_buffers() -> None:
    from superlocalmemory.storage.migrations import (
        M047_fisher_vectors_are_stored_like_every_other_vector,
    )

    conn = _conn()
    conn.execute(
        "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, "
        "fisher_mean TEXT, fisher_variance TEXT)"
    )
    conn.execute(
        "INSERT INTO atomic_facts VALUES ('f1', '[0.5, 1.5]', '[2.0]')"
    )
    conn.commit()  # close the fixture transaction: modules manage their own

    assert not M047_fisher_vectors_are_stored_like_every_other_vector.verify(conn)
    M047_fisher_vectors_are_stored_like_every_other_vector.repair(conn)
    assert M047_fisher_vectors_are_stored_like_every_other_vector.verify(conn)

    kinds = {
        r[0]
        for r in conn.execute(
            "SELECT typeof(fisher_mean) FROM atomic_facts WHERE fact_id = 'f1'"
        )
    }
    assert kinds == {"blob"}
    M047_fisher_vectors_are_stored_like_every_other_vector.repair(conn)
    assert M047_fisher_vectors_are_stored_like_every_other_vector.verify(conn)
    conn.close()


# --- M028: dropped state table returns ------------------------------------------

def test_m028_dropped_state_table_returns() -> None:
    from superlocalmemory.storage.migrations import M028_fact_entity_associations

    conn = _conn()
    conn.execute(
        "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT)"
    )
    M028_fact_entity_associations.apply(conn)
    assert M028_fact_entity_associations.verify(conn)

    conn.execute("DROP TABLE fact_entity_association_repair_state")
    assert not M028_fact_entity_associations.verify(conn)
    M028_fact_entity_associations.repair(conn)
    assert M028_fact_entity_associations.verify(conn)
    conn.close()


# --- M049: duplicate version rows collapse ---------------------------------------

def test_m049_duplicate_versions_collapse_to_one() -> None:
    from superlocalmemory.storage.migrations import (
        M049_a_schema_version_marker_is_one_row,
    )

    conn = _conn()
    conn.execute(
        "CREATE TABLE schema_version (version TEXT NOT NULL, "
        "description TEXT, applied_at TEXT)"
    )
    conn.execute(
        "INSERT INTO schema_version VALUES "
        "('4.1.0', 'first', '2026-01-01'), "
        "('4.1.0', 'second', '2026-01-02')"
    )
    conn.commit()  # close the fixture transaction: modules manage their own

    assert not M049_a_schema_version_marker_is_one_row.verify(conn)
    M049_a_schema_version_marker_is_one_row.repair(conn)
    assert M049_a_schema_version_marker_is_one_row.verify(conn)

    rows = conn.execute("SELECT version FROM schema_version").fetchall()
    assert [r[0] for r in rows] == ["4.1.0"]
    M049_a_schema_version_marker_is_one_row.repair(conn)
    assert M049_a_schema_version_marker_is_one_row.verify(conn)
    conn.close()


# --- M020: backfill fills only empty hashes ----------------------------------

def test_m020_backfill_fills_only_empty_hashes() -> None:
    from superlocalmemory.storage.migrations import M020_model_state_integrity

    conn = _conn()
    conn.execute(
        "CREATE TABLE learning_model_state ("
        "id INTEGER PRIMARY KEY, state_bytes BLOB, bytes_sha256 TEXT)"
    )
    conn.execute(
        "INSERT INTO learning_model_state (id, state_bytes, bytes_sha256)"
        " VALUES (1, X'0102', ''), (2, X'0304', ?)",
        ("0" * 64,),
    )

    M020_model_state_integrity.apply(conn)
    assert M020_model_state_integrity.verify(conn)

    import hashlib

    rows = {
        r[0]: r[1]
        for r in conn.execute("SELECT id, bytes_sha256 FROM learning_model_state")
    }
    assert rows[1] == hashlib.sha256(b"\x01\x02").hexdigest()
    assert rows[2] == "0" * 64  # pre-filled hash untouched
    M020_model_state_integrity.repair(conn)  # idempotent re-run
    assert M020_model_state_integrity.verify(conn)
    conn.close()


# --- M022: parent-profile backfill, orphans default ---------------------------

def test_m022_backfill_uses_parent_profile_orphans_default() -> None:
    from superlocalmemory.storage.migrations import M022_entity_aliases_profile

    conn = _conn()
    conn.execute(
        "CREATE TABLE canonical_entities (entity_id TEXT PRIMARY KEY, "
        "profile_id TEXT NOT NULL)"
    )
    conn.execute(
        "CREATE TABLE entity_aliases (alias_id INTEGER PRIMARY KEY, "
        "entity_id TEXT NOT NULL, alias TEXT NOT NULL)"
    )
    conn.execute(
        "INSERT INTO canonical_entities VALUES ('e1', 'work')"
    )
    conn.execute(
        "INSERT INTO entity_aliases (entity_id, alias) "
        "VALUES ('e1', 'a1'), ('missing-parent', 'a2')"
    )

    M022_entity_aliases_profile.apply(conn)
    assert M022_entity_aliases_profile.verify(conn)

    rows = {
        r[0]: r[1]
        for r in conn.execute("SELECT alias, profile_id FROM entity_aliases")
    }
    assert rows == {"a1": "work", "a2": "default"}
    M022_entity_aliases_profile.repair(conn)  # idempotent re-run
    assert M022_entity_aliases_profile.verify(conn)
    conn.close()


# --- Framework: justified drift skips instead of failing ----------------------

def _justified_drift_db() -> "sqlite3.Connection":
    import sqlite3

    from superlocalmemory.storage.migrations import (
        M003_migration_log,
        M048_upcoming_holds_only_what_is_upcoming as M048,
    )
    from superlocalmemory.storage import _migration_internals as mi

    conn = sqlite3.connect(":memory:")
    conn.executescript(M003_migration_log.DDL)
    conn.execute(
        "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT, "
        "fact_type TEXT)"
    )
    conn.execute(
        "INSERT INTO atomic_facts VALUES ('f1', 'The sky is blue', 'prospective')"
    )
    conn.commit()
    assert not M048.verify(conn)
    ddl_hash = mi._ddl_hash(M048.DDL)
    mi._upsert_log(conn, M048.NAME, ddl_hash, "complete")
    return conn


def test_justified_drift_skips_with_reason() -> None:
    """4.1.14 audit (critical): a justified module's routine drift is a
    SKIP, not a failure — failing here wrote migration-error logs and
    doctor FAILs for by-design drift."""
    from superlocalmemory.storage import _migration_internals as mi
    from superlocalmemory.storage.migrations import (
        M048_upcoming_holds_only_what_is_upcoming as M048,
    )

    conn = _justified_drift_db()
    mig = mi.Migration(name=M048.NAME, db_target="memory", ddl=M048.DDL)

    outcome, detail = mi._apply_single(conn, mig, dry_run=False)

    assert outcome == "skipped", detail
    assert "by design" in detail
    conn.close()


def test_unjustified_drift_still_fails(monkeypatch) -> None:
    from superlocalmemory.storage import _migration_internals as mi
    from superlocalmemory.storage.migrations import (
        M048_upcoming_holds_only_what_is_upcoming as M048,
    )

    conn = _justified_drift_db()
    monkeypatch.delattr(M048, "REPAIR_NOT_APPLICABLE", raising=True)
    mig = mi.Migration(name=M048.NAME, db_target="memory", ddl=M048.DDL)

    outcome, _ = mi._apply_single(conn, mig, dry_run=False)

    assert outcome == "failed"
    conn.close()


def test_justified_blocking_drift_still_fails(monkeypatch) -> None:
    """4.1.14 audit (critical): justification excuses repair, not the
    serving gate — a justified module without a non-blocking answer
    (M002 schema holes) keeps failing instead of booting silently."""
    import types

    from superlocalmemory.storage import _migration_internals as mi
    from superlocalmemory.storage.migrations import M003_migration_log

    justified = types.SimpleNamespace(
        NAME="MXXX_justified_blocking",
        DDL="-- nothing",
        REPAIR_NOT_APPLICABLE="destructive rebuild; restore from snapshot",
        verify=lambda conn: False,
    )
    monkeypatch.setitem(mi._MODULES, justified.NAME, justified)

    conn = __import__("sqlite3").connect(":memory:")
    conn.executescript(M003_migration_log.DDL)
    mi._upsert_log(
        conn, justified.NAME, mi._ddl_hash(justified.DDL), "complete",
    )
    mig = mi.Migration(name=justified.NAME, db_target="memory", ddl=justified.DDL)

    outcome, detail = mi._apply_single(conn, mig, dry_run=False)

    assert outcome == "failed", detail
    assert "destructive rebuild" in detail
    conn.close()
