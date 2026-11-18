# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M080: registered, additive, repairable, and its invariants held by the store itself."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M080_versions_and_proposals as M080


def test_m080_is_deferred_additive_and_repairable() -> None:
    assert _MODULES[M080.NAME] is M080
    assert M080.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    assert M080.DOWNGRADE_FLOOR == 51 and M080.DB_TARGET == "memory"
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    assert not M080.verify(conn)
    M080.apply(conn)
    M080.apply(conn)
    assert M080.verify(conn)
    conn.execute("DROP TRIGGER trg_proposals_fact_erased")
    assert not M080.verify(conn)
    M080.repair(conn)
    assert M080.verify(conn)


def test_m080_builds_without_facts_and_holds_the_applied_needs_a_basis_invariant() -> None:
    conn = sqlite3.connect(":memory:")
    M080.apply(conn)
    assert M080.verify(conn)
    conn.execute("INSERT INTO proposal_sets (set_id, profile_id, origin, state, created_at) "
                 "VALUES ('s1', 'p', 'dream', 'anything-new', 'now')")
    for bad in ("('s2', 'p', 'dream', 'applied', NULL, NULL)",
                "('s3', 'p', 'dream', 'proposed', 'person', NULL)"):
        with pytest.raises(sqlite3.IntegrityError):
            conn.execute("INSERT INTO proposal_sets (set_id, profile_id, origin, state, "
                         f"basis_kind, basis_ref, created_at) VALUES {bad[:-1]}, 'now')")
    conn.execute("INSERT INTO proposal_sets (set_id, profile_id, origin, state, basis_kind, "
                 "basis_ref, created_at) VALUES ('s4', 'p', 'dream', 'applied', 'person', "
                 "'alice@cli', 'now')")


def test_deleting_a_set_or_a_version_takes_its_members_and_links_with_it() -> None:
    conn = sqlite3.connect(":memory:")
    M080.apply(conn)
    conn.execute("INSERT INTO proposal_sets (set_id, profile_id, origin, state, created_at) "
                 "VALUES ('s', 'p', 'o', 'proposed', 'n')")
    conn.execute("INSERT INTO proposals (proposal_id, set_id, profile_id, seq, kind, payload_json, "
                 "state, created_at) VALUES ('q', 's', 'p', 0, 'retire', '{}', 'proposed', 'n')")
    conn.execute("INSERT INTO proposal_facts VALUES ('q', 'f1')")
    conn.execute("INSERT INTO artifact_versions VALUES ('v', 'p', 'persona', '[\"p\"]', 1, 'h', "
                 "'{}', 'overwrite', NULL, 'n')")
    conn.execute("INSERT INTO artifact_version_facts VALUES ('v', 'f1')")
    conn.execute("DELETE FROM proposal_sets")
    conn.execute("DELETE FROM artifact_versions")
    for table in ("proposals", "proposal_facts", "artifact_version_facts"):
        assert conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] == 0  # noqa: S608
