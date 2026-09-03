# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Drift recovery for the exact #128 Bug 4 scenario.

The reporter dropped and recreated ``atomic_facts``, losing every
migration-added column and index while the migration log still read
complete. Each affected migration's ``repair()`` must restore its own
end-state so ``verify()`` passes again — no hand-written ALTERs, no DB
surgery.
"""
from __future__ import annotations

import sqlite3

from superlocalmemory.storage.migrations import (
    M011_archive_and_merge,
    M013_bi_temporal_columns,
    M014_v345_scale_ready,
    M015_add_pinned_column,
    M016_add_scope_support,
)


def _drifted_atomic_facts() -> sqlite3.Connection:
    """An atomic_facts as the reporter's looked: recreated, columns lost."""
    conn = sqlite3.connect(":memory:")
    conn.execute(
        "CREATE TABLE atomic_facts ("
        "fact_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, content TEXT)"
    )
    return conn


def _assert_repaired(module, conn) -> None:
    assert not module.verify(conn)
    module.repair(conn)
    assert module.verify(conn), module.NAME
    module.repair(conn)  # second run is a no-op proof
    assert module.verify(conn), module.NAME


def test_reporter_drift_m011_columns_restored() -> None:
    conn = _drifted_atomic_facts()
    _assert_repaired(M011_archive_and_merge, conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
    assert {"archive_status", "merged_into", "retrieval_prior"} <= cols
    conn.close()


def test_reporter_drift_m013_columns_restored() -> None:
    conn = _drifted_atomic_facts()
    _assert_repaired(M013_bi_temporal_columns, conn)
    conn.close()


def test_reporter_drift_m014_column_and_indexes_restored() -> None:
    conn = _drifted_atomic_facts()
    conn.execute("CREATE TABLE graph_edges (source_id TEXT, target_id TEXT)")
    _assert_repaired(M014_v345_scale_ready, conn)
    conn.close()


def test_reporter_drift_m015_column_restored() -> None:
    conn = _drifted_atomic_facts()
    _assert_repaired(M015_add_pinned_column, conn)
    conn.close()


def test_reporter_drift_m016_scope_restored() -> None:
    conn = _drifted_atomic_facts()
    _assert_repaired(M016_add_scope_support, conn)
    cols = {row[1] for row in conn.execute("PRAGMA table_info(atomic_facts)")}
    assert {"scope", "shared_with"} <= cols
    conn.close()
