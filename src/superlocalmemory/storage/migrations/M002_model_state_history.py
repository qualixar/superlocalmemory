# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.21 — LLD-07 §3.2

"""M002 — rebuild learning_model_state without UNIQUE(profile_id).

Enables shadow testing (old + new model live side-by-side). SQLite cannot
drop a UNIQUE constraint directly, so we use the new-table-rename pattern
wrapped in a single transaction. Existing rows are copied forward and
marked ``is_active = 1``.

SEC-02-02: ``bytes_sha256`` integrity column added. The daemon will
compute and verify this on every load to block corrupted model BLOBs from
reaching the LightGBM deserialiser.
"""

from __future__ import annotations

import hashlib
import sqlite3

NAME = "M002_model_state_history"
DB_TARGET = "learning"

_REQUIRED_COLS = frozenset({
    "model_version", "bytes_sha256", "trained_on_count",
    "feature_names", "metrics_json", "is_active",
    "trained_at", "updated_at",
})


def verify(conn: sqlite3.Connection) -> bool:
    """Return True if the rebuilt model_state schema is in place."""
    try:
        cols = {r[1] for r in conn.execute(
            "PRAGMA table_info(learning_model_state)"
        ).fetchall()}
    except sqlite3.Error:
        return False
    return _REQUIRED_COLS <= cols


# S9-W1 M-DATA-01: prior version of this migration hardcoded ``is_active=1``
# for every copied row. v3.4.19 mainline has ``UNIQUE(profile_id)`` which
# guarantees one row per profile so the hardcode worked — but a non-mainline
# dev-build could have multiple rows and the partial unique index created
# after rebuild would fail transactionally. We now mark only the row with
# the MAX(id) per profile as active; everything else becomes history.
DDL = """
BEGIN IMMEDIATE;

CREATE TABLE learning_model_state_new (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    profile_id        TEXT NOT NULL,
    model_version     TEXT NOT NULL DEFAULT '3.4.21',
    state_bytes       BLOB NOT NULL,
    bytes_sha256      TEXT NOT NULL DEFAULT '',
    trained_on_count  INTEGER NOT NULL DEFAULT 0,
    feature_names     TEXT NOT NULL DEFAULT '[]',
    metrics_json      TEXT NOT NULL DEFAULT '{}',
    is_active         INTEGER NOT NULL DEFAULT 0,
    trained_at        TEXT NOT NULL,
    updated_at        TEXT NOT NULL
);

INSERT INTO learning_model_state_new
    (profile_id, state_bytes, is_active, trained_at, updated_at)
SELECT lms.profile_id, lms.state_bytes,
       CASE WHEN lms.id = (
             SELECT MAX(lms2.id)
             FROM learning_model_state lms2
             WHERE lms2.profile_id = lms.profile_id
           ) THEN 1 ELSE 0 END,
       lms.updated_at, lms.updated_at
FROM learning_model_state lms;

DROP TABLE learning_model_state;
ALTER TABLE learning_model_state_new RENAME TO learning_model_state;

CREATE UNIQUE INDEX idx_model_active
    ON learning_model_state(profile_id)
    WHERE is_active = 1;

CREATE INDEX idx_model_profile_time
    ON learning_model_state(profile_id, trained_at);

COMMIT;
"""


def post_ddl_hook(conn: sqlite3.Connection) -> None:
    """S9-W1 H-DATA-01: backfill ``bytes_sha256`` for every row copied forward.

    The DDL INSERT could not list ``bytes_sha256`` because SQLite cannot
    call a Python function inside an ``executescript`` block unless the
    function is registered beforehand, and registering a UDF mid-DDL is
    fragile. Instead, we run one UPDATE pass after the DDL commits.

    Without this backfill, ``model_cache._parse_row`` calls
    ``verify_sha256(state_bytes, '')`` which raises IntegrityError, the
    parser tombstones the cache entry, and EVERY 18,000+ user who had a
    trained model on v3.4.19 loses usable learned-ranker state on upgrade.

    The fix is safe: SHA-256 of the already-persisted blob is
    deterministic, adds <1 ms per profile, and never alters
    ``state_bytes``. Runs inside the same connection so any UPDATE error
    surfaces to the runner as ``post_ddl_hook`` failed.
    """
    try:
        rows = conn.execute(
            "SELECT id, state_bytes FROM learning_model_state "
            "WHERE bytes_sha256 = '' OR bytes_sha256 IS NULL"
        ).fetchall()
    except sqlite3.Error:
        return  # table empty or schema not yet present — nothing to do.

    if not rows:
        return

    updates = []
    for row_id, state_bytes in rows:
        if state_bytes is None:
            continue
        sha = hashlib.sha256(state_bytes).hexdigest()
        updates.append((sha, row_id))

    if updates:
        conn.executemany(
            "UPDATE learning_model_state SET bytes_sha256 = ? WHERE id = ?",
            updates,
        )
        conn.commit()
