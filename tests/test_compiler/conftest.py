"""Shared store for the compiler tests: a fully migrated store with a few facts and an entity."""

from __future__ import annotations

import json
import sqlite3

import pytest

from tests.test_core.test_coverage_backfill import _db
from tests.test_truth.test_slot_pass import _fact

PROFILE = "default"
ENTITY = "ent-slm"


@pytest.fixture
def store(tmp_path):
    """(db, [fact ids]) on a store migrated like a live one, M080 included."""
    from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
    from superlocalmemory.storage.schema_v347 import apply_v347_schema
    from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
    from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

    db = _db(tmp_path / "memory.db")
    for apply_schema in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                         apply_v3410_schema, apply_v3411_schema):
        apply_schema(str(db.db_path))
    ids = [
        _fact(db, f"SuperLocalMemory shipped release {i} with a measured recall gain",
              f"2026-0{i + 1}-15T10:00:00+00:00")
        for i in range(3)
    ]
    db.execute("INSERT OR IGNORE INTO canonical_entities (entity_id, profile_id, canonical_name) "
               "VALUES (?, ?, 'SuperLocalMemory')", (ENTITY, PROFILE))
    for fid in ids:
        db.execute("UPDATE atomic_facts SET canonical_entities_json = ? WHERE fact_id = ?",
                   (json.dumps([ENTITY]), fid))
    return db, ids


def versions_of(db, artifact):
    return [dict(r) for r in db.execute(
        "SELECT * FROM artifact_versions WHERE artifact = ? ORDER BY seq", (artifact,))]


def table_rows(db, table, order="1"):
    """Every row of a table as plain dicts (BLOBs left out): the row-level snapshot."""
    out = []
    for r in db.execute(f"SELECT * FROM {table} ORDER BY {order}"):  # noqa: S608
        out.append({k: v for k, v in dict(r).items() if not isinstance(v, bytes)})
    return out


def raw(db):
    conn = sqlite3.connect(db.db_path, isolation_level=None)
    conn.row_factory = sqlite3.Row
    return conn
