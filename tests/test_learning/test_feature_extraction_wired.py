# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-02 §6.2

"""Verify that feature extraction writes all 20 features per candidate.

Checks:
    - ``len(features_json) == 20``
    - Keys match ``FEATURE_NAMES``
    - Grep guard: identity-mapping regression cannot return.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

import pytest

from superlocalmemory.learning.features import FEATURE_DIM, FEATURE_NAMES
from superlocalmemory.learning.signals import record_signal_batch
from tests.test_learning._signal_fixtures import (
    make_db_with_migrations,
    make_batch,
    open_conn,
)


def _learning_dir() -> Path:
    return (Path(__file__).resolve().parent.parent.parent
            / "src" / "superlocalmemory" / "learning")


def test_recall_writes_20_features(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    record_signal_batch(conn, make_batch(n_candidates=5))
    rows = conn.execute(
        "SELECT features_json FROM learning_features"
    ).fetchall()
    assert len(rows) == 5
    for r in rows:
        parsed = json.loads(dict(r)["features_json"])
        assert len(parsed) == FEATURE_DIM == 20
    conn.close()


def test_feature_names_match_FEATURE_NAMES(tmp_path):
    db = make_db_with_migrations(tmp_path)
    conn = open_conn(db)
    record_signal_batch(conn, make_batch(n_candidates=1))
    row = conn.execute(
        "SELECT features_json FROM learning_features LIMIT 1"
    ).fetchone()
    parsed = json.loads(dict(row)["features_json"])
    assert set(parsed.keys()) == set(FEATURE_NAMES)
    conn.close()


def test_identity_mapping_regression_blocked():
    """§6.2 static check — ``signal_value.*label`` must not return."""
    pat = re.compile(r"signal_value.*label")
    for p in _learning_dir().rglob("*.py"):
        text = p.read_text(encoding="utf-8")
        assert not pat.search(text), f"{p} contains an identity-mapping regression"
