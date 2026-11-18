# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
"""OPS-01: a future/invalid hash version cannot inherit v2's verified badge."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.compliance.audit import AuditChain


# 3 is the keyed version since 4.2.30 (compliance/audit_keyed.py); see test_keyed_audit_chain.
@pytest.mark.parametrize("version", [5, 4, -1, 0, "unknown"])
def test_an_unknown_hash_version_is_reported_as_unverified(tmp_path, version):
    path = tmp_path / "audit_chain.db"
    chain = AuditChain(path)
    chain.log("store", agent_id="fixture-actor", content_hash="sha256:fixture")
    with sqlite3.connect(path) as conn:
        conn.execute("UPDATE audit_chain SET hash_version=?", (version,))
    report = chain.verify_report()
    assert report["ok"] is False
    assert report["failed_at"] == 1
    assert report["fully_covered"] == 0
    assert "unsupported hash version" in report["reason"]
    assert chain.verify_integrity() is False


# 3 is the keyed version since 4.2.30 (compliance/audit_keyed.py); see test_keyed_audit_chain.
@pytest.mark.parametrize("version", [5, 4, -1, 0, "unknown"])
def test_unknown_versions_never_receive_a_covered_badge(version):
    from superlocalmemory.server.audit_view import present_event

    event = present_event({"id": 1, "hash_version": version, "operation": "store",
                           "metadata": "{}"})
    assert event["fully_covered"] is False
