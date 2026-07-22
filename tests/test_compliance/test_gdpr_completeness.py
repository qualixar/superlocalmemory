# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — GDPR erasure completeness (audit cycle 1)

"""GDPR erasure must clear EVERY profile-scoped table (Art. 17).

Regression pin for the audit finding that forget_profile() missed the mesh,
RBAC, event-bus and tool tables. Seeds a throwaway tenant across those tables
via the real subsystem APIs, erases it, and asserts zero residue — while a
second tenant is untouched.
"""

from __future__ import annotations

import pytest


@pytest.fixture
def wired(engine_with_mock_deps):
    from superlocalmemory.access.rbac import RbacEngine
    from superlocalmemory.compliance.gdpr import GDPRCompliance
    from superlocalmemory.mesh.broker import MeshBroker

    engine = engine_with_mock_deps
    db_path = str(engine._config.db_path)
    for pid in ("wipe_me", "keep_me"):
        engine._db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            (pid, pid),
        )
    return engine, MeshBroker(db_path), RbacEngine(db_path), GDPRCompliance(engine._db)


def _seed_tenant(engine, broker, rbac, pid):
    # atomic_facts + memories
    engine._db.execute(
        "INSERT INTO memories (memory_id, profile_id, content, session_id, speaker, "
        "role, created_at, metadata_json, scope) VALUES (?,?,?,?,?,?,?,?,?)",
        (f"mem_{pid}", pid, "secret note", "s1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    engine._db.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, lifecycle, "
        "created_at, scope) VALUES (?,?,?,?,?,?,?)",
        (f"f_{pid}", f"mem_{pid}", pid, "secret fact", "active",
         "2026-01-01T00:00:00Z", "personal"),
    )
    # mesh: peer + broadcast message + shared state
    peer = broker.register_peer(f"sess_{pid}", profile_id=pid)["peer_id"]
    broker.send_message("other", "broadcast", "mesh secret", profile_id=pid)
    broker.set_state("flag", "value", "setter", profile_id=pid)
    # rbac membership
    u = rbac.create_user(f"user_{pid}", "password-1234")
    rbac.set_membership(pid, u["user_id"], "admin")
    return peer


def test_erase_clears_every_profile_scoped_table(wired):
    engine, broker, rbac, gdpr = wired
    _seed_tenant(engine, broker, rbac, "wipe_me")
    _seed_tenant(engine, broker, rbac, "keep_me")

    scoped = gdpr._profile_scoped_tables()
    # The tables the audit flagged as missed MUST be in the erase set.
    for t in ("mesh_messages", "mesh_peers", "mesh_state", "rbac_memberships",
              "atomic_facts", "memories"):
        assert t in scoped, f"{t} not in profile-scoped erase set"

    gdpr.forget_profile("wipe_me")

    # Zero residue for the erased tenant across every scoped table...
    for t in scoped:
        n = engine._db.execute(
            f"SELECT COUNT(*) AS c FROM {t} WHERE profile_id = ?", ("wipe_me",)
        )
        assert int(dict(n[0])["c"]) == 0, f"{t} still has wipe_me rows"

    # ...and the other tenant is fully intact.
    keep_facts = engine._db.execute(
        "SELECT COUNT(*) AS c FROM atomic_facts WHERE profile_id='keep_me'")
    assert int(dict(keep_facts[0])["c"]) == 1
    keep_msgs = engine._db.execute(
        "SELECT COUNT(*) AS c FROM mesh_messages WHERE profile_id='keep_me'")
    assert int(dict(keep_msgs[0])["c"]) == 1
    keep_mem = engine._db.execute(
        "SELECT COUNT(*) AS c FROM rbac_memberships WHERE profile_id='keep_me'")
    assert int(dict(keep_mem[0])["c"]) == 1


def test_erase_logged_to_tamper_proof_chain(wired, monkeypatch, tmp_path):
    """The erasure event must land in the audit chain (survives the wipe)."""
    engine, broker, rbac, gdpr = wired
    chain_db = tmp_path / "audit_chain.db"
    monkeypatch.setattr(
        "superlocalmemory.infra.data_root.state_path",
        lambda name: chain_db if name == "audit_chain.db" else tmp_path / name,
    )
    _seed_tenant(engine, broker, rbac, "wipe_me")
    gdpr.forget_profile("wipe_me")

    from superlocalmemory.compliance.audit import AuditChain
    events = AuditChain(str(chain_db)).query(operation="gdpr_erase")
    assert any(e.get("profile_id") == "wipe_me" for e in events), \
        "erasure not recorded in tamper-proof audit chain"
