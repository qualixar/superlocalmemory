# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The one-time repair restores blanked memories and removes only empty entities."""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.storage.store_repair import (
    apply_repair,
    is_blanked_copy,
    plan_repair,
)

ORIGINAL = "Backups go to Agentic_official/slm-backups/pre-4.1.18-<ts> with sha256."
BLANKED = "Backups go to [REDACTED:ENTROPY:.18-]<ts> with sha256."
MISMATCH = "queryable ingestion memory content mismatch"


@pytest.fixture()
def conn(tmp_path) -> sqlite3.Connection:
    c = sqlite3.connect(tmp_path / "m.db", isolation_level=None)
    c.executescript("""
        CREATE TABLE memories(memory_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE atomic_facts(fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,
                                  canonical_entities_json TEXT);
        CREATE TABLE ingestion_operations(operation_id TEXT PRIMARY KEY, raw_content TEXT,
            queryable_fact_ids_json TEXT, state TEXT, last_error TEXT, attempt_count INT,
            next_retry_at REAL, lease_owner TEXT, lease_expires_at REAL, updated_at TEXT);
        CREATE TABLE canonical_entities(entity_id TEXT PRIMARY KEY, canonical_name TEXT,
                                        fact_count INT);
        CREATE TABLE entity_aliases(entity_id TEXT, alias TEXT, source TEXT);
        CREATE TABLE fact_temporal_validity(fact_id TEXT PRIMARY KEY, valid_until TEXT,
            system_expired_at TEXT, invalidated_by TEXT, invalidation_reason TEXT);
    """)
    c.execute("INSERT INTO memories VALUES ('m1', ?)", (BLANKED,))
    c.execute("INSERT INTO atomic_facts VALUES ('f1', 'm1', ?, '[\"e_used\"]')", (BLANKED,))
    c.execute("INSERT INTO ingestion_operations VALUES ('op1', ?, ?, 'failed', ?, 9, 1e12, "
              "'', 0, '')", (ORIGINAL, json.dumps(["f1"]), MISMATCH))
    # A different stored text: not a blanked copy of its original -> left alone.
    c.execute("INSERT INTO memories VALUES ('m2', 'something else entirely')")
    c.execute("INSERT INTO atomic_facts VALUES ('f2', 'm2', 'something else entirely', '[]')")
    c.execute("INSERT INTO ingestion_operations VALUES ('op2', 'the real text', ?, 'failed', "
              "?, 9, 1e12, '', 0, '')", (json.dumps(["f2"]), MISMATCH))
    for eid, name, n in (("e_zorblax", "Zorblax", 0), ("e_used", "Used", 0),
                         ("e_alias", "Aliased", 0), ("e_real", "Real", 3)):
        c.execute("INSERT INTO canonical_entities VALUES (?, ?, ?)", (eid, name, n))
    c.execute("INSERT INTO entity_aliases VALUES ('e_alias', 'Al', 'llm')")
    # Every entity carries its own name as a 'canonical' self-alias.
    for eid, name in (("e_zorblax", "Zorblax"), ("e_alias", "Aliased"), ("e_used", "Used")):
        c.execute("INSERT INTO entity_aliases VALUES (?, ?, 'canonical')", (eid, name))
    T0 = "2026-07-20T12:49:05+00:00"
    c.execute("INSERT INTO fact_temporal_validity VALUES ('f_llm', ?, ?, 'f_new', "
              "'LLM-verified contradiction (sheaf pre-filter severity=0.5)')", (T0, T0))
    c.execute("INSERT INTO fact_temporal_validity VALUES ('f_sheaf', ?, ?, 'f_new', "
              "'Sheaf coboundary 0.493 > 0.45 along semantic edge')", (T0, T0))
    c.execute("INSERT INTO fact_temporal_validity VALUES ('f_user', NULL, ?, 'f_fix', "
              "'direct_content_correction')", (T0,))
    c.execute("INSERT INTO fact_temporal_validity VALUES ('f_live', NULL, NULL, NULL, NULL)")
    return c


def test_a_blanked_copy_is_recognised_and_nothing_else_is() -> None:
    assert is_blanked_copy(BLANKED, ORIGINAL)
    assert not is_blanked_copy(ORIGINAL, ORIGINAL)            # nothing blanked
    assert not is_blanked_copy("totally different", ORIGINAL)  # no marker
    assert not is_blanked_copy("Backups go to [REDACTED:X:1] with md5.", ORIGINAL)


def test_the_plan_reads_only(conn) -> None:
    before = conn.execute("SELECT * FROM memories ORDER BY 1").fetchall()
    plan = plan_repair(conn)
    assert plan.summary() == {"memories_to_restore": 1, "memories_left_alone": 1,
                              "empty_entities_to_remove": 1,
                              "wrong_replaced_marks_to_undo": 2}
    assert plan.wrong_replacements == ("f_llm", "f_sheaf")
    assert plan.empty_entities == ("e_zorblax",)
    assert conn.execute("SELECT * FROM memories ORDER BY 1").fetchall() == before


def test_apply_restores_the_memory_and_requeues_its_enrichment(conn) -> None:
    result = apply_repair(conn, plan_repair(conn))
    assert (result.memories_restored, result.memories_skipped, result.entities_removed) == (1, 0, 1)
    assert conn.execute("SELECT content FROM memories WHERE memory_id='m1'").fetchone()[0] == ORIGINAL
    assert conn.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'").fetchone()[0] == ORIGINAL
    op = conn.execute("SELECT state, attempt_count, last_error, next_retry_at "
                      "FROM ingestion_operations WHERE operation_id='op1'").fetchone()
    assert op == ("queryable", 0, "", 0)
    # The one that was not a blanked copy is untouched.
    assert conn.execute("SELECT state FROM ingestion_operations WHERE operation_id='op2'"
                        ).fetchone()[0] == "failed"
    remaining = {r[0] for r in conn.execute("SELECT entity_id FROM canonical_entities")}
    assert remaining == {"e_used", "e_alias", "e_real"}
    assert conn.execute("SELECT COUNT(*) FROM entity_aliases WHERE entity_id='e_zorblax'"
                        ).fetchone()[0] == 0


def test_a_memory_edited_after_planning_is_skipped_not_overwritten(conn) -> None:
    plan = plan_repair(conn)
    conn.execute("UPDATE memories SET content='edited by the user' WHERE memory_id='m1'")
    result = apply_repair(conn, plan)
    assert (result.memories_restored, result.memories_skipped) == (0, 1)
    assert conn.execute("SELECT content FROM memories WHERE memory_id='m1'"
                        ).fetchone()[0] == "edited by the user"


def test_wrong_replaced_marks_are_undone_and_a_users_correction_is_kept(conn) -> None:
    result = apply_repair(conn, plan_repair(conn))
    assert result.replaced_marks_undone == 2
    rows = dict((r[0], r[1:]) for r in conn.execute(
        "SELECT fact_id, valid_until, system_expired_at, invalidation_reason "
        "FROM fact_temporal_validity"))
    for fact_id in ("f_llm", "f_sheaf"):
        valid_until, expired, reason = rows[fact_id]
        assert valid_until is None and expired is None
        assert reason.startswith("reverted in 4.1.19")      # the old reason is kept
    assert rows["f_user"][1] is not None                     # correction untouched
    assert rows["f_user"][2] == "direct_content_correction"


# ---------------------------------------------------------------------------
# L1-08: a memory over 24,000 chars has a CLAMPED fact (core/ingest_gate.py),
# so the fact's stored content never equals the memory's blanked text byte
# for byte. The repair must match and write the clamped forms, not the raw
# memory text, or the fact stays blanked forever.
# ---------------------------------------------------------------------------

def _long_blanked_and_original() -> tuple[str, str]:
    original = ("Build notes. Artifacts live under "
                "/Users/me/work/repo/build/out/release-4.1.18/x86 "
                + "lorem ipsum dolor sit amet " * 1200)
    blanked = original.replace(
        "/Users/me/work/repo/build/out/release-4.1.18/x86", "[REDACTED:PATH:x86]")
    return blanked, original


@pytest.fixture()
def long_conn(tmp_path) -> tuple[sqlite3.Connection, str, str]:
    blanked, original = _long_blanked_and_original()
    from superlocalmemory.core.ingest_gate import apply_ingest_gate
    clamped_fact = apply_ingest_gate(blanked).fact_content
    assert clamped_fact != blanked, "fixture must exercise the >24k clamp"

    c = sqlite3.connect(tmp_path / "long.db", isolation_level=None)
    c.executescript("""
        CREATE TABLE memories(memory_id TEXT PRIMARY KEY, content TEXT);
        CREATE TABLE atomic_facts(fact_id TEXT PRIMARY KEY, memory_id TEXT, content TEXT,
                                  canonical_entities_json TEXT);
        CREATE TABLE ingestion_operations(operation_id TEXT PRIMARY KEY, raw_content TEXT,
            queryable_fact_ids_json TEXT, state TEXT, last_error TEXT, attempt_count INT,
            next_retry_at REAL, lease_owner TEXT, lease_expires_at REAL, updated_at TEXT);
        CREATE TABLE canonical_entities(entity_id TEXT PRIMARY KEY, canonical_name TEXT,
                                        fact_count INT);
        CREATE TABLE entity_aliases(entity_id TEXT, alias TEXT, source TEXT);
        CREATE TABLE fact_temporal_validity(fact_id TEXT PRIMARY KEY, valid_until TEXT,
            system_expired_at TEXT, invalidated_by TEXT, invalidation_reason TEXT);
    """)
    c.execute("INSERT INTO memories VALUES ('m1', ?)", (blanked,))
    c.execute("INSERT INTO atomic_facts VALUES ('f1', 'm1', ?, '[]')", (clamped_fact,))
    c.execute("INSERT INTO ingestion_operations VALUES ('op1', ?, ?, 'failed', ?, 9, 1e12, "
              "'', 0, '')", (original, json.dumps(["f1"]), MISMATCH))
    return c, blanked, original


def test_a_clamped_fact_is_matched_and_restored_too(long_conn) -> None:
    from superlocalmemory.core.ingest_gate import apply_ingest_gate

    c, _blanked, original = long_conn
    plan = plan_repair(c)
    assert plan.summary()["memories_to_restore"] == 1
    result = apply_repair(c, plan)
    assert (result.memories_restored, result.memories_skipped) == (1, 0)
    assert c.execute("SELECT content FROM memories WHERE memory_id='m1'"
                     ).fetchone()[0] == original
    restored_fact = c.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'"
                              ).fetchone()[0]
    assert "[REDACTED:" not in restored_fact
    assert restored_fact == apply_ingest_gate(original).fact_content


def test_a_fact_that_no_longer_matches_its_clamp_is_left_alone(long_conn) -> None:
    # Something changed the fact's content between planning and apply (or the
    # clamp assumption does not hold). Report must count it as skipped, and
    # neither the memory nor the fact may be partially restored.
    c, blanked, _original = long_conn
    c.execute("UPDATE atomic_facts SET content='drifted elsewhere' WHERE fact_id='f1'")
    plan = plan_repair(c)
    result = apply_repair(c, plan)
    assert (result.memories_restored, result.memories_skipped) == (0, 1)
    assert c.execute("SELECT content FROM memories WHERE memory_id='m1'"
                     ).fetchone()[0] == blanked
    assert c.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'"
                     ).fetchone()[0] == "drifted elsewhere"


# ---------------------------------------------------------------------------
# A real credential the old redactor blanked must never be put back, by the
# one-time repair or by any other caller of the shared restore.
# ---------------------------------------------------------------------------

_TOKEN = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
SECRET_ORIGINAL = f"deploy with token {_TOKEN} from the ci box"
SECRET_BLANKED = "deploy with token [REDACTED:GITHUB:q7R8] from the ci box"


@pytest.fixture()
def secret_conn(conn) -> sqlite3.Connection:
    conn.execute("INSERT INTO memories VALUES ('ms', ?)", (SECRET_BLANKED,))
    conn.execute("INSERT INTO atomic_facts VALUES ('fs', 'ms', ?, '[]')", (SECRET_BLANKED,))
    conn.execute("INSERT INTO ingestion_operations VALUES ('ops', ?, ?, 'failed', ?, 9, 1e12, "
                 "'', 0, '')", (SECRET_ORIGINAL, json.dumps(["fs"]), MISMATCH))
    return conn


def test_the_plan_leaves_a_secret_holding_original_and_counts_it(secret_conn) -> None:
    from superlocalmemory.core.security_primitives import detect_secrets

    assert is_blanked_copy(SECRET_BLANKED, SECRET_ORIGINAL) and detect_secrets(SECRET_ORIGINAL)
    plan = plan_repair(secret_conn)
    assert [i.operation_id for i in plan.restorable] == ["op1"]
    assert "ops" in plan.not_restorable
    assert plan.summary()["memories_left_alone"] == 2


def test_apply_never_restores_a_secret_holding_original(secret_conn) -> None:
    result = apply_repair(secret_conn, plan_repair(secret_conn))
    assert (result.memories_restored, result.memories_skipped) == (1, 0)
    assert secret_conn.execute("SELECT content FROM memories WHERE memory_id='ms'"
                               ).fetchone()[0] == SECRET_BLANKED
    assert secret_conn.execute("SELECT content FROM atomic_facts WHERE fact_id='fs'"
                               ).fetchone()[0] == SECRET_BLANKED
    assert secret_conn.execute("SELECT state FROM ingestion_operations WHERE operation_id='ops'"
                               ).fetchone()[0] == "failed"


def test_the_shared_restore_refuses_a_secret_as_a_last_guard(secret_conn) -> None:
    # A hand-built plan, as a caller that skipped the planner would make.
    from superlocalmemory.storage.store_repair import (
        RepairPlan, RestoreItem, restore_blanked_text,
    )

    run = lambda sql, params=(): secret_conn.execute(sql, params).fetchall()  # noqa: E731
    assert restore_blanked_text(run, "ms", SECRET_BLANKED, SECRET_ORIGINAL, ("fs",)) is False
    item = RestoreItem("ops", "ms", ("fs",), SECRET_BLANKED, SECRET_ORIGINAL)
    result = apply_repair(secret_conn, RepairPlan((item,), (), (), ()))
    assert (result.memories_restored, result.memories_skipped) == (0, 1)
    assert secret_conn.execute("SELECT content FROM memories WHERE memory_id='ms'"
                               ).fetchone()[0] == SECRET_BLANKED
