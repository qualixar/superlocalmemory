# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
from __future__ import annotations

import json
import sqlite3

from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M072_judge_receipts as M072


def test_registered_additive_idempotent_repairable() -> None:
    assert _MODULES[M072.NAME] is M072 and M072.DOWNGRADE_FLOOR == 51
    assert M072.NAME in {m.name for m in DEFERRED_MIGRATIONS}
    conn = sqlite3.connect(":memory:")
    assert not M072.verify(conn)
    M072.apply(conn)
    M072.apply(conn)
    assert M072.verify(conn)
    conn.execute("DROP INDEX idx_judge_receipts_inputs")
    assert not M072.verify(conn)
    M072.repair(conn)
    assert M072.verify(conn)


def test_erasing_a_fact_erases_the_receipts_that_judged_it() -> None:
    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M072.apply(conn)
    conn.executemany("INSERT INTO atomic_facts VALUES (?)", [("f1",), ("f2",), ("f3",)])
    for rid, facts in (("r1", '["f1", "f2"]'), ("r2", '["f3"]')):
        conn.executemany("INSERT INTO judge_receipt_facts VALUES (?, ?)",
                         [(rid, f) for f in json.loads(facts)])
        conn.execute("INSERT INTO judge_receipts (receipt_id, profile_id, question_id, "
                     "question_version, subject_kind, subject_id, fact_ids_json, tier, model, "
                     "verdict, confidence, threshold, accepted, calibration_id, inputs_hash, "
                     "decided_at) VALUES (?, 'p', 'Q', '2', 's', 'x', ?, 'jev', 'm', 'same', 0.9, "
                     "0.7, 1, 'c', 'h', 't')", (rid, facts))
        conn.execute("INSERT INTO judge_labels (receipt_id, label, source, labelled_at) "
                     "VALUES (?, 'same', 'audit', 't')", (rid,))
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f2'")
    assert [r[0] for r in conn.execute("SELECT receipt_id FROM judge_receipts")] == ["r2"]
    assert [r[0] for r in conn.execute("SELECT receipt_id FROM judge_labels")] == ["r2"]
    assert conn.execute("SELECT receipt_id, fact_id FROM judge_receipt_facts").fetchall() == [
        ("r2", "f3")]


def test_an_underscore_in_an_id_is_not_a_wildcard_and_the_ladder_writes_the_links() -> None:
    from superlocalmemory.core.judge_ladder import Context, JudgeLadder, TierAnswer, write_receipts
    from superlocalmemory.core.judge_questions import SLOT_RELATION
    from tests.test_core.test_audit_sampler import Fixed, _items

    conn = sqlite3.connect(":memory:")
    conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY)")
    M072.apply(conn)
    items = _items(2)  # fact ids a0, b0, a1, b1
    _, rows = JudgeLadder([Fixed(lambda xs: [TierAnswer("same", 0.9, model="m")] * len(xs))]
                          ).decide(SLOT_RELATION, items, Context(population="slot_dispute"))
    write_receipts(conn, rows)
    assert conn.execute("SELECT COUNT(*) FROM judge_receipt_facts").fetchone() == (4,)
    conn.executemany("INSERT INTO atomic_facts VALUES (?)", [("a_",), ("a0",)])
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'a_'")  # LIKE '%"a_"%' matched a0
    assert conn.execute("SELECT COUNT(*) FROM judge_receipts").fetchone() == (2,)
    conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'a0'")
    assert conn.execute("SELECT COUNT(*) FROM judge_receipts").fetchone() == (1,)
