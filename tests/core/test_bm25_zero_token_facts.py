# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Issue #319: a fact whose text has no searchable words (a Markdown divider
of 99 hyphens) can never be indexed by BM25, so the verifier must not demand
an index row for it. Before the fix every such fact left a projection
obligation that failed verification forever, and each repair skipped it
again."""

from __future__ import annotations

from superlocalmemory.core.transactions import OperationContext
from superlocalmemory.core.transactions.concrete_owners import Bm25Owner
from superlocalmemory.retrieval.bm25_channel import BM25Channel, tokenize
from superlocalmemory.storage.models import AtomicFact, FactType, MemoryRecord

DIVIDER = "-" * 99


def _db(tmp_path):
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db = DatabaseManager(tmp_path / "memory.db")
    db.initialize(schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
               ("default", "default"))
    db.store_memory(MemoryRecord(memory_id="m1", profile_id="default", content="doc"))
    return db


def _fact(db, fact_id, content):
    db.store_fact(AtomicFact(fact_id=fact_id, memory_id="m1", content=content,
                             fact_type=FactType.SEMANTIC))


def _ctx(*fact_ids):
    return OperationContext(operation_id="op", profile_id="default",
                            subject_id=fact_ids[0], fact_ids=fact_ids)


class _Retrieval:
    def __init__(self, db):
        self._bm25 = BM25Channel(db)


def test_divider_has_no_tokens():
    assert tokenize(DIVIDER) == []


def test_zero_token_fact_is_not_required_and_verifies(tmp_path):
    db = _db(tmp_path)
    _fact(db, "divider", DIVIDER)
    owner = Bm25Owner(db, retrieval=_Retrieval(db))
    assert owner.verify(_ctx("divider")).ok is True
    applied = owner.apply(_ctx("divider"))
    assert applied.ok is True
    assert "missing" not in applied.detail


def test_word_facts_beside_a_divider_are_still_required(tmp_path):
    db = _db(tmp_path)
    _fact(db, "divider", DIVIDER)
    _fact(db, "words", "Varun prefers dark roast coffee")
    owner = Bm25Owner(db, retrieval=_Retrieval(db))
    before = owner.verify(_ctx("divider", "words"))
    assert before.ok is False and before.detail["missing"] == ["words"]
    assert owner.apply(_ctx("divider", "words")).ok is True


def test_missing_canonical_row_still_fails_closed(tmp_path):
    db = _db(tmp_path)
    result = Bm25Owner(db, retrieval=_Retrieval(db)).verify(_ctx("ghost"))
    assert result.ok is False and result.detail["missing"] == ["ghost"]


def test_stale_tokens_of_a_fact_now_without_words_are_removed(tmp_path):
    db = _db(tmp_path)
    _fact(db, "edited", DIVIDER)
    db.store_bm25_tokens("edited", "default", ["old", "words"])
    owner = Bm25Owner(db, retrieval=_Retrieval(db))
    assert owner.apply(_ctx("edited")).ok is True
    rows = db.execute("SELECT COUNT(*) AS c FROM bm25_tokens WHERE fact_id = ?",
                      ("edited",))
    assert dict(rows[0])["c"] == 0
