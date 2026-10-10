"""Erasing page memories erases the pages, and the document once none are left."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.core.transactions.erasure import ErasureService
from superlocalmemory.core.transactions.owners import OperationContext
from superlocalmemory.documents.status import remove_document
from superlocalmemory.media import files
from superlocalmemory.media.erasure import MediaErasureOwner
from tests.test_documents.index_support import add_document, make_root


def _erase(db, root, fact_ids, profile="p1", op="op1"):
    svc = ErasureService({"media": MediaErasureOwner(db, data_root=root)})
    ctx = OperationContext(operation_id=op, profile_id=profile, subject_id="x", fact_ids=tuple(fact_ids))
    removed = svc.remove(db, ctx)
    return svc.finalize(db, ctx, subject_type="entity", subject_id="x", remove_result=removed)


def _count(store, table, where="1=1"):
    return store._read().execute(f"SELECT COUNT(*) FROM {table} WHERE {where}").fetchone()[0]


PAGES = [("text_layer", ("A",), "00"), ("text_layer", ("B",), "11"), ("none", (), None)]


def test_erasing_one_page_memory_removes_only_that_page(tmp_path):
    root, db, store = make_root(tmp_path)
    sha, rel = add_document(root, store, db, "d1", pages=PAGES)
    receipt = _erase(db, root, ["f-d1-1"])
    assert receipt.all_erased
    assert [p["page_no"] for p in store.get_pages("d1")] == [2, 3]
    assert _count(store, "media_items", "document_id = 'd1'") == 2
    assert _count(store, "media_vector_rows") == 2
    assert store.get_document("d1") is not None
    assert (files.media_root(root) / rel).exists()


def test_erasing_every_page_memory_removes_the_document_and_its_file(tmp_path):
    root, db, store = make_root(tmp_path)
    sha, rel = add_document(root, store, db, "d1", pages=PAGES)
    receipt = _erase(db, root, ["f-d1-1", "f-d1-2"])
    assert receipt.all_erased and receipt.proofs[0].erased
    assert store.get_document("d1") is None
    assert _count(store, "doc_pages") == 0 and _count(store, "media_items") == 0
    assert _count(store, "media_vector_rows") == 0
    assert not (files.media_root(root) / rel).exists()


def test_the_file_stays_while_another_document_uses_it(tmp_path):
    root, db, store = make_root(tmp_path)
    data = b"%PDF-same"
    _, rel = add_document(root, store, db, "d1", pages=[("text_layer", ("A",), None)], data=data)
    add_document(root, store, db, "d2", profile="p2", pages=[("text_layer", ("A",), None)], data=data)
    _erase(db, root, ["f-d1-1"])
    assert store.get_document("d1") is None and store.get_document("d2") is not None
    assert (files.media_root(root) / rel).exists()
    _erase(db, root, ["f-d2-1"], profile="p2", op="op2")
    assert not (files.media_root(root) / rel).exists()


def test_a_missing_file_is_fine_and_other_profiles_are_untouched(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=[("text_layer", ("A",), None)], with_file=False)
    add_document(root, store, db, "d2", profile="p2", pages=[("text_layer", ("A",), None)])
    assert _erase(db, root, ["f-d1-1"]).all_erased
    assert store.get_document("d1") is None and store.get_document("d2") is not None


def test_the_cached_index_is_dropped(tmp_path, monkeypatch):
    from superlocalmemory.cache.keys import CacheKey
    from superlocalmemory.cache.sqlite_store import SqliteDeriveCache

    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=[("text_layer", ("A",), None)])
    cache = SqliteDeriveCache(root / "derive_cache.db")
    key = CacheKey("a" * 64, "doc.index", "1")
    cache.put(key, b'{"documents": [{"title": "Secret"}]}', kind="json")
    _erase(db, root, ["f-d1-1"])
    assert cache.get(key) is None


class Eraser:
    def __init__(self, db, root, complete=True):
        self.db, self.root, self.complete, self.calls = db, root, complete, []

    def __call__(self, profile_id, fact_ids, subject):
        self.calls.append((profile_id, sorted(fact_ids), subject))
        if self.complete:
            _erase(self.db, self.root, fact_ids, profile_id, op=f"hard{len(self.calls)}")
        return {"erasure_complete": 1 if self.complete else 0}


def test_hard_removal_erases_every_memory_then_the_document(tmp_path):
    root, db, store = make_root(tmp_path)
    _, rel = add_document(root, store, db, "d1", pages=PAGES)
    store.update_document("d1", memory_id="m-doc", fact_ids_json=json.dumps(["f-doc"]))
    db.fact("f-doc", "m-doc", "p1", ("T",))
    eraser = Eraser(db, root)
    assert remove_document("d1", "p1", hard=True, eraser=eraser, store=store) is True
    assert eraser.calls == [("p1", ["f-d1-1", "f-d1-2", "f-doc"], "d1")]
    assert store.get_document("d1") is None and _count(store, "media_items") == 0
    assert not (files.media_root(root) / rel).exists()


def test_hard_removal_with_only_blank_pages_still_removes_the_document(tmp_path):
    root, db, store = make_root(tmp_path)
    _, rel = add_document(root, store, db, "d1", pages=[("none", (), None)])
    assert remove_document("d1", "p1", hard=True, eraser=Eraser(db, root), store=store) is True
    assert store.get_document("d1") is None and not (files.media_root(root) / rel).exists()


def test_hard_removal_checks_profile_and_completeness(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=PAGES)
    assert remove_document("d1", "p2", hard=True, eraser=Eraser(db, root), store=store) is False
    assert remove_document("nope", "p1", hard=True, eraser=Eraser(db, root), store=store) is False
    bad = Eraser(db, root, complete=False)
    assert remove_document("d1", "p1", hard=True, eraser=bad, store=store) is False
    assert store.get_document("d1") is not None             # nothing is dropped when the erasure was incomplete


def test_hard_removal_needs_an_eraser(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=PAGES)
    with pytest.raises(ValueError):
        remove_document("d1", "p1", hard=True, store=store)


def test_a_tombstoned_document_can_still_be_erased(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=PAGES)
    store.tombstone_document("d1")
    assert remove_document("d1", "p1", hard=True, eraser=Eraser(db, root), store=store) is True
    assert store.get_document("d1") is None
