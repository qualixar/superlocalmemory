"""The document index: rows, top entities, paging, profile scope, caching."""

from __future__ import annotations

from superlocalmemory.cache.sqlite_store import SqliteDeriveCache
from superlocalmemory.documents.index import document_index
from tests.test_documents.index_support import add_document, make_root


def _setup(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", title="Alpha", created="2026-02-01T00:00:00Z",
                 pages=[("text_layer", ("Ann", "Bob", "Cy"), None), ("text_layer", ("Ann", "Bob", "Di", "Ed", "Flo"), None),
                        ("none", (), None)])
    add_document(root, store, db, "d2", title="Beta", created="2026-02-02T00:00:00Z",
                 pages=[("ocr", ("Zed",), None)])
    add_document(root, store, db, "d3", title="Other", profile="p2", pages=[("text_layer", ("Qux",), None)])
    return root, db, store


def test_rows_are_newest_first_with_counts_and_top_five_entities(tmp_path):
    root, db, store = _setup(tmp_path)
    out = document_index("p1", 10, "", store=store, db=db, cache=SqliteDeriveCache(root / "c.db"))
    assert [d["document_id"] for d in out["documents"]] == ["d2", "d1"]
    d1 = out["documents"][1]
    assert d1["title"] == "Alpha" and d1["page_count"] == 3 and d1["pages_empty"] == 1 and d1["state"] == "ready"
    names = [e["name"] for e in d1["entities"]]
    assert len(names) == 5 and names[:2] == ["Ann", "Bob"] and d1["entities"][0]["facts"] == 2
    assert out["next_cursor"] is None
    assert "Qux" not in str(out) and "d3" not in str(out)


def test_paging_with_a_cursor_visits_every_document_once(tmp_path):
    root, db, store = _setup(tmp_path)
    cache = SqliteDeriveCache(root / "c.db")
    first = document_index("p1", 1, "", store=store, db=db, cache=cache)
    assert [d["document_id"] for d in first["documents"]] == ["d2"] and first["next_cursor"]
    second = document_index("p1", 1, first["next_cursor"], store=store, db=db, cache=cache)
    assert [d["document_id"] for d in second["documents"]] == ["d1"] and second["next_cursor"] is None


def test_a_bad_cursor_or_limit_is_handled(tmp_path):
    root, db, store = _setup(tmp_path)
    cache = SqliteDeriveCache(root / "c.db")
    assert len(document_index("p1", 10, "garbage!!", store=store, db=db, cache=cache)["documents"]) == 2
    assert len(document_index("p1", 0, "", store=store, db=db, cache=cache)["documents"]) == 1
    assert len(document_index("p1", 99999, "", store=store, db=db, cache=cache)["documents"]) == 2


def test_removed_documents_are_left_out(tmp_path):
    root, db, store = _setup(tmp_path)
    store.tombstone_document("d2")
    out = document_index("p1", 10, "", store=store, db=db, cache=SqliteDeriveCache(root / "c.db"))
    assert [d["document_id"] for d in out["documents"]] == ["d1"]


def test_second_call_is_served_from_the_cache_until_something_changes(tmp_path):
    root, db, store = _setup(tmp_path)
    cache = SqliteDeriveCache(root / "c.db")
    first = document_index("p1", 10, "", store=store, db=db, cache=cache)
    db.queries = 0
    again = document_index("p1", 10, "", store=store, db=db, cache=cache)
    assert again == first
    assert db.queries <= 1                       # only the cheap change signature, no entity lookups
    db.fact("fnew", "mnew", "p1", ("Late",), created="2026-03-01T00:00:00Z")
    db.queries = 0
    document_index("p1", 10, "", store=store, db=db, cache=cache)
    assert db.queries > 1                        # a new fact changed the signature: recomputed


def test_no_media_store_gives_an_empty_index(tmp_path):
    out = document_index("p1", 10, "", store=None, db=None, cache=SqliteDeriveCache(tmp_path / "c.db"), data_root=tmp_path)
    assert out == {"documents": [], "next_cursor": None}
