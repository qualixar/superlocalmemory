"""Document health checks."""

from __future__ import annotations

from superlocalmemory.documents.lint import document_lint
from tests.test_documents.index_support import add_document, make_root


def _lint(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", title="One", pages=[
        ("text_layer", ("Ann",), "ffffffffffffffff"), ("none", (), None), ("text_layer", (), "0000000000000000")])
    add_document(root, store, db, "d2", title="Two", pages=[("ocr", ("Bob",), "fffffffffffffff0")])
    add_document(root, store, db, "d3", title="Three", profile="p2", pages=[("none", (), "ffffffffffffffff")])
    return root, db, store


def test_pages_with_no_text(tmp_path):
    root, db, store = _lint(tmp_path)
    out = document_lint("p1", store=store, db=db)
    assert out["empty_pages"] == [{"document_id": "d1", "title": "One", "page_no": 2}]


def test_documents_whose_facts_link_to_no_entity(tmp_path):
    root, db, store = _lint(tmp_path)
    out = document_lint("p1", store=store, db=db)
    assert [d["document_id"] for d in out["no_entities"]] == []   # d1 page 1 has an entity
    add_document(root, store, db, "d4", title="Bare", pages=[("text_layer", (), None)])
    out = document_lint("p1", store=store, db=db)
    assert [d["document_id"] for d in out["no_entities"]] == ["d4"]


def test_duplicate_pages_across_documents_by_picture_hash(tmp_path):
    root, db, store = _lint(tmp_path)
    out = document_lint("p1", store=store, db=db)
    assert out["duplicate_pages"] == [{"a": {"document_id": "d1", "page_no": 1, "title": "One"},
                                       "b": {"document_id": "d2", "page_no": 1, "title": "Two"}, "distance": 4}]


def test_pages_further_apart_than_four_bits_are_not_duplicates(tmp_path):
    root, db, store = make_root(tmp_path)
    add_document(root, store, db, "d1", pages=[("text_layer", ("A",), "ffffffffffffffff")])
    add_document(root, store, db, "d2", pages=[("text_layer", ("A",), "ffffffffffffff00")])
    assert document_lint("p1", store=store, db=db)["duplicate_pages"] == []


def test_documents_with_replaced_or_contradicted_facts(tmp_path):
    root, db, store = _lint(tmp_path)
    db.edge("f-d1-1", "other-fact", "supersedes")
    db.edge("x", "f-d2-1", "contradiction")
    db.edge("f-d1-3", "y", "entity")
    out = document_lint("p1", store=store, db=db)
    assert {d["document_id"]: d["edges"] for d in out["contradicted"]} == {"d1": 1, "d2": 1}


def test_other_profiles_are_never_included(tmp_path):
    root, db, store = _lint(tmp_path)
    out = document_lint("p2", store=store, db=db)
    assert [p["document_id"] for p in out["empty_pages"]] == ["d3"] and out["duplicate_pages"] == []


def test_a_missing_edge_table_only_drops_that_check(tmp_path):
    root, db, store = _lint(tmp_path)
    db.execute("DROP TABLE graph_edges")
    out = document_lint("p1", store=store, db=db)
    assert out["contradicted"] is None and len(out["empty_pages"]) == 1
