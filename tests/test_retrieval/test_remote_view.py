# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""What a remote caller sees of pictures, document pages and connected folders."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from superlocalmemory.media.store import MediaStore
from superlocalmemory.retrieval import remote_view, visibility

from ._media_support import MemoryDb

_SHA = "ab" * 32
GOOD, DIRTY, ORPHAN = "1" * 32, "2" * 32, "3" * 32
DOC_OK, DOC_DIRTY = "d" * 32, "e" * 32


def _fact(memory_id: str) -> SimpleNamespace:
    return SimpleNamespace(fact_id="f-" + memory_id, memory_id=memory_id)


def _world(tmp_path: Path, *, with_media_db: bool = True) -> tuple[MemoryDb, dict]:
    db = MemoryDb()
    db.db_path = tmp_path / "memory.db"
    sources = {
        "pic-good": {"type": "media", "media_id": GOOD, "origin": "tool"},
        "pic-dirty": {"type": "media", "media_id": DIRTY, "origin": "tool"},
        "pic-orphan": {"type": "media", "media_id": ORPHAN, "origin": "tool"},
        "page-good": {"type": "document", "document_id": DOC_OK, "page": 1},
        "page-dirty": {"type": "document", "document_id": DOC_DIRTY, "page": 1},
        "doc-good": {"type": "document", "document_id": DOC_OK},
        "doc-dirty": {"type": "document", "document_id": DOC_DIRTY},
        "folder-note": {"type": "folder", "source_id": "s1", "relpath": "a.md"},
        "folder-pic": {"type": "folder", "source_id": "s1", "media_id": GOOD, "relpath": "p.jpg"},
        "plain": None,
    }
    for memory_id, source in sources.items():
        db.add(memory_id, "f-" + memory_id, source)
    if with_media_db:
        store = MediaStore(tmp_path / "media.db")
        for media_id, ok in ((GOOD, 1), (DIRTY, 0)):
            store.insert_item(media_id=media_id, profile_id="default", kind="image",
                              source_sha256=_SHA, mime="image/png", bytes=1, origin="tool",
                              remote_ok=ok)
        for doc, ok in ((DOC_OK, 1), (DOC_DIRTY, 0)):
            store.insert_document(document_id=doc, profile_id="default", sha256=_SHA, title="t",
                                  mime="application/pdf", bytes=1, source_relpath="")
            store.insert_item(media_id=("5" if ok else "6") * 32, profile_id="default", kind="page",
                              source_sha256=_SHA, mime="image/png", bytes=0, origin="document",
                              document_id=doc, page_no=1, remote_ok=ok)
            store.put_page(doc, 1, media_id=None, memory_ids=[], fact_ids=[], text_origin="ocr",
                           char_count=5)
        store.close()
    return db, sources


def _shown(db: MemoryDb, view: str, sources: dict) -> set[str]:
    ctx = remote_view.context_for(view, db, "default")
    facts = {"f-" + m: _fact(m) for m in sources}
    with visibility.use(ctx):
        return {f.memory_id for f in visibility.drop_hidden_facts(facts, db).values()}


def test_a_local_caller_has_no_context(tmp_path: Path) -> None:
    db, _ = _world(tmp_path)
    assert remote_view.context_for("", db, "default") is None
    assert remote_view.parse_view("anything else") == ""
    assert remote_view.parse_view(" Remote ") == remote_view.REMOTE


def test_without_the_media_permission_only_plain_memories_show(tmp_path: Path) -> None:
    db, sources = _world(tmp_path)
    assert _shown(db, remote_view.REMOTE, sources) == {"plain"}


def test_with_the_permission_only_vetted_pictures_and_pages_show(tmp_path: Path) -> None:
    db, sources = _world(tmp_path)
    assert _shown(db, remote_view.REMOTE_MEDIA, sources) == {
        "plain", "pic-good", "page-good", "doc-good"}


def test_folders_stay_hidden_even_with_the_permission(tmp_path: Path) -> None:
    db, sources = _world(tmp_path)
    shown = _shown(db, remote_view.REMOTE_MEDIA, sources)
    assert not shown & {"folder-note", "folder-pic"}


def test_a_picture_with_no_record_is_not_shown(tmp_path: Path) -> None:
    db, sources = _world(tmp_path)
    assert "pic-orphan" not in _shown(db, remote_view.REMOTE_MEDIA, sources)


def test_without_media_db_no_picture_or_page_shows(tmp_path: Path) -> None:
    db, sources = _world(tmp_path, with_media_db=False)
    assert _shown(db, remote_view.REMOTE_MEDIA, sources) == {"plain"}


def test_a_lookup_costs_a_fixed_number_of_queries_not_one_per_fact(tmp_path: Path) -> None:
    db, _ = _world(tmp_path)
    for n in range(1200):
        db.add(f"bulk-{n}", f"f-bulk-{n}", {"type": "media", "media_id": f"{n:032x}"})
    facts = {f"f-bulk-{n}": _fact(f"bulk-{n}") for n in range(1200)}
    db.queries.clear()
    with visibility.use(remote_view.context_for(remote_view.REMOTE_MEDIA, db, "default")):
        assert visibility.drop_hidden_facts(facts, db) == {}
    assert len(db.queries) <= 3


def test_a_local_request_changes_nothing(tmp_path: Path) -> None:
    db, sources = _world(tmp_path)
    facts = {"f-" + m: _fact(m) for m in sources}
    db.queries.clear()
    assert visibility.is_empty()
    assert visibility.drop_hidden_facts(facts, db) is facts
    assert db.queries == []


@pytest.mark.parametrize("view", [remote_view.REMOTE, remote_view.REMOTE_MEDIA])
def test_hidden_among_matches_the_fact_filter(tmp_path: Path, view: str) -> None:
    db, sources = _world(tmp_path)
    ids = ["f-" + m for m in sources]
    with visibility.use(remote_view.context_for(view, db, "default")):
        hidden = visibility.hidden_among(db, "default", ids)
    assert {i[2:] for i in ids} - {i[2:] for i in hidden} == _shown(db, view, sources)
