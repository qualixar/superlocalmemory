"""Housekeeping: rows with no memory, files with no row, memories whose image row is gone."""

from __future__ import annotations

from superlocalmemory.media import files, media_db_path
from superlocalmemory.media.gc import gc

from tests.test_media._erase_support import add_image, make_root


def _setup(tmp_path, file_age=3600):
    root, db, store = make_root(tmp_path)
    db.add_memory("m1", fact_id="f1")
    _, rel_ok = add_image(root, store, media_id="a" * 32, memory_id="m1", data=b"ok")
    _, rel_orph = add_image(root, store, media_id="b" * 32, memory_id="deleted", data=b"orph")
    stray = files.media_root(root) / "ab" / ("ab" + "0" * 62 + ".png")
    stray.parent.mkdir(parents=True, exist_ok=True)
    stray.write_bytes(b"x")
    import os, time
    old = time.time() - file_age
    os.utime(stray, (old, old))
    db.add_memory("m9", media_id="c" * 32)  # media-sourced memory, no row
    return root, db, store, rel_ok, rel_orph, stray


def test_dry_run_reports_and_changes_nothing(tmp_path):
    root, db, store, rel_ok, rel_orph, stray = _setup(tmp_path)
    rep = gc("p1", dry_run=True, data_root=root)
    assert rep.dry_run and rep.rows_without_memory == ["b" * 32]
    assert rep.files_without_row == [stray.name] and rep.memories_without_row == ["m9"]
    assert len(store.list_items("p1")) == 2 and stray.exists()
    assert (files.media_root(root) / rel_orph).exists()
    store.close()


def test_real_run_fixes_rows_and_files_but_never_memories(tmp_path):
    root, db, store, rel_ok, rel_orph, stray = _setup(tmp_path)
    rep = gc("p1", dry_run=False, data_root=root)
    assert not rep.dry_run
    assert [r["media_id"] for r in store.list_items("p1")] == ["a" * 32]
    assert not stray.exists()
    assert not (files.media_root(root) / rel_orph).exists()
    assert (files.media_root(root) / rel_ok).exists()
    assert db.execute("SELECT 1 FROM memories WHERE memory_id='m9'")  # user memory is kept
    assert rep.memories_without_row == ["m9"]
    store.close()


def test_young_files_are_skipped(tmp_path):
    root, db, store, *_rest, stray = _setup(tmp_path, file_age=5)
    rep = gc("p1", dry_run=False, data_root=root)
    assert stray.exists() and rep.files_without_row == [] and rep.files_skipped_young == 1
    store.close()


def test_no_media_db_is_a_no_op(tmp_path):
    rep = gc("p1", dry_run=False, data_root=tmp_path)
    assert rep.rows_without_memory == [] and not media_db_path(tmp_path).exists()


def test_a_real_run_keeps_the_stored_pdf_of_a_document(tmp_path):
    import hashlib
    import os
    import time

    root, _db, store = make_root(tmp_path)
    data = b"%PDF-1.4 kept"
    sha = hashlib.sha256(data).hexdigest()
    scratch = tmp_path / "upload.pdf"
    scratch.write_bytes(data)
    relpath = files.place_original(root, scratch, "p1", "pdf")
    old = time.time() - 3600
    os.utime(files.media_root(root) / relpath, (old, old))
    store.insert_document(document_id="d" * 32, profile_id="p1", sha256=sha, title="kept.pdf",
                          mime="application/pdf", bytes=len(data), source_relpath=relpath)
    rep = gc("p1", dry_run=False, data_root=root)
    assert rep.files_without_row == []
    assert (files.media_root(root) / relpath).exists()
    store.close()


def test_operational_files_under_media_are_never_collected(tmp_path):
    """Audit round 2 (CX7): uploads.db lives in media/ and is not an original; gc must never touch it."""
    import os, time
    root, db, store, *_rest, stray = _setup(tmp_path)
    base = files.media_root(root)
    kept = [base / "uploads.db", base / "uploads.db-wal", base / "uploads.db-shm",
            base / "ab" / "notes.txt", base / "zz" / ("ab" + "0" * 62 + ".png")]
    old = time.time() - 3600
    for path in kept:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"keep")
        os.utime(path, (old, old))
    rep = gc("p1", dry_run=False, data_root=root)
    assert rep.files_without_row == [stray.name]
    assert all(p.exists() for p in kept)
    assert not stray.exists()
    store.close()
