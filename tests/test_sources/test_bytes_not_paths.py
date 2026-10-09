"""Pictures and PDFs are saved from the bytes that were hashed, never re-opened by path."""

from __future__ import annotations

import hashlib
import os

from tests.test_sources.test_real_picture import pics, png  # noqa: F401


def _swap_after_hashing(monkeypatch, path, outside):
    import superlocalmemory.sources.reconcile as rc

    real = rc._hash_all

    def swapped(p, entries):
        out = real(p, entries)  # the window between the hash and the save
        os.unlink(path)
        os.symlink(outside, path)
        return out

    monkeypatch.setattr(rc, "_hash_all", swapped)


def test_a_picture_swapped_for_a_link_after_the_hash_is_never_stored(pics, tmp_path, monkeypatch):
    env = pics
    outside = tmp_path / "outside.png"
    outside.write_bytes(png("secret-outside"))
    path = env.write("p.png", png("a"))
    env.write("keep.canvas", "{}")
    _swap_after_hashing(monkeypatch, path, outside)
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    from superlocalmemory.media import open_media_store

    store = open_media_store(data_root=env.data)
    try:
        shas = [r[0] for r in store._read().execute("SELECT source_sha256 FROM media_items")]
    finally:
        store.close()
    assert hashlib.sha256(png("secret-outside")).hexdigest() not in shas
    assert stats.errors == 1 and env.files(sid)["p.png"]["state"] == "error"


def test_a_pdf_is_handed_over_as_bytes_and_a_swapped_link_is_refused(env, tmp_path, monkeypatch):
    seen = []

    def fake(inp, **kw):
        seen.append(inp)
        raise AssertionError("must not be reached for a swapped file")

    monkeypatch.setattr("superlocalmemory.documents.submit_document", fake)
    outside = tmp_path / "outside.pdf"
    outside.write_bytes(b"%PDF-1.4\nsecret")
    path = env.write("a.pdf", b"%PDF-1.4\n" + b"0" * 50)
    _swap_after_hashing(monkeypatch, path, outside)
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert seen == [] and stats.errors == 1
