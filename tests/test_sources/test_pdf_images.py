"""PDFs and pictures go through the document and picture paths, tagged as coming from a folder."""

from __future__ import annotations

from types import SimpleNamespace

import pytest

PDF = b"%PDF-1.4\n" + b"0" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def test_pdf_is_submitted_as_a_folder_document(env, monkeypatch):
    calls = []

    def fake(inp, **kw):
        calls.append((inp, kw))
        return SimpleNamespace(status="processing", document_id="d1", job_id="j", reason="")

    monkeypatch.setattr("superlocalmemory.documents.submit_document", fake)
    env.write("papers/a.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    [(inp, kw)] = calls
    assert inp.path == env.root.resolve() / "papers/a.pdf" and kw["actor_id"] == "test-actor"
    assert kw["folder"] == {"type": "folder", "source_id": sid, "relpath": "papers/a.pdf",
                            "version": kw["folder"]["version"], "origin": "folder"}
    row = env.files(sid)["papers/a.pdf"]
    assert row["state"] == "indexed" and row["document_id"] == "d1" and row["reason"] is None


def test_pdf_that_is_a_duplicate_of_another_document_is_marked_shared(env, monkeypatch):
    monkeypatch.setattr("superlocalmemory.documents.submit_document",
                        lambda inp, **kw: SimpleNamespace(status="duplicate", document_id="d9",
                                                          job_id=None, reason=""))
    env.write("a.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["a.pdf"]["reason"] == "shared"
    (env.root / "a.pdf").unlink()
    removed = []
    monkeypatch.setattr("superlocalmemory.documents.remove_document",
                        lambda *a, **k: removed.append(a))
    env.scan(sid)
    assert removed == []  # a document someone else saved is left alone


def test_deleted_pdf_is_soft_removed(env, monkeypatch):
    monkeypatch.setattr("superlocalmemory.documents.submit_document",
                        lambda inp, **kw: SimpleNamespace(status="processing", document_id="d1",
                                                          job_id="j", reason=""))
    removed = []
    monkeypatch.setattr("superlocalmemory.documents.remove_document",
                        lambda *a, **k: removed.append((a, k.get("hard", False))))
    path = env.write("a.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    path.unlink()
    env.scan(sid)
    assert removed and removed[0][0][0] == "d1" and removed[0][1] is False
    assert env.files(sid)["a.pdf"]["state"] == "tombstoned"


def test_refused_pdf_is_skipped_with_the_reason(env, monkeypatch):
    monkeypatch.setattr("superlocalmemory.documents.submit_document",
                        lambda inp, **kw: SimpleNamespace(status="refused", document_id=None, job_id=None,
                                                          reason="Images and documents are turned off."))
    env.write("a.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.files(sid)["a.pdf"]
    assert row["state"] == "skipped" and "turned off" in row["reason"]


def test_picture_is_remembered_as_folder_media(env, monkeypatch):
    calls = []

    def fake(inp, **kw):
        calls.append(kw)
        return SimpleNamespace(status="stored", media_id="i1", memory_id="mem-i1")

    monkeypatch.setattr("superlocalmemory.media.ingest.remember_media", fake)
    env.write("pics/a.png", PNG)
    sid = env.add_and_confirm()
    env.scan(sid)
    [kw] = calls
    assert kw["folder"]["origin"] == "folder" and kw["folder"]["relpath"] == "pics/a.png"
    assert kw["runtime"] is env.runtime and kw["profile_id"] == "default"
    row = env.files(sid)["pics/a.png"]
    assert row["state"] == "indexed" and row["media_id"] == "i1"


def test_warming_picture_is_retried_next_pass(env, monkeypatch):
    monkeypatch.setattr("superlocalmemory.media.ingest.remember_media",
                        lambda inp, **kw: SimpleNamespace(status="warming", media_id=None, memory_id=None,
                                                          reason=""))
    env.write("a.png", PNG)
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert stats.deferred == 1 and "a.png" not in env.files(sid)


def test_canvas_is_recorded_but_not_read(env):
    env.write("board.canvas", '{"nodes": []}')
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.files(sid)["board.canvas"]["state"] == "skipped" and env.runtime.saved == []
