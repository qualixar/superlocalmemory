"""A folder copy never owns a memory or document the user saved another way."""

from __future__ import annotations

import json
import os
import sqlite3
from types import SimpleNamespace

from superlocalmemory.sources.store import SourceStore

PDF = b"%PDF-1.4\n" + b"0" * 100
PNG = b"\x89PNG\r\n\x1a\n" + b"0" * 100


def bump(path, seconds=1):
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + seconds * 10**9))


def user_memory(env):
    """The user's own picture memory: one fact the writer can archive."""
    db = sqlite3.connect(":memory:", check_same_thread=False)
    db.row_factory = sqlite3.Row
    db.execute("CREATE TABLE atomic_facts(fact_id, memory_id, lifecycle DEFAULT 'active')")
    db.execute("INSERT INTO atomic_facts(fact_id, memory_id) VALUES ('user-fact', 'user-mem')")
    env.runtime._db = db
    plain = env.runtime.archive_fact

    def archive(profile_id, fact_id, *, idempotency_key=None):
        db.execute("UPDATE atomic_facts SET lifecycle = 'archived' WHERE fact_id = ?", (fact_id,))
        return plain(profile_id, fact_id, idempotency_key=idempotency_key)

    env.runtime.archive_fact = archive
    return db


def test_the_users_picture_memory_survives_delete_edit_and_purge(env, monkeypatch):
    user_memory(env)
    monkeypatch.setattr(
        "superlocalmemory.media.ingest.remember_media",
        lambda inp, **kw: SimpleNamespace(status="duplicate", media_id="md1", memory_id="user-mem", reason=""))
    path = env.write("pic.png", PNG)
    env.write("keep.canvas", "{}")  # the folder is never empty
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.files(sid)["pic.png"]
    assert row["reason"] == "shared" and row["media_id"] is None
    assert json.loads(row["memory_ids_json"]) == [{"shared_m": "user-mem"}]

    path.write_bytes(PNG + b"edited")  # editing the folder copy
    bump(path)
    env.scan(sid)
    path.unlink()  # deleting it
    env.scan(sid)
    env.host.purge_after_s = -1.0  # and the grace being over
    env.scan(sid)
    assert env.runtime.archived == [] and env.erased == []
    assert "pic.png" not in env.files(sid)


def test_a_shared_document_is_never_removed_by_the_folder_copy(env, monkeypatch):
    monkeypatch.setattr("superlocalmemory.documents.submit_document",
                        lambda inp, **kw: SimpleNamespace(status="duplicate", document_id="d9",
                                                          job_id=None, reason=""))
    removed = []
    monkeypatch.setattr("superlocalmemory.documents.remove_document",
                        lambda *a, **k: removed.append((a, k)) or True)
    path = env.write("a.pdf", PDF)
    env.write("keep.canvas", "{}")  # the folder is never empty
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.files(sid)["a.pdf"]
    assert row["document_id"] is None
    assert json.loads(row["memory_ids_json"]) == [{"shared_doc": "d9"}]
    path.unlink()
    env.scan(sid)
    env.host.purge_after_s = -1.0
    env.scan(sid)
    assert removed == [] and env.erased == []


def test_identical_pdf_copies_the_second_becomes_the_owner_when_the_first_goes(env, monkeypatch):
    docs: dict[bytes, str] = {}  # the file's bytes -> live document id
    saved = []

    def submit(inp, **kw):
        sha = inp.data
        if sha in docs:
            return SimpleNamespace(status="duplicate", document_id=docs[sha], job_id=None, reason="")
        docs[sha] = f"d{len(saved) + 1}"
        saved.append((inp.file_name, docs[sha]))
        return SimpleNamespace(status="processing", document_id=docs[sha], job_id="j", reason="")

    def remove(document_id, profile_id, **kw):
        for key in [k for k, v in docs.items() if v == document_id]:
            del docs[key]
        return True

    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)
    monkeypatch.setattr("superlocalmemory.documents.remove_document", remove)
    first = env.write("a.pdf", PDF)
    env.write("b.pdf", PDF)
    env.write("keep.canvas", "{}")  # the folder is never empty
    sid = env.add_and_confirm()
    env.scan(sid)
    files = env.files(sid)
    assert files["a.pdf"]["document_id"] == "d1" and files["b.pdf"]["reason"] == "shared"

    media = env.store()
    SourceStore(media).cancel_scans(sid)  # nothing waiting: only the hand-over may queue a scan
    media.close()
    first.unlink()
    env.scan(sid)  # a.pdf is tombstoned and its document hidden; b.pdf is queued to take over
    assert env.files(sid)["b.pdf"]["state"] == "pending"
    media = env.store()
    queued = media.list_jobs("default", ["queued"])
    media.close()
    assert len(queued) == 1 and json.loads(queued[0]["payload_json"])["source_id"] == sid
    env.scan(sid)
    b = env.files(sid)["b.pdf"]
    assert b["state"] == "indexed" and b["reason"] is None and b["document_id"] == "d2"
    assert docs == {PDF: "d2"} and saved == [("a.pdf", "d1"), ("b.pdf", "d2")]


def test_a_changed_owner_hands_ownership_to_its_identical_copy(env, monkeypatch):
    docs: dict[bytes, str] = {}
    made = []

    def submit(inp, **kw):
        data = inp.data
        if data in docs:
            return SimpleNamespace(status="duplicate", document_id=docs[data], job_id=None, reason="")
        made.append(1)
        docs[data] = f"d{len(made)}"
        return SimpleNamespace(status="processing", document_id=docs[data], job_id="j", reason="")

    def remove(document_id, profile_id, **kw):
        for key in [k for k, v in docs.items() if v == document_id]:
            del docs[key]
        return True

    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)
    monkeypatch.setattr("superlocalmemory.documents.remove_document", remove)
    owner = env.write("a.pdf", PDF)
    env.write("b.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    owner.write_bytes(PDF + b"changed")
    bump(owner)
    env.scan(sid)
    env.scan(sid)
    files = env.files(sid)
    assert files["b.pdf"]["state"] == "indexed" and files["b.pdf"]["document_id"]
    assert files["b.pdf"]["reason"] is None
    assert files["a.pdf"]["document_id"] and files["a.pdf"]["document_id"] != files["b.pdf"]["document_id"]
