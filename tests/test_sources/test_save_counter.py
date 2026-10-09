"""A file edited back to earlier content is saved again: every save of a path has its own key."""

from __future__ import annotations

import json
import os
from types import SimpleNamespace

from tests.test_sources.test_shared_ownership import PDF, PNG, bump, user_memory


def live(env, sid, relpath):
    entries = json.loads(env.files(sid)[relpath]["memory_ids_json"])
    return [e for e in entries if e.get("m") and not e.get("sup")]


def rewrite(path, data, seconds):
    path.write_bytes(data if isinstance(data, bytes) else data.encode())
    st = os.stat(path)
    os.utime(path, ns=(st.st_atime_ns, 1_000_000_000_000_000_000 + seconds * 10**9))


def test_text_edited_back_to_earlier_content_is_visible(env):
    path = env.write("n.md", "version A of the note")
    sid = env.add_and_confirm()
    env.scan(sid)
    rewrite(path, "version B, quite different", 1)
    env.scan(sid)
    rewrite(path, "version A of the note", 2)
    env.scan(sid)
    assert env.runtime.contents() == ["version A of the note", "version B, quite different",
                                      "version A of the note"]
    assert len({r["key"] for r in env.runtime.saved}) == 3
    assert [e["m"] for e in live(env, sid, "n.md")] == ["m3"]
    assert env.runtime.archived == ["f1", "f2"]


def test_pdf_edited_back_to_earlier_content_is_saved_again(env, monkeypatch):
    docs: dict[bytes, str] = {}
    keys = []

    def submit(inp, **kw):
        data = inp.path.read_bytes()
        keys.append(kw["idempotency_key"])
        if data in docs:
            return SimpleNamespace(status="duplicate", document_id=docs[data], job_id=None, reason="")
        docs[data] = f"d{len(keys)}"
        return SimpleNamespace(status="processing", document_id=docs[data], job_id="j", reason="")

    def remove(document_id, profile_id, **kw):
        for key in [k for k, v in docs.items() if v == document_id]:
            del docs[key]
        return True

    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)
    monkeypatch.setattr("superlocalmemory.documents.remove_document", remove)
    path = env.write("a.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    rewrite(path, PDF + b"B", 1)
    env.scan(sid)
    rewrite(path, PDF, 2)
    env.scan(sid)
    assert len(set(keys)) == 3 and [k.split(":")[-2] for k in keys] == ["1", "2", "3"]
    row = env.files(sid)["a.pdf"]
    assert row["document_id"] == "d3" and docs[PDF] == "d3"


def test_picture_edited_back_to_earlier_content_is_saved_again(env, monkeypatch):
    db = user_memory(env)
    known: dict[bytes, str] = {}
    made = []

    def remember(inp, **kw):
        data = inp.path.read_bytes()
        made.append(kw["idempotency_key"])
        if data in known and not known[data].startswith("again:"):
            old = known[data]  # the library still knows these bytes and hands back the old, archived save
            known[data] = "again:" + old
            return SimpleNamespace(status="duplicate", media_id="md-old", memory_id=old, reason="")
        mem = f"mem{len(made)}"
        known[data] = mem
        db.execute("INSERT INTO atomic_facts(fact_id, memory_id) VALUES (?, ?)", (f"f-{mem}", mem))
        return SimpleNamespace(status="stored", media_id=f"md{len(made)}", memory_id=mem, reason="")

    monkeypatch.setattr("superlocalmemory.media.ingest.remember_media", remember)
    path = env.write("p.png", PNG)
    sid = env.add_and_confirm()
    env.scan(sid)
    rewrite(path, PNG + b"B", 1)
    env.scan(sid)
    rewrite(path, PNG, 2)
    env.scan(sid)
    [now] = live(env, sid, "p.png")
    archived = {r[0] for r in db.execute("SELECT memory_id FROM atomic_facts WHERE lifecycle = 'archived'")}
    assert now["m"] not in archived and {"mem1", "mem2"} <= archived
    assert env.files(sid)["p.png"]["state"] == "indexed"


def test_file_restored_after_a_purge_is_saved_fresh(env):
    path = env.write("a.md", "same words")
    sid = env.add_and_confirm()
    env.scan(sid)
    path.unlink()
    env.scan(sid)
    env.host.purge_after_s = -1.0
    env.scan(sid)
    assert "a.md" not in env.files(sid)
    env.host.purge_after_s = 7 * 86400.0
    env.write("a.md", "same words")
    env.scan(sid)
    first, second = env.runtime.saved
    assert first["key"] != second["key"]
    assert [e["m"] for e in live(env, sid, "a.md")] == [second["mid"]]


def test_a_file_that_comes_back_after_delete_counts_on(env):
    for n in range(3):
        path = env.write("a.md", "same words")
        sid = env.add_and_confirm() if n == 0 else sid
        env.scan(sid)
        path.unlink()
        env.scan(sid)
    assert [r["key"].split(":")[-2] for r in env.runtime.saved] == ["1", "2", "3"]
