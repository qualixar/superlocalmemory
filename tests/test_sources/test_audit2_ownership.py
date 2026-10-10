"""Audit round 2: a folder never loses track of what it saved, and removal never lies."""

from __future__ import annotations

import json
import sqlite3
import types

import pytest

from superlocalmemory import sources
from superlocalmemory.sources import ingest
from superlocalmemory.sources.api import SourceRefused

LONG = ("word " * 6000).strip()  # several parts


def _fail_once_on_part(env, part: int):
    real = env.runtime.remember
    calls = {"n": 0}

    def remember(admission, actor, deadline_ms=0, accept_after_ms=0):
        calls["n"] += 1
        if calls["n"] == part:
            raise RuntimeError("writer busy")
        return real(admission, actor, deadline_ms=deadline_ms, accept_after_ms=accept_after_ms)

    env.runtime.remember = remember


def test_a_file_saved_in_part_keeps_the_parts_it_saved_and_hides_them_on_the_next_save(env):
    """CX5: chunk 1 saved, chunk 2 failed; chunk 1 must stay owned by the file row."""
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    _fail_once_on_part(env, 2)
    env.scan(sid)
    row = env.files(sid)["long.txt"]
    assert row["state"] == "error"
    kept = [e["m"] for e in json.loads(row["memory_ids_json"])]
    assert kept == ["m1"]
    env.write("long.txt", LONG + " more")
    env.scan(sid)
    assert env.files(sid)["long.txt"]["state"] == "indexed"
    assert "f1" in env.runtime.archived  # the half-saved copy is hidden, not left recallable


def test_removing_a_folder_after_a_partial_save_hides_the_saved_part(env):
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    _fail_once_on_part(env, 2)
    env.scan(sid)
    sources.remove_source(sid)
    assert "f1" in env.runtime.archived


def test_a_folder_removal_that_could_not_hide_a_memory_refuses_and_can_be_retried(env):
    """CX4: no 'removed' while a memory is still recallable; the retry finishes the job."""
    env.write("a.md", "one")
    env.write("b.md", "two")
    sid = env.add_and_confirm()
    env.scan(sid)
    real = env.runtime.archive_fact
    state = {"failed": False}

    def archive_fact(profile_id, fact_id, *, idempotency_key=None):
        if not state["failed"]:
            state["failed"] = True
            raise RuntimeError("writer busy")
        return real(profile_id, fact_id, idempotency_key=idempotency_key)

    env.runtime.archive_fact = archive_fact
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid)
    assert refused.value.code == "removal_incomplete"
    sources.remove_source(sid)
    assert sorted(env.runtime.archived) == ["f1", "f2"]


def test_a_pdf_skipped_while_media_was_off_is_read_once_media_is_ready(env, monkeypatch):
    """CX6: the skip is temporary; turning media on must not need the file to change."""
    ready = {"on": False}
    monkeypatch.setattr(ingest, "media_ready", lambda: ready["on"])
    handed = []

    def fake_submit(inp, **kw):
        handed.append(kw["idempotency_key"])
        return types.SimpleNamespace(status="processing", document_id="d" * 32, reason="")

    import superlocalmemory.documents as documents

    monkeypatch.setattr(documents, "submit_document", fake_submit)
    env.write("paper.pdf", b"%PDF-1.4 test")
    sid = env.add_and_confirm()
    env.scan(sid)
    row = env.files(sid)["paper.pdf"]
    assert row["state"] == "skipped" and row["reason"] == ingest.MEDIA_NOT_READY and handed == []
    env.scan(sid)
    assert handed == []  # still off: left alone, not re-read every pass
    ready["on"] = True
    env.scan(sid)
    assert len(handed) == 1 and env.files(sid)["paper.pdf"]["document_id"] == "d" * 32


def test_a_queued_folder_save_is_owned_by_its_key_and_resolves_to_its_facts(env, monkeypatch):
    """CX1: an accepted (queued) save has no ids yet; the row keeps its key instead."""
    from superlocalmemory.memory_core import submit as submit_mod

    monkeypatch.setattr(submit_mod, "SETTLE_WAIT_S", 0.0)

    def remember(admission, actor, deadline_ms=0, accept_after_ms=0):
        return types.SimpleNamespace(payload={"status": "accepted", "operation_id": None, "fact_ids": []})

    env.runtime.remember = remember
    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    entries = json.loads(env.files(sid)["a.md"]["memory_ids_json"])
    assert len(entries) == 1 and entries[0]["m"] is None and entries[0]["k"].startswith("src:")
    db = sqlite3.connect(":memory:")
    db.execute("CREATE TABLE ingestion_operations (profile_id TEXT, source_type TEXT, idempotency_key TEXT,"
               " state TEXT, next_retry_at REAL, attempt_count INTEGER, queryable_fact_ids_json TEXT,"
               " final_fact_ids_json TEXT)")
    db.execute("INSERT INTO ingestion_operations VALUES ('default', 'folder', ?, 'complete', 0, 0, '[\"q1\"]',"
               " '[\"q1\", \"q2\"]')",
               (entries[0]["k"],))
    runtime = types.SimpleNamespace(_db=db)
    assert sorted(set(ingest.facts_of_keys(runtime, "default", [entries[0]["k"]]))) == ["q1", "q2"]
