"""Fable audit, package FB: folder bookkeeping that must not lie about what is hidden or saved."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from superlocalmemory.sources import ingest
from tests.test_sources.conftest import ListDb


def _runtime(*rows):
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.execute("CREATE TABLE atomic_facts(fact_id, memory_id, lifecycle)")
    conn.executemany("INSERT INTO atomic_facts VALUES (?, ?, ?)", rows)
    return SimpleNamespace(_db=ListDb(conn))


def test_any_archived_sees_archived_memories_in_the_list_the_database_returns():
    """F-2: the database hands back a list; reading it as a cursor made this always False."""
    runtime = _runtime(("f1", "m1", "archived"), ("f2", "m2", "active"))
    assert ingest.any_archived(runtime, ["m1"]) is True
    assert ingest.any_archived(runtime, ["m2"]) is False
    assert ingest.any_archived(runtime, ["m1", "m2"]) is True
    assert ingest.any_archived(runtime, []) is False


# -- F-5 / F-6: a removal never reports done while something is still recallable -----------------

import json  # noqa: E402

import pytest  # noqa: E402

from superlocalmemory import sources  # noqa: E402
from superlocalmemory.sources.api import SourceRefused  # noqa: E402
from tests.test_sources.test_shared_ownership import PDF  # noqa: E402

DOC_ID = "d" * 32


def _real_document(env, monkeypatch):
    """A folder PDF whose document is real (one page, one memory) in the throwaway media store."""
    def submit(inp, **kw):
        media = env.store()
        try:
            media.insert_document(document_id=DOC_ID, profile_id="default", sha256="s" * 64, title="T",
                                  mime="application/pdf", bytes=len(inp.data), source_relpath=None)
            media.put_page(DOC_ID, 1, media_id=None, memory_ids=["pm1"], fact_ids=["pf1"],
                           text_origin="text_layer", char_count=5)
        finally:
            media.close()
        return SimpleNamespace(status="processing", document_id=DOC_ID, job_id="j", reason="")

    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)


class _Archive:
    """Makes ``archive_fact`` fail for one fact until ``heal()`` (a writer that is busy, then not)."""

    def __init__(self, env, fact_id):
        self.failing, self._real, self._fact = True, env.runtime.archive_fact, fact_id
        env.runtime.archive_fact = self

    def __call__(self, profile_id, fact, *, idempotency_key=None):
        if self.failing and fact == self._fact:
            raise RuntimeError("writer busy")
        return self._real(profile_id, fact, idempotency_key=idempotency_key)

    def heal(self):
        self.failing = False


def _document_state(env):
    media = env.store()
    try:
        return media.get_document(DOC_ID)["state"]
    finally:
        media.close()


def test_a_second_removal_retries_the_document_whose_hide_failed(env, monkeypatch):
    """F-5: the row was tombstoned although its document was not removed; nothing retried it."""
    _real_document(env, monkeypatch)
    env.write("paper.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    writer = _Archive(env, "pf1")
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid)
    assert refused.value.code == "removal_incomplete" and _document_state(env) != "tombstoned"
    writer.heal()
    sources.remove_source(sid)  # the retry must hide the document, not report a clean removal
    assert _document_state(env) == "tombstoned" and "pf1" in env.runtime.archived
    assert sources.list_sources("default") == []


def test_a_scan_finishes_a_document_hide_that_failed_when_the_file_vanished(env, monkeypatch):
    _real_document(env, monkeypatch)
    env.write("paper.pdf", PDF)
    env.write("keep.md", "stays")
    sid = env.add_and_confirm()
    env.scan(sid)
    writer = _Archive(env, "pf1")
    (env.root / "paper.pdf").unlink()
    assert env.scan(sid).errors == 1 and _document_state(env) != "tombstoned"
    writer.heal()
    assert env.scan(sid).errors == 0
    assert _document_state(env) == "tombstoned" and "pf1" in env.runtime.archived


def test_forget_empty_refuses_when_a_memory_could_not_be_hidden_and_keeps_its_state(env):
    """F-6: hide_file's failures were thrown away and the folder was set active."""
    from tests.test_sources.test_forget_empty import emptied, reason, row

    sid = emptied(env)
    writer = _Archive(env, "f1")
    with pytest.raises(SourceRefused) as refused:
        sources.forget_empty(sid)
    assert refused.value.code == "removal_incomplete"
    assert row(env, sid)["state"] == "offline" and reason(env, sid) == "empty_folder"
    writer.heal()
    sources.forget_empty(sid)  # the retry hides what is left
    assert row(env, sid)["state"] == "active" and sorted(env.runtime.archived) == ["f1", "f2"]
