"""Fable audit, package FB: a removed document hides every memory its pages saved, recorded or not."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

import pytest

from superlocalmemory.documents.status import remove_document
from superlocalmemory.documents.submit import submit_document
from superlocalmemory.media import open_media_store
from tests.test_documents.support import FakeClient, Runtime, fake_script, make_service, pdf_input

CFG = SimpleNamespace(pii_redaction=False)


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm"))
    s = open_media_store(create=True, data_root=tmp_path / "slm")
    yield s
    s.close()


class _Db:
    """The writer's database as ``DatabaseManager`` answers: ``execute`` returns a list of rows."""

    def __init__(self):
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute(
            "CREATE TABLE ingestion_operations (profile_id TEXT, source_type TEXT, idempotency_key TEXT,"
            " state TEXT, queryable_fact_ids_json TEXT DEFAULT '[]', final_fact_ids_json TEXT DEFAULT '[]')")

    def execute(self, sql, params=()):
        return self.conn.execute(sql, params).fetchall()

    def commit_save(self, key, facts, source_type="document"):
        import json

        self.conn.execute("INSERT INTO ingestion_operations VALUES ('p1', ?, ?, 'queryable', ?, '[]')",
                          (source_type, key, json.dumps(facts)))


def _runtime():
    runtime = Runtime()
    runtime._db = _Db()
    return runtime


def _submitted(store):
    r = submit_document(pdf_input(("one", "two")), profile_id="p1", actor_id="a", config=CFG, store=store)
    assert r.status == "processing"
    return r


def test_a_cancelled_job_hides_a_page_save_that_committed_after_its_page_was_deferred(store, tmp_path):
    """F-3 (documents): the page was never recorded, so removal could not list its memory."""
    r = _submitted(store)
    runtime = _runtime()
    runtime._db.commit_save(f"doc:{r.document_id}:1:1", ["late-fact"])
    runtime._db.commit_save("doc:someone-else:1:1", ["not-mine"])
    store.tombstone_document(r.document_id)  # removed while the job waited in the queue
    make_service(store, runtime, FakeClient(), fake_script(tmp_path, ["a" * 40, "b" * 40])).process_next()
    assert store.get_job(r.job_id)["state"] == "cancelled"
    assert runtime.archived == [("p1", "late-fact")]


def test_removing_a_document_hides_a_committed_page_save_that_has_no_page_row(store):
    r = _submitted(store)
    runtime = _runtime()
    runtime._db.commit_save(f"doc:{r.document_id}:2:1", ["unrecorded"])
    assert remove_document(r.document_id, "p1", runtime=runtime, store=store) is True
    assert ("p1", "unrecorded") in runtime.archived


def test_a_hard_removal_erases_a_committed_page_save_that_has_no_page_row(store):
    r = _submitted(store)
    runtime = _runtime()
    runtime._db.commit_save(f"doc:{r.document_id}:2:1", ["unrecorded"])
    erased = []

    def eraser(profile_id, facts, subject):
        erased.append(list(facts))
        return {"erasure_complete": 1}

    assert remove_document(r.document_id, "p1", hard=True, eraser=eraser, runtime=runtime, store=store) is True
    assert erased == [["unrecorded"]]
