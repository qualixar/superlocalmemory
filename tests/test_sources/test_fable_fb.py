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


# -- F-3: a queued save (key only) is never called hidden or erased before it has committed ------

import sqlite3 as _sqlite3  # noqa: E402

from tests.test_sources.conftest import ListDb as _ListDb  # noqa: E402


class _Operations:
    """The writer's ingestion_operations table, with the real column names."""

    def __init__(self, env):
        self.conn = _sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = _sqlite3.Row
        self.conn.execute(
            "CREATE TABLE ingestion_operations (profile_id TEXT, source_type TEXT, idempotency_key TEXT,"
            " state TEXT, next_retry_at REAL DEFAULT 0, attempt_count INTEGER DEFAULT 0,"
            " queryable_fact_ids_json TEXT DEFAULT '[]', final_fact_ids_json TEXT DEFAULT '[]')")
        env.runtime._db = _ListDb(self.conn)

    def put(self, key, state, facts=(), source_type="folder", attempts=0, retry_at=0.0):
        self.conn.execute(
            "INSERT INTO ingestion_operations(profile_id, source_type, idempotency_key, state,"
            " queryable_fact_ids_json, attempt_count, next_retry_at) VALUES ('default', ?, ?, ?, ?, ?, ?)",
            (source_type, key, state, json.dumps(list(facts)), attempts, retry_at))


def _queued_folder_file(env, monkeypatch):
    """One text file whose save stayed queued past the settle budget: the row holds only its key."""
    from superlocalmemory.memory_core import submit as submit_mod

    monkeypatch.setattr(submit_mod, "SETTLE_WAIT_S", 0.0)
    ops = _Operations(env)
    env.runtime.remember = lambda *a, **k: SimpleNamespace(
        payload={"status": "accepted", "operation_id": None, "fact_ids": []})
    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    [entry] = json.loads(env.files(sid)["a.md"]["memory_ids_json"])
    assert entry["m"] is None and entry["k"]
    return sid, entry["k"], ops


def test_removing_a_folder_with_a_queued_save_is_incomplete_until_the_save_commits(env, monkeypatch):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid)  # no operation row yet: nothing can be hidden
    assert refused.value.code == "removal_incomplete"
    entries = json.loads(env.files(sid)["a.md"]["memory_ids_json"])
    assert entries[0].get("old") and not entries[0].get("sup")
    ops.put(key, "raw")  # admitted but the facts are not written yet: still not committed
    with pytest.raises(SourceRefused):
        sources.remove_source(sid)
    ops.conn.execute("UPDATE ingestion_operations SET state = 'queryable', queryable_fact_ids_json = '[\"q1\"]'")
    sources.remove_source(sid)  # the commit landed: the retry hides it
    assert env.runtime.archived == ["q1"]
    assert sources.list_sources("default") == []


def test_a_purge_of_a_queued_save_is_incomplete_until_the_save_commits(env, monkeypatch):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid, purge=True)
    assert refused.value.code == "erasure_incomplete" and env.erased == []
    ops.put(key, "complete", ["q1"])
    sources.remove_source(sid, purge=True)
    assert env.erased and env.erased[0][1] == ("q1",) and env.files(sid) == {}


def test_a_scan_hides_a_queued_save_that_was_replaced_once_it_commits(env, monkeypatch):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    env.write("a.md", "two, different")
    import os as _os
    st = _os.stat(env.root / "a.md")
    _os.utime(env.root / "a.md", ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
    env.scan(sid)  # the old queued save is replaced; it cannot be hidden yet and stays flagged
    old = [e for e in json.loads(env.files(sid)["a.md"]["memory_ids_json"]) if e.get("k") == key]
    assert old and old[0].get("old") and not old[0].get("sup")
    ops.put(key, "queryable", ["q1"])
    env.scan(sid)
    assert "q1" in env.runtime.archived


# -- CX5 residual: a retry of the same file version re-sends the same keys --------------------

from tests.test_sources.test_audit2_ownership import LONG, _fail_once_on_part  # noqa: E402


def _live(env, sid, relpath):
    entries = json.loads(env.files(sid)[relpath]["memory_ids_json"])
    return [e for e in entries if e.get("m") and not e.get("sup")]


def test_a_retry_after_a_partial_save_re_sends_the_same_keys_and_saves_nothing_twice(env):
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    _fail_once_on_part(env, 2)
    env.scan(sid)
    assert env.files(sid)["long.txt"]["state"] == "error"
    first_attempt = len(env.runtime.saved)
    env.scan(sid)  # same bytes: the retry
    row = env.files(sid)["long.txt"]
    assert row["state"] == "indexed"
    keys = [r["key"] for r in env.runtime.saved]
    assert len(keys) == len(set(keys)) and len({k.rsplit(":", 2)[1] for k in keys}) == 1  # one save number
    assert len(keys) > first_attempt  # the parts that had not been saved yet were saved
    assert env.runtime.archived == []  # nothing was hidden: the re-sent parts are the file's live memories
    assert len(_live(env, sid, "long.txt")) == len(keys)


def test_the_save_number_moves_on_once_the_version_is_saved_or_changes(env):
    path = env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    _fail_once_on_part(env, 2)
    env.scan(sid)
    env.scan(sid)
    path.write_bytes((LONG + " changed").encode())
    import os as _os
    st = _os.stat(path)
    _os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
    env.scan(sid)
    numbers = [r["key"].split(":")[-2] for r in env.runtime.saved]
    assert sorted(set(numbers)) == ["1", "2"]
    first = [r["key"] for r in env.runtime.saved if r["key"].split(":")[-2] == "1"]
    assert len(first) == len(set(first))
    assert len(_live(env, sid, "long.txt")) == numbers.count("2")  # only the new version is live


def test_a_retry_that_fails_again_keeps_the_same_keys(env):
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    real = env.runtime.remember
    state = {"down": True}

    def remember(admission, actor, deadline_ms=0, accept_after_ms=0):
        if state["down"] and admission.idempotency_key.endswith(":2"):
            raise RuntimeError("writer busy")
        return real(admission, actor, deadline_ms=deadline_ms, accept_after_ms=accept_after_ms)

    env.runtime.remember = remember
    env.scan(sid)
    env.scan(sid)
    assert env.files(sid)["long.txt"]["state"] == "error"
    state["down"] = False
    env.scan(sid)
    keys = [r["key"] for r in env.runtime.saved]
    assert env.files(sid)["long.txt"]["state"] == "indexed" and len({k.split(":")[-2] for k in keys}) == 1
    assert env.runtime.archived == [] and len(_live(env, sid, "long.txt")) == len(keys)


# -- CX1 residual: the settle budget is per file, not per part ------------------------------------

def _writer_that_only_queues(env):
    sent = []

    def remember(admission, actor, deadline_ms=0, accept_after_ms=0):
        sent.append(admission.idempotency_key)
        return SimpleNamespace(payload={"status": "accepted", "operation_id": None, "fact_ids": []})

    env.runtime.remember = remember
    return sent


def test_a_busy_writer_costs_a_file_one_settle_wait_not_one_per_part(env, monkeypatch):
    """A 100-part note under contention used to hold the scan for 100 x 20 s."""
    from superlocalmemory.memory_core import submit as submit_mod

    monkeypatch.setattr(submit_mod, "SETTLE_WAIT_S", 0.0)
    sent = _writer_that_only_queues(env)
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    parts = len(sent)  # what was sent before the budget ran out
    assert parts == 1 and stats.deferred == 1 and stats.errors == 0
    row = env.files(sid)["long.txt"]
    entries = json.loads(row["memory_ids_json"])
    assert row["state"] == "error" and row["reason"] == "save_queued"
    assert [e["k"] for e in entries] == sent  # the queued part stays owned by its key


def test_the_rest_of_a_deferred_file_is_saved_under_the_same_keys_once_the_writer_catches_up(env, monkeypatch):
    from superlocalmemory.memory_core import submit as submit_mod

    monkeypatch.setattr(submit_mod, "SETTLE_WAIT_S", 0.0)
    real = env.runtime.remember
    sent = _writer_that_only_queues(env)
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    env.scan(sid)
    env.runtime.remember = real  # the writer catches up; the queued part has committed under its key
    env.scan(sid)
    keys = [r["key"] for r in env.runtime.saved]
    assert env.files(sid)["long.txt"]["state"] == "indexed" and len(keys) > 1
    assert len({k.split(":")[-2] for k in keys}) == 1 and keys[0] == sent[0]
    assert env.runtime.archived == []


def test_a_slow_file_still_gets_the_whole_budget_across_its_parts(env, monkeypatch):
    """The budget is shared by the parts of a file: time spent on part 1 is not given again to part 2."""
    from superlocalmemory.memory_core import submit as submit_mod
    from superlocalmemory.sources import ingest as ingest_mod

    waits = []
    real = submit_mod.submit_memory_settled

    def spy(runtime, request, *, config, wait_s=None, **kw):
        waits.append(wait_s)
        return real(runtime, request, config=config, wait_s=wait_s, **kw)

    monkeypatch.setattr(ingest_mod, "submit_memory_settled", spy)
    env.write("long.txt", LONG)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert len(waits) > 1 and all(w is not None for w in waits)
    assert waits == sorted(waits, reverse=True) and waits[0] <= submit_mod.SETTLE_WAIT_S


# -- a queued save that can never commit is not a dead end ---------------------------------------

class _PlainCodec:
    def encrypt(self, plaintext: bytes) -> bytes:
        return plaintext

    def decrypt(self, ciphertext: bytes) -> bytes:
        return ciphertext


def _journal(env, tmp_path, key, final_state=None):
    """The real admission journal holding the queued request (and, optionally, how it ended)."""
    from superlocalmemory.storage.admission_journal import Actor, AdmissionJournal, RememberRequest

    journal = AdmissionJournal(tmp_path / "admission_journal.db", codec=_PlainCodec())
    entry = journal.prepare(
        RememberRequest(content="one", profile_id="default", source_type="folder", idempotency_key=key,
                        trusted_actor_id="test-actor"),
        Actor(principal_id="test-actor", allowed_profiles=frozenset({"default"}),
              allowed_scopes=frozenset({"personal"})))
    if final_state == "rejected":
        journal.mark_rejected(entry.journal_id, "COMMAND_REJECTED")
    elif final_state == "dispatched":
        journal.mark_dispatched(entry.journal_id)
    env.runtime.journal = journal
    return journal


def test_a_queued_save_the_journal_rejected_no_longer_blocks_removal(env, monkeypatch, tmp_path):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    _journal(env, tmp_path, key, "rejected")
    sources.remove_source(sid)  # nothing was ever saved under that key: nothing to hide
    assert env.runtime.archived == [] and sources.list_sources("default") == []


def test_a_queued_save_the_journal_still_holds_as_pending_blocks_removal(env, monkeypatch, tmp_path):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    for state in (None, "dispatched"):
        _journal(env, tmp_path / (state or "prepared"), key, state)
        with pytest.raises(SourceRefused) as refused:
            sources.remove_source(sid)
        assert refused.value.code == "removal_incomplete"


def test_a_purge_of_a_rejected_queued_save_completes(env, monkeypatch, tmp_path):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    _journal(env, tmp_path, key, "rejected")
    sources.remove_source(sid, purge=True)
    assert env.files(sid) == {}


def test_an_operation_that_failed_for_good_with_no_facts_saved_nothing(env, monkeypatch):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    ops.put(key, "failed", attempts=1, retry_at=1.0)  # failed, but a retry is still scheduled
    with pytest.raises(SourceRefused):
        sources.remove_source(sid)
    ops.conn.execute("UPDATE ingestion_operations SET attempt_count = 10, next_retry_at = 0")  # attempts used up
    sources.remove_source(sid)
    assert env.runtime.archived == []


def test_an_operation_the_reaper_gave_up_on_saved_nothing(env, monkeypatch):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    ops.put(key, "failed", attempts=3, retry_at=9_999_999_999.0)
    sources.remove_source(sid)
    assert env.runtime.archived == []


def test_a_journal_entry_that_committed_names_its_facts_in_the_receipt(env, monkeypatch, tmp_path):
    sid, key, ops = _queued_folder_file(env, monkeypatch)
    journal = _journal(env, tmp_path, key)
    journal.mark_committed(journal.get_by_idempotency_key("default", key).journal_id,
                           {"status": "queryable", "fact_ids": ["r1"], "operation_id": None})
    sources.remove_source(sid)  # the operations table has no row, but the receipt knows the facts
    assert env.runtime.archived == ["r1"]


# -- a replaced PDF whose old document could not be hidden stays reachable --------------------------

def _two_documents(env, monkeypatch):
    """Each submit makes a new real document (own page and memory): d1/pf1, then d2/pf2."""
    made = []

    def submit(inp, **kw):
        n = len(made) + 1
        doc_id = f"{n}" * 32
        media = env.store()
        try:
            media.insert_document(document_id=doc_id, profile_id="default", sha256=f"{n}" * 64, title="T",
                                  mime="application/pdf", bytes=len(inp.data), source_relpath=None)
            media.put_page(doc_id, 1, media_id=None, memory_ids=[f"pm{n}"], fact_ids=[f"pf{n}"],
                           text_origin="text_layer", char_count=5)
        finally:
            media.close()
        made.append(doc_id)
        return SimpleNamespace(status="processing", document_id=doc_id, job_id="j", reason="")

    monkeypatch.setattr("superlocalmemory.documents.submit_document", submit)


def _state_of(env, doc_id):
    media = env.store()
    try:
        return media.get_document(doc_id)["state"]
    finally:
        media.close()


def test_a_replaced_pdf_whose_old_document_could_not_be_hidden_is_hidden_by_a_later_pass(env, monkeypatch):
    import os as _os

    _two_documents(env, monkeypatch)
    path = env.write("paper.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    writer = _Archive(env, "pf1")
    path.write_bytes(PDF + b"B")
    st = _os.stat(path)
    _os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
    assert env.scan(sid).errors >= 1
    row = env.files(sid)["paper.pdf"]
    assert row["document_id"] == "2" * 32 and _state_of(env, "1" * 32) != "tombstoned"
    assert {"hd": "1" * 32} in json.loads(row["memory_ids_json"])  # the old document stays reachable
    writer.heal()
    assert env.scan(sid).errors == 0
    assert _state_of(env, "1" * 32) == "tombstoned" and "pf1" in env.runtime.archived
    assert _state_of(env, "2" * 32) != "tombstoned" and "pf2" not in env.runtime.archived
    assert {"hd": "1" * 32} not in json.loads(env.files(sid)["paper.pdf"]["memory_ids_json"])


def test_removing_the_source_finishes_a_pending_old_document_hide(env, monkeypatch):
    import os as _os

    _two_documents(env, monkeypatch)
    path = env.write("paper.pdf", PDF)
    sid = env.add_and_confirm()
    env.scan(sid)
    writer = _Archive(env, "pf1")
    path.write_bytes(PDF + b"B")
    st = _os.stat(path)
    _os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 5 * 10**9))
    env.scan(sid)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid)
    assert refused.value.code == "removal_incomplete"
    writer.heal()
    sources.remove_source(sid)
    assert _state_of(env, "1" * 32) == "tombstoned" and _state_of(env, "2" * 32) == "tombstoned"
