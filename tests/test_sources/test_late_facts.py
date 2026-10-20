"""Issue #297: a save returns its raw fact first; enrichment adds more facts to the same memory later.

Revising, deleting or purging a file must hide (or erase) every fact its save ended up with, not only
the ids the save returned. A save whose enrichment has not finished cannot be called hidden yet.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory import sources
from superlocalmemory.sources.api import SourceRefused

from .conftest import FakeRuntime


class EnrichingRuntime(FakeRuntime):
    """Like the real writer: the receipt names the raw fact; the enriched fact lands afterwards."""

    def __init__(self) -> None:
        super().__init__()
        self._db = sqlite3.connect(":memory:", check_same_thread=False)
        self._db.row_factory = sqlite3.Row
        self._db.execute("CREATE TABLE ingestion_operations (profile_id TEXT, source_type TEXT,"
                         " idempotency_key TEXT, state TEXT, next_retry_at REAL, attempt_count INTEGER,"
                         " queryable_fact_ids_json TEXT, final_fact_ids_json TEXT)")
        self._db.execute("CREATE TABLE atomic_facts (fact_id TEXT, memory_id TEXT, lifecycle TEXT)")
        self.enrich_state = "complete"

    def remember(self, admission, actor, deadline_ms=0, accept_after_ms=0):
        result = super().remember(admission, actor, deadline_ms, accept_after_ms)
        row = self._by_key[admission.idempotency_key]
        raw, late = row["fid"], row["fid"] + "-late"
        final = [raw] if self.enrich_state != "complete" else [raw, late]
        self._db.execute("INSERT INTO ingestion_operations VALUES (?, 'folder', ?, ?, 0, 0, ?, ?)",
                         (admission.profile_id, admission.idempotency_key, self.enrich_state,
                          json.dumps([raw]), json.dumps(final)))
        for fact in final:
            self._db.execute("INSERT INTO atomic_facts VALUES (?, ?, 'active')", (fact, row["mid"]))
        return result

    def finish_enrichment(self) -> None:
        for row in self.saved:
            late = row["fid"] + "-late"
            self._db.execute("UPDATE ingestion_operations SET state = 'complete', final_fact_ids_json = ?"
                             " WHERE idempotency_key = ?", (json.dumps([row["fid"], late]), row["key"]))
            if not self._db.execute("SELECT 1 FROM atomic_facts WHERE fact_id = ?", (late,)).fetchall():
                self._db.execute("INSERT INTO atomic_facts VALUES (?, ?, 'active')", (late, row["mid"]))
        self.enrich_state = "complete"


@pytest.fixture
def enriching(env):
    env.runtime = EnrichingRuntime()
    return env


def test_a_revised_note_hides_its_enriched_facts_too(enriching):
    env = enriching
    env.write("a.md", "CopperLake v3 keeps audit logs for 11 days.")
    sid = env.add_and_confirm()
    env.scan(sid)
    env.write("a.md", "CopperLake v4 keeps audit logs for 14 days.")
    env.scan(sid)
    assert sorted(env.runtime.archived) == ["f1", "f1-late"]


def test_a_deleted_note_hides_its_enriched_facts_too(enriching):
    env = enriching
    path = env.write("a.md", "Bluefin ships on the second Tuesday.")
    env.write("keep.canvas", "{}")  # the folder is never empty
    sid = env.add_and_confirm()
    env.scan(sid)
    path.unlink()
    env.scan(sid)
    assert sorted(env.runtime.archived) == ["f1", "f1-late"]
    assert env.files(sid)["a.md"]["state"] == "tombstoned"


def test_removing_a_source_hides_its_enriched_facts_too(enriching):
    env = enriching
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    sources.remove_source(sid)
    assert sorted(env.runtime.archived) == ["f1", "f1-late"]


def test_purge_erases_the_enriched_facts_too(enriching):
    env = enriching
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    sources.remove_source(sid, purge=True)
    assert sorted(env.erased[0][1]) == ["f1", "f1-late"]
    assert env.files(sid) == {}


def test_purge_while_enrichment_runs_refuses_instead_of_reporting_done(enriching):
    env = enriching
    env.runtime.enrich_state = "enriching"
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid, purge=True)
    assert refused.value.code == "erasure_incomplete"
    env.runtime.finish_enrichment()
    sources.remove_source(sid, purge=True)
    assert sorted(env.erased[-1][1]) == ["f1", "f1-late"]


def test_removal_while_enrichment_runs_refuses_and_hides_everything_on_retry(enriching):
    env = enriching
    env.runtime.enrich_state = "enriching"
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    with pytest.raises(SourceRefused) as refused:
        sources.remove_source(sid)
    assert refused.value.code == "removal_incomplete"
    env.runtime.finish_enrichment()
    sources.remove_source(sid)
    assert "f1-late" in env.runtime.archived
    assert sources.list_sources("default") == []
