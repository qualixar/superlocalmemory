# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""M065: the made-up dates of builds before 4.1.6 leave the store at its first 4.2.8 start.

Temporary stores only. Each test starts from a fully migrated store with M065
struck from the log, so ``apply_deferred`` is the daemon's first 4.2.8 start.
"""

from __future__ import annotations

import json
import sqlite3

import pytest

from superlocalmemory.storage import data_repair_undo as undo_log
from superlocalmemory.storage._migration_internals import _MODULES
from superlocalmemory.storage.migration_runner import DEFERRED_MIGRATIONS
from superlocalmemory.storage.migrations import M065_fabricated_dates_are_removed as M065
from tests.test_migrations._repair_store import (
    add_event,
    add_ingestion,
    add_fact,
    first_start,
    forget_migration,
    fts_ids,
    migrated_store,
)

OLD = "2026-05-10T07:14:22+00:00"      # before 4.1.6
NEW = "2026-09-02T10:00:00+00:00"      # after 4.1.6


def _seed(memory) -> None:
    conn = sqlite3.connect(memory)
    add_fact(conn, "f_clause", "The file CHANGELOG.md was modified on 2024-01-15",
             referenced_date="2024-01-15", embedding=b"\x00\x01")
    add_event(conn, "ev_fab", "f_clause", "The file CHANGELOG.md was modified on 2024-01-15",
              referenced="2024-01-15")
    add_event(conn, "ev_mixed", "f_clause", "The file CHANGELOG.md was modified on 2024-01-15",
              referenced="2024-01-15", observed="2026-05-10")
    add_event(conn, "ev_quote", "f_clause", "The file CHANGELOG.md was modified on 2024-01-15",
              observed="2026-05-11")
    conn.execute("INSERT INTO bm25_tokens (fact_id, profile_id, tokens) VALUES "
                 "('f_clause', 'default', ?)", (json.dumps(["file", "2024-01-15"]),))
    conn.execute("INSERT INTO fact_context (fact_id, profile_id, contextual_description, "
                 "keywords) VALUES ('f_clause', 'default', ?, ?)",
                 ("This event records: The file CHANGELOG.md was modified on 2024-01-15.",
                  json.dumps(["changelog", "2024-01-15", "modified"])))
    conn.execute("INSERT INTO fact_temporal_validity (fact_id, profile_id, valid_from) "
                 "VALUES ('f_clause', 'default', '2024-01-15')")
    add_fact(conn, "f_subject", "2024-01-15 was a Tuesday", referenced_date="2024-01-15")
    add_fact(conn, "f_dateonly", "The reranker test passed", referenced_date="2024-01-15")
    add_fact(conn, "f_after", "The prompt hardcoded a worked example (session date "
             "2024-01-15) until 4.1.6", created_at=NEW, referenced_date="2024-01-15")
    add_fact(conn, "f_source", "The invoice was paid on 2024-01-15",
             source="I paid the invoice on 2024-01-15.", referenced_date="2024-01-15")
    add_fact(conn, "f_near", "The demo ran on 2026-01-15",
             created_at="2026-01-16T09:00:00+00:00", referenced_date="2026-01-15")
    add_fact(conn, "f_plain", "Nothing dated here", referenced_date="2026-05-01")
    # The source names the day in another written form: the date is real.
    add_fact(conn, "f_spelled", "Mel was seen on January 15, 2024",
             source="Mel was seen at the park on Jan 15.", referenced_date="2024-01-15")
    add_fact(conn, "f_dayfirst", "The demo ran on 2024-01-15",
             source="We ran the demo 15 Jan 2024.", referenced_date="2024-01-15")
    # No source anywhere: nothing can prove the date made up, so it stays.
    add_fact(conn, "f_nosource", "The deploy happened on 2024-01-15", source=None,
             referenced_date="2024-01-15")
    # The memory row is gone but the raw text survives in its ingestion record.
    add_fact(conn, "f_lineage", "The build broke on 2024-01-15", source=None,
             referenced_date="2024-01-15")
    add_ingestion(conn, "f_lineage", "the build broke again")
    add_fact(conn, "f_lineage_dated", "The cache was cleared on 2024-01-15", source=None)
    add_ingestion(conn, "f_lineage_dated", "cleared the cache on 2024-01-15",
                  linked_by="fact_list")
    conn.commit()
    conn.close()


@pytest.fixture()
def store(tmp_path):
    memory, learning = migrated_store(tmp_path)
    _seed(memory)
    forget_migration(memory, M065.NAME)
    return memory, learning


def _fact(conn, fact_id):
    conn.row_factory = sqlite3.Row
    return dict(conn.execute("SELECT * FROM atomic_facts WHERE fact_id = ?",
                             (fact_id,)).fetchone())


def _snapshot(memory) -> dict:
    conn = sqlite3.connect(memory)
    out = {
        "facts": conn.execute("SELECT fact_id, content, referenced_date FROM atomic_facts "
                              "ORDER BY fact_id").fetchall(),
        "events": conn.execute("SELECT * FROM temporal_events ORDER BY event_id").fetchall(),
        "validity": conn.execute("SELECT fact_id, valid_from, valid_until FROM "
                                 "fact_temporal_validity ORDER BY fact_id").fetchall(),
        "bm25": conn.execute("SELECT fact_id, tokens FROM bm25_tokens").fetchall(),
    }
    conn.close()
    return out


def test_registered_deferred_with_a_floor() -> None:
    assert _MODULES[M065.NAME] is M065
    assert any(m.name == M065.NAME for m in DEFERRED_MIGRATIONS)
    assert M065.DB_TARGET == "memory" and M065.DOWNGRADE_FLOOR == 51
    assert M065.REPAIR_NOT_APPLICABLE.strip()


def test_first_start_removes_the_clause_and_its_dates(store) -> None:
    memory, learning = store
    result = first_start(memory, learning)
    assert M065.NAME in result["applied"], result

    conn = sqlite3.connect(memory)
    fact = _fact(conn, "f_clause")
    assert fact["content"] == "The file CHANGELOG.md was modified"
    assert fact["referenced_date"] is None
    assert fact["embedding"] is None  # queued for the background re-embedding
    assert fts_ids(conn, "2024-01-15") >= {"f_after", "f_source"}
    assert "f_clause" not in fts_ids(conn, "2024-01-15")
    assert "f_clause" in fts_ids(conn, "CHANGELOG.md was modified")
    tokens = json.loads(conn.execute("SELECT tokens FROM bm25_tokens WHERE fact_id='f_clause'"
                                     ).fetchone()[0])
    assert "2024-01-15" not in tokens and "changelog" in " ".join(tokens).lower()
    events = {r[0]: r for r in conn.execute(
        "SELECT event_id, referenced_date, observation_date, description FROM temporal_events "
        "WHERE fact_id='f_clause'")}
    assert set(events) == {"ev_mixed", "ev_quote"}  # the date-only event is gone
    assert events["ev_mixed"][1:] == (None, "2026-05-10", "The file CHANGELOG.md was modified")
    assert events["ev_quote"][1:] == (None, "2026-05-11", "The file CHANGELOG.md was modified")
    assert conn.execute("SELECT COUNT(*) FROM fact_temporal_validity WHERE fact_id='f_clause'"
                        ).fetchone()[0] == 0
    assert conn.execute("SELECT 1 FROM projection_outbox WHERE fact_id='f_clause'").fetchone()


def test_a_statement_about_the_date_keeps_its_words_and_loses_the_date(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    subject = _fact(conn, "f_subject")
    assert subject["content"] == "2024-01-15 was a Tuesday" and subject["referenced_date"] is None
    dateonly = _fact(conn, "f_dateonly")
    assert dateonly["content"] == "The reranker test passed"
    assert dateonly["referenced_date"] is None


def test_legitimate_mentions_are_never_touched(store) -> None:
    memory, learning = store
    before = _snapshot(memory)
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    kept = ("f_after", "f_source", "f_near", "f_plain", "f_spelled", "f_dayfirst",
            "f_nosource", "f_lineage_dated")
    for fact_id in kept:
        row = conn.execute("SELECT content, referenced_date FROM atomic_facts WHERE fact_id=?",
                           (fact_id,)).fetchone()
        assert (fact_id, *row) in [(f[0], f[1], f[2]) for f in before["facts"]], fact_id
    marks = ",".join("?" * len(kept))
    assert not conn.execute(f"SELECT 1 FROM data_repair_undo WHERE fact_id IN ({marks})",
                            kept).fetchone()


def test_a_source_reached_through_its_ingestion_record_counts(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    assert _fact(conn, "f_lineage")["content"] == "The build broke"


def test_the_survey_counts_each_kept_group(store) -> None:
    from superlocalmemory.storage.fabricated_dates_repair import survey

    memory, _learning = store
    found = survey(sqlite3.connect(memory))
    assert {p.fact_id for p in found.repairs} == {"f_clause", "f_subject", "f_dateonly",
                                                   "f_lineage"}
    assert set(found.in_source) == {"f_source", "f_spelled", "f_dayfirst", "f_lineage_dated"}
    assert set(found.no_source) == {"f_nosource"}


def test_every_fact_is_kept(store) -> None:
    memory, learning = store
    before = len(_snapshot(memory)["facts"])
    first_start(memory, learning)
    assert len(_snapshot(memory)["facts"]) == before


def test_it_runs_once_and_apply_is_idempotent(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    again = first_start(memory, learning)
    assert M065.NAME in again["skipped"] and M065.NAME not in again["applied"]
    after_once = _snapshot(memory)
    conn = sqlite3.connect(memory, isolation_level=None)
    undo_rows = conn.execute("SELECT COUNT(*) FROM data_repair_undo").fetchone()[0]
    M065.apply(conn)
    assert conn.execute("SELECT COUNT(*) FROM data_repair_undo").fetchone()[0] == undo_rows
    conn.close()
    assert _snapshot(memory) == after_once


def test_the_undo_log_puts_everything_back(store) -> None:
    memory, learning = store
    before = _snapshot(memory)
    first_start(memory, learning)
    conn = sqlite3.connect(memory, isolation_level=None)
    restored = undo_log.undo(conn, M065.NAME)
    assert restored > 0
    conn.close()
    after = _snapshot(memory)
    assert after["facts"] == before["facts"]
    assert after["events"] == before["events"]
    assert after["validity"] == before["validity"]
    assert sorted(after["bm25"]) == sorted(before["bm25"])
    conn = sqlite3.connect(memory)
    assert "f_clause" in fts_ids(conn, "2024-01-15")  # the keyword index follows the text
    assert undo_log.undo(sqlite3.connect(memory, isolation_level=None), M065.NAME) == 0


def test_a_crash_mid_run_changes_nothing_and_the_next_start_finishes(store, monkeypatch) -> None:
    memory, learning = store
    before = _snapshot(memory)
    from superlocalmemory.storage import fabricated_dates_repair as repair

    calls = {"n": 0}
    real = repair._repair_fact

    def dies_on_the_second(conn, plan, writer):
        calls["n"] += 1
        if calls["n"] == 2:
            raise sqlite3.OperationalError("disk I/O error (simulated crash)")
        return real(conn, plan, writer)

    monkeypatch.setattr(repair, "_repair_fact", dies_on_the_second)
    result = first_start(memory, learning)
    assert M065.NAME in result["failed"], result
    assert _snapshot(memory) == before  # one transaction: nothing half-done

    monkeypatch.setattr(repair, "_repair_fact", real)
    result = first_start(memory, learning)
    assert M065.NAME in result["applied"], result
    assert _fact(sqlite3.connect(memory), "f_clause")["referenced_date"] is None


def test_summary_reports_what_the_repair_did(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    summary = undo_log.summary(conn, M065.NAME)
    assert summary["facts"] == 4  # f_clause, f_subject, f_dateonly, f_lineage
    assert summary["text_changed"] == 2
    assert summary["undone"] == 0


def test_the_context_line_follows_the_new_text(store) -> None:
    memory, learning = store
    first_start(memory, learning)
    conn = sqlite3.connect(memory)
    description, keywords = conn.execute(
        "SELECT contextual_description, keywords FROM fact_context WHERE fact_id='f_clause'"
    ).fetchone()
    assert description == "This event records: The file CHANGELOG.md was modified."
    assert json.loads(keywords) == ["changelog", "modified"]
