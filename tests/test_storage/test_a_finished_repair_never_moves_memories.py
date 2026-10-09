# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""A repair recorded complete never moves a memory again, on any later start.

Observed on a scratch copy of a real 4.1.17 store, whose own migration log
recorded M043 and M047 complete, booted by 4.1.18:

    M043 memory repair: 0 summaries preserved for display, 0 withheld from
    recall, 3452 memories restored to recall
    M047 verify: fisher_variance on fact ... is still text ...

The hypothesis was that a missing learning.db made memory.db's repairs look
pending. It does not: memory.db's repairs are decided by memory.db's own log,
and these tests run every case with learning.db present AND absent to show the
outcome is identical. What actually happens is that both migrations re-check
their end-state on EVERY start, and act on it:

  * M043 re-ran its "restore" step on every start. That step un-hides any
    archived memory whose score is high, or that a summary ever drew on -- so
    it overrode every lifecycle decision made after the upgrade: the memories a
    consolidation pass archived behind its gist, the ones the tier manager
    archived after a year untouched. A memory the owner forgot was counted as
    "restored" without moving, and because it then still looked hidden, the
    migration was reported failed on every start, forever.
  * M047 found Fisher vectors in the old text form because the maintenance
    cycle writes them in that form, undoing the conversion one fact at a time.

The one-time repair still runs, once, when the migration is first applied.
"""

from __future__ import annotations

import json
import logging
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.core.lifecycle_state import set_fact_lifecycle_zone
from superlocalmemory.storage.migration_runner import apply_all, apply_deferred
from superlocalmemory.storage.migrations import (
    M043_quarantine_display_summaries as M043,
)
from superlocalmemory.storage.migrations import (
    M047_fisher_vectors_are_stored_like_every_other_vector as M047,
)
from superlocalmemory.storage.schema import create_all_tables
from superlocalmemory.storage.schema_v343 import apply_v343_schema, apply_v346_schema
from superlocalmemory.storage.schema_v347 import apply_v347_schema
from superlocalmemory.storage.schema_v3410 import apply_v3410_schema
from superlocalmemory.storage.schema_v3411 import apply_v3411_schema

_PROFILE = "default"


def _engine_schema_bootstrap(memory_db: Path) -> None:
    conn = sqlite3.connect(str(memory_db))
    try:
        create_all_tables(conn)
        conn.commit()
    finally:
        conn.close()
    for step in (apply_v343_schema, apply_v346_schema, apply_v347_schema,
                 apply_v3410_schema, apply_v3411_schema):
        step(str(memory_db))


def _start(learning_db: Path, memory_db: Path) -> dict:
    apply_all(learning_db, memory_db)
    _engine_schema_bootstrap(memory_db)
    apply_all(learning_db, memory_db)
    return apply_deferred(learning_db, memory_db)


def _open(db: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db))
    conn.row_factory = sqlite3.Row
    return conn


def _add_memory(conn: sqlite3.Connection, fact_id: str, score: float) -> None:
    conn.execute(
        "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
        " lifecycle, created_at) VALUES (?, 'mem1', ?, ?, 'active',"
        " '2026-03-01T00:00:00+00:00')",
        (fact_id, _PROFILE, f"A real memory called {fact_id}."),
    )
    conn.execute(
        "INSERT INTO fact_retention (fact_id, profile_id, retention_score,"
        " lifecycle_zone) VALUES (?, ?, ?, 'active')",
        (fact_id, _PROFILE, score),
    )


def _live_use_after_the_upgrade(memory_db: Path) -> None:
    """Lifecycle decisions the running product makes after M043 completed."""
    conn = _open(memory_db)
    try:
        conn.execute(
            "INSERT INTO memories (memory_id, profile_id, content) "
            "VALUES ('mem1', ?, 'a conversation')", (_PROFILE,),
        )
        _add_memory(conn, "behind-a-gist", 1.0)
        _add_memory(conn, "untouched-for-a-year", 0.5)
        _add_memory(conn, "forgotten-by-owner", 0.9)
        # The current consolidator writes display-only summaries and records
        # their sources in the ledger. It archives nothing.
        conn.execute(
            "INSERT INTO fact_consolidations (consolidation_id, profile_id,"
            " consolidated_fact_id, source_fact_ids, strategy, created_at) "
            "VALUES ('d1', ?, 'summary-1', ?, 'display_summary',"
            " '2026-09-01T00:00:00+00:00')",
            (_PROFILE, json.dumps(["untouched-for-a-year", "forgotten-by-owner"])),
        )
        # A consolidation pass replaced a cluster with its gist and archived it.
        set_fact_lifecycle_zone(conn, ["behind-a-gist"], "archive", profile_id=_PROFILE)
        # The tier manager archived a memory nobody had opened for a year.
        set_fact_lifecycle_zone(
            conn, ["untouched-for-a-year"], "archive", profile_id=_PROFILE,
        )
        # The owner asked for this one to be forgotten.
        set_fact_lifecycle_zone(
            conn, ["forgotten-by-owner"], "forgotten", profile_id=_PROFILE,
        )
        conn.execute(
            "UPDATE fact_retention SET retention_score = 0.0 "
            "WHERE fact_id = 'forgotten-by-owner'"
        )
        conn.commit()
    finally:
        conn.close()


def _memory_state(memory_db: Path) -> list[tuple]:
    conn = _open(memory_db)
    try:
        return [tuple(r) for r in conn.execute(
            "SELECT af.fact_id, af.lifecycle, COALESCE(af.quarantined, 0),"
            "       r.lifecycle_zone, r.retention_score, typeof(af.fisher_variance)"
            "  FROM atomic_facts af LEFT JOIN fact_retention r"
            "    ON r.fact_id = af.fact_id ORDER BY af.fact_id"
        )]
    finally:
        conn.close()


@pytest.fixture()
def upgraded(tmp_path: Path) -> tuple[Path, Path]:
    learning_db = tmp_path / "learning.db"
    memory_db = tmp_path / "memory.db"
    _start(learning_db, memory_db)
    _start(learning_db, memory_db)
    log = _open(memory_db)
    try:
        status = log.execute(
            "SELECT status FROM migration_log WHERE name = ?", (M043.NAME,),
        ).fetchone()
    finally:
        log.close()
    assert status is not None and status[0] == "complete", (
        "fixture is wrong: M043 is not recorded complete"
    )
    _live_use_after_the_upgrade(memory_db)
    return learning_db, memory_db


@pytest.mark.parametrize("learning_db_present", [True, False])
class TestALaterStartMovesNothing:
    def test_lifecycle_decisions_made_after_the_upgrade_stand(
        self, upgraded, learning_db_present, caplog,
    ) -> None:
        learning_db, memory_db = upgraded
        if not learning_db_present:
            for suffix in ("", "-wal", "-shm"):
                Path(f"{learning_db}{suffix}").unlink(missing_ok=True)
        before = _memory_state(memory_db)

        with caplog.at_level(logging.INFO):
            deferred = _start(learning_db, memory_db)

        assert _memory_state(memory_db) == before, (
            "a start re-ran a completed repair and moved memories"
        )
        assert M043.NAME not in deferred["failed"], deferred["details"].get(M043.NAME)
        assert "memories restored to recall" not in caplog.text


class TestTheFirstRepairCountsOnlyWhatMoved:
    def _store(self, tmp_path: Path) -> Path:
        db = tmp_path / "memory.db"
        conn = _open(db)
        try:
            create_all_tables(conn)
            conn.commit()
            conn.execute("PRAGMA foreign_keys=OFF")
            for table_sql in (
                "CREATE TABLE IF NOT EXISTS fact_consolidations ("
                " consolidation_id TEXT PRIMARY KEY, profile_id TEXT,"
                " consolidated_fact_id TEXT NOT NULL, source_fact_ids TEXT NOT NULL,"
                " strategy TEXT DEFAULT 'entity_cluster', created_at TEXT NOT NULL)",
                "CREATE TABLE IF NOT EXISTS fact_retention ("
                " fact_id TEXT PRIMARY KEY, profile_id TEXT, retention_score REAL,"
                " memory_strength REAL, access_count INTEGER, last_accessed_at TEXT,"
                " last_computed_at TEXT, lifecycle_zone TEXT)",
            ):
                conn.execute(table_sql)
            conn.execute(
                "INSERT INTO memories (memory_id, profile_id, content) "
                "VALUES ('mem1', ?, 'a conversation')", (_PROFILE,),
            )
            for fact_id, score in (("victim", 1.0), ("faded", 0.1), ("shown", 0.5)):
                _add_memory(conn, fact_id, score)
                conn.execute(
                    "UPDATE fact_retention SET lifecycle_zone = 'archive' "
                    "WHERE fact_id = ?", (fact_id,),
                )
                conn.execute(
                    "UPDATE atomic_facts SET lifecycle = 'archived' "
                    "WHERE fact_id = ?", (fact_id,),
                )
            # The old consolidator wrote a summary into recall and archived its
            # sources; 'faded' has since decayed for real.
            conn.execute(
                "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content,"
                " lifecycle, created_at) VALUES ('old-summary', '', ?,"
                " 'A model summary of three memories.', 'active', '2026-08-01')",
                (_PROFILE,),
            )
            conn.execute(
                "INSERT INTO fact_consolidations VALUES ('c1', ?, 'old-summary',"
                " ?, 'entity_cluster', '2026-08-01')",
                (_PROFILE, json.dumps(["victim", "faded"])),
            )
            # The current consolidator only displays; it never archived 'shown'.
            conn.execute(
                "INSERT INTO fact_consolidations VALUES ('d1', ?, 'display-1',"
                " ?, 'display_summary', '2026-09-01')",
                (_PROFILE, json.dumps(["shown"])),
            )
            conn.commit()
        finally:
            conn.close()
        return db

    def _zone(self, db: Path, fact_id: str) -> str:
        conn = _open(db)
        try:
            return conn.execute(
                "SELECT lifecycle_zone FROM fact_retention WHERE fact_id = ?",
                (fact_id,),
            ).fetchone()[0]
        finally:
            conn.close()

    def test_the_log_counts_only_memories_that_came_back(
        self, tmp_path: Path, caplog,
    ) -> None:
        db = self._store(tmp_path)
        conn = sqlite3.connect(str(db), isolation_level=None)
        try:
            with caplog.at_level(logging.INFO):
                M043.apply(conn)
        finally:
            conn.close()
        assert self._zone(db, "victim") == "active"
        assert self._zone(db, "faded") == "archive"
        assert "1 memories restored to recall" in caplog.text, caplog.text

    def test_a_display_summarys_sources_were_never_hidden_by_it(
        self, tmp_path: Path,
    ) -> None:
        db = self._store(tmp_path)
        conn = sqlite3.connect(str(db), isolation_level=None)
        try:
            M043.apply(conn)
        finally:
            conn.close()
        assert self._zone(db, "shown") == "archive", (
            "a memory archived by something other than consolidation was "
            "un-hidden because a display summary had listed it"
        )

    def test_the_end_state_holds_after_the_repair(self, tmp_path: Path) -> None:
        """A faded source stays hidden, so it must not keep verify failing."""
        db = self._store(tmp_path)
        conn = sqlite3.connect(str(db), isolation_level=None)
        try:
            M043.apply(conn)
            assert M043.verify(conn) is True, M043.unmet(conn)
        finally:
            conn.close()


class TestMaintenanceKeepsTheConvertedForm:
    def test_the_fisher_update_writes_the_binary_form(self, tmp_path: Path) -> None:
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.core.maintenance import run_maintenance
        from superlocalmemory.storage import schema
        from superlocalmemory.storage.database import DatabaseManager
        from superlocalmemory.storage.models import AtomicFact, MemoryRecord

        db = DatabaseManager(tmp_path / "memory.db")
        db.initialize(schema)
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) "
            "VALUES ('default', 'default')"
        )
        mem = MemoryRecord(profile_id=_PROFILE, content="Something said once.")
        db.store_memory(mem)
        fact = AtomicFact(
            memory_id=mem.memory_id, profile_id=_PROFILE, content="The sky is blue.",
            embedding=[0.1] * 8, fisher_mean=[0.1] * 8, fisher_variance=[1.0] * 8,
            confidence=0.9, importance=0.5, evidence_count=1, access_count=3,
        )
        db.store_fact(fact)

        counts = run_maintenance(db, SLMConfig(), profile_id=_PROFILE)

        assert counts["fisher_posterior_updated"] == 1, "fixture is wrong"
        kind = db.execute(
            "SELECT typeof(fisher_variance) AS kind FROM atomic_facts "
            "WHERE fact_id = ?", (fact.fact_id,),
        )[0]["kind"]
        assert kind == "blob", (
            "maintenance wrote the Fisher variance back as text, undoing the "
            "conversion and re-triggering it on the next start"
        )
        conn = sqlite3.connect(str(tmp_path / "memory.db"))
        try:
            assert M047.verify(conn) is True
        finally:
            conn.close()


class TestTheHealthReportPromisesOnlyWhatHappens:
    def test_it_does_not_say_a_restart_will_re_file_hidden_memories(self) -> None:
        """A start no longer re-files them, so saying it will is a false promise."""
        from superlocalmemory.core.memory_health import MemoryHealth, describe

        text = " ".join(describe(MemoryHealth(
            live_facts=10, findable_by_meaning=10, inconsistently_hidden=3,
        )))

        assert "next time the service starts" not in text
        assert "slm doctor --refile-hidden" in text
        assert "decay" not in text

    def test_doctor_points_at_the_pass_that_re_files_them(self) -> None:
        import argparse
        from unittest.mock import patch

        from superlocalmemory.cli.commands import cmd_doctor
        from superlocalmemory.core.memory_health import MemoryHealth

        captured: list[dict] = []
        health = MemoryHealth(live_facts=10, findable_by_meaning=10,
                              inconsistently_hidden=3)
        with patch(
            "superlocalmemory.cli.json_output.json_print",
            side_effect=lambda event, data=None, **kw: captured.append(data or {}),
        ), patch(
            "superlocalmemory.core.memory_health.measure", return_value=health,
        ):
            cmd_doctor(argparse.Namespace(json=True, quick=True))

        check = next(
            c for c in captured[-1]["checks"] if c["name"] == "Memory answer-ability"
        )
        assert check["status"] == "WARN"
        assert check["fix"] == "slm doctor --refile-hidden", check
