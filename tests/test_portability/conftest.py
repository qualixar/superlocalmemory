"""Shared harness for the portability pack: a real engine, the daemon's writer, the flag on."""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core import decisions
from superlocalmemory.core.feature_flags import env_name
from superlocalmemory.portability import readers, service
from superlocalmemory.portability.jobs import JOBS
from tests.test_server.test_per_request_profile import _daemon

WHO = decisions.Decider("agent:test-importer", "cli")

#: Tables that are not memory. Append-only ledgers record that something happened and are never
#: rewritten (receipts of writes, the ids-only change feed, queued removals of vector rows,
#: the decision log and the import record that this feature owns); two are cache-version counters
#: that move on every write, a rollback's included; the others (below) are the search index's own
#: internal segments and the vector store's, which hold no row a person can read and are emptied
#: by the daemon's maintenance sweep.
LEDGERS = {"write_commits", "fact_search_changes", "projection_outbox", "decision_log",
           "import_runs", "import_run_items", "temporal_retrieval_version", "graph_generation",
           "memory_events"}
INTERNALS = ("atomic_facts_fts_", "fact_embeddings_")


def snapshot(db) -> dict[str, list[str]]:
    """Every row of every table, as text, ledgers and index internals aside."""
    conn = sqlite3.connect(db.db_path)
    conn.row_factory = sqlite3.Row
    out: dict[str, list[str]] = {}
    names = [r[0] for r in conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'")]
    for name in names:
        if name in LEDGERS or name.startswith(INTERNALS):
            continue
        try:
            rows = conn.execute(f"SELECT * FROM {name}").fetchall()  # noqa: S608
        except sqlite3.Error:
            continue
        out[name] = sorted(repr({k: (v if not isinstance(v, bytes) else len(v))
                                 for k, v in dict(r).items()}) for r in rows)
    conn.close()
    return out


def diff(before: dict, after: dict) -> dict[str, tuple[int, int]]:
    return {t: (len(before.get(t, [])), len(after.get(t, [])))
            for t in sorted(set(before) | set(after)) if before.get(t) != after.get(t)}


class World:
    """An engine, the daemon app with its writer, and helpers to run and wait for imports."""

    def __init__(self, engine, client, app) -> None:
        self.engine, self.client, self.app = engine, client, app
        self.db = engine._db
        self.profile = engine.profile_id
        self.runtime = app.state.canonical_remember_runtime

    def docs(self, name: str, data: bytes):
        return readers.from_bytes(name, data)

    def run(self, name: str, data: bytes, fmt: str | None = None, *, wait: bool = True) -> dict:
        run = service.start_run(self.engine, self.runtime, profile_id=self.profile,
                                docs=self.docs(name, data), fmt=fmt, label=name, decider=WHO)
        if wait:
            assert JOBS.join(run["run_id"], 60), "the import did not finish"
        return service.get_run(self.engine, self.profile, run["run_id"])

    def enrich(self) -> None:
        """What the daemon's background materializer does to every save, done now."""
        from superlocalmemory.core.engine_ingestion import build_engine_ingestion_command

        command = build_engine_ingestion_command(self.engine)
        for row in self.db.execute("SELECT operation_id FROM ingestion_operations "
                                   "WHERE state != 'complete'"):
            command.materialize(dict(row)["operation_id"])

    def undo(self, run_id: str, wait: float = 60.0) -> dict:
        return service.start_undo(self.engine, self.runtime, profile_id=self.profile,
                                  run_id=run_id, decider=WHO, wait_s=wait)

    def count(self, table: str, where: str = "1=1", params: tuple = ()) -> int:
        return int(dict(self.db.execute(f"SELECT COUNT(*) AS c FROM {table} WHERE {where}",  # noqa: S608
                                        params)[0])["c"])


@pytest.fixture
def world(engine_with_mock_deps, monkeypatch, tmp_path):
    monkeypatch.setenv(env_name("portability.import"), "on")
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "slm-home"))
    (tmp_path / "slm-home").mkdir()
    with _daemon(engine_with_mock_deps, profiles=("default", "work")) as (client, app):
        yield World(engine_with_mock_deps, client, app)
