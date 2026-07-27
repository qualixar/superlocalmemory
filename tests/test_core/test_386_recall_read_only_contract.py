# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""3.8.6 regression contracts: recall is a query, never a memory mutation.

These tests deliberately exercise the public engine and MCP entry points.  A
``readOnlyHint`` annotation or a best-effort ``try/except`` is not sufficient:
recall must not *attempt* canonical-state mutation.
"""

from __future__ import annotations

import asyncio
import sqlite3
from unittest.mock import MagicMock, patch

from superlocalmemory.storage.models import AtomicFact, RecallResponse, RetrievalResult


class _MockServer:
    def __init__(self) -> None:
        self.tools: dict[str, object] = {}

    def tool(self, *args, **kwargs):
        def decorate(fn):
            self.tools[fn.__name__] = fn
            return fn
        return decorate


def _response() -> RecallResponse:
    fact = AtomicFact(
        fact_id="386-read-only-fact",
        memory_id="386-memory",
        profile_id="default",
        content="The 3.8.6 recall contract is query-only.",
    )
    return RecallResponse(
        query="read-only contract",
        mode="A",
        results=[RetrievalResult(
            fact=fact,
            score=0.9,
            channel_scores={"semantic": 0.9, "bm25": 0.8},
            confidence=0.9,
        )],
        query_type="factual",
    )


def _is_mutation(sql: str) -> bool:
    """Recognise DML and schema changes, including CTE-wrapped writes."""
    normalized = " ".join(sql.upper().split())
    return any(
        token in normalized
        for token in (
            "INSERT ", "UPDATE ", "DELETE ", "REPLACE ", "CREATE ",
            "ALTER ", "DROP ", "VACUUM", "PRAGMA ",
        )
    )


def test_386_engine_recall_attempts_no_canonical_dml(engine_with_mock_deps, monkeypatch):
    """Engine recall must be physically pure even when it returns a fact.

    The guard is intentionally installed at the canonical database facade.  A
    write swallowed by a best-effort telemetry block is still a contract
    violation because it contends with foreground remember calls.
    """
    engine = engine_with_mock_deps
    attempted: list[str] = []
    original_execute = engine._db.execute
    # ``data_version`` is a second, independent witness: it changes whenever
    # another connection commits to the canonical database.
    observer = sqlite3.connect(engine._db.db_path)
    before_version = observer.execute("PRAGMA data_version").fetchone()[0]

    def query_only_execute(sql, params=()):
        if _is_mutation(str(sql)):
            attempted.append(" ".join(str(sql).split()))
            raise AssertionError(f"recall attempted canonical DML: {sql}")
        return original_execute(sql, params)

    monkeypatch.setattr(engine._db, "execute", query_only_execute)
    monkeypatch.setattr(engine._retrieval_engine, "recall", lambda *args, **kwargs: _response())
    monkeypatch.setattr(
        "superlocalmemory.infra.local_diagnostics.record_recall", lambda *args, **kwargs: None,
    )

    try:
        result = engine.recall("read-only contract", fast=True)
        after_version = observer.execute("PRAGMA data_version").fetchone()[0]
    finally:
        observer.close()

    assert result.results
    assert attempted == [], "\n".join(attempted)
    assert after_version == before_version


def test_386_engine_recall_skips_write_capable_bandit(engine_with_mock_deps, monkeypatch):
    """The recall path cannot create a learning-db bandit play row."""
    engine = engine_with_mock_deps
    monkeypatch.setattr(engine._retrieval_engine, "recall", lambda *args, **kwargs: _response())
    monkeypatch.setattr(
        "superlocalmemory.core.recall_pipeline.apply_v2_bandit_ensemble",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            AssertionError("recall invoked the write-capable bandit"),
        ),
    )

    result = engine.recall("read-only contract", fast=True)

    assert result.results


def test_386_v1_adaptive_ranking_reads_feedback_without_schema_writes(
    tmp_path, monkeypatch,
):
    """Legacy adaptive ranking may inspect feedback but never initialize it."""
    from superlocalmemory.core import recall_pipeline

    learning_db = tmp_path / "learning.db"
    bootstrap = sqlite3.connect(learning_db)
    try:
        bootstrap.execute(
            "CREATE TABLE learning_feedback ("
            "id INTEGER PRIMARY KEY, profile_id TEXT NOT NULL)"
        )
        bootstrap.execute(
            "INSERT INTO learning_feedback(profile_id) VALUES ('default')"
        )
        bootstrap.commit()
    finally:
        bootstrap.close()

    original_connect = sqlite3.connect
    opens: list[tuple[object, dict]] = []
    denied: list[int] = []

    def audited_connect(path, *args, **kwargs):
        opens.append((path, dict(kwargs)))
        connection = original_connect(path, *args, **kwargs)

        def authorizer(action, _p1, _p2, _db, _source):
            if action in {
                sqlite3.SQLITE_INSERT,
                sqlite3.SQLITE_UPDATE,
                sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_TABLE,
                sqlite3.SQLITE_CREATE_INDEX,
                sqlite3.SQLITE_DROP_TABLE,
                sqlite3.SQLITE_DROP_INDEX,
            }:
                denied.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        connection.set_authorizer(authorizer)
        return connection

    monkeypatch.setattr(
        "superlocalmemory.infra.data_root.state_path",
        lambda name: learning_db,
    )
    monkeypatch.setattr(recall_pipeline.sqlite3, "connect", audited_connect)

    response = _response()
    result = recall_pipeline.apply_adaptive_ranking(
        response,
        "read-only contract",
        "default",
        config=MagicMock(),
    )

    assert result is response
    assert opens
    path, kwargs = opens[0]
    assert kwargs.get("uri") is True
    assert "mode=ro" in str(path)
    assert denied == []


def test_386_mcp_recall_has_no_post_query_write_side_effects():
    """MCP recall must not emit events or feedback into canonical state."""
    from superlocalmemory.mcp.tools_core import register_core_tools

    server = _MockServer()
    get_engine = MagicMock()
    register_core_tools(server, get_engine)
    recall = server.tools["recall"]
    pool = MagicMock()
    pool.recall.return_value = {
        "ok": True,
        "results": [{"fact_id": "386-read-only-fact", "content": "contract"}],
        "result_count": 1,
        "query_type": "semantic",
    }
    side_effects: list[str] = []

    def note_event(*_args, **_kwargs) -> None:
        side_effects.append("event")

    with (
        patch("superlocalmemory.mcp._daemon_proxy.choose_pool", return_value=pool),
        patch("superlocalmemory.mcp.tools_core._emit_event", side_effect=note_event),
        patch("superlocalmemory.hooks.session_registry.lookup_by_parent", return_value=None),
        patch("superlocalmemory.hooks.session_registry.most_recent_active", return_value=None),
    ):
        result = asyncio.run(recall("read-only contract"))

    assert result["success"] is True
    assert side_effects == []
    from superlocalmemory.mcp import tools_core
    assert not hasattr(tools_core, "_record_recall_hits")


def test_386_session_init_has_no_registration_or_event_side_effects():
    """Session start is a recall path and must not mutate user memory state."""
    from superlocalmemory.mcp._pool_adapter import PoolFact, PoolRecallItem, PoolRecallResponse
    from superlocalmemory.mcp.tools_active import register_active_tools

    server = _MockServer()
    engine = MagicMock()
    engine.profile_id = "default"
    engine._adaptive_learner.get_feedback_count.return_value = 0
    server_engine = MagicMock(return_value=engine)
    register_active_tools(server, server_engine)
    session_init = server.tools["session_init"]
    response = PoolRecallResponse(results=[PoolRecallItem(
        fact=PoolFact(fact_id="386-read-only-fact", content="contract", memory_id="m-1"),
        score=0.9,
    )])
    side_effects: list[str] = []
    rules = MagicMock()
    rules.should_recall.return_value = True
    rules.get_recall_config.return_value = {"relevance_threshold": 0.3}

    def note_event(*_args, **_kwargs) -> None:
        side_effects.append("event")

    with (
        patch("superlocalmemory.hooks.rules_engine.RulesEngine", return_value=rules),
        patch("superlocalmemory.mcp._pool_adapter.pool_recall", return_value=response),
        patch("superlocalmemory.mcp.tools_active._emit_event", side_effect=note_event),
    ):
        result = asyncio.run(session_init(query="read-only contract"))

    assert result["success"] is True
    assert side_effects == []


def test_386_emergency_session_recall_opens_memory_db_read_only(tmp_path, monkeypatch):
    """The emergency FTS fallback must use URI read-only mode plus an authorizer."""
    from superlocalmemory.mcp import tools_active

    db_path = tmp_path / "memory.db"
    bootstrap = sqlite3.connect(db_path)
    try:
        bootstrap.execute(
            "CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, content TEXT, "
            "memory_id TEXT, created_at TEXT, profile_id TEXT)"
        )
        bootstrap.execute(
            "CREATE VIRTUAL TABLE atomic_facts_fts USING fts5(fact_id, content)"
        )
        bootstrap.commit()
    finally:
        bootstrap.close()

    original_connect = sqlite3.connect
    opens: list[tuple[object, dict]] = []
    denied: list[int] = []

    def audited_connect(path, *args, **kwargs):
        opens.append((path, dict(kwargs)))
        conn = original_connect(path, *args, **kwargs)

        def authorizer(action, _p1, _p2, _db, _source):
            if action in {
                sqlite3.SQLITE_INSERT, sqlite3.SQLITE_UPDATE, sqlite3.SQLITE_DELETE,
                sqlite3.SQLITE_CREATE_TABLE, sqlite3.SQLITE_DROP_TABLE,
            }:
                denied.append(action)
                return sqlite3.SQLITE_DENY
            return sqlite3.SQLITE_OK

        conn.set_authorizer(authorizer)
        return conn

    from superlocalmemory.storage import read_connection

    monkeypatch.setattr(tools_active, "state_path", lambda name: db_path)
    monkeypatch.setattr(read_connection.sqlite3, "connect", audited_connect)

    tools_active._sqlite_emergency_recall("read-only contract", limit=3)

    assert opens, "session-init fallback did not open its test database"
    path, kwargs = opens[0]
    assert kwargs.get("uri") is True
    assert "mode=ro" in str(path)
    assert denied == []
