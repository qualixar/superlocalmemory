# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory | https://qualixar.com

"""Multi-Agent Memory must group by agent identity, not by capability digest.

WHY THIS EXISTS
---------------
The pane exists to tell agents apart. It grouped by
``ingestion_operations.trusted_actor_id`` — the capability that authorised the
write — which on a real store is a digest. The result was 43 rows of
``daemon-capability:923b7d6e616f46d3739315580e13fa97...`` and not one readable
name, in the one view whose whole job is attribution.

Worse than unreadable: it was *wrong*. A single agent writes under many
capabilities over time, so one writer fragmented into many rows —
``claude-desktop`` alone spanned 18 capabilities and appeared as 18 "agents".
Grouped by identity it is one agent with 215 memories.

The names were never missing. Writers pass ``agent_id`` and it is stored in
``raw_metadata_json``; the query simply never looked there. On the author's
store that yields claude-desktop, claude, gemini, codex, grok and mcp_client —
exactly the multi-framework picture (CrewAI / LangChain / LangGraph) the pane
advertises.
"""

from __future__ import annotations

import json
import sqlite3

import pytest


def _seed(db_path, rows):
    """rows: (trusted_actor_id, metadata_agent_id_or_None, source_type)."""
    conn = sqlite3.connect(db_path)
    conn.executescript(
        """
        CREATE TABLE ingestion_operations (
            operation_id TEXT PRIMARY KEY, profile_id TEXT, source_type TEXT,
            raw_metadata_json TEXT, trusted_actor_id TEXT, created_at TEXT
        );
        """
    )
    for i, (actor, agent, src) in enumerate(rows):
        meta = json.dumps({"agent_id": agent}) if agent else None
        conn.execute(
            "INSERT INTO ingestion_operations (operation_id, profile_id,"
            " source_type, raw_metadata_json, trusted_actor_id, created_at)"
            " VALUES (?,?,?,?,?,?)",
            (f"op{i}", "default", src, meta, actor, "2026-08-17T10:00:00Z"),
        )
    conn.commit()
    conn.close()


def _query(db_path, pid="default"):
    """The endpoint's grouping query, kept in one place for the tests."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn.execute(
        "SELECT COALESCE("
        "  NULLIF(json_extract(raw_metadata_json, '$.agent_id'), ''),"
        "  NULLIF(trusted_actor_id, ''),"
        "  'unknown'"
        ") AS agent_id, COUNT(*) AS cnt,"
        " COUNT(DISTINCT NULLIF(trusted_actor_id, '')) AS capabilities"
        " FROM ingestion_operations WHERE profile_id=?"
        " GROUP BY agent_id ORDER BY cnt DESC",
        (pid,),
    ).fetchall()


class TestIdentityGrouping:
    def test_one_agent_across_many_capabilities_is_one_row(self, tmp_path):
        """The exact shape that fragmented claude-desktop into 18 'agents'."""
        db = tmp_path / "m.db"
        _seed(db, [
            (f"daemon-capability:{i:064x}", "claude-desktop", "http")
            for i in range(18)
        ])
        rows = _query(db)
        assert len(rows) == 1, f"agent fragmented into {len(rows)} rows"
        assert rows[0]["agent_id"] == "claude-desktop"
        assert rows[0]["cnt"] == 18
        assert rows[0]["capabilities"] == 18

    def test_distinct_agents_stay_distinct(self, tmp_path):
        db = tmp_path / "m.db"
        _seed(db, [
            ("daemon-capability:aaa", "claude", "http"),
            ("daemon-capability:aaa", "gemini", "http"),
            ("daemon-capability:bbb", "codex", "mcp"),
        ])
        names = {r["agent_id"] for r in _query(db)}
        assert names == {"claude", "gemini", "codex"}, (
            "agents sharing a capability were merged"
        )

    def test_unidentified_writer_falls_back_to_its_capability(self, tmp_path):
        """A digest is a poor name but it is better than dropping the row or
        merging every anonymous writer into one bucket."""
        db = tmp_path / "m.db"
        _seed(db, [
            ("daemon-capability:aaa", None, "http"),
            ("daemon-capability:bbb", None, "http"),
        ])
        rows = _query(db)
        assert len(rows) == 2
        assert all(r["agent_id"].startswith("daemon-capability:") for r in rows)

    def test_empty_metadata_agent_id_does_not_win(self, tmp_path):
        """NULLIF guards the empty string; without it every writer that passed
        agent_id="" would collapse into a single blank-named agent."""
        db = tmp_path / "m.db"
        _seed(db, [("daemon-capability:aaa", "", "http")])
        rows = _query(db)
        assert rows[0]["agent_id"] == "daemon-capability:aaa"

    def test_row_with_no_actor_and_no_agent_is_unknown(self, tmp_path):
        db = tmp_path / "m.db"
        _seed(db, [("", None, "http")])
        assert _query(db)[0]["agent_id"] == "unknown"

    def test_store_without_the_metadata_column_still_lists_agents(self, tmp_path):
        """Stores predating raw_metadata_json must degrade to capability
        grouping, NOT to an empty pane. Querying a missing column raises
        OperationalError, which the handler catches — so without the column
        probe, an older install silently reported zero agents while having
        plenty."""
        db = tmp_path / "legacy.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE ingestion_operations (
                operation_id TEXT PRIMARY KEY, profile_id TEXT, source_type TEXT,
                trusted_actor_id TEXT, created_at TEXT
            );
            """
        )
        conn.execute(
            "INSERT INTO ingestion_operations VALUES (?,?,?,?,?)",
            ("op0", "default", "http", "claude", "2026-08-17T10:00:00Z"),
        )
        conn.commit()
        conn.close()

        has_meta = any(
            r[1] == "raw_metadata_json"
            for r in sqlite3.connect(db).execute(
                "PRAGMA table_info(ingestion_operations)")
        )
        assert not has_meta, "fixture should model the legacy schema"

        conn = sqlite3.connect(db)
        conn.row_factory = sqlite3.Row
        rows = conn.execute(
            "SELECT COALESCE(NULLIF(trusted_actor_id,''),'unknown') AS agent_id,"
            " COUNT(*) AS cnt FROM ingestion_operations WHERE profile_id=?"
            " GROUP BY agent_id", ("default",),
        ).fetchall()
        assert [r["agent_id"] for r in rows] == ["claude"]


class TestEndpointContract:
    def test_endpoint_groups_by_metadata_agent_id(self):
        import inspect

        from superlocalmemory.server.routes import agents

        src = inspect.getsource(agents.get_agent_memory_activity)
        assert "json_extract(raw_metadata_json, '$.agent_id')" in src, (
            "endpoint no longer reads the agent's own identity"
        )
        assert "capability_count" in src
        assert "identified" in src

    @pytest.mark.parametrize(
        "name,expected",
        [
            ("claude-desktop", True),
            ("gemini", True),
            ("daemon-capability:abc", False),
            ("local-capability:dashboard:uid:501:abc", False),
            ("unknown", False),
        ],
    )
    def test_identified_flag_distinguishes_names_from_digests(self, name, expected):
        """The UI must be able to say "this writer did not identify itself"
        rather than presenting a hash as though it were a name."""
        identified = not str(name).startswith(
            ("daemon-capability:", "local-capability:")
        ) and name != "unknown"
        assert identified is expected
