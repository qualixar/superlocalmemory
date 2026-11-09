# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An erasure that erases a second person is a breach, not thoroughness.

The code graph holds repository paths, file names and symbol names, and no
table in it carries a profile. Erasure wiped the whole file, which is exactly
right when one person's store is the only thing in it — and destroys everybody
else's records when it is not.

What is unambiguously one person's, even here, is the link between a code node
and one of their memories: those rows name a fact, and facts carry a profile.
Those go. The shape of the source code stays, because it describes a repository
rather than a person, and the receipt says so instead of implying a clean sweep.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

from superlocalmemory.compliance.gdpr import GDPRCompliance


class _Rows(list):
    """A result set whose rows answer to both a key and an index."""


class _FakeDB:
    """Just the two queries the code-graph decision asks."""

    def __init__(self, profiles: list[str], facts: dict[str, list[str]]):
        self._profiles = profiles
        self._facts = facts
        self.raise_on_profiles = False

    def execute(self, sql: str, params: tuple = ()):
        if "FROM profiles" in sql:
            if self.raise_on_profiles:
                raise sqlite3.OperationalError("no such table: profiles")
            return _Rows({"profile_id": p} for p in self._profiles)
        if "FROM atomic_facts" in sql:
            owner = params[0]
            return _Rows({"fact_id": f} for f in self._facts.get(owner, []))
        raise AssertionError(f"unexpected query: {sql}")


def _code_graph(tmp_path: Path) -> Path:
    """A graph shared by two people: shared structure, per-person links."""
    conn = sqlite3.connect(tmp_path / "code_graph.db")
    conn.executescript(
        """
        CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, name TEXT);
        CREATE TABLE graph_edges (edge_id TEXT PRIMARY KEY, source_node_id TEXT);
        CREATE TABLE graph_files (file_path TEXT PRIMARY KEY);
        CREATE TABLE code_memory_links (
            link_id TEXT PRIMARY KEY, code_node_id TEXT, slm_fact_id TEXT);
        INSERT INTO graph_nodes VALUES ('n1','parse'),('n2','render');
        INSERT INTO graph_edges VALUES ('e1','n1');
        INSERT INTO graph_files VALUES ('src/app.py');
        INSERT INTO code_memory_links VALUES
            ('L1','n1','alice-fact-1'),
            ('L2','n1','alice-fact-2'),
            ('L3','n2','bob-fact-1');
        """
    )
    conn.commit()
    conn.close()
    return tmp_path


def _counts(root: Path) -> dict[str, int]:
    conn = sqlite3.connect(root / "code_graph.db")
    try:
        return {
            t: conn.execute(f"SELECT COUNT(*) FROM {t}").fetchone()[0]
            for t in ("graph_nodes", "graph_edges", "graph_files", "code_memory_links")
        }
    finally:
        conn.close()


def test_a_sole_profile_still_gets_the_whole_graph_erased(tmp_path) -> None:
    root = _code_graph(tmp_path)
    db = _FakeDB(["alice"], {"alice": ["alice-fact-1", "alice-fact-2"]})
    gdpr = GDPRCompliance(db)

    result = gdpr._erase_code_graph(root, profile_id="alice", sole_profile=True)

    assert result["scope"] == "whole_graph"
    assert _counts(root) == {
        "graph_nodes": 0, "graph_edges": 0, "graph_files": 0, "code_memory_links": 0,
    }


def test_a_shared_graph_keeps_the_other_persons_records(tmp_path) -> None:
    root = _code_graph(tmp_path)
    db = _FakeDB(["alice", "bob"], {"alice": ["alice-fact-1", "alice-fact-2"]})
    gdpr = GDPRCompliance(db)

    result = gdpr._erase_code_graph(root, profile_id="alice", sole_profile=False)

    assert result["scope"] == "links_only"
    assert result["retained_reason"]
    after = _counts(root)
    assert after["code_memory_links"] == 1, "alice's links should be the only ones gone"
    assert after["graph_nodes"] == 2, "the repository structure is not alice's to erase"

    conn = sqlite3.connect(root / "code_graph.db")
    survivors = {r[0] for r in conn.execute("SELECT slm_fact_id FROM code_memory_links")}
    conn.close()
    assert survivors == {"bob-fact-1"}


def test_being_unable_to_count_profiles_errs_toward_keeping(tmp_path) -> None:
    """Not knowing is not a reason to delete somebody else's records."""
    db = _FakeDB(["alice", "bob"], {})
    db.raise_on_profiles = True
    gdpr = GDPRCompliance(db)
    assert gdpr._is_sole_profile("alice") is False


def test_one_profile_is_recognised_as_sole(tmp_path) -> None:
    gdpr = GDPRCompliance(_FakeDB(["alice"], {}))
    assert gdpr._is_sole_profile("alice") is True


def test_two_profiles_are_not_sole(tmp_path) -> None:
    gdpr = GDPRCompliance(_FakeDB(["alice", "bob"], {}))
    assert gdpr._is_sole_profile("alice") is False


def test_a_missing_graph_is_not_an_error(tmp_path) -> None:
    gdpr = GDPRCompliance(_FakeDB(["alice"], {}))
    result = gdpr._erase_code_graph(tmp_path, profile_id="alice", sole_profile=True)
    assert result["rows_deleted"] == 0
    assert "error" not in result
