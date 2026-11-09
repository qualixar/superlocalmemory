# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for ChangeAnalyzer — risk scoring, review context, git diff parsing."""

from __future__ import annotations

import time

import pytest

from superlocalmemory.code_graph.changes import (
    SECURITY_KEYWORDS,
    ChangeAnalyzer,
    ChangedNode,
    DiffHunk,
    ReviewContext,
    _parse_diff_output,
)
from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.models import (
    EdgeKind,
    GraphEdge,
    GraphNode,
    NodeKind,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def change_db(db: CodeGraphDatabase) -> CodeGraphDatabase:
    """DB with nodes and edges for change analysis."""
    now = time.time()
    nodes = [
        GraphNode(
            node_id="n1", kind=NodeKind.FUNCTION,
            name="authenticate_user",
            qualified_name="src/auth.py::authenticate_user",
            file_path="src/auth.py", line_start=10, line_end=30,
            language="python", community_id=0,
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n2", kind=NodeKind.FUNCTION,
            name="get_user",
            qualified_name="src/users.py::get_user",
            file_path="src/users.py", line_start=5, line_end=25,
            language="python", community_id=1,
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="n3", kind=NodeKind.FUNCTION,
            name="process_data",
            qualified_name="src/data.py::process_data",
            file_path="src/data.py", line_start=1, line_end=50,
            language="python", community_id=2,
            created_at=now, updated_at=now,
        ),
        GraphNode(
            node_id="test1", kind=NodeKind.FUNCTION,
            name="test_auth", is_test=True,
            qualified_name="tests/test_auth.py::test_auth",
            file_path="tests/test_auth.py", line_start=1, line_end=20,
            language="python", created_at=now, updated_at=now,
        ),
    ]
    for node in nodes:
        db.upsert_node(node)

    edges = [
        # n1 is tested by test1
        GraphEdge(
            edge_id="e1", kind=EdgeKind.TESTED_BY,
            source_node_id="n1", target_node_id="test1",
            file_path="src/auth.py", line=10,
            created_at=now, updated_at=now,
        ),
        # n2 calls n1 (cross-community)
        GraphEdge(
            edge_id="e2", kind=EdgeKind.CALLS,
            source_node_id="n2", target_node_id="n1",
            file_path="src/users.py", line=15,
            created_at=now, updated_at=now,
        ),
        # n3 calls n2
        GraphEdge(
            edge_id="e3", kind=EdgeKind.CALLS,
            source_node_id="n3", target_node_id="n2",
            file_path="src/data.py", line=20,
            created_at=now, updated_at=now,
        ),
    ]
    for edge in edges:
        db.upsert_edge(edge)

    return db


@pytest.fixture
def analyzer(change_db: CodeGraphDatabase) -> ChangeAnalyzer:
    """ChangeAnalyzer instance with populated DB."""
    return ChangeAnalyzer(change_db)


# ---------------------------------------------------------------------------
# Risk Scoring Tests
# ---------------------------------------------------------------------------

class TestRiskScoring:
    """Tests for compute_risk_score()."""

    def test_security_keyword_increases_risk(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """Nodes with security keywords have higher risk."""
        # authenticate_user contains "auth" (security keyword)
        node = {
            "node_id": "n1", "name": "authenticate_user",
            "kind": "function", "file_path": "src/auth.py",
            "line_start": 10, "line_end": 30,
        }
        risk = analyzer.compute_risk_score(node)
        assert risk >= 0.20  # Security keyword contributes 0.20

    def test_untested_node_higher_risk(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """Untested nodes have higher test_coverage risk component."""
        # n2 (get_user) has no TESTED_BY edge
        untested = {
            "node_id": "n2", "name": "get_user",
            "kind": "function", "file_path": "src/users.py",
            "line_start": 5, "line_end": 25,
        }
        # n1 (authenticate_user) has TESTED_BY edge
        tested = {
            "node_id": "n1", "name": "authenticate_user",
            "kind": "function", "file_path": "src/auth.py",
            "line_start": 10, "line_end": 30,
        }
        # Untested contributes 0.30 vs tested 0.05 for test component
        untested_risk = analyzer.compute_risk_score(untested)
        tested_risk = analyzer.compute_risk_score(tested)
        # The tested node has security keywords (+0.20) but lower test risk
        # So total difference depends on all factors
        assert untested_risk > 0  # At minimum has test gap component
        assert untested_risk > tested_risk

    def test_risk_score_bounded(self, analyzer: ChangeAnalyzer) -> None:
        """Risk score is between 0 and 1."""
        node = {
            "node_id": "n3", "name": "process_data",
            "kind": "function", "file_path": "src/data.py",
            "line_start": 1, "line_end": 50,
        }
        risk = analyzer.compute_risk_score(node)
        assert 0.0 <= risk <= 1.0

    def test_caller_count_contributes(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """Nodes with more callers score higher on caller_count."""
        # n1 has 1 caller (n2), n3 has 0 callers
        node_with_caller = {
            "node_id": "n1", "name": "safe_func",
            "kind": "function", "file_path": "src/auth.py",
            "line_start": 10, "line_end": 30,
        }
        # caller_count_score for 1 caller = min(1/20, 0.10) = 0.05
        risk = analyzer.compute_risk_score(node_with_caller)
        assert risk > 0


# ---------------------------------------------------------------------------
# Analyze Changes Tests
# ---------------------------------------------------------------------------

class TestAnalyzeChanges:
    """Tests for analyze_changes()."""

    def test_analyze_single_file(self, analyzer: ChangeAnalyzer) -> None:
        """Analyze changes to a single file."""
        result = analyzer.analyze_changes(["src/auth.py"])
        assert isinstance(result, ReviewContext)
        assert len(result.changed_nodes) >= 1
        # authenticate_user is in src/auth.py
        node_ids = {n.node_id for n in result.changed_nodes}
        assert "n1" in node_ids

    def test_analyze_multiple_files(self, analyzer: ChangeAnalyzer) -> None:
        """Analyze changes to multiple files."""
        result = analyzer.analyze_changes(["src/auth.py", "src/users.py"])
        assert len(result.changed_nodes) >= 2

    def test_analyze_empty_files(self, analyzer: ChangeAnalyzer) -> None:
        """Empty file list returns empty context."""
        result = analyzer.analyze_changes([])
        assert result.summary == "No changes detected."
        assert result.changed_nodes == ()
        assert result.overall_risk == 0.0

    def test_analyze_nonexistent_file(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """Nonexistent file returns empty nodes."""
        result = analyzer.analyze_changes(["nonexistent.py"])
        assert result.changed_nodes == ()

    def test_test_gaps_identified(self, analyzer: ChangeAnalyzer) -> None:
        """Nodes without test coverage are flagged as test gaps."""
        result = analyzer.analyze_changes(
            ["src/auth.py", "src/users.py", "src/data.py"]
        )
        gap_ids = {n.node_id for n in result.test_gaps}
        # n2 (get_user) and n3 (process_data) have no tests
        assert "n2" in gap_ids or "n3" in gap_ids

    def test_review_priorities_sorted(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """Review priorities are sorted by risk (highest first)."""
        result = analyzer.analyze_changes(
            ["src/auth.py", "src/users.py", "src/data.py"]
        )
        priorities = result.review_priorities
        for i in range(len(priorities) - 1):
            assert priorities[i].risk_score >= priorities[i + 1].risk_score

    def test_overall_risk_is_max(self, analyzer: ChangeAnalyzer) -> None:
        """Overall risk is the max of individual node risks."""
        result = analyzer.analyze_changes(
            ["src/auth.py", "src/users.py"]
        )
        if result.changed_nodes:
            max_risk = max(n.risk_score for n in result.changed_nodes)
            assert result.overall_risk == max_risk


# ---------------------------------------------------------------------------
# Review Context Tests
# ---------------------------------------------------------------------------

class TestReviewContext:
    """Tests for get_review_context()."""

    def test_review_context_same_as_analyze(
        self, analyzer: ChangeAnalyzer
    ) -> None:
        """get_review_context is equivalent to analyze_changes."""
        r1 = analyzer.analyze_changes(["src/auth.py"])
        r2 = analyzer.get_review_context(["src/auth.py"])
        assert r1.summary == r2.summary
        assert len(r1.changed_nodes) == len(r2.changed_nodes)


# ---------------------------------------------------------------------------
# Git Diff Parsing Tests
# ---------------------------------------------------------------------------

class TestGitDiffParsing:
    """Tests for _parse_diff_output()."""

    def test_parse_simple_diff(self) -> None:
        """Parse a simple unified diff."""
        diff = (
            "diff --git a/src/auth.py b/src/auth.py\n"
            "--- a/src/auth.py\n"
            "+++ b/src/auth.py\n"
            "@@ -10,3 +10,5 @@ def authenticate():\n"
            "+    new line 1\n"
            "+    new line 2\n"
        )
        hunks = _parse_diff_output(diff)
        assert len(hunks) == 1
        assert hunks[0].file_path == "src/auth.py"
        assert hunks[0].start_line == 10
        assert hunks[0].end_line == 14

    def test_parse_multiple_files(self) -> None:
        """Parse diff spanning multiple files."""
        diff = (
            "diff --git a/a.py b/a.py\n"
            "--- a/a.py\n"
            "+++ b/a.py\n"
            "@@ -5,2 +5,3 @@\n"
            "+new\n"
            "diff --git a/b.py b/b.py\n"
            "--- a/b.py\n"
            "+++ b/b.py\n"
            "@@ -1,1 +1,2 @@\n"
            "+another\n"
        )
        hunks = _parse_diff_output(diff)
        assert len(hunks) == 2
        files = {h.file_path for h in hunks}
        assert "a.py" in files
        assert "b.py" in files

    def test_parse_single_line_addition(self) -> None:
        """Parse diff with single line addition."""
        diff = (
            "+++ b/file.py\n"
            "@@ -0,0 +1 @@\n"
            "+new line\n"
        )
        hunks = _parse_diff_output(diff)
        assert len(hunks) == 1
        assert hunks[0].start_line == 1
        assert hunks[0].end_line == 1

    def test_parse_empty_diff(self) -> None:
        """Empty diff returns empty list."""
        assert _parse_diff_output("") == []

    def test_parse_multiple_hunks(self) -> None:
        """Parse diff with multiple hunks in same file."""
        diff = (
            "+++ b/file.py\n"
            "@@ -5,3 +5,4 @@\n"
            "+line\n"
            "@@ -20,2 +21,3 @@\n"
            "+another\n"
        )
        hunks = _parse_diff_output(diff)
        assert len(hunks) == 2
        assert hunks[0].start_line == 5
        assert hunks[1].start_line == 21


# ---------------------------------------------------------------------------
# Security Keywords Tests
# ---------------------------------------------------------------------------

class TestSecurityKeywords:
    """Tests for security keyword constant."""

    def test_security_keywords_frozen(self) -> None:
        """Security keywords are a frozenset."""
        assert isinstance(SECURITY_KEYWORDS, frozenset)

    def test_security_keywords_comprehensive(self) -> None:
        """Contains expected security terms."""
        expected = {"auth", "password", "token", "sql", "encrypt", "admin"}
        assert expected.issubset(SECURITY_KEYWORDS)


# ---------------------------------------------------------------------------
# Frozen Dataclass Tests
# ---------------------------------------------------------------------------

class TestDataclasses:
    """Test that result types are frozen."""

    def test_diff_hunk_frozen(self) -> None:
        """DiffHunk is immutable."""
        h = DiffHunk(file_path="a.py", start_line=1, end_line=5)
        with pytest.raises(AttributeError):
            h.start_line = 10  # type: ignore[misc]

    def test_changed_node_frozen(self) -> None:
        """ChangedNode is immutable."""
        n = ChangedNode(
            node_id="n1", name="func", kind="function",
            file_path="a.py", line_start=1, line_end=10,
            risk_score=0.5,
        )
        with pytest.raises(AttributeError):
            n.risk_score = 0.9  # type: ignore[misc]

    def test_review_context_frozen(self) -> None:
        """ReviewContext is immutable."""
        r = ReviewContext(
            summary="test", changed_nodes=(), test_gaps=(),
            review_priorities=(), overall_risk=0.0,
        )
        with pytest.raises(AttributeError):
            r.overall_risk = 1.0  # type: ignore[misc]
