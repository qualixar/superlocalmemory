# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Tests for IncrementalUpdater — hash-based change detection."""

# ruff: noqa: E402 (the imports below deliberately follow the rustworkx availability guard and skip marker)

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest

try:
    import rustworkx  # noqa: F401
    HAS_RUSTWORKX = True
except ImportError:
    HAS_RUSTWORKX = False

pytestmark = pytest.mark.skipif(
    not HAS_RUSTWORKX, reason="rustworkx not installed"
)

from superlocalmemory.code_graph.database import CodeGraphDatabase
from superlocalmemory.code_graph.graph_engine import GraphEngine
from superlocalmemory.code_graph.graph_store import GraphStore
from superlocalmemory.code_graph.incremental import (
    IncrementalUpdater,
    UpdateResult,
)
from superlocalmemory.code_graph.models import (
    EdgeKind,
    FileRecord,
    GraphEdge,
    GraphNode,
    NodeKind,
    ParseResult,
)

# ---------------------------------------------------------------------------
# Mock parser
# ---------------------------------------------------------------------------

class MockParser:
    """A mock parser that returns deterministic results based on file content."""

    def __init__(self, repo_root: Path) -> None:
        self._repo_root = repo_root
        self.parse_count = 0

    def parse_file(self, file_path: str, repo_root: Path) -> ParseResult:
        self.parse_count += 1
        abs_path = repo_root / file_path
        content = abs_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()

        # Generate a simple node from the file
        node = GraphNode(
            node_id=f"node_{file_path}",
            kind=NodeKind.FUNCTION,
            name=Path(file_path).stem,
            qualified_name=f"mod.{Path(file_path).stem}",
            file_path=file_path,
            line_start=1,
            line_end=5,
            language="python",
        )

        file_record = FileRecord(
            file_path=file_path,
            content_hash=content_hash,
            mtime=abs_path.stat().st_mtime,
            language="python",
            node_count=1,
            edge_count=0,
        )

        return ParseResult(
            file_path=file_path,
            nodes=(node,),
            edges=(),
            file_record=file_record,
        )


class MockParserWithEdges:
    """Parser that creates import edges between files."""

    def __init__(self, edges_map: dict[str, list[tuple[str, str]]]) -> None:
        # edges_map: file_path -> [(target_file, target_qname)]
        self._edges_map = edges_map
        self.parse_count = 0

    def parse_file(self, file_path: str, repo_root: Path) -> ParseResult:
        self.parse_count += 1
        abs_path = repo_root / file_path
        content = abs_path.read_bytes()
        content_hash = hashlib.sha256(content).hexdigest()

        node = GraphNode(
            node_id=f"node_{file_path}",
            kind=NodeKind.FUNCTION,
            name=Path(file_path).stem,
            qualified_name=f"mod.{Path(file_path).stem}",
            file_path=file_path,
            line_start=1,
            line_end=5,
            language="python",
        )

        edges: list[GraphEdge] = []
        for i, (tgt_file, tgt_qname) in enumerate(
            self._edges_map.get(file_path, [])
        ):
            edges.append(GraphEdge(
                edge_id=f"edge_{file_path}_{i}",
                kind=EdgeKind.IMPORTS,
                source_node_id=f"node_{file_path}",
                target_node_id=f"node_{tgt_file}",
                file_path=file_path,
                line=1,
                confidence=1.0,
            ))

        file_record = FileRecord(
            file_path=file_path,
            content_hash=content_hash,
            mtime=abs_path.stat().st_mtime,
            language="python",
            node_count=1,
            edge_count=len(edges),
        )

        return ParseResult(
            file_path=file_path,
            nodes=(node,),
            edges=tuple(edges),
            file_record=file_record,
        )


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def store(db: CodeGraphDatabase) -> GraphStore:
    return GraphStore(db)


@pytest.fixture
def engine(store: GraphStore) -> GraphEngine:
    return GraphEngine(store)


@pytest.fixture
def updater(store: GraphStore, engine: GraphEngine) -> IncrementalUpdater:
    return IncrementalUpdater(store, engine)


@pytest.fixture
def repo_root(tmp_path: Path) -> Path:
    repo = tmp_path / "repo"
    repo.mkdir()
    return repo


# ---------------------------------------------------------------------------
# Tests: basic update
# ---------------------------------------------------------------------------

class TestBasicUpdate:
    def test_parse_new_file(
        self, updater: IncrementalUpdater, repo_root: Path
    ) -> None:
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/foo.py").write_text("def foo(): pass", encoding="utf-8")

        parser = MockParser(repo_root)
        result = updater.update(["src/foo.py"], parser, repo_root)

        assert result.parsed == 1
        assert result.skipped == 0
        assert result.deleted == 0

    def test_skip_unchanged_file(
        self, updater: IncrementalUpdater, repo_root: Path,
        store: GraphStore,
    ) -> None:
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/foo.py").write_text("def foo(): pass", encoding="utf-8")

        parser = MockParser(repo_root)
        # First update
        updater.update(["src/foo.py"], parser, repo_root)
        # Second update — same content
        result = updater.update(["src/foo.py"], parser, repo_root)

        assert result.parsed == 0
        assert result.skipped == 1

    def test_reparse_changed_file(
        self, updater: IncrementalUpdater, repo_root: Path,
    ) -> None:
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/foo.py").write_text("def foo(): pass", encoding="utf-8")

        parser = MockParser(repo_root)
        updater.update(["src/foo.py"], parser, repo_root)

        # Modify file
        (repo_root / "src/foo.py").write_text("def foo(): return 42", encoding="utf-8")
        result = updater.update(["src/foo.py"], parser, repo_root)

        assert result.parsed == 1
        assert result.skipped == 0


# ---------------------------------------------------------------------------
# Tests: file deletion
# ---------------------------------------------------------------------------

class TestFileDeletion:
    def test_delete_missing_file(
        self, updater: IncrementalUpdater, repo_root: Path,
        store: GraphStore,
    ) -> None:
        # Pre-populate a file
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/foo.py").write_text("def foo(): pass", encoding="utf-8")

        parser = MockParser(repo_root)
        updater.update(["src/foo.py"], parser, repo_root)

        # Delete the file from disk
        (repo_root / "src/foo.py").unlink()

        # Update should detect deletion
        result = updater.update(["src/foo.py"], parser, repo_root)

        assert result.deleted == 1
        assert store.get_file_record("src/foo.py") is None


# ---------------------------------------------------------------------------
# Tests: dependent tracing
# ---------------------------------------------------------------------------

class TestDependentTracing:
    def test_dependents_reparsed(
        self, updater: IncrementalUpdater, repo_root: Path,
        store: GraphStore,
    ) -> None:
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/a.py").write_text("def a(): pass", encoding="utf-8")
        (repo_root / "src/b.py").write_text("from a import a", encoding="utf-8")

        # b.py imports from a.py
        parser = MockParserWithEdges({
            "src/b.py": [("src/a.py", "mod.a")],
        })

        # First: index both files
        updater.update(["src/a.py", "src/b.py"], parser, repo_root)
        initial_count = parser.parse_count

        # Now change a.py → b.py should be reparsed as dependent
        (repo_root / "src/a.py").write_text("def a(): return 1", encoding="utf-8")
        result = updater.update(["src/a.py"], parser, repo_root)

        assert result.parsed == 1  # a.py
        assert result.dependents_parsed == 1  # b.py
        assert parser.parse_count == initial_count + 2  # a.py and its dependent b.py


# ---------------------------------------------------------------------------
# Tests: empty input
# ---------------------------------------------------------------------------

class TestEmptyInput:
    def test_empty_list(self, updater: IncrementalUpdater, repo_root: Path) -> None:
        parser = MockParser(repo_root)
        result = updater.update([], parser, repo_root)
        assert result == UpdateResult()


# ---------------------------------------------------------------------------
# Tests: error handling
# ---------------------------------------------------------------------------

class TestErrorHandling:
    def test_unreadable_file(
        self, updater: IncrementalUpdater, repo_root: Path,
    ) -> None:
        # File path exists but as a directory (can't read_bytes)
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/bad.py").mkdir()

        parser = MockParser(repo_root)
        result = updater.update(["src/bad.py"], parser, repo_root)

        assert result.parsed == 0
        assert len(result.errors) > 0

    def test_engine_invalidated(
        self, updater: IncrementalUpdater, repo_root: Path,
        engine: GraphEngine,
    ) -> None:
        (repo_root / "src").mkdir(parents=True)
        (repo_root / "src/foo.py").write_text("def foo(): pass", encoding="utf-8")

        parser = MockParser(repo_root)
        # Build the graph first
        engine.build_graph()

        # Update triggers invalidation
        updater.update(["src/foo.py"], parser, repo_root)

        # Next build_graph should rebuild (version mismatch)
        # Just verify it doesn't crash
        g = engine.build_graph()
        assert g.num_nodes() >= 1
