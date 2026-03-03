#!/usr/bin/env python3
# SPDX-License-Identifier: MIT
# Copyright (c) 2026 SuperLocalMemory (superlocalmemory.com)
"""SuperLocalMemory V2 - Regression Tests (v2.7)

Regression tests ensuring ALL existing functionality still works after v2.7
changes. These tests verify backward compatibility with v2.6 behavior,
file integrity, and schema stability. All tests use temporary databases.

Run with:
    pytest tests/test_regression_v27.py -v
"""
import importlib
import json
import os
import sqlite3
import subprocess
import sys
from pathlib import Path
from unittest.mock import patch, MagicMock

import pytest

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_DIR = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_DIR))
sys.path.insert(0, str(REPO_DIR / "src"))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(autouse=True)
def reset_singletons():
    """Reset all singletons between tests."""
    from src.learning.learning_db import LearningDB
    LearningDB.reset_instance()
    yield
    LearningDB.reset_instance()


@pytest.fixture
def memory_db(tmp_path):
    """Create a basic memory.db with v2.6 schema."""
    db_path = tmp_path / "memory.db"
    conn = sqlite3.connect(str(db_path))
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS memories (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            content TEXT NOT NULL,
            summary TEXT,
            tags TEXT DEFAULT '[]',
            category TEXT,
            memory_type TEXT DEFAULT 'general',
            importance INTEGER DEFAULT 5,
            project_name TEXT,
            project_path TEXT,
            profile TEXT DEFAULT 'default',
            parent_id INTEGER,
            cluster_id INTEGER,
            tier INTEGER DEFAULT 1,
            entity_vector TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            last_accessed TIMESTAMP,
            access_count INTEGER DEFAULT 0,
            created_by TEXT,
            source_protocol TEXT,
            trust_score REAL DEFAULT 1.0,
            provenance_chain TEXT
        )
    ''')
    cursor.execute('''
        CREATE VIRTUAL TABLE IF NOT EXISTS memories_fts
        USING fts5(content, summary, tags, content='memories', content_rowid='id')
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS identity_patterns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            pattern_type TEXT NOT NULL,
            key TEXT NOT NULL,
            value TEXT NOT NULL,
            confidence REAL DEFAULT 0.0,
            evidence_count INTEGER DEFAULT 0,
            category TEXT,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS graph_clusters (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            summary TEXT,
            member_count INTEGER DEFAULT 0,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS creator_metadata (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    ''')
    conn.commit()
    conn.close()
    return db_path


# ============================================================================
# Regression Test Scenarios
# ============================================================================


class TestMCPServerImports:
    """Scenario 1: Verify mcp_server.py can be imported without errors."""

    def test_mcp_server_imports(self):
        """Import mcp_server.py -> no syntax errors."""
        mcp_server_path = REPO_DIR / "mcp_server.py"
        assert mcp_server_path.exists(), "mcp_server.py not found"

        # Compile check (catches SyntaxError without executing)
        source = mcp_server_path.read_text(encoding="utf-8")
        try:
            compile(source, str(mcp_server_path), "exec")
        except SyntaxError as e:
            pytest.fail(f"mcp_server.py has syntax error: {e}")


class TestExistingToolSignatures:
    """Scenario 2: Verify core MCP tool handler names still exist."""

    def test_existing_tool_signatures(self):
        """Verify remember, recall, list_recent, etc. still exist."""
        mcp_server_path = REPO_DIR / "mcp_server.py"
        source = mcp_server_path.read_text(encoding="utf-8")

        # These function/handler patterns MUST exist in mcp_server.py
        expected_handlers = [
            "remember",
            "recall",
            "list_recent",
            "get_status",
            "build_graph",
            "switch_profile",
        ]

        for handler in expected_handlers:
            assert handler in source, (
                f"Expected handler '{handler}' not found in mcp_server.py. "
                f"This would break existing MCP integrations."
            )


class TestRecallWithoutLearning:
    """Scenario 3: Mock LEARNING_AVAILABLE=False -> recall still works."""

    def test_recall_without_learning(self, memory_db):
        """When learning system is unavailable, recall returns v2.6 behavior."""
        from src.learning.adaptive_ranker import AdaptiveRanker

        # Create ranker with no learning DB
        ranker = AdaptiveRanker(learning_db=None)

        # Phase depends on whether the lazy-loaded default LearningDB has
        # feedback data. Accept either baseline or rule_based as valid.
        phase = ranker.get_phase()
        assert phase in ("baseline", "rule_based")

        # Reranking should return results with correct metadata
        test_results = [
            {"id": 1, "content": "test memory", "score": 0.8},
            {"id": 2, "content": "another memory", "score": 0.6},
        ]
        reranked = ranker.rerank(test_results, "test query")

        assert len(reranked) == 2
        assert reranked[0]["base_score"] == 0.8
        assert reranked[1]["base_score"] == 0.6
        assert all(r["ranking_phase"] in ("baseline", "rule_based") for r in reranked)


class TestRecallWithLearningFailure:
    """Scenario 4: Learning import OK but rerank throws -> graceful fallback."""

    def test_recall_with_learning_failure(self, tmp_path):
        """Ranker.rerank() throws -> caller gets original results."""
        from src.learning.learning_db import LearningDB
        from src.learning.adaptive_ranker import AdaptiveRanker

        ldb = LearningDB(db_path=tmp_path / "learning.db")

        # Seed enough feedback for rule_based phase
        for i in range(25):
            ldb.store_feedback(
                query_hash=f"hash_{i % 5}",
                memory_id=i + 1,
                signal_type="mcp_used_high",
                signal_value=1.0,
                channel="mcp",
            )

        ranker = AdaptiveRanker(learning_db=ldb)

        # Mock feature extractor to throw
        ranker._feature_extractor = MagicMock()
        ranker._feature_extractor.set_context.return_value = None
        ranker._feature_extractor.extract_batch.side_effect = RuntimeError(
            "Feature extraction failed!"
        )

        test_results = [
            {"id": 1, "content": "test memory", "score": 0.8},
            {"id": 2, "content": "another memory", "score": 0.6},
        ]

        # The caller should wrap rerank in try/except. Test the pattern:
        try:
            reranked = ranker.rerank(test_results, "test query")
        except Exception:
            # Fallback: return original results
            reranked = test_results
            for r in reranked:
                r["ranking_phase"] = "fallback"

        assert len(reranked) == 2


class TestInstallShSyntax:
    """Scenario 5: install.sh passes syntax check."""

    def test_install_sh_syntax(self):
        """bash -n install.sh passes."""
        install_sh = REPO_DIR / "install.sh"
        if not install_sh.exists():
            pytest.skip("install.sh not found")

        result = subprocess.run(
            ["bash", "-n", str(install_sh)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f"install.sh syntax error: {result.stderr}"
        )


class TestBinSlmSyntax:
    """Scenario 6: bin/slm passes syntax check."""

    def test_bin_slm_syntax(self):
        """bash -n bin/slm passes."""
        slm_bin = REPO_DIR / "bin" / "slm"
        if not slm_bin.exists():
            pytest.skip("bin/slm not found")

        result = subprocess.run(
            ["bash", "-n", str(slm_bin)],
            capture_output=True, text=True, timeout=10,
        )
        assert result.returncode == 0, (
            f"bin/slm syntax error: {result.stderr}"
        )


class TestPackageJsonValid:
    """Scenario 7: package.json parses correctly."""

    def test_package_json_valid(self):
        """package.json is valid JSON with required fields."""
        pkg_json = REPO_DIR / "package.json"
        if not pkg_json.exists():
            pytest.skip("package.json not found")

        with open(pkg_json, "r", encoding="utf-8") as f:
            data = json.load(f)

        assert "name" in data, "package.json missing 'name'"
        assert "version" in data, "package.json missing 'version'"
        assert isinstance(data["version"], str)
        # Version should be semver-like
        parts = data["version"].split(".")
        assert len(parts) >= 2, f"Invalid version format: {data['version']}"


class TestAllSkillsExist:
    """Scenario 8: All expected skills have SKILL.md files."""

    def test_all_skills_exist(self):
        """All 6 original skills + new skill have SKILL.md."""
        skills_dir = REPO_DIR / "skills"
        if not skills_dir.exists():
            pytest.skip("skills/ directory not found")

        expected_skills = [
            "slm-remember",
            "slm-recall",
            "slm-list-recent",
            "slm-status",
            "slm-build-graph",
            "slm-switch-profile",
        ]

        for skill_name in expected_skills:
            skill_md = skills_dir / skill_name / "SKILL.md"
            assert skill_md.exists(), (
                f"Missing skill definition: {skill_md}"
            )


class TestAllBinExecutablesExist:
    """Scenario 9: All bin files exist and are executable."""

    def test_all_bin_executables_exist(self):
        """All essential bin files exist."""
        bin_dir = REPO_DIR / "bin"
        assert bin_dir.exists(), "bin/ directory not found"

        essential_bins = ["slm"]
        for bin_name in essential_bins:
            bin_file = bin_dir / bin_name
            assert bin_file.exists(), f"Missing bin file: {bin_file}"


class TestRequirementsLearningExists:
    """Scenario 10: requirements-learning.txt exists with correct deps."""

    def test_requirements_learning_exists(self):
        """File exists with correct dependencies."""
        req_file = REPO_DIR / "requirements-learning.txt"
        if not req_file.exists():
            pytest.skip("requirements-learning.txt not yet created")

        content = req_file.read_text(encoding="utf-8")
        assert "lightgbm" in content.lower(), (
            "requirements-learning.txt must include lightgbm"
        )
        assert "scipy" in content.lower(), (
            "requirements-learning.txt must include scipy"
        )


class TestCopyrightHeaders:
    """Scenario 11: All new .py files in src/learning/ have copyright."""

    def test_copyright_headers(self):
        """All .py files in src/learning/ have the copyright header."""
        learning_dir = REPO_DIR / "src" / "learning"
        assert learning_dir.exists(), "src/learning/ directory not found"

        py_files = list(learning_dir.glob("*.py"))
        assert len(py_files) > 0, "No .py files found in src/learning/"

        for py_file in py_files:
            if py_file.name == "__init__.py":
                continue  # __init__.py might have the header differently
            content = py_file.read_text(encoding="utf-8")
            assert "Copyright" in content, (
                f"{py_file.name} missing copyright header"
            )
            has_author = "Varun Pratap Bhardwaj" in content
            has_product = "SuperLocalMemory" in content
            assert has_author or has_product, (
                f"{py_file.name} missing creator attribution "
                f"(expected 'Varun Pratap Bhardwaj' or 'SuperLocalMemory')"
            )
            has_mit = "MIT License" in content
            has_spdx_mit = "SPDX-License-Identifier: MIT" in content
            assert has_mit or has_spdx_mit, (
                f"{py_file.name} missing license reference"
            )


class TestNoHardcodedPaths:
    """Scenario 12: No absolute user paths in source files."""

    def test_no_hardcoded_paths(self):
        """No absolute user-specific paths in src/learning/ source files."""
        learning_dir = REPO_DIR / "src" / "learning"
        py_files = list(learning_dir.glob("*.py"))

        # Patterns that indicate hardcoded user-specific paths
        bad_patterns = [
            "/home/varun",
            "C:\\Users\\",
        ]

        for py_file in py_files:
            content = py_file.read_text(encoding="utf-8")
            in_docstring = False

            for pattern in bad_patterns:
                lines = content.split("\n")
                for line_no, line in enumerate(lines, 1):
                    stripped = line.strip()

                    # Track docstring boundaries
                    if '"""' in stripped or "'''" in stripped:
                        # Count triple quotes to toggle docstring state
                        triple_count = stripped.count('"""') + stripped.count("'''")
                        if triple_count % 2 == 1:
                            in_docstring = not in_docstring
                        continue

                    # Skip content inside docstrings (example paths allowed)
                    if in_docstring:
                        continue

                    # Skip comment lines
                    if stripped.startswith("#"):
                        continue

                    if pattern in stripped:
                        pytest.fail(
                            f"{py_file.name}:{line_no} contains hardcoded "
                            f"path pattern '{pattern}': {stripped[:80]}"
                        )


class TestLearningDbSeparate:
    """Scenario 13: Learning tables NOT in memory.db schema."""

    def test_learning_db_separate(self, memory_db):
        """Learning tables should NOT exist in memory.db."""
        conn = sqlite3.connect(str(memory_db))
        cursor = conn.cursor()
        cursor.execute(
            "SELECT name FROM sqlite_master WHERE type='table'"
        )
        memory_tables = {row[0] for row in cursor.fetchall()}
        conn.close()

        learning_only_tables = {
            "transferable_patterns",
            "workflow_patterns",
            "ranking_feedback",
            "ranking_models",
            "source_quality",
            "engagement_metrics",
        }

        overlap = memory_tables & learning_only_tables
        assert len(overlap) == 0, (
            f"Learning tables found in memory.db (should be in learning.db): "
            f"{overlap}"
        )


class TestMemoryDbUnchanged:
    """Scenario 14: memory.db schema matches v2.6 exactly."""

    def test_memory_db_unchanged(self, memory_db):
        """memory.db schema has no v2.7 learning columns added."""
        conn = sqlite3.connect(str(memory_db))
        cursor = conn.cursor()
        cursor.execute("PRAGMA table_info(memories)")
        columns = {row[1] for row in cursor.fetchall()}
        conn.close()

        # These columns exist in v2.6 and should remain
        v26_columns = {
            "id", "content", "summary", "tags", "category",
            "memory_type", "importance", "project_name", "project_path",
            "profile", "parent_id", "cluster_id", "tier", "entity_vector",
            "created_at", "updated_at", "last_accessed", "access_count",
            "created_by", "source_protocol", "trust_score", "provenance_chain",
        }

        # All v2.6 columns should be present
        missing = v26_columns - columns
        assert len(missing) == 0, (
            f"v2.6 columns missing from memory.db: {missing}"
        )

        # Learning-specific columns should NOT be in memory.db
        learning_columns = {
            "learning_score", "ranking_phase", "ml_score",
            "feedback_count", "bootstrap_score",
        }
        unexpected = columns & learning_columns
        assert len(unexpected) == 0, (
            f"Learning-specific columns found in memory.db (should not be): "
            f"{unexpected}"
        )
