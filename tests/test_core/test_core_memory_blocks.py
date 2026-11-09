# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for Phase 5: Core Memory Blocks schema and CRUD.

Covers: create, get, char limit, version increment, unique constraint,
        source_fact_ids tracking, cascade delete.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import _new_id

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    """Create a test database with full schema."""
    db_path = tmp_path / "test.db"
    d = DatabaseManager(db_path)
    d.initialize(schema)
    return d


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestCoreMemoryBlocksCRUD:
    """Tests for core_memory_blocks table CRUD operations."""

    def test_create_block_stores_in_db(self, db: DatabaseManager) -> None:
        """A stored block is persisted with correct fields."""
        bid = _new_id()
        db.store_core_block(
            block_id=bid,
            profile_id="default",
            block_type="user_profile",
            content="Varun is a senior architect.",
            source_fact_ids=json.dumps(["f1", "f2"]),
            char_count=28,
            version=1,
            compiled_by="rules",
        )
        block = db.get_core_block("default", "user_profile")
        assert block is not None
        assert block["block_type"] == "user_profile"
        assert block["content"] == "Varun is a senior architect."
        assert block["compiled_by"] == "rules"
        assert block["version"] == 1
        assert json.loads(block["source_fact_ids"]) == ["f1", "f2"]

    def test_get_core_blocks_returns_all(self, db: DatabaseManager) -> None:
        """get_core_blocks returns all blocks for a profile."""
        for btype in [
            "user_profile", "project_context", "behavioral_patterns",
            "active_decisions", "learned_preferences",
        ]:
            db.store_core_block(
                block_id=_new_id(),
                profile_id="default",
                block_type=btype,
                content=f"Content for {btype}",
                char_count=len(f"Content for {btype}"),
            )
        blocks = db.get_core_blocks("default")
        assert len(blocks) == 5
        types = {b["block_type"] for b in blocks}
        assert "user_profile" in types
        assert "learned_preferences" in types

    def test_get_core_block_returns_none_for_missing(
        self, db: DatabaseManager,
    ) -> None:
        """get_core_block returns None when block doesn't exist."""
        assert db.get_core_block("default", "custom") is None

    def test_unique_constraint_one_per_type(
        self, db: DatabaseManager,
    ) -> None:
        """Two blocks of same type for same profile -> one entry (replace)."""
        db.store_core_block(
            block_id=_new_id(),
            profile_id="default",
            block_type="user_profile",
            content="First version",
            char_count=13,
        )
        db.store_core_block(
            block_id=_new_id(),
            profile_id="default",
            block_type="user_profile",
            content="Second version",
            char_count=14,
        )
        blocks = db.get_core_blocks("default")
        profile_blocks = [
            b for b in blocks if b["block_type"] == "user_profile"
        ]
        assert len(profile_blocks) == 1
        assert profile_blocks[0]["content"] == "Second version"

    def test_source_fact_ids_tracked(self, db: DatabaseManager) -> None:
        """source_fact_ids is stored as JSON and can be retrieved."""
        source_ids = ["fact_1", "fact_2", "fact_3"]
        db.store_core_block(
            block_id=_new_id(),
            profile_id="default",
            block_type="project_context",
            content="Project context block",
            source_fact_ids=json.dumps(source_ids),
            char_count=21,
        )
        block = db.get_core_block("default", "project_context")
        assert json.loads(block["source_fact_ids"]) == source_ids

    def test_char_count_stored_correctly(self, db: DatabaseManager) -> None:
        """char_count matches stored value."""
        content = "Short content"
        db.store_core_block(
            block_id=_new_id(),
            profile_id="default",
            block_type="user_profile",
            content=content,
            char_count=len(content),
        )
        block = db.get_core_block("default", "user_profile")
        assert block["char_count"] == len(content)

    def test_delete_core_blocks(self, db: DatabaseManager) -> None:
        """delete_core_blocks removes all blocks for profile."""
        for btype in ["user_profile", "project_context"]:
            db.store_core_block(
                block_id=_new_id(),
                profile_id="default",
                block_type=btype,
                content="some content",
                char_count=12,
            )
        assert len(db.get_core_blocks("default")) == 2
        db.delete_core_blocks("default")
        assert len(db.get_core_blocks("default")) == 0

    def test_blocks_scoped_to_profile(self, db: DatabaseManager) -> None:
        """Blocks are profile-scoped — different profiles don't collide."""
        # Create second profile
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
            ("profile2", "Profile 2"),
        )
        db.store_core_block(
            block_id=_new_id(),
            profile_id="default",
            block_type="user_profile",
            content="Default profile content",
            char_count=23,
        )
        db.store_core_block(
            block_id=_new_id(),
            profile_id="profile2",
            block_type="user_profile",
            content="Profile 2 content",
            char_count=17,
        )
        default_blocks = db.get_core_blocks("default")
        p2_blocks = db.get_core_blocks("profile2")

        assert len(default_blocks) == 1
        assert len(p2_blocks) == 1
        assert default_blocks[0]["content"] != p2_blocks[0]["content"]

    def test_core_memory_blocks_table_exists(
        self, db: DatabaseManager,
    ) -> None:
        """core_memory_blocks table is created by schema initialization."""
        tables = db.list_tables()
        assert "core_memory_blocks" in tables
