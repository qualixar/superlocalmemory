# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Tests for core.key_expander + fact_expansion_fts (T3b)."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.core.key_expander import KeyExpander
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import (
    AtomicFact,
    CanonicalEntity,
    EntityAlias,
    FactType,
    MemoryRecord,
)


@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    mgr = DatabaseManager(tmp_path / "test.db")
    mgr.initialize(real_schema)
    return mgr


def _seed_entity_with_aliases(db: DatabaseManager) -> None:
    db.store_entity(CanonicalEntity(
        entity_id="e1", profile_id="default",
        canonical_name="New York City", entity_type="place",
    ))
    for alias in ("NYC", "Big Apple"):
        db.store_alias(
            EntityAlias(entity_id="e1", alias=alias, source="test"), "default",
        )


def _fact(content: str, entities: list[str]) -> AtomicFact:
    return AtomicFact(
        fact_id="f1", memory_id="m0", profile_id="default",
        content=content, fact_type=FactType.SEMANTIC,
        canonical_entities=entities,
    )


class TestKeyExpanderModeA:
    def test_aliases_become_keys(self, db: DatabaseManager) -> None:
        _seed_entity_with_aliases(db)
        fact = _fact("I visited the city last summer", ["New York City"])
        keys = KeyExpander(db).expand(fact, "default", mode="a")
        assert "NYC" in keys
        assert "Big Apple" in keys
        assert "New York City" in keys  # canonical name included

    def test_keys_already_in_content_are_dropped(self, db: DatabaseManager) -> None:
        _seed_entity_with_aliases(db)
        # Content already says NYC -> NYC must NOT be re-indexed as an alt-key.
        fact = _fact("I love NYC food", ["New York City"])
        keys = KeyExpander(db).expand(fact, "default", mode="a").split()
        assert "NYC" not in keys
        assert "Apple" in " ".join(keys)  # Big Apple still added

    def test_no_entities_yields_empty(self, db: DatabaseManager) -> None:
        fact = _fact("a plain undecorated memory", [])
        assert KeyExpander(db).expand(fact, "default", mode="a") == ""

    def test_unknown_entity_name_is_safe(self, db: DatabaseManager) -> None:
        fact = _fact("about something", ["Nonexistent Entity"])
        assert KeyExpander(db).expand(fact, "default", mode="a") == ""


class TestKeyExpanderModeBC:
    def test_llm_paraphrases_included(self, db: DatabaseManager) -> None:
        class _LLM:
            def is_available(self) -> bool:
                return True

            def generate(self, prompt: str) -> str:
                return "automobile, motorcar, vehicle"

        fact = _fact("my car needs service", [])
        keys = KeyExpander(db, llm=_LLM()).expand(fact, "default", mode="b")
        assert "automobile" in keys
        assert "vehicle" in keys

    def test_llm_failure_is_fail_open(self, db: DatabaseManager) -> None:
        class _LLM:
            def is_available(self) -> bool:
                return True

            def generate(self, prompt: str) -> str:
                raise RuntimeError("model down")

        fact = _fact("my car needs service", [])
        # Must not raise; returns Mode-A keys (none here) => empty.
        assert KeyExpander(db, llm=_LLM()).expand(fact, "default", mode="b") == ""

    def test_mode_a_ignores_llm(self, db: DatabaseManager) -> None:
        class _LLM:
            def is_available(self) -> bool:
                return True

            def generate(self, prompt: str) -> str:
                return "should-not-appear"

        fact = _fact("my car needs service", [])
        keys = KeyExpander(db, llm=_LLM()).expand(fact, "default", mode="a")
        assert "should-not-appear" not in keys


class TestFactExpansionUpsert:
    def _count(self, db: DatabaseManager, fact_id: str) -> int:
        rows = db.execute(
            "SELECT COUNT(*) AS n FROM fact_expansion_fts WHERE fact_id = ?",
            (fact_id,),
        )
        return dict(rows[0])["n"]

    def test_upsert_and_replace(self, db: DatabaseManager) -> None:
        db.upsert_fact_expansion("f1", "nyc big apple")
        assert self._count(db, "f1") == 1
        # Replace: still exactly one row.
        db.upsert_fact_expansion("f1", "gotham metropolis")
        assert self._count(db, "f1") == 1
        rows = db.execute(
            "SELECT alt_keys FROM fact_expansion_fts WHERE fact_id = ?", ("f1",)
        )
        assert "gotham" in dict(rows[0])["alt_keys"]

    def test_upsert_empty_clears(self, db: DatabaseManager) -> None:
        db.upsert_fact_expansion("f1", "nyc")
        assert self._count(db, "f1") == 1
        db.upsert_fact_expansion("f1", "")
        assert self._count(db, "f1") == 0

    def test_fts_match_finds_row(self, db: DatabaseManager) -> None:
        db.upsert_fact_expansion("f1", "automobile motorcar vehicle")
        rows = db.execute(
            "SELECT fact_id FROM fact_expansion_fts "
            "WHERE fact_expansion_fts MATCH ?",
            ('"automobile"',),
        )
        assert [dict(r)["fact_id"] for r in rows] == ["f1"]
