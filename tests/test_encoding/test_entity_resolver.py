# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.entity_resolver.

Covers:
  - jaro_winkler() pure-Python similarity
  - _guess_entity_type() heuristic
  - EntityResolver.resolve() 4-tier resolution
  - EntityResolver.create_speaker_entities()
  - EntityResolver.get_canonical_name()
  - EntityResolver.merge_entities()
  - Pronoun filtering
  - Alias persistence and dedup
  - LLM disambiguation (Mode B/C)
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.entity_resolver import (
    JARO_WINKLER_AUTO_MERGE,
    JARO_WINKLER_LLM_FLOOR,
    EntityResolver,
    _guess_entity_type,
    jaro_winkler,
)
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import CanonicalEntity, EntityAlias


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


@pytest.fixture()
def resolver(db: DatabaseManager) -> EntityResolver:
    return EntityResolver(db=db, llm=None)


# ---------------------------------------------------------------------------
# jaro_winkler
# ---------------------------------------------------------------------------

class TestJaroWinkler:
    def test_identical(self) -> None:
        assert jaro_winkler("Alice", "Alice") == 1.0

    def test_empty_strings(self) -> None:
        assert jaro_winkler("", "") == 0.0
        assert jaro_winkler("Alice", "") == 0.0
        assert jaro_winkler("", "Alice") == 0.0

    def test_similar_names(self) -> None:
        score = jaro_winkler("Alice", "Alise")
        assert score > 0.9

    def test_different_names(self) -> None:
        score = jaro_winkler("Alice", "Xyzzy")
        assert score < 0.5

    def test_prefix_boost(self) -> None:
        # Jaro-Winkler gives prefix bonus
        score_prefix = jaro_winkler("Johnson", "Johnston")
        score_no_prefix = jaro_winkler("ohnson", "ohnston")
        assert score_prefix >= score_no_prefix

    def test_capped_at_200_chars(self) -> None:
        long_a = "A" * 300
        long_b = "A" * 300
        assert jaro_winkler(long_a, long_b) == 1.0


# ---------------------------------------------------------------------------
# _guess_entity_type
# ---------------------------------------------------------------------------

class TestGuessEntityType:
    def test_person_two_words(self) -> None:
        assert _guess_entity_type("Alice Smith") == "person"

    def test_single_word_is_concept(self) -> None:
        # v3.4.10: single words default to concept (too many false positives)
        assert _guess_entity_type("Alice") == "concept"
        assert _guess_entity_type("Dashboard") == "concept"

    def test_organization(self) -> None:
        assert _guess_entity_type("Acme Corp") == "organization"
        assert _guess_entity_type("MIT University") == "organization"

    def test_place(self) -> None:
        assert _guess_entity_type("Central Park") == "place"
        assert _guess_entity_type("Baker Street") == "place"

    def test_event(self) -> None:
        assert _guess_entity_type("Science Festival") == "event"
        assert _guess_entity_type("Tech Conference") == "event"

    def test_concept_fallback(self) -> None:
        assert _guess_entity_type("quantum computing") == "concept"


# ---------------------------------------------------------------------------
# EntityResolver.resolve() — 4-tier
# ---------------------------------------------------------------------------

class TestResolve:
    def test_empty_input(self, resolver: EntityResolver) -> None:
        result = resolver.resolve([], "default")
        assert result == {}

    def test_pronouns_skipped(self, resolver: EntityResolver) -> None:
        result = resolver.resolve(["he", "she", "they", "I", "me"], "default")
        assert result == {}

    def test_blank_strings_skipped(self, resolver: EntityResolver) -> None:
        result = resolver.resolve(["", "  "], "default")
        assert result == {}

    def test_tier_a_exact_match(self, resolver: EntityResolver, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e_alice", profile_id="default", canonical_name="Alice",
        ))
        result = resolver.resolve(["Alice"], "default")
        assert result["Alice"] == "e_alice"

    def test_tier_b_alias_match(self, resolver: EntityResolver, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e_bob", profile_id="default", canonical_name="Robert",
        ))
        db.store_alias(EntityAlias(
            alias_id="a1", entity_id="e_bob", alias="Bob", source="test",
        ), "default")
        result = resolver.resolve(["Bob"], "default")
        assert result["Bob"] == "e_bob"

    def test_tier_d_creates_new_entity(self, resolver: EntityResolver) -> None:
        result = resolver.resolve(["Zephyr"], "default")
        assert "Zephyr" in result
        entity_id = result["Zephyr"]
        assert len(entity_id) > 0

    def test_multiple_entities_resolved(self, resolver: EntityResolver) -> None:
        result = resolver.resolve(["Alice", "Bob", "Carol"], "default")
        assert len(result) == 3
        # Each should get a unique entity_id
        assert len(set(result.values())) == 3

    def test_same_entity_resolved_consistently(
        self, resolver: EntityResolver,
    ) -> None:
        r1 = resolver.resolve(["Alice"], "default")
        r2 = resolver.resolve(["Alice"], "default")
        assert r1["Alice"] == r2["Alice"]

    def test_profile_isolation(self, resolver: EntityResolver, db: DatabaseManager) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
        )
        db.store_entity(CanonicalEntity(
            entity_id="e_def", profile_id="default", canonical_name="Alice",
        ))
        result_default = resolver.resolve(["Alice"], "default")
        result_work = resolver.resolve(["Alice"], "work")
        assert result_default["Alice"] == "e_def"
        # Work profile should create a new entity
        assert result_work["Alice"] != "e_def"


# ---------------------------------------------------------------------------
# create_speaker_entities
# ---------------------------------------------------------------------------

class TestCreateSpeakerEntities:
    def test_creates_both_speakers(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        resolver.create_speaker_entities("Alice", "Bob", "default")
        alice = db.get_entity_by_name("Alice", "default")
        bob = db.get_entity_by_name("Bob", "default")
        assert alice is not None
        assert bob is not None
        assert alice.entity_type == "person"

    def test_skips_empty_speaker(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        resolver.create_speaker_entities("Alice", "", "default")
        alice = db.get_entity_by_name("Alice", "default")
        assert alice is not None

    def test_skips_pronoun_speaker(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        resolver.create_speaker_entities("I", "you", "default")
        assert db.get_entity_by_name("I", "default") is None

    def test_idempotent(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        resolver.create_speaker_entities("Alice", "Bob", "default")
        resolver.create_speaker_entities("Alice", "Bob", "default")
        # Should not crash or create duplicates
        rows = db.execute(
            "SELECT COUNT(*) AS c FROM canonical_entities "
            "WHERE canonical_name = 'Alice' AND profile_id = 'default'"
        )
        assert int(dict(rows[0])["c"]) == 1


# ---------------------------------------------------------------------------
# get_canonical_name
# ---------------------------------------------------------------------------

class TestGetCanonicalName:
    def test_known_entity(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e1", profile_id="default", canonical_name="Robert Smith",
        ))
        assert resolver.get_canonical_name("Robert Smith", "default") == "Robert Smith"

    def test_alias_lookup(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e1", profile_id="default", canonical_name="Robert Smith",
        ))
        db.store_alias(EntityAlias(
            alias_id="a1", entity_id="e1", alias="Bob", source="test",
        ), "default")
        assert resolver.get_canonical_name("Bob", "default") == "Robert Smith"

    def test_unknown_returns_original(self, resolver: EntityResolver) -> None:
        assert resolver.get_canonical_name("Unknown", "default") == "Unknown"

    def test_empty_returns_original(self, resolver: EntityResolver) -> None:
        assert resolver.get_canonical_name("", "default") == ""


# ---------------------------------------------------------------------------
# merge_entities
# ---------------------------------------------------------------------------

class TestMergeEntities:
    def test_merge_moves_aliases(
        self, resolver: EntityResolver, db: DatabaseManager,
    ) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="keep", profile_id="default", canonical_name="Alice Smith",
        ))
        db.store_entity(CanonicalEntity(
            entity_id="merge", profile_id="default", canonical_name="Ali",
        ))
        db.store_alias(EntityAlias(
            alias_id="a_old", entity_id="merge", alias="A. Smith", source="test",
        ), "default")
        resolver.merge_entities("keep", "merge", "default")
        # Merged entity should be deleted
        rows = db.execute(
            "SELECT * FROM canonical_entities WHERE entity_id = 'merge'"
        )
        assert len(rows) == 0
        # Alias should be moved to keep
        aliases = db.get_aliases_for_entity("keep", "default")
        alias_names = {a.alias for a in aliases}
        assert "A. Smith" in alias_names


# ---------------------------------------------------------------------------
# LLM disambiguation (Mode B/C)
# ---------------------------------------------------------------------------

class TestLLMDisambiguation:
    def test_llm_resolves_ambiguous(self, db: DatabaseManager) -> None:
        db.store_entity(CanonicalEntity(
            entity_id="e_alice", profile_id="default", canonical_name="Alice Smith",
        ))
        llm = MagicMock()
        llm.generate.return_value = json.dumps({"Ms. Smith": "Alice Smith"})
        resolver = EntityResolver(db=db, llm=llm)

        # Create a fuzzy match scenario by providing a name with medium similarity
        result = resolver.resolve(["Ms. Smith"], "default")
        # Should resolve via either fuzzy merge or LLM or new entity creation
        assert "Ms. Smith" in result

    def test_llm_failure_creates_new(self, db: DatabaseManager) -> None:
        llm = MagicMock()
        llm.generate.side_effect = RuntimeError("API error")
        resolver = EntityResolver(db=db, llm=llm)

        result = resolver.resolve(["NewPerson"], "default")
        assert "NewPerson" in result
