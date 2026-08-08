# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Tests for superlocalmemory.encoding.scene_builder.

Covers:
  - SceneBuilder.assign_to_scene() — new scene creation
  - SceneBuilder.assign_to_scene() — assign to existing scene
  - SceneBuilder.get_scene_for_fact()
  - SceneBuilder.get_all_scenes()
  - Persistence via DB
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock

import pytest

from superlocalmemory.encoding.scene_builder import SceneBuilder, _cosine
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import AtomicFact, MemoryRecord, MemoryScene

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def db(tmp_path: Path) -> DatabaseManager:
    db_path = tmp_path / "test.db"
    mgr = DatabaseManager(db_path)
    mgr.initialize(real_schema)
    return mgr


def _make_fact(
    fact_id: str = "f1", content: str = "Alice works at Google",
    canonical_entities: list[str] | None = None,
) -> AtomicFact:
    return AtomicFact(
        fact_id=fact_id, content=content,
        canonical_entities=canonical_entities or [],
    )


def _persist_fact(db: DatabaseManager, fact: AtomicFact) -> None:
    fact.memory_id = f"memory-{fact.fact_id}"
    db.store_memory(MemoryRecord(memory_id=fact.memory_id, content=fact.content))
    db.store_fact(fact)


# ---------------------------------------------------------------------------
# _cosine
# ---------------------------------------------------------------------------

class TestCosine:
    def test_identical(self) -> None:
        assert _cosine([1.0, 0.0], [1.0, 0.0]) == pytest.approx(1.0)

    def test_zero(self) -> None:
        assert _cosine([0.0, 0.0], [1.0, 1.0]) == 0.0


# ---------------------------------------------------------------------------
# New scene creation
# ---------------------------------------------------------------------------

class TestCreateScene:
    def test_creates_scene_no_embedder(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        fact = _make_fact("f1", "Alice works at Google", ["ent_alice"])
        scene = builder.assign_to_scene(fact, "default")
        assert isinstance(scene, MemoryScene)
        assert "f1" in scene.fact_ids
        assert scene.profile_id == "default"

    def test_creates_scene_no_existing_scenes(self, db: DatabaseManager) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]
        builder = SceneBuilder(db=db, embedder=embedder)
        fact = _make_fact("f1", "Alice works at Google")
        scene = builder.assign_to_scene(fact, "default")
        assert "f1" in scene.fact_ids

    def test_scene_persisted(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        fact = _make_fact("f1", "Alice works at Google")
        scene = builder.assign_to_scene(fact, "default")
        # Verify persisted in DB
        rows = db.execute(
            "SELECT * FROM memory_scenes WHERE scene_id = ?",
            (scene.scene_id,),
        )
        assert len(rows) == 1

    def test_theme_is_content(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        fact = _make_fact("f1", "Alice works at Google as an engineer")
        scene = builder.assign_to_scene(fact, "default")
        assert "Alice works at Google" in scene.theme

    def test_reuses_fact_embedding_when_creating_scene(
        self, db: DatabaseManager,
    ) -> None:
        embedder = MagicMock()
        builder = SceneBuilder(db=db, embedder=embedder)
        fact = _make_fact("f1", "Alice works at Google")
        fact.embedding = [1.0, 0.0, 0.0]

        builder.assign_to_scene(fact, "default")

        embedder.embed.assert_not_called()


# ---------------------------------------------------------------------------
# Assign to existing scene
# ---------------------------------------------------------------------------

class TestAssignToExistingScene:
    def test_assignment_bounds_mature_store_scene_candidates(self) -> None:
        db = MagicMock()
        db.execute.side_effect = [
            [
                {
                    "scene_id": f"scene-{index}",
                    "profile_id": "default",
                    "theme": f"Theme {index}",
                    "fact_ids_json": f'["fact-{index}"]',
                    "entity_ids_json": "[]",
                    "created_at": "2026-01-01T00:00:00+00:00",
                    "last_updated": "2026-01-01T00:00:00+00:00",
                }
                for index in range(300)
            ],
            [],
            [],
        ]
        incoming = _make_fact("incoming", "Bound mature scene selection")
        incoming.embedding = [1.0, 0.0, 0.0]

        SceneBuilder(db=db, embedder=MagicMock()).assign_to_scene(
            incoming,
            "default",
        )

        candidate_sql, candidate_params = db.execute.call_args_list[0].args
        assert "LIMIT ?" in candidate_sql
        assert candidate_params == ("default", 256)

        live_sql, live_params = db.execute.call_args_list[1].args
        assert "ms.scene_id IN" in live_sql
        assert len(live_params) == 257

    def test_vector_candidates_keep_old_relevant_scene_eligible(
        self, db: DatabaseManager,
    ) -> None:
        """Nearest-fact retrieval must beat a recency-only mature-store cap."""
        relevant = _make_fact("old-anchor", "Long-lived relevant subject")
        relevant.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, relevant)
        seed = SceneBuilder(db=db, embedder=None)
        old_scene = seed.assign_to_scene(relevant, "default")
        db.execute(
            "UPDATE memory_scenes SET last_updated = ? WHERE scene_id = ?",
            ("2020-01-01T00:00:00+00:00", old_scene.scene_id),
        )

        for index in range(300):
            fact = _make_fact(f"recent-{index}", f"Unrelated {index}")
            fact.embedding = [0.0, 1.0, 0.0]
            _persist_fact(db, fact)
            seed.assign_to_scene(fact, "default")

        vector_store = MagicMock()
        vector_store.search.return_value = [(relevant.fact_id, 0.99)]
        incoming = _make_fact("incoming", "The same long-lived subject")
        incoming.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, incoming)

        assigned = SceneBuilder(
            db=db,
            embedder=MagicMock(),
            vector_store=vector_store,
        ).assign_to_scene(incoming, "default")

        assert assigned.scene_id == old_scene.scene_id
        vector_store.search.assert_called_once_with(
            incoming.embedding,
            top_k=256,
            profile_id="default",
        )

    def test_scene_membership_projection_tracks_scene_updates(
        self, db: DatabaseManager,
    ) -> None:
        first = _make_fact("member-1", "First member")
        second = _make_fact("member-2", "Second member")
        _persist_fact(db, first)
        _persist_fact(db, second)
        builder = SceneBuilder(db=db, embedder=None)

        scene = builder.assign_to_scene(first, "default")
        builder._save_scene(MemoryScene(
            scene_id=scene.scene_id,
            profile_id="default",
            theme=scene.theme,
            fact_ids=[first.fact_id, second.fact_id],
            created_at=scene.created_at,
            last_updated=scene.last_updated,
        ))

        rows = db.execute(
            "SELECT fact_id, position FROM scene_fact_members "
            "WHERE scene_id = ? ORDER BY position",
            (scene.scene_id,),
        )
        assert [(row["fact_id"], row["position"]) for row in rows] == [
            (first.fact_id, 0),
            (second.fact_id, 1),
        ]

    def test_assigns_to_similar_scene(self, db: DatabaseManager) -> None:
        embedder = MagicMock()
        # Both facts produce similar embeddings
        embedder.embed.return_value = [1.0, 0.0, 0.0]

        builder = SceneBuilder(db=db, embedder=embedder)
        fact1 = _make_fact("f1", "Alice works at Google")
        _persist_fact(db, fact1)
        scene1 = builder.assign_to_scene(fact1, "default")

        fact2 = _make_fact("f2", "Alice loves Google")
        _persist_fact(db, fact2)
        scene2 = builder.assign_to_scene(fact2, "default")

        # Since embeddings are identical (sim=1.0 > 0.6), should merge
        assert scene2.scene_id == scene1.scene_id
        assert "f1" in scene2.fact_ids
        assert "f2" in scene2.fact_ids

    def test_creates_new_for_dissimilar(self, db: DatabaseManager) -> None:
        call_count = [0]
        def embed(text: str) -> list[float]:
            call_count[0] += 1
            if call_count[0] <= 2:  # First scene creation + comparison
                return [1.0, 0.0, 0.0]
            return [0.0, 0.0, 1.0]  # Very different

        embedder = MagicMock()
        embedder.embed = embed

        builder = SceneBuilder(db=db, embedder=embedder)
        fact1 = _make_fact("f1", "Alice works at Google")
        _persist_fact(db, fact1)
        scene1 = builder.assign_to_scene(fact1, "default")

        fact2 = _make_fact("f2", "Cats are mammals")
        _persist_fact(db, fact2)
        scene2 = builder.assign_to_scene(fact2, "default")

        assert scene2.scene_id != scene1.scene_id

    def test_cold_cache_reuses_persisted_anchor_fact_embeddings(
        self, db: DatabaseManager,
    ) -> None:
        anchor = _make_fact("f1", "Alice works at Google")
        anchor.memory_id = "m1"
        anchor.embedding = [1.0, 0.0, 0.0]
        db.store_memory(MemoryRecord(memory_id="m1", content=anchor.content))
        db.store_fact(anchor)
        SceneBuilder(db=db, embedder=None).assign_to_scene(anchor, "default")

        embedder = MagicMock()
        builder = SceneBuilder(db=db, embedder=embedder)
        incoming = _make_fact("f2", "Alice leads reliability at Google")
        incoming.embedding = [1.0, 0.0, 0.0]

        scene = builder.assign_to_scene(incoming, "default")

        assert set(scene.fact_ids) == {"f1", "f2"}
        embedder.embed.assert_not_called()
        embedder.embed_batch.assert_not_called()

    def test_duplicate_theme_does_not_make_stale_scene_eligible(
        self, db: DatabaseManager,
    ) -> None:
        theme = "Shared scene theme"
        live = _make_fact("f-live", theme)
        live.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, live)

        seed = SceneBuilder(db=db, embedder=None)
        seed._save_scene(MemoryScene(
            scene_id="scene-live",
            profile_id="default",
            theme=theme,
            fact_ids=[live.fact_id],
            created_at="2026-01-01T00:00:00+00:00",
            last_updated="2026-01-01T00:00:00+00:00",
        ))
        seed._save_scene(MemoryScene(
            scene_id="scene-stale",
            profile_id="default",
            theme=theme,
            fact_ids=["deleted-fact"],
            created_at="2026-01-02T00:00:00+00:00",
            last_updated="2026-01-02T00:00:00+00:00",
        ))

        incoming = _make_fact("f-new", "Related incoming fact")
        incoming.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, incoming)
        scene = SceneBuilder(
            db=db,
            embedder=MagicMock(),
        ).assign_to_scene(incoming, "default")

        assert scene.scene_id == "scene-live"
        assert scene.fact_ids == ["f-live", "f-new"]

    def test_cached_scene_is_ineligible_after_last_fact_is_deleted(
        self, db: DatabaseManager,
    ) -> None:
        embedder = MagicMock()
        embedder.embed.return_value = [1.0, 0.0, 0.0]
        builder = SceneBuilder(db=db, embedder=embedder)

        original = _make_fact("f-old", "Reusable theme")
        original.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, original)
        stale_scene = builder.assign_to_scene(original, "default")
        db.delete_fact(original.fact_id)

        incoming = _make_fact("f-new", "Related replacement")
        incoming.embedding = [1.0, 0.0, 0.0]
        _persist_fact(db, incoming)
        new_scene = builder.assign_to_scene(incoming, "default")

        assert new_scene.scene_id != stale_scene.scene_id
        assert new_scene.fact_ids == ["f-new"]


# ---------------------------------------------------------------------------
# get_scene_for_fact
# ---------------------------------------------------------------------------

class TestGetSceneForFact:
    def test_finds_scene(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        fact = _make_fact("f1", "Alice works at Google")
        scene = builder.assign_to_scene(fact, "default")
        found = builder.get_scene_for_fact("f1", "default")
        assert found is not None
        assert found.scene_id == scene.scene_id

    def test_returns_none_for_missing(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        found = builder.get_scene_for_fact("nonexistent", "default")
        assert found is None


# ---------------------------------------------------------------------------
# get_all_scenes
# ---------------------------------------------------------------------------

class TestGetAllScenes:
    def test_returns_all(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        builder.assign_to_scene(_make_fact("f1", "Alice at Google"), "default")
        builder.assign_to_scene(_make_fact("f2", "Bob at Apple"), "default")
        scenes = builder.get_all_scenes("default")
        assert len(scenes) == 2

    def test_empty_profile(self, db: DatabaseManager) -> None:
        builder = SceneBuilder(db=db, embedder=None)
        scenes = builder.get_all_scenes("default")
        assert scenes == []

    def test_profile_isolation(self, db: DatabaseManager) -> None:
        db.execute(
            "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('work', 'Work')"
        )
        builder = SceneBuilder(db=db, embedder=None)
        builder.assign_to_scene(_make_fact("f1", "Default fact"), "default")
        builder.assign_to_scene(_make_fact("f2", "Work fact"), "work")
        default_scenes = builder.get_all_scenes("default")
        work_scenes = builder.get_all_scenes("work")
        assert len(default_scenes) == 1
        assert len(work_scenes) == 1
