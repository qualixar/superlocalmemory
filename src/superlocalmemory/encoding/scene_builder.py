# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""SuperLocalMemory V3 — Scene Builder (Memory Clustering).

Groups related facts into thematic scenes (EverMemOS MemScene pattern).
Scenes provide contextual retrieval — related facts come together.

V1 had this module but NEVER CALLED it. Now wired into the encoding pipeline.

Part of Qualixar | Author: Varun Pratap Bhardwaj
"""

from __future__ import annotations

import json
import logging
from datetime import UTC, datetime

from superlocalmemory.storage.models import AtomicFact, MemoryScene

logger = logging.getLogger(__name__)

# Similarity threshold for assigning fact to existing scene
_ASSIGN_THRESHOLD = 0.6


class SceneBuilder:
    """Cluster related facts into thematic scenes.

    When a new fact arrives:
    1. Compute similarity to existing scenes (via scene theme embedding)
    2. If above threshold: assign to nearest scene, update scene
    3. If below threshold: create new scene
    """

    def __init__(self, db, embedder=None) -> None:
        self._db = db
        self._embedder = embedder
        # Key by scene ID, never theme. Themes are deliberately non-unique,
        # while eligibility and durable anchor membership are scene-specific.
        self._scene_embeddings_cache: dict[str, list[float]] = {}

    def assign_to_scene(
        self,
        new_fact: AtomicFact,
        profile_id: str,
    ) -> MemoryScene:
        """Assign a fact to an existing scene or create a new one.

        Always embeds the incoming fact content (when embedder is available)
        so that the embedding is ready for comparison against existing scenes.
        """
        if self._embedder is None:
            return self._create_scene(new_fact, profile_id)

        # Canonical ingestion already embeds the fact before scene assignment.
        # Reuse that vector so scene clustering does not issue a duplicate model
        # request for every remembered fact.
        fact_emb = new_fact.embedding
        if fact_emb is None:
            fact_emb = self._embedder.embed(new_fact.content)

        # v3.4.38: Defensive None guard. embedder.embed() returns None when
        # the embedding worker is unavailable (timeout, crash). Without this
        # guard, _cosine(None, theme_emb) → zip(None, ...) → 'NoneType'
        # object is not iterable, propagating up to engine.store() and
        # causing the entire memory to be lost. Better to skip scene
        # assignment than lose the memory.
        if fact_emb is None:
            return self._create_scene(new_fact, profile_id)

        scenes = self._get_scenes(profile_id)
        if not scenes:
            return self._create_scene(new_fact, profile_id)

        live_scene_embeddings = self._load_live_scene_embeddings(profile_id)
        live_scene_ids = set(live_scene_embeddings)
        self._scene_embeddings_cache.update({
            scene_id: embedding
            for scene_id, embedding in live_scene_embeddings.items()
            if embedding is not None
        })
        # Old consolidation/deletion paths left scene rows whose fact IDs no
        # longer exist. They are not evidence and must not trigger thousands of
        # replacement model calls after restart. A cache hit cannot prove that
        # a scene still has a surviving fact, so eligibility is always derived
        # from the current database state.
        scenes = [
            scene for scene in scenes if scene.scene_id in live_scene_ids
        ]
        if not scenes:
            return self._create_scene(new_fact, profile_id)

        # Find best matching scene
        best_scene: MemoryScene | None = None
        best_sim = -1.0

        # A scene's theme is derived from its first (anchor) fact. On daemon
        # restart the in-memory cache is empty, but the anchor embeddings remain
        # durable in atomic_facts. Prime from those vectors before calling the
        # model; otherwise a mature database re-embeds thousands of themes and
        # repeatedly recycles the shared foreground worker.
        # V3.3.27: Batch-embed all still-uncached scene themes in ONE call.
        # Previously: 200+ individual embed() calls per fact (30s on Mode B).
        # Now: 1 batch call for all uncached themes, then cache hits for the rest.
        uncached_scenes = [
            scene for scene in scenes
            if scene.scene_id not in self._scene_embeddings_cache
        ]
        if uncached_scenes and hasattr(self._embedder, 'embed_batch'):
            try:
                batch_embs = self._embedder.embed_batch(
                    [scene.theme for scene in uncached_scenes]
                )
                for scene, emb in zip(uncached_scenes, batch_embs):
                    if emb is not None:
                        self._scene_embeddings_cache[scene.scene_id] = emb
            except Exception:
                pass  # Fall through to individual embeds below

        for scene in scenes:
            if scene.scene_id in self._scene_embeddings_cache:
                theme_emb = self._scene_embeddings_cache[scene.scene_id]
            else:
                theme_emb = self._embedder.embed(scene.theme)
                if theme_emb is not None:
                    self._scene_embeddings_cache[scene.scene_id] = theme_emb
            if theme_emb is None:
                continue
            sim = _cosine(fact_emb, theme_emb)
            if sim > best_sim:
                best_sim = sim
                best_scene = scene

        if best_scene is not None and best_sim >= _ASSIGN_THRESHOLD:
            return self._add_to_scene(best_scene, new_fact, profile_id)

        return self._create_scene(new_fact, profile_id)

    def get_scene_for_fact(self, fact_id: str, profile_id: str) -> MemoryScene | None:
        """Get the scene containing a specific fact."""
        rows = self._db.execute(
            "SELECT * FROM memory_scenes WHERE profile_id = ?", (profile_id,)
        )
        for row in rows:
            d = dict(row)
            fids = json.loads(d.get("fact_ids_json", "[]"))
            if fact_id in fids:
                return self._row_to_scene(d)
        return None

    def get_all_scenes(self, profile_id: str) -> list[MemoryScene]:
        """Get all scenes for a profile."""
        return self._get_scenes(profile_id)

    # -- Internal ----------------------------------------------------------

    def _create_scene(self, fact: AtomicFact, profile_id: str) -> MemoryScene:
        """Create a new scene from a single fact.

        Pre-computes and caches the theme embedding for efficient later
        comparisons in assign_to_scene.
        """
        theme = fact.content[:200]
        scene = MemoryScene(
            profile_id=profile_id,
            theme=theme,
            fact_ids=[fact.fact_id],
            entity_ids=list(fact.canonical_entities),
            created_at=datetime.now(UTC).isoformat(),
            last_updated=datetime.now(UTC).isoformat(),
        )
        # Pre-compute theme embedding for future comparisons. The canonical fact
        # vector represents this exact theme and avoids a duplicate model call.
        if self._embedder is not None:
            theme_embedding = fact.embedding
            if theme_embedding is None:
                theme_embedding = self._embedder.embed(theme)
            if theme_embedding is not None:
                self._scene_embeddings_cache[scene.scene_id] = theme_embedding
        self._save_scene(scene)
        return scene

    def _add_to_scene(
        self, scene: MemoryScene, fact: AtomicFact, profile_id: str
    ) -> MemoryScene:
        """Add a fact to an existing scene."""
        new_fact_ids = [*scene.fact_ids, fact.fact_id]
        new_entity_ids = list(set(scene.entity_ids) | set(fact.canonical_entities))
        updated = MemoryScene(
            scene_id=scene.scene_id,
            profile_id=profile_id,
            theme=scene.theme,
            fact_ids=new_fact_ids,
            entity_ids=new_entity_ids,
            created_at=scene.created_at,
            last_updated=datetime.now(UTC).isoformat(),
        )
        self._save_scene(updated)
        return updated

    def _get_scenes(self, profile_id: str) -> list[MemoryScene]:
        """Load all scenes from DB."""
        rows = self._db.execute(
            "SELECT * FROM memory_scenes WHERE profile_id = ? ORDER BY last_updated DESC",
            (profile_id,),
        )
        return [self._row_to_scene(dict(r)) for r in rows]

    def _load_live_scene_embeddings(
        self,
        profile_id: str,
    ) -> dict[str, list[float] | None]:
        """Load one durable anchor embedding for every live scene.

        ``json_each`` resolves the first still-existing fact in each scene, so
        scenes whose original anchor was consolidated away can still reuse a
        surviving member. The result also identifies fully stale scene rows,
        which are ignored by assignment instead of being re-embedded.
        """
        try:
            rows = self._db.execute(
                """
                WITH live_scene_facts AS (
                    SELECT
                        ms.scene_id,
                        ms.theme,
                        af.embedding,
                        ROW_NUMBER() OVER (
                            PARTITION BY ms.scene_id
                            ORDER BY CAST(member.key AS INTEGER)
                        ) AS member_rank
                    FROM memory_scenes AS ms
                    JOIN json_each(ms.fact_ids_json) AS member
                    JOIN atomic_facts AS af
                      ON af.fact_id = member.value
                     AND af.profile_id = ms.profile_id
                    WHERE ms.profile_id = ?
                )
                SELECT scene_id, embedding
                FROM live_scene_facts
                WHERE member_rank = 1
                """,
                (profile_id,),
            )
        except Exception:
            return {}

        result: dict[str, list[float] | None] = {}
        for row in rows:
            data = dict(row)
            raw_embedding = data.get("embedding")
            embedding = None
            if raw_embedding:
                try:
                    embedding = json.loads(raw_embedding)
                except (TypeError, ValueError, json.JSONDecodeError):
                    embedding = None
            result[str(data["scene_id"])] = embedding
        return result

    def _save_scene(self, scene: MemoryScene) -> None:
        """Upsert scene to DB."""
        self._db.execute(
            """INSERT OR REPLACE INTO memory_scenes
               (scene_id, profile_id, theme, fact_ids_json, entity_ids_json,
                created_at, last_updated)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                scene.scene_id, scene.profile_id, scene.theme,
                json.dumps(scene.fact_ids), json.dumps(scene.entity_ids),
                scene.created_at, scene.last_updated,
            ),
        )

    @staticmethod
    def _row_to_scene(d: dict) -> MemoryScene:
        return MemoryScene(
            scene_id=d["scene_id"],
            profile_id=d["profile_id"],
            theme=d.get("theme", ""),
            fact_ids=json.loads(d.get("fact_ids_json", "[]")),
            entity_ids=json.loads(d.get("entity_ids_json", "[]")),
            created_at=d.get("created_at", ""),
            last_updated=d.get("last_updated", ""),
        )


def _cosine(a: list[float] | None, b: list[float] | None) -> float:
    # v3.4.38: Defensive None guard — embedder can return None on worker
    # unavailability. Returning 0.0 is correct: zero similarity means no
    # match, which falls back to creating a new scene.
    if a is None or b is None:
        return 0.0
    dot = sum(x * y for x, y in zip(a, b))
    na = sum(x * x for x in a) ** 0.5
    nb = sum(x * x for x in b) ** 0.5
    if na == 0 or nb == 0:
        return 0.0
    return dot / (na * nb)
