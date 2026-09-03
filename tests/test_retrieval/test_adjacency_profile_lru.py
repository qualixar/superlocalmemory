# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Adjacency cache: profile-keyed LRU replaces single-slot reload thrash.

The single-slot cache reloaded the whole graph (edges + entity maps +
metrics + snapshot) on every profile switch, so interleaved per-request
profiles paid a full rebuild per request. The LRU keeps the last
``SLM_ADJ_CACHE_PROFILES`` scopes warm instead.

Fixture convention: real SQLite ``DatabaseManager`` + ``schema.create_all_tables``
via ``raw_connection`` (as in ``test_remaining_scope_contract.scoped_db``);
``EntityGraphChannel`` needs no embedder. Each profile owns one visible fact
so the staleness check's ``(_adj or _visible_fact_ids)`` guard sees a
non-empty corpus and a hot hit is possible at all.
"""

from __future__ import annotations

import json
from unittest.mock import patch

from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

PROFILES = ("a", "b", "c")


def _channel(tmp_path) -> EntityGraphChannel:
    db = DatabaseManager(tmp_path / "adj_lru.db")
    with db.raw_connection() as conn:
        schema.create_all_tables(conn)
        for profile_id in PROFILES:
            fact_id = f"fact_{profile_id}"
            conn.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                (profile_id, profile_id),
            )
            conn.execute(
                "INSERT INTO memories "
                "(memory_id, profile_id, scope, shared_with, content) "
                "VALUES (?, ?, 'personal', NULL, ?)",
                (f"m_{profile_id}", profile_id, fact_id),
            )
            conn.execute(
                "INSERT INTO atomic_facts "
                "(fact_id, memory_id, profile_id, scope, shared_with, content, "
                " fact_type, confidence, importance, evidence_count, access_count, "
                " canonical_entities_json, embedding, created_at) "
                "VALUES (?, ?, ?, 'personal', NULL, ?, 'semantic', 0.9, 0.5, 1, 0, "
                "'[]', ?, datetime('now'))",
                (
                    fact_id,
                    f"m_{profile_id}",
                    profile_id,
                    fact_id,
                    json.dumps([1.0, 0.0, 0.0, 0.0]),
                ),
            )
    return EntityGraphChannel(db)


class TestProfileLRU:
    def test_interleaved_profiles_do_not_reload(self, tmp_path):
        ch = _channel(tmp_path)
        with patch.object(
            EntityGraphChannel,
            "_load_adjacency_from_db",
            wraps=ch._load_adjacency_from_db,
        ) as load:
            ch._ensure_adjacency("a", include_global=False, include_shared=False)
            ch._ensure_adjacency("b", include_global=False, include_shared=False)
            ch._ensure_adjacency("a", include_global=False, include_shared=False)  # hot hit
            ch._ensure_adjacency("b", include_global=False, include_shared=False)  # hot hit
        assert load.call_count == 2  # single-slot implementation reloads 4 times here

    def test_lru_eviction_reloads_evicted_profile(self, tmp_path, monkeypatch):
        monkeypatch.setenv("SLM_ADJ_CACHE_PROFILES", "2")
        ch = _channel(tmp_path)
        ch._ensure_adjacency("a", include_global=False, include_shared=False)
        ch._ensure_adjacency("b", include_global=False, include_shared=False)
        ch._ensure_adjacency("c", include_global=False, include_shared=False)  # evicts a
        with patch.object(
            EntityGraphChannel,
            "_load_adjacency_from_db",
            wraps=ch._load_adjacency_from_db,
        ) as load:
            ch._ensure_adjacency("a", include_global=False, include_shared=False)
        assert load.call_count == 1  # a was evicted and must reload

    def test_hot_hit_refreshes_the_compatibility_attributes(self, tmp_path):
        """Zero-change contract: readers of the legacy single-slot attributes
        (search(), score_candidates(), _resolve_entities()) must observe the
        scope of the call they are serving, not whichever profile loaded last."""
        ch = _channel(tmp_path)
        ch._ensure_adjacency("a", include_global=False, include_shared=False)
        ch._ensure_adjacency("b", include_global=False, include_shared=False)
        assert ch._adj_profile == "b"
        assert ch._visible_fact_ids == {"fact_b"}

        ch._ensure_adjacency("a", include_global=False, include_shared=False)  # hot hit
        assert ch._adj_profile == "a"
        assert ch._adj_scope_key == ("a", False, False)
        assert ch._visible_fact_ids == {"fact_a"}
        assert ch._entity_to_facts == {}  # fact_a carries no entities
        # "a" was re-entered last, so it is the most-recently-used slot.
        assert list(ch._adj_slots)[-1] == ("a", False, False)
