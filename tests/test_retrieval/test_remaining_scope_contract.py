"""Scope-contract proof for the remaining retrieval candidate paths.

These tests intentionally use a mixed-owner corpus.  The requester may see its
own facts plus explicitly opted-in global and authorized-shared facts.  Another
owner's personal/project facts and shared facts not addressed to the requester
must never become candidates, including after an opted-in cache has warmed.
"""

from __future__ import annotations

import json
from concurrent.futures import ThreadPoolExecutor
from unittest.mock import MagicMock

import numpy as np
import pytest

from superlocalmemory.math.hopfield import HopfieldConfig
from superlocalmemory.retrieval.bridge_discovery import BridgeDiscovery
from superlocalmemory.retrieval.entity_channel import EntityGraphChannel
from superlocalmemory.retrieval.hopfield_channel import HopfieldChannel
from superlocalmemory.retrieval.semantic_channel import SemanticChannel
from superlocalmemory.retrieval.scope_policy import authorized_fact_ids
from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)
from superlocalmemory.retrieval.temporal_channel import TemporalChannel
from superlocalmemory.storage import schema
from superlocalmemory.storage.database import DatabaseManager

REQUESTER = "requester"
OWNER = "owner"
VISIBLE = {"local", "global", "shared_ok"}
PRIVATE = {"personal", "project", "shared_denied"}


@pytest.fixture
def scoped_db(tmp_path) -> DatabaseManager:
    db = DatabaseManager(tmp_path / "scope.db")
    with db.raw_connection() as conn:
        schema.create_all_tables(conn)
        for profile_id in (REQUESTER, OWNER):
            conn.execute(
                "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
                (profile_id, profile_id),
            )
        conn.execute(
            "INSERT INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type) "
            "VALUES ('entity_requester', ?, 'Scope', 'concept')",
            (REQUESTER,),
        )
        conn.execute(
            "INSERT INTO canonical_entities "
            "(entity_id, profile_id, canonical_name, entity_type) "
            "VALUES ('entity_owner', ?, 'Scope', 'concept')",
            (OWNER,),
        )

        rows = (
            ("local", REQUESTER, "personal", None),
            ("global", OWNER, "global", None),
            ("shared_ok", OWNER, "shared", json.dumps([REQUESTER])),
            ("personal", OWNER, "personal", None),
            ("project", OWNER, "project", None),
            ("shared_denied", OWNER, "shared", json.dumps(["someone_else"])),
        )
        embedding = json.dumps([1.0, 0.0, 0.0, 0.0])
        for fact_id, profile_id, scope, shared_with in rows:
            memory_id = f"m_{fact_id}"
            conn.execute(
                "INSERT INTO memories "
                "(memory_id, profile_id, scope, shared_with, content) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, profile_id, scope, shared_with, fact_id),
            )
            conn.execute(
                "INSERT INTO atomic_facts "
                "(fact_id, memory_id, profile_id, scope, shared_with, content, "
                " fact_type, confidence, importance, evidence_count, access_count, "
                " canonical_entities_json, embedding, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, 'semantic', 0.9, 0.5, 1, 0, ?, ?, "
                " datetime('now'))",
                (
                    fact_id,
                    memory_id,
                    profile_id,
                    scope,
                    shared_with,
                    fact_id,
                    json.dumps([
                        "entity_requester"
                        if profile_id == REQUESTER
                        else "entity_owner"
                    ]),
                    embedding,
                ),
            )
    return db


def _ids(results: list[tuple[str, float]]) -> set[str]:
    return {fact_id for fact_id, _score in results}


def test_entity_graph_search_and_candidate_scoring_enforce_scope(scoped_db) -> None:
    resolver = MagicMock()
    resolver.resolve.return_value = {"Scope": "entity_requester"}
    channel = EntityGraphChannel(scoped_db, entity_resolver=resolver, max_hops=1)
    channel.include_global = True
    channel.include_shared = True

    assert _ids(channel.search("Scope", REQUESTER, top_k=20)) == VISIBLE
    scores = channel.score_candidates(
        "Scope",
        sorted(VISIBLE | PRIVATE),
        REQUESTER,
        include_global=True,
        include_shared=True,
    )
    assert set(scores) == VISIBLE

    # A cache warmed with cross-profile data cannot weaken a later private call.
    channel.include_global = False
    channel.include_shared = False
    assert _ids(channel.search("Scope", REQUESTER, top_k=20)) == {"local"}


def test_entity_candidate_scoring_is_scope_safe_under_concurrency(scoped_db) -> None:
    resolver = MagicMock()
    resolver.resolve.return_value = {"Scope": "entity_requester"}
    channel = EntityGraphChannel(scoped_db, entity_resolver=resolver, max_hops=1)
    candidates = sorted(VISIBLE | PRIVATE)

    def score(include_cross_profile: bool) -> set[str]:
        return set(
            channel.score_candidates(
                "Scope",
                candidates,
                REQUESTER,
                include_global=include_cross_profile,
                include_shared=include_cross_profile,
            )
        )

    with ThreadPoolExecutor(max_workers=2) as pool:
        for _ in range(20):
            opted_in = pool.submit(score, True)
            private = pool.submit(score, False)
            assert opted_in.result() == VISIBLE
            assert private.result() == {"local"}


def test_entity_sql_fallback_discovers_entities_only_from_visible_facts(
    scoped_db,
) -> None:
    channel = EntityGraphChannel(scoped_db, max_hops=2)

    discovered = channel._discover_entities(
        {"local", "personal"},
        REQUESTER,
        set(),
    )

    assert set(discovered) == {"entity_requester"}


def test_authorization_boundary_fails_closed() -> None:
    db = MagicMock()
    db.get_facts_by_ids.side_effect = RuntimeError("authorization unavailable")
    db.execute.side_effect = RuntimeError("authorization unavailable")

    assert authorized_fact_ids(
        db,
        ["candidate"],
        REQUESTER,
        include_global=True,
        include_shared=True,
    ) == set()


def test_hopfield_full_matrix_enforces_scope_and_warm_cache(scoped_db) -> None:
    vector_store = MagicMock(available=False)
    channel = HopfieldChannel(
        scoped_db,
        vector_store,
        HopfieldConfig(dimension=4, cache_ttl_seconds=60.0),
    )
    channel.include_global = True
    channel.include_shared = True
    query = [1.0, 0.0, 0.0, 0.0]

    assert _ids(channel.search(query, REQUESTER, top_k=20)) == VISIBLE

    channel.include_global = False
    channel.include_shared = False
    assert _ids(channel.search(query, REQUESTER, top_k=20)) == {"local"}


def test_hopfield_prefilter_supplements_visible_external_candidates(scoped_db) -> None:
    vector_store = MagicMock(available=True)
    vector_store.count.return_value = 20
    vector_store.search.return_value = [
        ("local", 1.0),
        ("personal", 1.0),
        ("project", 1.0),
        ("shared_denied", 1.0),
    ]
    channel = HopfieldChannel(
        scoped_db,
        vector_store,
        HopfieldConfig(dimension=4, prefilter_candidates=2),
    )
    channel.include_global = True
    channel.include_shared = True

    assert _ids(channel.search([1.0, 0.0, 0.0, 0.0], REQUESTER, top_k=20)) == VISIBLE


def test_semantic_fast_path_never_returns_all_unauthorized_ann_ids(
    scoped_db,
) -> None:
    vector_store = MagicMock(available=True)
    vector_store.search.return_value = [("personal", 0.99)]
    channel = SemanticChannel(scoped_db, vector_store=vector_store)

    results = channel.search(
        [1.0, 0.0, 0.0, 0.0],
        REQUESTER,
        top_k=20,
    )

    # An empty canonical authorization result must fall back to the scoped DB
    # corpus; it must never return the raw, untrusted ANN candidate list.
    assert _ids(results) == {"local"}


def test_temporal_event_scope_cannot_override_private_fact_scope(scoped_db) -> None:
    with scoped_db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO temporal_events "
            "(event_id, profile_id, scope, entity_id, fact_id, "
            " observation_date, referenced_date, description) "
            "VALUES ('forged_event', ?, 'global', 'entity_owner', "
            " 'personal', '2026-01-01', '2026-01-01', 'forged')",
            (OWNER,),
        )
    channel = TemporalChannel(scoped_db)
    channel.include_global = True

    date_results = channel.search("2026-01-01", REQUESTER, top_k=20)
    entity_results = channel.search("When did Scope happen?", REQUESTER, top_k=20)

    assert "personal" not in _ids(date_results)
    assert "personal" not in _ids(entity_results)


def test_spreading_activation_enforces_scope_and_scope_keyed_cache(scoped_db) -> None:
    vector_store = MagicMock(available=True)
    # The owner-partitioned ANN returns a maliciously broad candidate list;
    # the channel must authorize it and independently supplement visible peers.
    vector_store.search.return_value = [
        ("local", 1.0),
        ("personal", 1.0),
        ("project", 1.0),
        ("shared_denied", 1.0),
    ]
    channel = SpreadingActivation(
        scoped_db,
        vector_store,
        SpreadingActivationConfig(max_iterations=0, tau_gate=0.0, top_m=20),
    )
    channel.include_global = True
    channel.include_shared = True
    query = np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32)

    assert _ids(channel.search(query, REQUESTER, top_k=20)) == VISIBLE

    channel.include_global = False
    channel.include_shared = False
    assert _ids(channel.search(query, REQUESTER, top_k=20)) == {"local"}


def test_spreading_activation_traverses_only_visible_edges_and_endpoints(
    scoped_db,
) -> None:
    with scoped_db.raw_connection() as conn:
        conn.execute(
            "UPDATE atomic_facts SET embedding = ? WHERE profile_id = ?",
            (json.dumps([0.0, 0.0, 0.0, 0.0]), OWNER),
        )
        for target_id, scope, shared_with in (
            ("global", "global", None),
            ("shared_ok", "shared", json.dumps([REQUESTER])),
            # A visible edge never authorizes a private endpoint.
            ("personal", "global", None),
            ("project", "global", None),
            ("shared_denied", "global", None),
        ):
            conn.execute(
                "INSERT INTO graph_edges "
                "(edge_id, profile_id, scope, shared_with, source_id, target_id, "
                " edge_type, weight, created_at) "
                "VALUES (?, ?, ?, ?, 'local', ?, 'semantic', 1.0, datetime('now'))",
                (f"edge_{target_id}", OWNER, scope, shared_with, target_id),
            )

    vector_store = MagicMock(available=True)
    vector_store.search.return_value = [("local", 1.0)]
    channel = SpreadingActivation(
        scoped_db,
        vector_store,
        SpreadingActivationConfig(max_iterations=1, tau_gate=0.0, top_m=20),
    )
    channel.include_global = True
    channel.include_shared = True

    neighbors = channel._get_unified_neighbors(
        "local",
        REQUESTER,
        include_global=True,
        include_shared=True,
    )
    assert {fact_id for fact_id, _weight in neighbors} >= {
        "global",
        "shared_ok",
    }

    results = _ids(
        channel.search(
            np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32),
            REQUESTER,
            top_k=20,
        )
    )
    assert {"global", "shared_ok"}.issubset(results)
    assert results.isdisjoint(PRIVATE)


def test_bridge_expansion_enforces_scope_for_seeds_and_candidates(scoped_db) -> None:
    with scoped_db.raw_connection() as conn:
        conn.execute(
            "UPDATE atomic_facts SET canonical_entities_json = ? "
            "WHERE fact_id = 'local'",
            (json.dumps(["seed_a"]),),
        )
        conn.execute(
            "UPDATE atomic_facts SET canonical_entities_json = ? "
            "WHERE fact_id = 'global'",
            (json.dumps(["seed_b"]),),
        )
        conn.execute(
            "UPDATE atomic_facts SET canonical_entities_json = ? "
            "WHERE fact_id NOT IN ('local', 'global')",
            (json.dumps(["seed_a", "seed_b"]),),
        )
    bridge = BridgeDiscovery(scoped_db)

    allowed = bridge.discover(
        ["local", "global"],
        REQUESTER,
        max_bridges=20,
        include_global=True,
        include_shared=True,
    )
    assert _ids(allowed) == {"shared_ok"}

    private = bridge.discover(
        ["local", "global"],
        REQUESTER,
        max_bridges=20,
    )
    # The global seed itself is unauthorized without opt-in, so expansion
    # fails closed instead of reading it through unscoped get_fact().
    assert private == []


def test_bridge_spreading_activation_rejects_unauthorized_seed(scoped_db) -> None:
    with scoped_db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO graph_edges "
            "(edge_id, profile_id, scope, source_id, target_id, edge_type, "
            " weight, created_at) VALUES "
            "('forged_bridge', ?, 'global', 'personal', 'global', "
            " 'semantic', 1.0, datetime('now'))",
            (OWNER,),
        )
    bridge = BridgeDiscovery(scoped_db)

    assert bridge.spreading_activation(
        ["personal"],
        REQUESTER,
        max_depth=1,
        include_global=True,
    ) == []
