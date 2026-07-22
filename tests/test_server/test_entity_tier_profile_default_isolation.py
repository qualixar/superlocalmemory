# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later — see LICENSE file
# Part of SuperLocalMemory V3
"""Regression: dashboard read endpoints must default to the ACTIVE profile.

ROOT CAUSE
----------
Several read handlers declared the profile as a route default of the *literal*
string "default":

    async def list_entities(..., profile: str = Query(default="default"))
    async def tier_stats(profile_id: str = "default")
    async def get_brain(profile_id: str = "default")

The dashboard calls these WITHOUT a ?profile= query arg, so every request ran
`WHERE profile_id = 'default'` regardless of which profile was active. Result:
switching to an empty profile still showed the *default* profile's entities and
tier counts — a cross-profile data leak (confirmed empirically: /api/entity/list
returned 448 entities under a freshly-created empty profile).

FIX
---
Default the param to None and resolve `profile or get_active_profile()` — the
same request-runtime-truth helper the correctly-isolated behavioral/learning
routes already use. These tests seed one profile, switch to an empty one, and
assert the read endpoints return the EMPTY profile's data (0), not the seeded
profile's.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient


def _add_profile(engine, profile_id: str) -> None:
    engine._db.execute(
        "INSERT OR IGNORE INTO profiles (profile_id, name) VALUES (?, ?)",
        (profile_id, profile_id),
    )
    from superlocalmemory.server.routes.helpers import ensure_profile_in_json
    ensure_profile_in_json(profile_id)


def _daemon_headers(app) -> dict[str, str]:
    descriptor = app.state.daemon_descriptor
    return {
        "X-SLM-Daemon-Capability": descriptor.capability,
        "X-SLM-Target-Instance": descriptor.instance_id,
    }


def _seed_entity(engine, profile_id: str, name: str) -> None:
    engine._db.execute(
        "INSERT INTO canonical_entities "
        "(entity_id, profile_id, canonical_name, entity_type, fact_count) "
        "VALUES (?, ?, ?, ?, ?)",
        (f"ent-{profile_id}-{name}", profile_id, name, "person", 3),
    )


def _seed_fact(engine, profile_id: str, fid: str) -> None:
    mem_id = f"mem-{fid}"
    # atomic_facts.memory_id FKs memories; seed the parent memory first.
    engine._db.execute(
        "INSERT INTO memories "
        "(memory_id, profile_id, content, session_id, speaker, role, "
        " created_at, metadata_json, scope) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (mem_id, profile_id, "seed memory", "sess-1", "user", "user",
         "2026-01-01T00:00:00Z", "{}", "personal"),
    )
    engine._db.execute(
        "INSERT INTO atomic_facts "
        "(fact_id, memory_id, profile_id, content, lifecycle) "
        "VALUES (?, ?, ?, ?, ?)",
        (fid, mem_id, profile_id, "seed fact", "active"),
    )


@pytest.fixture
def client_with_two_profiles(engine_with_mock_deps):
    """Daemon app on profile 'default' (seeded) + empty profile 'work'."""
    from superlocalmemory.server.profile_runtime import bind_profile_runtime
    from superlocalmemory.server.unified_daemon import create_app

    engine = engine_with_mock_deps
    engine.profile_id = "default"
    engine._config.active_profile = "default"
    _add_profile(engine, "default")
    _add_profile(engine, "work")

    # Seed data ONLY into 'default'.
    _seed_entity(engine, "default", "Alice")
    _seed_entity(engine, "default", "Bob")
    _seed_fact(engine, "default", "f1")
    _seed_fact(engine, "default", "f2")
    engine._db.commit() if hasattr(engine._db, "commit") else None

    app = create_app()
    app.state.engine = engine
    app.state.config = engine._config
    bind_profile_runtime(app.state, engine, engine._config)
    return TestClient(app), _daemon_headers(app)


def test_entity_list_defaults_to_active_profile_not_literal_default(
    client_with_two_profiles,
):
    client, headers = client_with_two_profiles

    # On 'default' (active), the seeded entities are visible.
    r = client.get("/api/entity/list")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 2, "default profile should see its 2 seeded entities"

    # Switch to the EMPTY 'work' profile.
    sw = client.post("/api/profiles/work/switch", headers=headers)
    assert sw.status_code == 200, sw.text

    # REGRESSION GUARD: without ?profile=, this must reflect 'work' (empty),
    # NOT the literal 'default' profile's 2 entities.
    r2 = client.get("/api/entity/list")
    assert r2.status_code == 200, r2.text
    assert r2.json()["total"] == 0, (
        "entity/list leaked the default profile's entities into the empty "
        "'work' profile — the handler defaulted profile to literal 'default'."
    )


def test_tier_stats_defaults_to_active_profile_not_literal_default(
    client_with_two_profiles,
):
    client, headers = client_with_two_profiles

    r = client.get("/api/tiers/stats")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 2, "default profile should count its 2 seeded facts"

    sw = client.post("/api/profiles/work/switch", headers=headers)
    assert sw.status_code == 200, sw.text

    r2 = client.get("/api/tiers/stats")
    assert r2.status_code == 200, r2.text
    assert r2.json()["total"] == 0, (
        "tiers/stats counted the default profile's facts under the empty "
        "'work' profile — the handler defaulted profile_id to literal 'default'."
    )


def test_explicit_profile_query_arg_still_honored(client_with_two_profiles):
    """An explicit ?profile= override must still work (for tooling/tests)."""
    client, headers = client_with_two_profiles
    # Active profile is 'default'; explicitly request the empty 'work'.
    r = client.get("/api/entity/list?profile=work")
    assert r.status_code == 200, r.text
    assert r.json()["total"] == 0
    # And explicitly request 'default' while... still on default.
    r2 = client.get("/api/entity/list?profile=default")
    assert r2.json()["total"] == 2
