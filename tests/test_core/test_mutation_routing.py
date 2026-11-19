# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Which mutations may reach a profile other than the daemon's active one.

Only the kinds listed in ``core/mutation_routing.py`` may, and only to a
profile that exists. Every other kind stays bound to the active profile, so a
route that forwards a client's ``profile_id`` by mistake still cannot rewrite,
archive, merge or re-scope another profile's memories.
"""

from __future__ import annotations

import sqlite3

import pytest

from superlocalmemory.core.mutation_routing import (
    ROUTABLE_MUTATIONS,
    MutationTarget,
    classify_target,
)
from superlocalmemory.storage.write_coordinator import CommandKind
from tests.test_server.test_per_request_profile import _daemon


@pytest.fixture
def conn():
    connection = sqlite3.connect(":memory:")
    connection.execute("CREATE TABLE profiles (profile_id TEXT PRIMARY KEY)")
    connection.executemany("INSERT INTO profiles VALUES (?)", [("default",), ("work",)])
    yield connection
    connection.close()


def test_the_routable_kinds_are_exactly_those_with_a_routed_route() -> None:
    """Adding a kind is a decision with its own route change, never a drive-by.

    4.1.21: delete, propose-correction and set-kind, for DELETE/PATCH
    /api/memories/{id} and /api/memory-kinds with a ``profile_id``.
    """
    assert ROUTABLE_MUTATIONS == frozenset({
        CommandKind.REPLACE_BY_CALLER,
        CommandKind.APPLY_CORRECTION,
        CommandKind.REJECT_CORRECTION,
        CommandKind.ROLLBACK_CORRECTION,
        CommandKind.DELETE_FACT,
        CommandKind.PROPOSE_CORRECTION,
        CommandKind.SET_FACT_KIND,
        CommandKind.PROMOTE_DECLARED,  # 4.2.30: POST /remember with a ``profile_id`` and a kind
    })


@pytest.mark.parametrize("kind", list(CommandKind))
def test_the_active_profile_is_always_a_valid_target(conn, kind) -> None:
    assert classify_target(conn, kind, "default", "default") is MutationTarget.BOUND


@pytest.mark.parametrize("kind", sorted(ROUTABLE_MUTATIONS, key=lambda k: k.value))
def test_a_routable_kind_reaches_an_existing_profile(conn, kind) -> None:
    assert classify_target(conn, kind, "work", "default") is MutationTarget.ROUTED


@pytest.mark.parametrize("kind", sorted(ROUTABLE_MUTATIONS, key=lambda k: k.value))
def test_a_routable_kind_never_reaches_a_missing_profile(conn, kind) -> None:
    assert classify_target(conn, kind, "ghost", "default") is MutationTarget.UNKNOWN_PROFILE


@pytest.mark.parametrize(
    "kind", sorted(set(CommandKind) - ROUTABLE_MUTATIONS, key=lambda k: k.value))
def test_every_other_kind_stays_on_the_active_profile(conn, kind) -> None:
    assert classify_target(conn, kind, "work", "default") is MutationTarget.NOT_ROUTABLE


def test_existence_is_read_on_the_connection_it_is_given(conn) -> None:
    """The writer passes its own transaction connection, so the check and the
    mutation see the same snapshot. A profile visible only there must count."""
    conn.execute("INSERT INTO profiles VALUES ('fresh')")
    assert classify_target(
        conn, CommandKind.REPLACE_BY_CALLER, "fresh", "default") is MutationTarget.ROUTED


# ---------------------------------------------------------------------------
# The canonical writer enforces it
# ---------------------------------------------------------------------------

@pytest.fixture
def writer(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as (client, app):
        response = client.post("/remember", json={
            "content": "The northern depot closes at 18:00 on weekdays.",
            "profile_id": "work", "idempotency_key": "routing-seed"})
        assert response.status_code == 200, response.text
        [fact_id] = response.json()["fact_ids"]
        yield app.state.canonical_remember_runtime, app.state.engine, fact_id


def _fact(engine, fact_id: str) -> dict | None:
    rows = engine._db.execute(
        "SELECT profile_id, content, scope FROM atomic_facts WHERE fact_id=?", (fact_id,))
    return dict(rows[0]) if rows else None


@pytest.mark.parametrize("mutate", [
    lambda rt, fid: rt.update_fact("work", fid, {"content": "rewritten"}),
    lambda rt, fid: rt.archive_fact("work", fid),
    lambda rt, fid: rt.set_fact_scope("work", fid, "global", []),
], ids=["update", "archive", "scope"])
def test_non_routable_mutations_cannot_touch_a_routed_profile(writer, mutate) -> None:
    """Refused as a decision, not reported as an outage that invites a retry."""
    from superlocalmemory.core.remember_runtime import MutationNotRoutable

    runtime, engine, fact_id = writer
    before = _fact(engine, fact_id)
    with pytest.raises(MutationNotRoutable):
        mutate(runtime, fact_id)
    assert _fact(engine, fact_id) == before


def test_a_routed_replace_to_a_missing_profile_is_a_refusal_not_an_outage(writer) -> None:
    from superlocalmemory.core.remember_runtime import UnknownMutationProfile

    runtime, _engine, fact_id = writer
    with pytest.raises(UnknownMutationProfile):
        runtime.replace_by_caller("ghost", fact_id, fact_id,
                                  trusted_actor_id="tester", idempotency_key="ghost-1")


@pytest.mark.parametrize("mutate", [
    lambda rt, fid: rt.delete_fact("ghost", fid),
    lambda rt, fid: rt.set_fact_kinds("ghost", [(fid, "decision")]),
], ids=["delete", "kind"])
def test_a_routed_delete_or_kind_to_a_missing_profile_is_a_refusal(writer, mutate) -> None:
    from superlocalmemory.core.remember_runtime import UnknownMutationProfile

    runtime, engine, fact_id = writer
    before = _fact(engine, fact_id)
    with pytest.raises(UnknownMutationProfile):
        mutate(runtime, fact_id)
    assert _fact(engine, fact_id) == before


def test_a_kind_change_finds_only_facts_the_named_profile_owns(writer) -> None:
    """Named 'default', a 'work' fact is not found: the profile is a filter, so
    a routed change cannot reach a fact by id across profiles."""
    runtime, engine, fact_id = writer
    receipt = dict(runtime.set_fact_kinds("default", [(fact_id, "decision")]))
    assert [f.get("ok") for f in receipt.get("facts", [])] == [False], receipt
    assert _fact(engine, fact_id)["profile_id"] == "work"


def test_a_routed_review_of_a_missing_profile_is_a_refusal_not_an_outage(writer) -> None:
    from superlocalmemory.core.remember_runtime import UnknownMutationProfile

    runtime, _engine, _fact_id = writer
    with pytest.raises(UnknownMutationProfile):
        runtime.transition_correction("ghost", "abc", action="rollback",
                                      expected_version=1, actor_id="tester")
