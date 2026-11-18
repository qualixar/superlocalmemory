# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/delivered/recent``: review-read role, per profile, read-only, plain words."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest

from superlocalmemory.storage.migrations import M075_injection_receipts as M075
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from tests.test_server.test_per_request_profile import _daemon


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        with engine_with_mock_deps._db.raw_connection() as conn:
            M075.apply(conn)
            conn.commit()
        yield pair


def _delivery(app, profile, text, *, session="s1", path="hook_recall"):
    db = app.state.engine._db
    record = MemoryRecord(content=text, session_date="2026-11-15", profile_id=profile)
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=text, profile_id=profile)
    db.store_fact(fact)
    at = (datetime.now(UTC) - timedelta(minutes=5)).isoformat()
    did = f"d-{fact.fact_id}"
    with db.raw_connection() as conn:
        conn.execute(
            "INSERT INTO injection_deliveries (delivery_id, profile_id, session_id, client, path, design,"
            " holdout_state, epsilon, eligible_count, candidates, delivered_at)"
            " VALUES (?, ?, ?, 'claude', ?, 'one-out-v2', 'on', 0.05, 1, 1, ?)",
            (did, profile, session, path, at))
        conn.execute(
            "INSERT INTO injection_receipts (delivery_id, fact_id, profile_id, role, position, score,"
            " eligible, exclusion, propensity, included, delivered)"
            " VALUES (?, ?, ?, 'recall', 0, 0.5, 1, NULL, 0.95, 1, 1)", (did, fact.fact_id, profile))
        conn.commit()
    return fact


def test_the_card_gets_groups_reasons_and_both_gate_numbers(routed) -> None:
    client, app = routed
    _delivery(app, "default", "The backup runs at night.")
    body = client.get("/api/delivered/recent").json()
    assert body["success"] and body["profile_id"] == "default" and body["memories"] == 1
    (group,) = body["groups"]
    (delivery,) = group["deliveries"]
    assert delivery["kind"] == "prompt" and delivery["memories"][0]["preview"] == "The backup runs at night."
    assert "Found by search" in delivery["memories"][0]["why"]["words"]
    assert set(body["need_gate"]) >= {"turns", "skipped", "rules_bound"}
    assert body["harm_gate"]["acting"] is False and body["harm_gate"]["would_withhold"] == []


def test_it_is_per_profile(routed) -> None:
    client, app = routed
    _delivery(app, "work", "A work memory.", session="w")
    _delivery(app, "default", "A default memory.", session="d")
    body = client.get("/api/delivered/recent", params={"profile_id": "work"}).json()
    assert body["profile_id"] == "work" and [g["session_id"] for g in body["groups"]] == ["w"]
    assert "default memory" not in str(body)


def test_an_unknown_profile_is_refused(routed) -> None:
    client, _app = routed
    r = client.get("/api/delivered/recent", params={"profile_id": "nobody"})
    assert r.status_code >= 400 or r.json().get("success") is False


def test_the_limit_is_bounded_and_a_quiet_store_answers(routed) -> None:
    client, _app = routed
    assert client.get("/api/delivered/recent", params={"limit": 10**6}).status_code == 200
    body = client.get("/api/delivered/recent").json()
    assert body["success"] and body["groups"] == [] and body["deliveries"] == 0


def test_a_remote_app_never_sees_a_memory_it_may_not(routed, monkeypatch) -> None:
    client, app = routed
    _delivery(app, "default", "Not for the remote app.")
    from superlocalmemory.retrieval import remote_view

    monkeypatch.setattr(remote_view, "parse_view", lambda raw: object() if raw else None)
    monkeypatch.setattr(remote_view, "hidden_among", lambda view, db, profile, ids: set(ids))
    body = client.get("/api/delivered/recent", params={"caller_view": "x"}).json()
    assert "Not for the remote app" not in str(body) and body["memories"] == 0


def test_the_trust_overview_route_is_per_profile_and_read_only(routed) -> None:
    client, app = routed
    _delivery(app, "default", "Anything.")
    body = client.get("/api/trust/overview").json()
    assert body["success"] and body["profile_id"] == "default"
    assert {"firewall", "secret_screen", "origins", "signatures", "audit_chain"} <= set(body)
    assert body["signatures"]["words"] and body["audit_chain"]["words"]
    r = client.get("/api/trust/overview", params={"profile_id": "nobody"})
    assert r.status_code >= 400 or r.json().get("success") is False


def test_the_trust_overview_never_asks_the_keychain(routed, monkeypatch) -> None:
    from superlocalmemory.trust import signing_key as sk

    client, _app = routed
    monkeypatch.setattr(sk, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keychain")))
    assert client.get("/api/trust/overview").json()["success"] is True
