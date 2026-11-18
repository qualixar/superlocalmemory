# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``GET /api/trust/why/{fact_id}``: review-read role, per profile, read-only, never the keychain."""

from __future__ import annotations

import pytest

from superlocalmemory.storage.migrations import M074_fact_origin, M075_injection_receipts, M082_fact_signatures
from superlocalmemory.storage.models import AtomicFact, MemoryRecord
from superlocalmemory.trust import signing_key as sk
from tests.test_server.test_per_request_profile import _daemon


@pytest.fixture
def routed(engine_with_mock_deps):
    with engine_with_mock_deps._db.raw_connection() as conn:
        for module in (M074_fact_origin, M075_injection_receipts, M082_fact_signatures):
            module.apply(conn)
        conn.commit()
    sk.install(None)
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair
    sk.install(None)


def _fact(app, profile, text):
    db = app.state.engine._db
    record = MemoryRecord(content=text, session_date="2026-11-15", profile_id=profile)
    db.store_memory(record)
    fact = AtomicFact(memory_id=record.memory_id, content=text, profile_id=profile)
    db.store_fact(fact)
    return fact


def test_the_report_has_every_section(routed) -> None:
    client, app = routed
    fact = _fact(app, "default", "The deploy key rotates monthly.")
    body = client.get(f"/api/trust/why/{fact.fact_id}").json()
    assert body["success"] and body["fact_id"] == fact.fact_id and body["profile_id"] == "default"
    assert {"origin", "firewall", "signature", "truth", "compiler", "deliveries"} <= set(body)
    assert body["signature"]["state"] == "unsigned"


def test_a_memory_in_another_profile_is_a_404_not_a_hint(routed) -> None:
    client, app = routed
    fact = _fact(app, "work", "A work memory.")
    assert client.get(f"/api/trust/why/{fact.fact_id}").status_code == 404
    assert client.get(f"/api/trust/why/{fact.fact_id}",
                      params={"profile_id": "work"}).json()["profile_id"] == "work"


def test_an_unknown_memory_is_a_404(routed) -> None:
    client, _app = routed
    assert client.get("/api/trust/why/nope").status_code == 404


def test_an_unknown_profile_is_refused(routed) -> None:
    client, app = routed
    fact = _fact(app, "default", "Anything.")
    r = client.get(f"/api/trust/why/{fact.fact_id}", params={"profile_id": "nobody"})
    assert r.status_code >= 400 or r.json().get("success") is False


def test_the_daemons_key_checks_the_signature_without_asking_the_keychain(routed, monkeypatch) -> None:
    client, app = routed
    signer = sk.Signer(bytes(range(32)))
    monkeypatch.setenv("SLM_FEATURE_TRUST_SIGNING", "on")
    sk.install(signer)
    fact = _fact(app, "default", "Signed through the daemon's key.")
    monkeypatch.setattr(sk, "load", lambda *a, **k: (_ for _ in ()).throw(AssertionError("keychain")))
    body = client.get(f"/api/trust/why/{fact.fact_id}").json()
    assert body["signature"]["state"] == "ok"


def test_the_deliveries_limit_is_bounded(routed) -> None:
    client, app = routed
    fact = _fact(app, "default", "Bounded.")
    assert client.get(f"/api/trust/why/{fact.fact_id}", params={"deliveries": 100000}).status_code == 200
    assert client.get(f"/api/trust/why/{fact.fact_id}", params={"deliveries": 0}).status_code == 200
