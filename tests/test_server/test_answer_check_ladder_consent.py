# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""W2, D-20: the background judges' consent to ask Jev, from the dashboard."""

from __future__ import annotations

import json

from superlocalmemory.core import consent_store
from superlocalmemory.infra.data_root import canonical_data_root
from tests.test_server.test_answer_check_api import (  # noqa: F401 — fixtures
    _FakeKeyStore,
    _reset_fake_key_store,
    _set_retrieval,
    call_order,
    client,
)
from tests.test_server.test_answer_check_rerank_api import _jev_on

URL = "/api/v3/answer-check/jev/ladder"
PURPOSE = "judge_ladder_hosted"


def _given() -> bool:
    return consent_store.given(canonical_data_root(), PURPOSE)


def _post(client, accepted=True, provider="typesafe"):  # noqa: F811
    return client.post(URL, json={"accepted": accepted, "provider": provider})


def test_agreeing_needs_the_online_answer_check_with_jev(client) -> None:  # noqa: F811
    response = _post(client)
    assert response.status_code == 400 and "answer check" in response.json()["error"]
    assert not _given()


def test_agreeing_with_jev_on_records_it_with_its_provider(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    body = _post(client).json()
    assert _given() and body["jev"]["ladder"]["consent"] is True
    entry = json.loads((canonical_data_root() / "consents.json").read_text())[PURPOSE]
    assert entry["via"] == "dashboard" and entry["provider"] == "typesafe"
    assert "one pair per request" in body["jev"]["ladder"]["notice"]
    assert "TypeSafe" in body["jev"]["ladder"]["notice"]


def test_another_provider_than_the_one_in_use_is_refused(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    assert _post(client, provider="openrouter").status_code == 409 and not _given()


def test_stopping_always_works(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    _post(client)
    _set_retrieval(tmp_path, sufficiency_judge="off")
    body = _post(client, accepted=False).json()
    assert not _given() and body["jev"]["ladder"]["consent"] is False


def test_withdrawing_the_answer_check_consent_withdraws_this_one(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    _post(client)
    response = client.post("/api/v3/answer-check/jev/consent",
                           json={"accepted": False, "provider": "typesafe"})
    assert response.status_code == 200, response.text
    assert not _given()


def test_moving_the_answer_check_to_another_provider_withdraws_it(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    _post(client)
    _FakeKeyStore._by_test["openrouter"] = "sk-test-openrouter-key-000000000000"
    client.post("/api/v3/answer-check/jev/consent",
                json={"accepted": True, "provider": "openrouter"})
    assert not _given()


def test_the_status_says_whether_the_judges_only_watch(client, tmp_path) -> None:  # noqa: F811
    _jev_on(tmp_path)
    ladder = _post(client).json()["jev"]["ladder"]
    assert ladder["flag"] in {"off", "shadow"} and ladder["since"]


def test_each_change_leaves_an_audit_receipt(client, tmp_path) -> None:  # noqa: F811
    import sqlite3

    from superlocalmemory.infra.data_root import state_path

    _jev_on(tmp_path)
    _post(client)
    _post(client, accepted=False)
    rows = sqlite3.connect(str(state_path("audit_chain.db"))).execute(
        "SELECT operation FROM audit_chain").fetchall()
    assert [r[0] for r in rows].count("consent." + PURPOSE) == 2


def test_no_receipt_no_change(client, tmp_path, monkeypatch) -> None:  # noqa: F811
    from superlocalmemory.compliance import audit

    def broken(self, *a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(audit.AuditChain, "log", broken)
    _jev_on(tmp_path)
    assert _post(client).status_code == 503 and not _given()
