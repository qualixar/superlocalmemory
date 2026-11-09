# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""``POST /api/overtaken-corrections/dismiss``: what ``slm corrections dismiss`` calls.

Dry run by default (a read); ``apply`` takes a backup of memory.db first, moves
the pending machine cases, and purges the profile's context cache.
"""

from __future__ import annotations

import sqlite3
from pathlib import Path

import pytest

from tests.test_server.test_per_request_profile import _daemon

URL = "/api/overtaken-corrections/dismiss"


@pytest.fixture
def routed(engine_with_mock_deps):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair


def _case(engine, case_id: str, kind="host_attested", reason="temporal_contradiction",
          profile="work") -> None:
    """A case row exactly as the ledger holds it; the facts are placeholders (FKs are
    not enforced on this connection) because dismissal reads ids only."""
    conn = sqlite3.connect(engine._db.db_path, isolation_level=None)
    conn.execute("INSERT INTO correction_cases (case_id, profile_id, scope, predecessor_fact_id, "
                 "successor_fact_id, reason_code, status, version, idempotency_key, "
                 "proposed_by_actor_id, proposed_by_actor_kind, proposed_by_trust_tier, "
                 "created_at, updated_at) VALUES (?, ?, 'personal', ?, 's', ?, 'proposed', 0, ?, "
                 "'w', ?, 't', '2026-10-01T00:00:00+00:00', '2026-10-01T00:00:00+00:00')",
                 (case_id, profile, f"p-{case_id}", reason, f"k-{case_id}", kind))
    conn.close()


def _n(engine, sql: str, args: tuple = ()) -> int:
    return int(dict(engine._db.execute(sql, args)[0])["n"])


def test_a_dry_run_reports_and_changes_nothing(routed) -> None:
    client, app = routed
    engine = app.state.engine
    _case(engine, "a", reason="temporal_contradiction")
    _case(engine, "b", reason="consolidation_update")
    _case(engine, "person", kind="host_authenticated", reason="direct_content_correction")

    response = client.post(URL, json={"profile_id": "work"})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dry_run"] is True and body["eligible"] == 2
    assert body["by_reason"] == {"temporal_contradiction": 1, "consolidation_update": 1}
    assert "backup" not in body
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases") == 3


def test_apply_backs_up_first_moves_the_cases_and_purges_the_cache(routed, monkeypatch) -> None:
    client, app = routed
    engine = app.state.engine
    for i in range(3):
        _case(engine, f"m{i}")
    _case(engine, "person", kind="host_authenticated", reason="direct_content_correction")
    purged: list[str] = []
    monkeypatch.setattr("superlocalmemory.core.mutations.purge_profile_context_cache",
                        lambda eng, profile: purged.append(profile))

    response = client.post(URL, json={"profile_id": "work", "apply": True})

    assert response.status_code == 200, response.text
    body = response.json()
    assert body["dry_run"] is False and body["dismissed"] == 3
    backup = Path(body["backup"])
    assert backup.is_file() and backup.parent == Path(engine._db.db_path).parent
    with sqlite3.connect(str(backup)) as copy:    # taken BEFORE the write
        assert copy.execute("SELECT COUNT(*) FROM correction_cases").fetchone() == (4,)
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases") == 1
    assert _n(engine, "SELECT COUNT(*) AS n FROM correction_cases_overtaken "
                      "WHERE user_action = 'dismissed'") == 3
    assert purged == ["work"]


def test_apply_with_nothing_eligible_takes_no_backup(routed) -> None:
    client, app = routed
    folder = Path(app.state.engine._db.db_path).parent

    body = client.post(URL, json={"profile_id": "work", "apply": True}).json()

    assert body["dismissed"] == 0 and "backup" not in body
    assert list(folder.glob("*.before-*")) == []


def test_filters_reach_the_query(routed) -> None:
    client, app = routed
    engine = app.state.engine
    _case(engine, "t", reason="temporal_contradiction")
    _case(engine, "u", reason="consolidation_update")

    only = client.post(URL, json={"profile_id": "work", "reasons": ["consolidation_update"]})
    assert only.json()["by_reason"] == {"consolidation_update": 1}
    old = client.post(URL, json={"profile_id": "work", "before": "2026-09-01"})
    assert old.json()["eligible"] == 0
    new = client.post(URL, json={"profile_id": "work", "before": "2026-11-01"})
    assert new.json()["eligible"] == 2


def test_bad_input_is_refused_before_any_work(routed) -> None:
    client, _ = routed
    for body in ({"reasons": ["direct_content_correction"]}, {"before": "tomorrow-ish"},
                 {"older_than_days": -3}):
        response = client.post(URL, json={"profile_id": "work", "apply": True, **body})
        assert response.status_code == 422, (body, response.text)


def test_an_unknown_profile_is_not_found(routed) -> None:
    client, _ = routed
    assert client.post(URL, json={"profile_id": "nope"}).status_code == 404
