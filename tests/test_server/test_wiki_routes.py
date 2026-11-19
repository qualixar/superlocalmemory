# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The living wiki's doors: read pages, project to Markdown, and read edits back (D-17, D-18)."""

from __future__ import annotations

import json

import pytest

from superlocalmemory.compiler import page_sources, proposals
from tests.test_server.test_per_request_profile import _daemon
from tests.test_truth.test_slot_pass import _fact

PERSON = proposals.Basis.person("alice", "cli")
ENTITY = "ent-slm"


@pytest.fixture
def wiki_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SLM_WIKI_DIR", str(tmp_path / "wiki"))
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.delenv("SLM_4230_OFF", raising=False)
    # The wiki is on by default; these tests start from a person's own "off" and turn it on or to shadow
    # one test at a time (``_flag``).
    monkeypatch.setenv("SLM_FEATURE_COMPILER_WIKI", "off")
    return tmp_path / "wiki"


@pytest.fixture
def routed(engine_with_mock_deps, wiki_env):
    with _daemon(engine_with_mock_deps, profiles=("work",)) as pair:
        yield pair


@pytest.fixture
def seeded(routed):
    client, app = routed
    db = app.state.engine._db
    ids = [_fact(db, f"SuperLocalMemory shipped release {i} with a measured recall gain",
                 f"2026-0{i + 1}-15T10:00:00Z") for i in range(3)]
    db.execute("INSERT OR IGNORE INTO canonical_entities (entity_id, profile_id, canonical_name) "
               "VALUES (?, 'default', 'SuperLocalMemory')", (ENTITY,))
    for fid in ids:
        db.execute("UPDATE atomic_facts SET canonical_entities_json = ? WHERE fact_id = ?",
                   (json.dumps([ENTITY]), fid))
    return client, app, ids


def _flag(monkeypatch, state):
    monkeypatch.setenv("SLM_FEATURE_COMPILER_WIKI", state)


def _publish(app) -> None:
    db = app.state.engine._db
    with db.raw_connection() as conn:
        drafts = page_sources.build_drafts(conn, "default")
        set_id = proposals.create_set(conn, profile_id="default", origin="dream",
                                      proposals=[{"kind": "create_page", "payload": {"page": d}}
                                                 for d in drafts])
        proposals.apply_set(conn, profile_id="default", set_id=set_id, basis=PERSON)


def test_pages_are_listed_and_read_with_citations_and_the_withheld_count(seeded) -> None:
    client, app, ids = seeded
    assert client.get("/api/wiki/pages").json()["pages"] == []
    _publish(app)
    listed = client.get("/api/wiki/pages").json()
    assert listed["success"] and listed["flag"] == "off" and len(listed["pages"]) == 1
    assert listed["status"]["pages"] == 1
    page_id = listed["pages"][0]["page_id"]
    page = client.get(f"/api/wiki/pages/{page_id}").json()["page"]
    assert page["origin"] == "derived" and not page["stale"] and page["withheld"]["count"] == 0
    assert [c["fact_id"] for s in page["sentences"] for c in s["citations"]] == [
        i for i in reversed(ids)]
    assert client.get("/api/wiki/pages/nope").status_code == 404
    assert client.get("/api/wiki/pages", params={"kind": "story"}).json()["pages"] == []


def test_a_sentence_whose_memory_changed_is_withheld_when_the_page_is_read(seeded) -> None:
    client, app, ids = seeded
    _publish(app)
    page_id = client.get("/api/wiki/pages").json()["pages"][0]["page_id"]
    app.state.engine._db.execute("UPDATE atomic_facts SET content = 'Changed.' WHERE fact_id = ?",
                                 (ids[0],))
    page = client.get(f"/api/wiki/pages/{page_id}").json()["page"]
    assert page["stale"] and page["withheld"]["count"] == 1 and len(page["sentences"]) == 2


def test_a_caller_view_parameter_decides_nothing(seeded, monkeypatch) -> None:
    """D-49: locality is the connection's, never a query parameter the caller sends."""
    client, app, _ids = seeded
    _flag(monkeypatch, "on")
    _publish(app)
    remote = {"caller_view": "remote"}
    assert client.get("/api/wiki/pages", params=remote).status_code == 200
    assert client.get("/api/wiki/diff", params=remote).status_code == 200


def test_build_is_refused_while_the_wiki_is_off_and_only_reports_in_shadow(seeded, wiki_env,
                                                                         monkeypatch) -> None:
    client, app, _ids = seeded
    _publish(app)
    r = client.post("/api/wiki/build", json={"channel": "cli"})
    assert r.status_code == 409 and "compiler.wiki" in r.json()["detail"]
    _flag(monkeypatch, "shadow")
    shadow = client.post("/api/wiki/build", json={}).json()
    assert shadow["dry_run"] and len(shadow["written"]) == 1 and not wiki_env.exists()
    _flag(monkeypatch, "on")
    done = client.post("/api/wiki/build", json={}).json()
    assert not done["dry_run"] and list(wiki_env.rglob("*.md"))
    assert client.get("/api/wiki/diff").json()["items"] == []
    assert client.post("/api/wiki/build", json={}).json()["unchanged"]


def test_rebuild_is_a_recorded_decision_and_compiles_then_projects(seeded, wiki_env, monkeypatch) -> None:
    client, app, _ids = seeded
    _flag(monkeypatch, "on")
    r = client.post("/api/wiki/rebuild", json={"channel": "dashboard"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] == 1 and body["files"]["written"]
    listed = client.get("/api/wiki/pages").json()
    assert len(listed["pages"]) == 1
    done = client.get(f"/api/proposals/{body['set_id']}").json()["set"]
    assert done["state"] == "applied" and done["basis"]["kind"] == "decision"
    again = client.post("/api/wiki/rebuild", json={"channel": "cli"}).json()
    assert again["pages"] == 0 and again["set_id"] is None


def test_sync_reads_an_edit_as_a_proposal_and_a_deleted_line_as_nothing(seeded, wiki_env,
                                                                      monkeypatch) -> None:
    client, app, ids = seeded
    _flag(monkeypatch, "on")
    client.post("/api/wiki/rebuild", json={"channel": "cli"})
    path = next(wiki_env.rglob("topic/*.md"))
    text = path.read_text()
    path.write_text(text.replace("release 1 with a measured recall gain",
                                 "release 1 and a doubled recall gain"))
    db = app.state.engine._db
    live_before = db.execute("SELECT COUNT(*) FROM fact_temporal_validity "
                             "WHERE system_expired_at IS NULL")[0][0]
    r = client.post("/api/wiki/sync", json={"channel": "cli"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["found"] == 1 and len(body["queued"]) == 1 and body["skipped"] == []
    shown = client.get(f"/api/proposals/{body['queued'][0]['set_id']}").json()["set"]
    assert shown["state"] == "proposed" and shown["origin"] == "wiki"
    assert "doubled recall" in shown["proposals"][0]["summary"]
    # a proposal, not a write: the memory it edits is still current, the new words are hidden
    assert db.execute("SELECT COUNT(*) FROM fact_temporal_validity WHERE system_expired_at IS NULL"
                      )[0][0] == live_before + 1  # the hidden successor's own validity row
    case = db.execute("SELECT status, successor_fact_id FROM correction_cases")[0]
    assert case["status"] == "proposed"
    origin = db.execute("SELECT channel FROM fact_origin WHERE fact_id = ?",
                        (case["successor_fact_id"],))[0][0]
    assert origin == "external_edit"
    again = client.post("/api/wiki/sync", json={"channel": "cli"}).json()
    assert again["found"] == 0 and again["already_proposed"] == 1
    done = client.post(f"/api/proposals/{body['queued'][0]['set_id']}/apply",
                       json={"channel": "dashboard"})
    assert done.status_code == 200 and done.json()["set"]["state"] == "applied"
    assert db.execute("SELECT status FROM correction_cases")[0][0] == "applied"


def test_a_credential_in_an_edit_is_stored_as_a_placeholder(seeded, wiki_env, monkeypatch) -> None:
    client, app, ids = seeded
    _flag(monkeypatch, "on")
    monkeypatch.setenv("SLM_FEATURE_TRUST_SECRET_SCREEN", "on")
    client.post("/api/wiki/rebuild", json={"channel": "cli"})
    path = next(wiki_env.rglob("topic/*.md"))
    secret = "ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8"
    path.write_text(path.read_text().replace("release 2 with a measured recall gain",
                                             f"release 2 token {secret}"))
    body = client.post("/api/wiki/sync", json={"channel": "cli"}).json()
    assert len(body["queued"]) == 1
    db = app.state.engine._db
    succ = db.execute("SELECT successor_fact_id FROM correction_cases")[0][0]
    stored = db.execute("SELECT content FROM atomic_facts WHERE fact_id = ?", (succ,))[0][0]
    assert secret not in stored and "[REDACTED:" in stored


def test_sync_in_shadow_lists_and_queues_nothing_and_off_is_refused(seeded, wiki_env,
                                                                  monkeypatch) -> None:
    client, _app, _ids = seeded
    assert client.post("/api/wiki/sync", json={"channel": "cli"}).status_code == 409
    _flag(monkeypatch, "on")
    client.post("/api/wiki/rebuild", json={"channel": "cli"})
    path = next(wiki_env.rglob("topic/*.md"))
    path.write_text(path.read_text().replace("release 0 with a measured", "release 0 with a better"))
    _flag(monkeypatch, "shadow")
    got = client.post("/api/wiki/sync", json={"channel": "cli"}).json()
    assert got["dry_run"] and got["found"] == 1 and got["queued"] == []
    assert client.get("/api/proposals").json()["count"] == 0


def test_the_kill_switch_reads_the_wiki_as_off(seeded, monkeypatch) -> None:
    client, _app, _ids = seeded
    _flag(monkeypatch, "on")
    monkeypatch.setenv("SLM_4230_OFF", "1")
    assert client.post("/api/wiki/build", json={}).status_code == 409


def test_a_sync_that_failed_after_admitting_an_edit_is_replayed_not_duplicated(
        seeded, wiki_env, monkeypatch) -> None:
    client, app, _ids = seeded
    _flag(monkeypatch, "on")
    client.post("/api/wiki/rebuild", json={"channel": "cli"})
    path = next(wiki_env.rglob("topic/*.md"))
    path.write_text(path.read_text().replace("release 1 with a measured", "release 1 with a better"))
    real = proposals.create_set
    monkeypatch.setattr(proposals, "create_set",
                        lambda *a, **k: (_ for _ in ()).throw(RuntimeError("disk full")))
    assert client.post("/api/wiki/sync", json={"channel": "cli"}).status_code == 500
    monkeypatch.setattr(proposals, "create_set", real)
    body = client.post("/api/wiki/sync", json={"channel": "cli"}).json()
    assert len(body["queued"]) == 1 and body["skipped"] == []
    db = app.state.engine._db
    assert db.execute("SELECT COUNT(*) FROM correction_cases")[0][0] == 1  # one case, replayed
