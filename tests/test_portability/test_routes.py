# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""WS-9, D-41, D-49: the HTTP doors. Roles, the size cap, the switch named, the decision recorded."""

from __future__ import annotations

import base64
import time

from fastapi.testclient import TestClient

from superlocalmemory.core.agent_attribution import CHANNEL_HEADER, CONNECTION_HEADER
from superlocalmemory.core.feature_flags import env_name
from superlocalmemory.portability import service
from superlocalmemory.portability.jobs import JOBS
from tests.test_portability import fixtures as fx

AGENT = {CONNECTION_HEADER: "claude-code", CHANNEL_HEADER: "mcp"}


def b64(data: bytes) -> str:
    return base64.b64encode(data).decode()


def upload(name, payload) -> dict:
    return {"filename": name, "data_base64": b64(payload if isinstance(payload, bytes)
                                                 else fx.dumps(payload))}


def wait(world, run_id, seconds=60):
    assert JOBS.join(run_id, seconds)


def test_when_a_person_turned_it_off_every_call_names_the_switch(world, monkeypatch) -> None:
    monkeypatch.setenv(env_name("portability.import"), "off")  # on by default; this is a person's own off
    listing = world.client.get("/api/import/runs").json()
    assert listing["success"] and listing["enabled"] is False and listing["flag"] == "portability.import"
    for path in ("/api/import/preview", "/api/import/run"):
        r = world.client.post(path, json=upload("m.json", fx.MEM0_ITEMS))
        assert r.status_code == 409, r.text  # a state, not a role
        assert r.headers["X-SLM-Switch"] == "portability.import"
        assert "slm flags set portability.import on" in r.json()["detail"]
    assert world.count("import_runs") == 0


def test_preview_then_run_then_status_then_undo(world) -> None:
    pre = world.client.post("/api/import/preview", json=upload("conversations.json", fx.chatgpt_list()))
    assert pre.status_code == 200, pre.text
    body = pre.json()
    assert body["label"] == "ChatGPT" and body["would_save"] == 4 and body["conversations"] == 2
    assert body["summary"].startswith("We found 2 conversations from ChatGPT, November 2023.")
    assert len(body["upload_id"]) == 32 and world.count("memories") == 0
    started = world.client.post("/api/import/run", json={"upload_id": body["upload_id"],
                                                        "filename": "conversations.json"},
                                headers=AGENT)
    assert started.status_code == 200, started.text
    run = started.json()["run"]
    wait(world, run["run_id"])
    shown = world.client.get(f"/api/import/runs/{run['run_id']}").json()["run"]
    assert shown["state"] == "done" and shown["saved_items"] == 4 and shown["progress"] == 1.0
    assert (shown["actor"], shown["channel"]) == ("agent:claude-code", "mcp")
    listing = world.client.get("/api/import/runs").json()
    assert listing["runs"][0]["run_id"] == run["run_id"] and listing["limits"]["max_upload_mb"] == 64
    undone = world.client.post(f"/api/import/runs/{run['run_id']}/undo",
                               json={"wait_seconds": 30}, headers=AGENT)
    assert undone.status_code == 200 and undone.json()["run"]["state"] == "rolled_back"
    assert world.count("memories") == 0
    again = world.client.post(f"/api/import/runs/{run['run_id']}/undo", json={})
    assert again.status_code == 409 and "already undone" in again.json()["detail"]


def test_the_staged_upload_is_deleted_when_the_run_has_read_it(world, tmp_path) -> None:
    body = world.client.post("/api/import/preview", json=upload("m.json", fx.MEM0_ITEMS)).json()
    staged = tmp_path / "slm-home" / "import-staging" / body["upload_id"]
    assert staged.is_dir() and oct(staged.stat().st_mode & 0o777) == "0o700"
    run = world.client.post("/api/import/run", json={"upload_id": body["upload_id"]}).json()["run"]
    wait(world, run["run_id"])
    for _ in range(50):
        if not staged.exists():
            break
        time.sleep(0.05)
    assert not staged.exists()
    gone = world.client.post("/api/import/run", json={"upload_id": body["upload_id"]})
    assert gone.status_code == 422 and "no longer here" in gone.json()["detail"]


def test_an_import_is_a_decision_listed_and_undone_through_the_decision_log(world) -> None:
    run = world.client.post("/api/import/run", json=upload("m.json", fx.MEM0_ITEMS),
                            headers=AGENT).json()["run"]
    wait(world, run["run_id"])
    [row] = [d for d in world.client.get("/api/decisions").json()["decisions"] if d["kind"] == "import"]
    assert (row["actor"], row["channel"], row["can_undo"]) == ("agent:claude-code", "mcp", True)
    assert row["summary"] == "Imported 2 memories from Mem0"
    r = world.client.post(f"/api/decisions/{row['decision_id']}/undo", json={})
    assert r.status_code == 200, r.text
    for _ in range(200):
        if service.get_run(world.engine, world.profile, run["run_id"])["state"] == "rolled_back":
            break
        time.sleep(0.05)
    assert world.count("memories") == 0
    after = {d["action"]: d for d in world.client.get("/api/decisions").json()["decisions"]
             if d["kind"] == "import"}
    assert after["run"]["undone"] and after["rollback"]["actor"] == "client:local"


def test_the_dashboard_door_records_the_dashboard(world) -> None:
    from superlocalmemory.core.security_primitives import ensure_install_token

    page = TestClient(world.app, client=("127.0.0.1", 50000))
    page.headers["X-Install-Token"] = ensure_install_token()
    r = page.post("/api/import/run", json=upload("m.json", fx.MEM0_ITEMS),
                  headers={CHANNEL_HEADER: "dashboard"})
    assert r.status_code == 200, r.text
    run = r.json()["run"]
    wait(world, run["run_id"])
    assert (run["actor"], run["channel"]) == ("dashboard", "dashboard")


def test_the_size_cap_is_refused_before_the_body_is_read(world, monkeypatch) -> None:
    from superlocalmemory.server.routes import portability as route

    monkeypatch.setattr(route, "_BODY_CAP", 2_000)
    big = {"filename": "m.json", "data_base64": b64(b"x" * 5_000)}
    r = world.client.post("/api/import/preview", json=big)
    assert r.status_code == 413 and "larger than 64 MB" in r.json()["detail"]
    chunked = world.client.post("/api/import/run", content=iter([b'{"data_base64": "', b"A" * 3_000, b'"}']),
                                headers={"Content-Type": "application/json"})
    assert chunked.status_code == 413
    assert world.count("import_runs") == 0


def test_a_decoded_file_over_the_limit_is_refused_in_plain_words(world, monkeypatch) -> None:
    from superlocalmemory.portability import constants as C

    monkeypatch.setattr(C, "MAX_UPLOAD_BYTES", 100)
    r = world.client.post("/api/import/preview", json={"filename": "a.json",
                                                      "data_base64": b64(b"x" * 500)})
    assert r.status_code == 422 and "larger than" in r.json()["detail"]


def test_bad_input_gets_plain_words(world) -> None:
    unknown = world.client.post("/api/import/preview", json=upload("x.json", b'{"hello": 1}'))
    assert unknown.status_code == 422
    for name in ("ChatGPT", "Claude", "Mem0", "Letta", "OKF", "Markdown"):
        assert name in unknown.json()["detail"]
    assert world.client.post("/api/import/preview", json={}).status_code == 422
    bad = world.client.post("/api/import/preview", json={"filename": "a", "data_base64": "!!not base64"})
    assert bad.status_code == 422 and "could not be read" in bad.json()["detail"]
    named = world.client.post("/api/import/preview", json={**upload("a.md", b"# A\nbody here"),
                                                          "format": "notion"})
    assert named.status_code == 422 and "not a format I know" in named.json()["detail"]
    assert world.client.get("/api/import/runs/imp_nope").status_code == 404
    assert world.client.post("/api/import/runs/imp_nope/cancel", json={}).status_code == 404
    assert world.client.post("/api/import/runs/imp_nope/undo", json={}).status_code == 404


def test_a_second_import_while_one_runs_is_a_conflict(world) -> None:
    from superlocalmemory.portability import runs

    with world.db.raw_connection() as conn:
        runs.create(conn, profile_id=world.profile, fmt="mem0", source_label="x", actor="a",
                    channel="cli", conversations=0, total_items=1, skipped={}, first_at="", last_at="")
    r = world.client.post("/api/import/run", json=upload("m.json", fx.MEM0_ITEMS))
    assert r.status_code == 409 and "already running" in r.json()["detail"]


def test_a_path_is_taken_from_this_computer_only(world, tmp_path) -> None:
    folder = tmp_path / "notes"
    folder.mkdir()
    (folder / "AGENTS.md").write_text(fx.AGENTS_MD)
    local = world.client.post("/api/import/preview", json={"path": str(folder)})
    assert local.status_code == 200, local.text
    assert local.json()["format"] == "markdown" and local.json()["memories"] == 3
    remote = TestClient(world.app, client=("203.0.113.9", 50000))
    remote.headers.update(world.client.headers)
    refused = remote.post("/api/import/preview", json={"path": str(folder)})
    assert refused.status_code in (401, 403)
    own = world.client.post("/api/import/preview", json={"path": str(tmp_path / "slm-home")})
    assert own.status_code == 422 and "own data" in own.json()["detail"]


def test_a_viewer_can_preview_but_not_import_or_undo(world, monkeypatch) -> None:
    from superlocalmemory.access.rbac import RbacEngine

    world.app.state.rbac = RbacEngine(str(world.db.db_path))
    monkeypatch.setattr("superlocalmemory.infra.auth_middleware.verify_api_key",
                        lambda presented, key_file=None: presented == "good-key")
    user = world.app.state.rbac.create_user("vera", "password-1234")
    world.app.state.rbac.set_membership("default", user["user_id"], "viewer")
    headers = {"X-SLM-API-Key": "good-key", CHANNEL_HEADER: "dashboard",
               "X-SLM-User-Session": world.app.state.rbac.create_session(user["user_id"])}
    remote = TestClient(world.app, client=("203.0.113.9", 50000))
    assert remote.get("/api/import/runs", headers=headers).status_code == 200
    seen = remote.post("/api/import/preview", json=upload("m.json", fx.MEM0_ITEMS), headers=headers)
    assert seen.status_code == 200, seen.text
    named = remote.post("/api/import/preview", json={"path": "/etc"}, headers=headers)
    assert named.status_code == 403 and "only be named by a caller on this computer" in named.json()["detail"]
    assert remote.post("/api/import/run", json=upload("m.json", fx.MEM0_ITEMS),
                       headers=headers).status_code == 403
    run = world.client.post("/api/import/run", json=upload("m.json", fx.MEM0_ITEMS)).json()["run"]
    wait(world, run["run_id"])
    assert remote.post(f"/api/import/runs/{run['run_id']}/undo", json={},
                       headers=headers).status_code == 403
    assert remote.post("/api/import/switch", json={"state": "off"}, headers=headers).status_code == 403
    assert world.count("memories") == 2


def test_the_switch_turns_the_pack_on_and_off_and_respects_the_kill_switch(world, monkeypatch) -> None:
    monkeypatch.delenv(env_name("portability.import"))
    assert world.client.get("/api/import/runs").json()["enabled"] is True  # on by default
    assert world.client.post("/api/import/switch", json={"state": "maybe"}).status_code == 422
    off = world.client.post("/api/import/switch", json={"state": "off"})
    assert off.status_code == 200 and off.json()["enabled"] is False
    assert world.client.get("/api/import/runs").json()["enabled"] is False
    on = world.client.post("/api/import/switch", json={"state": "on"})
    assert on.status_code == 200 and on.json()["enabled"] is True
    assert world.client.get("/api/import/runs").json()["state"] == "on"
    monkeypatch.setenv("SLM_4230_OFF", "1")
    killed = world.client.post("/api/import/switch", json={"state": "on"})
    assert killed.status_code == 409 and "kill switch" in killed.json()["detail"]
    listing = world.client.get("/api/import/runs").json()
    assert listing["kill_switch"] is True and listing["enabled"] is False


def test_cancel_over_http(world, monkeypatch) -> None:
    import threading

    from superlocalmemory.portability import runner

    gate, reached = threading.Event(), threading.Event()
    real = runner._let_others_through

    def hold(ctx, stop):
        reached.set()
        gate.wait(20)
        real(ctx, stop)

    monkeypatch.setattr(runner.C, "BATCH", 1)
    monkeypatch.setattr(runner, "_let_others_through", hold)
    run = world.client.post("/api/import/run", json=upload("c.json", fx.chatgpt_list())).json()["run"]
    assert reached.wait(20)
    r = world.client.post(f"/api/import/runs/{run['run_id']}/cancel", json={})
    assert r.status_code == 200 and r.json()["run"]["cancel_requested"] is True
    gate.set()
    wait(world, run["run_id"])
    assert world.client.get(f"/api/import/runs/{run['run_id']}").json()["run"]["state"] == "cancelled"
