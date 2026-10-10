"""The folder routes: local only, credentials for changes, the remote rule, one profile's sources."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import sources as routes

LOCAL, REMOTE = ("127.0.0.1", 50000), ("10.1.2.3", 50000)


class Db:
    def execute(self, sql, params=()):
        return [{"one": 1}] if params and params[0] in ("default", "p2") else []


def client(env, *, actor="authenticated:test", peer=LOCAL):
    app = FastAPI()
    app.state.engine = SimpleNamespace(_profile_id="default", _config=SimpleNamespace(pii_redaction=False),
                                       _db=Db(), _hooks=None)

    @app.middleware("http")
    async def _actor(request, call_next):
        if actor:
            request.state.authenticated_actor = actor
        return await call_next(request)

    app.include_router(routes.router)
    return TestClient(app, client=peer)


@pytest.fixture(autouse=True)
def allow_everything(monkeypatch):
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_permission",
                        lambda *a, **k: None)


def test_add_returns_a_preview_and_writes_nothing(env):
    env.write("a.md", "x")
    c = client(env)
    r = c.post("/api/v3/sources", json={"path": str(env.root)})
    assert r.status_code == 200
    body = r.json()
    assert body["files_by_type"] == {".md": 1} and body["source_id"]
    assert not (env.data / "media.db").exists()


def test_unsafe_root_is_refused_with_a_code(env):
    c = client(env)
    r = c.post("/api/v3/sources", json={"path": "/"})
    assert r.status_code == 422 and r.json()["detail"]["code"] == "filesystem_root"


def test_confirm_connects_and_lists(env):
    env.write("a.md", "x")
    c = client(env)
    sid = c.post("/api/v3/sources", json={"path": str(env.root)}).json()["source_id"]
    assert c.post(f"/api/v3/sources/{sid}/confirm").status_code == 202
    listed = c.get("/api/v3/sources").json()["sources"]
    assert [s["source_id"] for s in listed] == [sid]
    assert c.post(f"/api/v3/sources/{sid}/rescan").status_code == 202
    assert c.get(f"/api/v3/sources/{sid}/report").status_code == 200


def test_confirm_is_refused_while_remote_access_is_on(env):
    c = client(env)
    sid = c.post("/api/v3/sources", json={"path": str(env.root)}).json()["source_id"]
    env.remote = True
    r = c.post(f"/api/v3/sources/{sid}/confirm")
    assert r.status_code == 409 and r.json()["detail"]["code"] == "remote_access_on"
    assert "remote" in r.json()["detail"]["message"].lower()
    assert not (env.data / "media.db").exists()


def test_a_raising_check_refuses_confirm(env):
    c = client(env)
    sid = c.post("/api/v3/sources", json={"path": str(env.root)}).json()["source_id"]

    def boom():
        raise OSError

    env.host.remote_check = boom
    assert c.post(f"/api/v3/sources/{sid}/confirm").status_code == 409


def test_changes_need_credentials(env):
    c = client(env, actor="")
    assert c.post("/api/v3/sources", json={"path": str(env.root)}).status_code == 403


def test_remote_callers_are_refused(env):
    c = client(env, peer=REMOTE)
    assert c.get("/api/v3/sources").status_code == 403
    assert c.post("/api/v3/sources", json={"path": str(env.root)}).status_code == 403


def test_another_profiles_source_is_not_found(env):
    sid = env.add_and_confirm()
    c = client(env)
    assert c.get(f"/api/v3/sources/{sid}/report").status_code == 200
    assert c.get(f"/api/v3/sources/{sid}/report", params={"profile_id": "p2"}).status_code == 404
    assert c.delete(f"/api/v3/sources/{sid}", params={"profile_id": "p2"}).status_code == 404
    assert c.get("/api/v3/sources/deadbeef/report").status_code == 404


def test_remove_and_release(env):
    env.write("keys.md", "AKIA" + "ABCDEFGHIJKLMNOP")
    sid = env.add_and_confirm()
    env.scan(sid)
    c = client(env)
    assert c.post(f"/api/v3/sources/{sid}/quarantine/release", json={"relpath": "keys.md"}).status_code == 200
    assert c.post(f"/api/v3/sources/{sid}/quarantine/release", json={"relpath": "none.md"}).status_code == 404
    r = c.delete(f"/api/v3/sources/{sid}", params={"purge": "true"})
    assert r.status_code == 200 and r.json()["purged"] is True
    assert c.get("/api/v3/sources").json()["sources"] == []


def test_hint_is_not_found_until_the_watcher_exists(env):
    sid = env.add_and_confirm()
    c = client(env)
    assert c.post(f"/api/v3/sources/{sid}/hint", json={"relpaths": ["a.md"]}).status_code == 404
