# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``/api/v3/features``: read what is on, turn images and documents on or off."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.core import security_primitives
from superlocalmemory.runtimes import features as feat
from superlocalmemory.runtimes.managed_env import EnvStatus
from superlocalmemory.server.routes import features as routes

TOKEN = "install-token-for-this-test-0123456789abcdef"
AUTH = {"X-Install-Token": TOKEN}


class _Env:
    def __init__(self, state="not_installed"):
        self.state = state

    def status(self):
        return EnvStatus(state=self.state, progress=0.25, step="Setting up")

    def precheck(self):
        return {"disk_ok": True, "free_bytes": 10 * 1024 ** 3}


@pytest.fixture()
def ctx(tmp_path, monkeypatch):
    token_file = tmp_path / ".install_token"
    token_file.write_text(TOKEN, encoding="utf-8")
    monkeypatch.setattr(security_primitives, "_install_token_path", lambda: token_file)
    root = tmp_path / "data"
    root.mkdir()
    env = _Env()
    monkeypatch.setattr(routes, "_data_root", lambda: root)
    monkeypatch.setattr(routes, "_media_env", lambda: env)
    calls = {"enable": [], "disable": []}

    def fake_enable(**kw):
        calls["enable"].append(kw)
        return {"enabled": True, "env": env.status().to_dict(), "precheck": env.precheck(), "media_db": True}

    def fake_disable(**kw):
        calls["disable"].append(kw)
        return {"enabled": False, "env": env.status().to_dict(), "precheck": env.precheck(), "media_db": True}

    monkeypatch.setattr(feat, "enable_media", fake_enable)
    monkeypatch.setattr(feat, "disable_media", fake_disable)
    feat._reset_media_loaded()
    app = FastAPI()
    app.include_router(routes.router)
    return TestClient(app), root, env, calls


def test_get_shape_when_everything_is_off_creates_nothing(ctx):
    client, root, _env, _calls = ctx
    body = client.get("/api/v3/features").json()
    media = body["media"]
    assert media["enabled"] is False and media["requested"] is False
    assert {"env_state", "progress", "step", "restart_required"} <= set(media)
    assert media["restart_required"] is False
    assert isinstance(body["mesh"]["apps_with_mesh"], int)
    assert list(root.iterdir()) == []


def test_a_request_shows_as_requested(ctx):
    import json
    client, root, _e, _c = ctx
    (root / "features.json").write_text(json.dumps({"media": {"requested": True}}))
    assert client.get("/api/v3/features").json()["media"]["requested"] is True


def test_enable_answers_202_and_records_the_dashboard(ctx):
    client, _r, _e, calls = ctx
    response = client.post("/api/v3/features/media/enable", json={"yes": True}, headers=AUTH)
    assert response.status_code == 202, response.text
    assert response.json()["media"]["enabled"] is True
    assert calls["enable"][0]["source"] == "dashboard"


def test_enable_records_the_given_source(ctx):
    client, _root, _env, calls = ctx
    body = {"yes": True, "source": "cli"}
    assert client.post("/api/v3/features/media/enable", json=body, headers=AUTH).status_code == 202
    assert calls["enable"][0]["source"] == "cli"


@pytest.mark.parametrize("source", ["npm", "bogus", 7])
def test_enable_with_a_bad_source_is_400(ctx, source):
    client, _root, _env, calls = ctx
    body = {"yes": True, "source": source}
    assert client.post("/api/v3/features/media/enable", json=body, headers=AUTH).status_code == 400
    assert calls["enable"] == []


@pytest.mark.parametrize("body", [{}, {"yes": False}, {"yes": "true"}])
def test_enable_without_a_literal_yes_is_400(ctx, body):
    client, _r, _e, calls = ctx
    assert client.post("/api/v3/features/media/enable", json=body, headers=AUTH).status_code == 400
    assert calls["enable"] == []


@pytest.mark.parametrize("body,expected", [({}, False), ({"remove_files": True}, True)])
def test_disable_passes_remove_files(ctx, body, expected):
    client, _r, _e, calls = ctx
    response = client.post("/api/v3/features/media/disable", json=body, headers=AUTH)
    assert response.status_code == 200, response.text
    assert calls["disable"][0]["remove_files"] is expected


@pytest.mark.parametrize("path,body", [("/api/v3/features/media/enable", {"yes": True}),
                                       ("/api/v3/features/media/disable", {})])
def test_changes_need_a_credential(ctx, path, body):
    client, _r, _e, calls = ctx
    assert client.post(path, json=body).status_code == 403
    assert client.post(path, json=body, headers={"X-Install-Token": "wrong"}).status_code == 403
    assert calls == {"enable": [], "disable": []}


def test_changes_from_off_the_machine_are_refused(ctx):
    client, _r, _e, calls = ctx
    remote = TestClient(client.app, client=("203.0.113.9", 5000))
    response = remote.post("/api/v3/features/media/enable", json={"yes": True}, headers=AUTH)
    assert response.status_code == 403
    assert calls["enable"] == []


def test_restart_required_follows_the_env_and_the_loaded_flag(ctx):
    client, root, env, _c = ctx
    import json
    (root / "features.json").write_text(json.dumps({"media": {"enabled": True}}))
    env.state = "ready"
    assert client.get("/api/v3/features").json()["media"]["restart_required"] is True
    feat.mark_media_loaded()
    assert client.get("/api/v3/features").json()["media"]["restart_required"] is False
    feat._reset_media_loaded()


# -- memory: the warning on small machines and the picture model's use -------------

class _Worker:
    """Stands in for a running picture worker: this very process is the 'worker'."""

    def __init__(self, pid, model="google/embeddinggemma-2", cap=4500):
        self.pid, self.model_id, self.rss_limit_mb = pid, model, cap


def test_ram_block_shape_when_the_worker_is_not_running(ctx, monkeypatch):
    from superlocalmemory.runtimes import worker_client

    client, _root, _env, _calls = ctx
    monkeypatch.setattr(worker_client, "_CLIENTS", {})
    ram = client.get("/api/v3/features").json()["media"]["ram"]
    assert set(ram) == {"system_total_mb", "worker_rss_mb", "worker_cap_mb", "model"}
    assert ram["system_total_mb"] > 0 and ram["worker_rss_mb"] is None


def test_ram_block_with_a_running_worker_reports_its_size_and_cap(ctx, monkeypatch):
    import os
    from superlocalmemory.runtimes import worker_client

    client, _root, _env, _calls = ctx
    monkeypatch.setattr(worker_client, "_CLIENTS", {("r", "m", ""): _Worker(os.getpid())})
    ram = client.get("/api/v3/features").json()["media"]["ram"]
    assert ram["worker_rss_mb"] > 0
    assert ram["worker_cap_mb"] == 4500 and ram["model"] == "google/embeddinggemma-2"


def test_ram_block_with_a_dead_pid_counts_as_not_running(ctx, monkeypatch):
    from superlocalmemory.runtimes import worker_client

    client, _root, _env, _calls = ctx
    monkeypatch.setattr(worker_client, "_CLIENTS", {("r", "m", ""): _Worker(None)})
    ram = client.get("/api/v3/features").json()["media"]["ram"]
    assert ram["worker_rss_mb"] is None and ram["worker_cap_mb"] == 4500


def test_reading_the_ram_block_creates_nothing(ctx, monkeypatch):
    from superlocalmemory.runtimes import worker_client

    client, root, _env, _calls = ctx
    monkeypatch.setattr(worker_client, "_CLIENTS", {})
    client.get("/api/v3/features")
    assert list(root.iterdir()) == [] and worker_client._CLIENTS == {}


def test_a_small_machine_gets_a_warning_in_the_features_reply(ctx, monkeypatch):
    client, _root, env, _calls = ctx
    monkeypatch.setattr(env, "precheck", lambda: {"disk_ok": True, "free_bytes": 10 * 1024 ** 3,
                                                   "ram_bytes": 4 * 1024 ** 3, "ram_warn": True})
    media = client.get("/api/v3/features").json()["media"]
    assert "4.0 GB" in media["ram_warning"] and "8 GB" in media["ram_warning"]


def test_a_big_machine_has_no_warning(ctx, monkeypatch):
    client, _root, env, _calls = ctx
    monkeypatch.setattr(env, "precheck", lambda: {"disk_ok": True, "free_bytes": 10 * 1024 ** 3,
                                                   "ram_bytes": 16 * 1024 ** 3, "ram_warn": False})
    assert client.get("/api/v3/features").json()["media"].get("ram_warning", "") == ""
