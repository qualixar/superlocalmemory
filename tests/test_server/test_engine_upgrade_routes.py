# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""``/api/v3/embedding/reindex/upgrade``: the plan (read) and the start (manage)."""

from __future__ import annotations

from types import SimpleNamespace

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from superlocalmemory.core import engine_upgrade as eu
from superlocalmemory.core.config import EmbeddingConfig
from superlocalmemory.core.embedding_reindex import NoChange, Refused
from superlocalmemory.server.routes import embedding_reindex as routes
from superlocalmemory.storage.embedding_reindex_jobs import JobConflict

URL = "/api/v3/embedding/reindex/upgrade"
NOMIC = EmbeddingConfig(provider="sentence-transformers", model_name="nomic-ai/nomic-embed-text-v1.5",
                        dimension=768)
EG2 = EmbeddingConfig(provider="slm-media", model_name="google/embeddinggemma-2", dimension=768)
JOB = {"job_id": 5, "kind": "switch", "state": "queued", "from": "nomic::768",
       "to": "google/embeddinggemma-2::768", "done": 0, "total": 40}


class _Runner:
    def __init__(self):
        self.switched = []
        self.error = None
        self.db_path = "unused"

    def request_switch(self, target, **kw):
        if self.error:
            raise self.error
        self.switched.append((target, kw))
        return dict(JOB)


@pytest.fixture()
def ctx(monkeypatch):
    state = {"live": NOMIC, "media": (True, "ready"), "job": False, "count": 40, "runner": _Runner()}
    monkeypatch.setattr(eu, "media_state", lambda data_root=None, env=None: state["media"])
    monkeypatch.setattr(eu, "count_memories", lambda db_path: state["count"])
    monkeypatch.setattr(eu, "job_is_running", lambda db_path: state["job"])
    monkeypatch.setattr(routes, "_live_config", lambda request: state["live"])
    app = FastAPI()
    app.include_router(routes.router)
    app.state.embedding_reindex = state["runner"]
    state["app"] = app
    return TestClient(app), state


def test_get_plan_is_available_and_complete(ctx):
    client, _state = ctx
    r = client.get(URL)
    assert r.status_code == 200
    body = r.json()
    assert body["available"] is True and body["memories"] == 40 and body["to"]["provider"] == "slm-media"
    assert {"reason", "ram_mb", "disk_mb", "minutes", "minutes_label", "already", "explain"} <= set(body)


def test_get_plan_says_why_when_images_are_off(ctx):
    client, state = ctx
    state["media"] = (False, "not_installed")
    body = client.get(URL).json()
    assert body["available"] is False and "slm media enable" in body["reason"]


def test_get_plan_without_the_daemon_runner_is_409(ctx):
    client, state = ctx
    state["app"].state.embedding_reindex = None
    r = client.get(URL)
    assert r.status_code == 409 and r.json()["error"] == "daemon_required"


def test_post_starts_the_switch_to_the_managed_model_and_answers_202(ctx, monkeypatch):
    client, state = ctx
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", lambda request: {})
    r = client.post(URL, json={})
    assert r.status_code == 202
    body = r.json()
    assert body["accepted"] is True and body["label"] == "upgrade" and body["job"]["job_id"] == 5
    (target, kw), = state["runner"].switched
    assert (target.provider, target.model_name, target.dimension) == ("slm-media", "google/embeddinggemma-2", 768)
    assert kw.get("kind", "switch") == "switch", "the runner only knows switch and rollback"
    assert "re-index" not in body["detail"].lower() and "roll back" in body["detail"].lower()


@pytest.mark.parametrize("mutate,needle", [
    (lambda s: s.update(media=(False, "not_installed")), "slm media enable"),
    (lambda s: s.update(media=(True, "installing")), "still being set up"),
    (lambda s: s.update(job=True), "slm embedder cancel"),
    (lambda s: s.update(live=EG2), "already"),
])
def test_post_refuses_with_409_and_the_plans_reason(ctx, monkeypatch, mutate, needle):
    client, state = ctx
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", lambda request: {})
    mutate(state)
    r = client.post(URL, json={})
    assert r.status_code == 409
    assert needle in r.json()["detail"] and r.json()["error"] == "upgrade_unavailable"
    assert state["runner"].switched == []


def test_post_needs_manage_permission(ctx, monkeypatch):
    client, state = ctx

    def deny(request):
        raise HTTPException(403, detail="manage permission required")

    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", deny)
    assert client.post(URL, json={}).status_code == 403
    assert state["runner"].switched == []


def test_get_does_not_need_manage(ctx, monkeypatch):
    client, _state = ctx

    def deny(request):
        raise HTTPException(403)

    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", deny)
    assert client.get(URL).status_code == 200


def test_a_race_with_another_job_is_the_usual_conflict(ctx, monkeypatch):
    client, state = ctx
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", lambda request: {})
    state["runner"].error = JobConflict({"job_id": 9, "state": "running",
                                         "from_signature": "a::1", "to_signature": "b::2"})
    r = client.post(URL, json={})
    assert r.status_code == 409 and r.json()["error"] == "reindex_running"


@pytest.mark.parametrize("error", [NoChange("same"), Refused("no space recorded")])
def test_a_runner_refusal_is_409_with_its_words(ctx, monkeypatch, error):
    client, state = ctx
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", lambda request: {})
    state["runner"].error = error
    r = client.post(URL, json={})
    assert r.status_code == 409 and str(error) in r.json()["detail"]


def test_post_without_the_daemon_runner_is_409(ctx, monkeypatch):
    client, state = ctx
    monkeypatch.setattr("superlocalmemory.server.rbac_enforce.require_manage", lambda request: {})
    state["app"].state.embedding_reindex = None
    assert client.post(URL, json={}).status_code == 409


def test_the_upgrade_path_does_not_shadow_the_other_routes():
    paths = {r.path for r in routes.router.routes}
    assert {"/api/v3/embedding/reindex/rollback", "/api/v3/embedding/reindex/forget-previous", URL} <= paths
