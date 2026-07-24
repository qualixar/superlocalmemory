# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3

"""Runtime behaviour config route (v3.8.2 UX-1).

GET/PUT /api/v3/runtime/config exposes the user-facing runtime knobs
(recall depth, reranker, memory injection) with live-apply + disk
persistence and fail-fast validation.
"""
from __future__ import annotations

import json

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import superlocalmemory.server.routes.v3_api as v3
from superlocalmemory.core.config import SLMConfig
from superlocalmemory.storage.models import Mode


@pytest.fixture()
def client(tmp_path, monkeypatch):
    # Bypass RBAC for the unit test; auth is exercised live elsewhere.
    monkeypatch.setattr(v3, "_require_manage", lambda request: None)
    cfg = SLMConfig.for_mode(Mode.A)
    cfg.base_dir = tmp_path
    cfg.db_path = tmp_path / "memory.db"
    app = FastAPI()
    app.state.config = cfg
    app.state.engine = None
    app.include_router(v3.router)
    return TestClient(app)


def test_get_returns_defaults(client):
    d = client.get("/api/v3/runtime/config").json()
    assert d["success"] is True
    assert set(d["config"]) == {"retrieval", "injection"}
    assert d["config"]["retrieval"]["top_k"] == 20
    assert d["config"]["retrieval"]["use_cross_encoder"] is True
    assert d["config"]["injection"]["core_block_max_facts"] == 5


def test_put_applies_live_and_persists(client, tmp_path):
    r = client.put("/api/v3/runtime/config", json={
        "retrieval": {"top_k": 42, "use_cross_encoder": False},
        "injection": {"enabled": False, "core_block_max_facts": 9},
    })
    assert r.status_code == 200
    body = r.json()
    assert body["success"] is True
    assert body["config"]["retrieval"]["top_k"] == 42
    assert body["config"]["retrieval"]["use_cross_encoder"] is False
    assert body["config"]["injection"]["enabled"] is False
    assert body["config"]["injection"]["core_block_max_facts"] == 9

    # Live: a subsequent GET reflects the change.
    live = client.get("/api/v3/runtime/config").json()["config"]
    assert live["retrieval"]["top_k"] == 42
    assert live["injection"]["enabled"] is False

    # Durable: persisted to config.json so it survives a restart.
    saved = json.loads((tmp_path / "config.json").read_text())
    assert saved["retrieval"]["top_k"] == 42
    assert saved["retrieval"]["use_cross_encoder"] is False
    assert saved["injection"]["enabled"] is False
    assert saved["injection"]["core_block_max_facts"] == 9


def test_partial_update_leaves_others_untouched(client):
    client.put("/api/v3/runtime/config", json={"retrieval": {"top_k": 33}})
    live = client.get("/api/v3/runtime/config").json()["config"]
    assert live["retrieval"]["top_k"] == 33
    # untouched
    assert live["retrieval"]["use_cross_encoder"] is True
    assert live["injection"]["enabled"] is True


def test_rejects_bad_bool_type(client):
    r = client.put("/api/v3/runtime/config", json={"injection": {"enabled": "yes"}})
    assert r.status_code == 400
    assert "true or false" in r.json()["error"]


def test_rejects_bool_as_int(client):
    # bool is an int subclass — must not slip through the int check.
    r = client.put("/api/v3/runtime/config", json={"retrieval": {"top_k": True}})
    assert r.status_code == 400


def test_rejects_out_of_range(client):
    assert client.put("/api/v3/runtime/config",
                      json={"retrieval": {"top_k": 0}}).status_code == 400
    assert client.put("/api/v3/runtime/config",
                      json={"retrieval": {"top_k": 999}}).status_code == 400
    assert client.put("/api/v3/runtime/config",
                      json={"injection": {"core_block_max_facts": 99}}).status_code == 400


def test_rejects_no_known_fields(client):
    r = client.put("/api/v3/runtime/config", json={"unknown": {"x": 1}})
    assert r.status_code == 400


def test_bad_value_does_not_apply(client):
    client.put("/api/v3/runtime/config", json={"retrieval": {"top_k": 0}})
    # top_k must remain the default — a rejected request applies nothing.
    assert client.get("/api/v3/runtime/config").json()["config"]["retrieval"]["top_k"] == 20
