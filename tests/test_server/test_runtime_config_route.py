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
    saved = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
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


def test_put_keeps_what_another_writer_changed_since_startup(client, tmp_path):
    # 4.2.14 (Codex review): the daemon persisted its start-up copy of the
    # whole config, so a dashboard change silently undid an earlier
    # `slm config set` (injection, cost settings, anything else on disk).
    path = tmp_path / "config.json"
    path.write_text(json.dumps({
        "mode": "b", "llm": {"provider": "ollama", "model": "gemma3:4b"},
        "injection": {"enabled": False, "core_block_enabled": True},
        "temporal_validator": {"max_llm_checks": 0},
        "encoding": {"context_generation_llm": False},
    }), encoding="utf-8")
    r = client.put("/api/v3/runtime/config", json={"retrieval": {"top_k": 33}})
    assert r.status_code == 200
    saved = json.loads(path.read_text(encoding="utf-8"))
    assert saved["retrieval"]["top_k"] == 33
    assert saved["mode"] == "b" and saved["llm"]["model"] == "gemma3:4b"
    assert saved["injection"] == {"enabled": False, "core_block_enabled": True}
    assert saved["temporal_validator"] == {"max_llm_checks": 0}
    assert saved["encoding"] == {"context_generation_llm": False}


def test_first_run_full_save_holds_the_config_lock(tmp_path):
    # 4.2.14 (Codex review): with no config.json yet the route fell back to an
    # unlocked full save, so a writer creating the file between the existence
    # check and the save was overwritten. Both now run under the config lock.
    import threading
    from types import SimpleNamespace

    from superlocalmemory.infra.config_lock import config_file_lock

    path = tmp_path / "config.json"
    probes: list[bool] = []

    def full_save(*, mode_change: bool) -> None:
        entered = threading.Event()

        def other_writer() -> None:
            with config_file_lock(path):
                entered.set()

        threading.Thread(target=other_writer, daemon=True).start()
        probes.append(entered.wait(0.3))
        path.write_text("{}\n", encoding="utf-8")

    live = SimpleNamespace(base_dir=tmp_path, save=full_save)
    v3._persist_runtime_updates(live, {"retrieval": {"top_k": 33}})
    assert probes == [False], "another writer got the lock during the first-run save"


def test_a_later_full_save_keeps_a_cli_change_made_after_the_dashboard(client, tmp_path):
    # Dashboard sets injection off; the CLI turns it back on; a later full save
    # of the daemon's object must not put the dashboard's older value back.
    from superlocalmemory.server.config_file import update_config

    path = tmp_path / "config.json"
    path.write_text(json.dumps({"mode": "a", "injection": {"enabled": True}}), encoding="utf-8")
    live = client.app.state.config
    live._record_sync(path, json.loads(path.read_text(encoding="utf-8")), from_load=True)
    assert client.put("/api/v3/runtime/config",
                      json={"injection": {"enabled": False}}).status_code == 200
    update_config(path, lambda d: d["injection"].update(enabled=True))   # slm config set
    live.save(mode_change=False)                                         # any later full save
    assert json.loads(path.read_text(encoding="utf-8"))["injection"]["enabled"] is True
