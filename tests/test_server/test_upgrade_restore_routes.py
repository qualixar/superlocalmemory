# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Updates & restore from the dashboard: readable by anyone who can see it,
changeable only with the product's credential, MANAGE permission and the typed word."""

from __future__ import annotations

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from superlocalmemory.server.routes import upgrade_restore as routes
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from tests.test_storage._upgrade_store import add_memory, current_store


@pytest.fixture()
def env(tmp_path, monkeypatch):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    spawned: list[int] = []
    monkeypatch.setattr(routes, "_spawn_restart", lambda: spawned.append(1) or True)
    app = FastAPI()
    routes.register(app, data_root=tmp_path)
    point = ur.list_restore_points(tmp_path)[0]
    return TestClient(app), tmp_path, point, spawned


def _allow(monkeypatch, *, credential=True, manage=True) -> None:
    def deny(_request):
        raise HTTPException(403, detail="denied")
    monkeypatch.setattr(routes, "_require_credential",
                        (lambda _r: None) if credential else deny)
    monkeypatch.setattr(routes, "_require_manage",
                        (lambda _r: {"username": "owner"}) if manage else deny)


def test_restore_needs_manage_and_the_word(env, monkeypatch) -> None:
    client, root, point, spawned = env
    body = {"restore_point_id": point.point_id, "confirm": "RESTORE", "reimport": True}

    assert client.post("/api/upgrade/restore", json=body).status_code == 403   # no credential
    _allow(monkeypatch, manage=False)
    assert client.post("/api/upgrade/restore", json=body).status_code == 403   # no MANAGE
    _allow(monkeypatch)
    for word in ("", "restore", "RESTORE ", "yes"):
        r = client.post("/api/upgrade/restore", json={**body, "confirm": word})
        assert r.status_code == 400 and "RESTORE" in r.json()["error"]
    assert not (root / "restore-intent.json").exists() and spawned == []

    r = client.post("/api/upgrade/restore", json=body)

    assert r.status_code == 200 and r.json()["status"] == "restarting"
    assert (root / "restore-intent.json").exists() and spawned == [1]


def test_cancel_before_restart_removes_intent(env, monkeypatch) -> None:
    client, root, point, _spawned = env
    _allow(monkeypatch)
    client.post("/api/upgrade/restore",
                json={"restore_point_id": point.point_id, "confirm": "RESTORE"})
    assert client.get("/api/upgrade/status").json()["pending_restore"]["point_id"] == point.point_id

    r = client.post("/api/upgrade/restore/cancel")

    assert r.status_code == 200 and r.json()["cancelled"] is True
    assert not (root / "restore-intent.json").exists()
    assert client.get("/api/upgrade/status").json()["pending_restore"] is None


def test_reads_list_and_preview_without_changing_anything(env, monkeypatch) -> None:
    client, root, point, _spawned = env
    listed = client.get("/api/upgrade/restore-points").json()["restore_points"]
    assert [p["point_id"] for p in listed] == [point.point_id]
    _allow(monkeypatch)
    preview = client.post("/api/upgrade/restore/preview",
                          json={"restore_point_id": point.point_id}).json()
    assert preview["verified"] is True and preview["facts_in_snapshot"] == 1
    bad = client.post("/api/upgrade/restore/preview", json={"restore_point_id": "../x"})
    assert bad.status_code == 404
    assert not (root / "restore-intent.json").exists()


def test_prepare_downgrade_needs_its_own_word(env, monkeypatch) -> None:
    client, root, _point, _spawned = env
    _allow(monkeypatch)
    r = client.post("/api/upgrade/prepare-downgrade", json={"confirm": "RESTORE"})
    assert r.status_code == 400 and not (root / ".downgrade-prepared").exists()
    r = client.post("/api/upgrade/prepare-downgrade", json={"confirm": "DOWNGRADE"})
    assert r.status_code == 200 and r.json()["prepared"] is True
    assert client.get("/api/upgrade/status").json()["downgrade"]["target_schema"] == 51
    r = client.post("/api/upgrade/prepare-downgrade/cancel")
    assert r.json()["cancelled"] is True and not (root / ".downgrade-prepared").exists()
