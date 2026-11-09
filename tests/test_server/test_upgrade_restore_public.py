# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Upgrade status and restore messages carry plain words only (L3-21).

In company or LAN mode a viewer of the dashboard must not learn the OS account
name, the data-directory layout or raw exception text. Paths are shown
relative to the data directory; details stay in the server log.
"""

from __future__ import annotations

import errno
import json
from argparse import Namespace
from pathlib import Path

from fastapi import FastAPI
from fastapi.testclient import TestClient

from superlocalmemory.cli import upgrade_cmd
from superlocalmemory.server.routes import upgrade_restore as routes
from superlocalmemory.storage import _restore_boot as rb
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from superlocalmemory.storage._durable_json import write_json_atomic
from tests.test_storage._upgrade_store import add_memory, current_store


def _client(root: Path, monkeypatch) -> TestClient:
    monkeypatch.setattr(routes, "_require_credential", lambda _r: None)
    monkeypatch.setattr(routes, "_require_manage", lambda _r: {"username": "owner"})
    monkeypatch.setattr(routes, "_spawn_restart", lambda: True)
    app = FastAPI()
    routes.register(app, data_root=root)
    return TestClient(app)


def _store(tmp_path: Path):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    return learning_db, memory_db, point


def _leaks(text: str, root: Path) -> list[str]:
    secrets = {str(root), str(root.resolve()), str(Path.home())}
    return [s for s in secrets if s and s in text]


def test_status_and_points_show_no_absolute_path(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, point = _store(tmp_path)
    add_memory(memory_db, "later", ["Written after the copy"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)
    assert ur.perform_pending_restore(tmp_path, memory_db, learning_db).status == "restored"
    client = _client(tmp_path, monkeypatch)

    status = client.get("/api/upgrade/status").json()
    points = client.get("/api/upgrade/restore-points").json()

    assert not _leaks(json.dumps(status), tmp_path), status
    assert not _leaks(json.dumps(points), tmp_path), points
    assert status["last_restore"]["safety_copies"][0].startswith("pre-restore/memory-")
    assert status["last_restore"]["delta_dir"].startswith("restore-delta/")
    assert points["restore_points"][0]["memory_snapshot"].startswith(
        "pre-migration-snapshots/memory-")


def test_a_failed_restore_says_plain_words(tmp_path, monkeypatch) -> None:
    learning_db, memory_db, point = _store(tmp_path)
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)

    def disk_full(*_a, **_k):
        raise OSError(errno.ENOSPC, f"No space left on device: {tmp_path / 'x.db'}")
    monkeypatch.setattr(rb, "_copy_quiescent", disk_full)

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "failed"
    assert not _leaks(outcome.message, tmp_path)
    assert "OSError" not in outcome.message and "No space left" not in outcome.message
    assert "were not changed" in outcome.message


def test_an_older_outcome_with_paths_is_scrubbed(tmp_path, monkeypatch) -> None:
    _store(tmp_path)
    write_json_atomic(tmp_path / "restore-outcome.json", {
        "status": "failed", "point_id": "p",
        "message": f"The copy cannot be used (snapshot is missing: {tmp_path}/pre-migration-"
                   f"snapshots/memory-x.db). A copy is in {Path.home()}/elsewhere/pre-restore.",
        "safety_copies": [str(Path.home() / "outside" / "memory-y.db")]})

    status = _client(tmp_path, monkeypatch).get("/api/upgrade/status").json()

    assert not _leaks(json.dumps(status), tmp_path), status
    assert status["last_restore"]["safety_copies"] == ["memory-y.db"]


def test_cli_json_paths_are_relative_to_the_data_directory(tmp_path, capsys) -> None:
    _store(tmp_path)

    upgrade_cmd.cmd_db_restore_points(Namespace(data_root=tmp_path, json=True))

    listed = json.loads(capsys.readouterr().out)["restore_points"][0]
    assert listed["memory_snapshot"].startswith("pre-migration-snapshots/memory-")
    assert not _leaks(json.dumps(listed), tmp_path)
