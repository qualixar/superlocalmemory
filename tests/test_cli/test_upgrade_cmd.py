# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""`slm db restore-points | restore | prepare-downgrade`: the same flow as the dashboard."""

from __future__ import annotations

import io
import json
import sqlite3
from argparse import Namespace
from contextlib import closing

import pytest

from superlocalmemory.cli import upgrade_cmd
from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur
from tests.test_storage._upgrade_store import add_memory, current_store, fact_ids


@pytest.fixture()
def store(tmp_path, monkeypatch):
    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    add_memory(memory_db, "m2", ["Written after the copy"], fact_ids=["f2"])
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.execute("UPDATE atomic_facts SET content='edited' WHERE fact_id='f1'")
        conn.commit()
    calls: list[str] = []
    running = {"now": True}

    def stop() -> bool:
        calls.append("stop")
        running["now"] = False
        return True

    def start() -> bool:
        calls.append("start")
        running["now"] = True
        return True

    monkeypatch.setattr(upgrade_cmd, "_daemon_running", lambda: running["now"])
    monkeypatch.setattr(upgrade_cmd, "_stop_daemon", stop)
    monkeypatch.setattr(upgrade_cmd, "_start_daemon", start)
    monkeypatch.setattr(upgrade_cmd, "_load_config", lambda: None)
    monkeypatch.setattr("sys.stdin", io.StringIO(""))
    return tmp_path, memory_db, calls


def _args(root, **kw) -> Namespace:
    base = {"data_root": root, "json": False, "yes": False, "cancel": False,
            "no_reimport": False, "point_id": None}
    return Namespace(**{**base, **kw})


def test_restore_points_json_lists_points(store, capsys) -> None:
    root, _memory_db, _calls = store
    assert upgrade_cmd.cmd_db_restore_points(_args(root, json=True)) == 0
    listed = json.loads(capsys.readouterr().out)["restore_points"]
    assert len(listed) == 1 and listed[0]["facts"] == 1


def test_restore_needs_yes_when_not_interactive(store, capsys) -> None:
    root, _memory_db, calls = store
    point = ur.list_restore_points(root)[0].point_id
    assert upgrade_cmd.cmd_db_restore(_args(root, point_id=point)) == 2
    assert "--yes" in capsys.readouterr().out
    assert not (root / "restore-intent.json").exists() and calls == []


def test_restore_inline_stops_restores_and_starts(store) -> None:
    root, memory_db, calls = store
    point = ur.list_restore_points(root)[0].point_id

    assert upgrade_cmd.cmd_db_restore(_args(root, point_id=point, yes=True)) == 0

    assert calls == ["stop", "start"]
    assert fact_ids(memory_db) == {"f1"}
    with closing(sqlite3.connect(memory_db)) as conn:
        assert conn.execute("SELECT content FROM atomic_facts").fetchone()[0] == \
            "Deploys go out on Tuesdays"
    outcome = json.loads((root / "restore-outcome.json").read_text(encoding="utf-8"))
    assert outcome["status"] == "restored" and outcome["reimport_pending"] is True


def test_restore_cancel(store) -> None:
    root, memory_db, _calls = store
    point = ur.list_restore_points(root)[0].point_id
    ur.request_restore(point, requested_by="t", data_root=root, memory_db=memory_db)
    assert upgrade_cmd.cmd_db_restore(_args(root, cancel=True)) == 0
    assert not (root / "restore-intent.json").exists()


def test_prepare_downgrade_needs_yes_then_prepares(store, capsys) -> None:
    root, _memory_db, _calls = store
    assert upgrade_cmd.cmd_db_prepare_downgrade(_args(root)) == 2
    assert not (root / ".downgrade-prepared").exists()
    assert upgrade_cmd.cmd_db_prepare_downgrade(_args(root, yes=True, json=True)) == 0
    assert json.loads(capsys.readouterr().out.splitlines()[-1])["prepared"] is True
    assert upgrade_cmd.cmd_db_prepare_downgrade(_args(root, cancel=True)) == 0
    assert not (root / ".downgrade-prepared").exists()
