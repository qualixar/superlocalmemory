# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The prepare-to-go-back message says what really happens (L1-11).

Chosen behaviour: restarting THIS version keeps the preparation (a daemon that
restarts by itself between "prepare" and "install the older version" must not
undo it, or the older version would refuse the store). It is undone by Cancel,
or by any other version starting. The message says exactly that.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import superlocalmemory.storage._schema_version as sv
from superlocalmemory.cli import upgrade_cmd
from superlocalmemory.storage import _snapshot_manifest as sm
from superlocalmemory.storage import migration_runner as mr
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import current_store


@pytest.fixture()
def store(tmp_path, monkeypatch):
    learning_db, memory_db = current_store(tmp_path)
    (tmp_path / ".last_version").write_text("4.1.19", encoding="utf-8")
    monkeypatch.setattr(sm, "package_version", lambda: "4.1.19")
    monkeypatch.setattr(sv, "_detect_all_installs", lambda: [])
    return tmp_path, learning_db, memory_db


def _next_start(learning_db: Path, memory_db: Path) -> None:
    mr.apply_all(learning_db, memory_db)
    mr.apply_deferred(learning_db, memory_db)


def test_message_matches_what_a_restart_does(store) -> None:
    root, learning_db, memory_db = store
    report = ur.prepare_downgrade(requested_by="t", data_root=root, memory_db=memory_db,
                                  learning_db=learning_db)

    assert "undoes the preparation" not in report.message
    assert "keeps" in report.message and "Cancel" in report.message
    for _ in range(2):                                     # this version restarts, twice
        _next_start(learning_db, memory_db)
        assert sv.read_schema_version(memory_db) == 51, "the preparation is kept"
    assert ur.cancel_downgrade(root) is True
    _next_start(learning_db, memory_db)
    assert sv.read_schema_version(memory_db) == sv.SUPPORTED_SCHEMA_VERSION, "Cancel undoes it at the next start"


def test_cli_cancel_message_is_true(store, capsys) -> None:
    from argparse import Namespace

    root, learning_db, memory_db = store
    ur.prepare_downgrade(requested_by="t", data_root=root, memory_db=memory_db,
                         learning_db=learning_db)
    upgrade_cmd.cmd_db_prepare_downgrade(Namespace(data_root=root, json=False, yes=True,
                                                   cancel=True))
    out = capsys.readouterr().out
    assert "next start" in out
    _next_start(learning_db, memory_db)
    assert sv.read_schema_version(memory_db) == sv.SUPPORTED_SCHEMA_VERSION


@pytest.mark.parametrize("current, expected", [
    ("4.1.20", "4.1.19"), ("4.1.19", "4.1.18"), ("4.1.25", "4.1.24"),
    ("not-a-version", "4.1.18"),
    # 4.2.30 follows 4.2.19 (4.2.20-4.2.29 were never published): never name "4.2.29".
    ("4.2.30", "4.2.19"),
])
def test_the_target_release_is_the_one_before_this(current, expected) -> None:
    from superlocalmemory.storage._downgrade import downgrade_target_release

    assert downgrade_target_release(51, current) == expected


def test_message_names_the_version_and_the_exact_install_command(store, monkeypatch) -> None:
    """Audit 4.1.20 L6: "install the older version" named neither."""
    root, learning_db, memory_db = store
    monkeypatch.setattr(sm, "package_version", lambda: "4.1.20")
    report = ur.prepare_downgrade(requested_by="t", data_root=root, memory_db=memory_db,
                                  learning_db=learning_db)

    assert "Ready to go back to 4.1.19" in report.message
    assert 'pipx install --force "superlocalmemory==4.1.19"' in report.message
    assert "npm install -g superlocalmemory@4.1.19" in report.message
    assert "any version from 4.1.18 on" in report.message
    data = report.as_dict()
    assert data["target_version"] == "4.1.19"
    assert 'pip install "superlocalmemory==4.1.19"' in data["install_commands"]
