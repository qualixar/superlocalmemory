# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Power lost in the middle of a restore: the store survives, and the next start finishes.

A real process is killed (``os._exit``, no cleanup, no finally blocks) after the
third step of writing a 50 MB copy into the live store -- the closest a test
gets to pulling the plug.
"""

from __future__ import annotations

import json
import os
import sqlite3
import subprocess
import sys
from contextlib import closing
from pathlib import Path

import pytest

from superlocalmemory.storage import backup
from superlocalmemory.storage import upgrade_restore as ur

from ._upgrade_store import (
    add_memory,
    current_store,
    delete_fact,
    fact_ids,
    integrity,
    logical_digest,
)

_REPO = Path(__file__).resolve().parents[2]
_KILL = """
import os, sys
from pathlib import Path
from superlocalmemory.storage import backup, upgrade_restore as ur
root, memory_db, learning_db = (Path(p) for p in sys.argv[1:4])
kill_at = int(sys.argv[4])
backup._RESTORE_PAGES_PER_STEP = 8
steps = {"n": 0}
def pull_the_plug(remaining, total):
    steps["n"] += 1
    if steps["n"] == kill_at:
        os._exit(9)
backup._restore_progress = pull_the_plug
ur.perform_pending_restore(root, memory_db, learning_db)
os._exit(0)
"""


def _big_store(root: Path) -> tuple[Path, Path, ur.RestorePoint, set[str]]:
    learning_db, memory_db = current_store(root)
    blob = "x" * 24_000
    with closing(sqlite3.connect(memory_db)) as conn:
        conn.executemany(
            "INSERT INTO memories (memory_id, profile_id, content) VALUES (?, 'default', ?)",
            [(f"m{i}", f"{i} {blob}") for i in range(2_000)])
        conn.executemany(
            "INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
            "VALUES (?, ?, 'default', ?)",
            [(f"f{i}", f"m{i}", f"fact {i}") for i in range(2_000)])
        conn.commit()
    assert memory_db.stat().st_size > 45 * 1024 * 1024
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=root / "pre-migration-snapshots")
    [point] = ur.list_restore_points(root)
    snapshot_facts = fact_ids(memory_db)
    add_memory(memory_db, "later", ["Written after the copy was taken"], fact_ids=["later-f"])
    delete_fact(memory_db, "f7")
    with closing(sqlite3.connect(memory_db)) as conn:       # a change only a restore undoes
        conn.execute("UPDATE atomic_facts SET content='edited' WHERE fact_id='f1'")
        conn.commit()
    ur.request_restore(point.point_id, requested_by="t", data_root=root, memory_db=memory_db)
    return learning_db, memory_db, point, snapshot_facts


def _pull_the_plug(root: Path, memory_db: Path, learning_db: Path,
                   kill_at: int) -> tuple[int, str]:
    env = {**os.environ, "PYTHONPATH": os.pathsep.join(
        [str(_REPO / "src"), os.environ.get("PYTHONPATH", "")])}
    done = subprocess.run(
        [sys.executable, "-c", _KILL, str(root), str(memory_db), str(learning_db), str(kill_at)],
        cwd=str(_REPO), env=env, capture_output=True, text=True, timeout=300)
    return done.returncode, done.stderr


#: The 3rd step (early), and step 1,200 (~38 MB in: past SQLite's page cache, so
#: uncommitted pages have already spilled into the live store's -wal).
@pytest.fixture(params=[3, 1_200], ids=["early", "late"])
def crashed(tmp_path, request):
    learning_db, memory_db, point, snapshot_facts = _big_store(tmp_path)
    before = logical_digest(memory_db)
    code, err = _pull_the_plug(tmp_path, memory_db, learning_db, request.param)
    assert code == 9, f"the restore was not interrupted: {code} {err[-2000:]}"
    return tmp_path, learning_db, memory_db, before, snapshot_facts


def test_kill_during_backup_step_leaves_target_intact(crashed) -> None:
    root, _learning_db, memory_db, before, _snap = crashed
    assert integrity(memory_db) == "ok"
    assert logical_digest(memory_db) == before, "a half-written restore reached the store"
    intent = json.loads((root / "restore-intent.json").read_text(encoding="utf-8"))
    assert intent["stage"] == "writing" and Path(intent["final_delta_dir"]).is_dir()
    assert any((root / "pre-restore").glob("memory-*-before-restore.db"))


def test_rerun_after_crash_completes(crashed) -> None:
    root, learning_db, memory_db, _before, snapshot_facts = crashed

    outcome = ur.perform_pending_restore(root, memory_db, learning_db)

    assert outcome.status == "restored", outcome.message
    assert integrity(memory_db) == "ok"
    assert fact_ids(memory_db) == snapshot_facts - {"f7"}
    with closing(sqlite3.connect(memory_db)) as conn:
        assert conn.execute("SELECT content FROM atomic_facts WHERE fact_id='f1'"
                            ).fetchone()[0] == "fact 1"
    # The export taken before the interrupted write is the one used afterwards:
    # the memory written after the copy is still on its way back.
    added = [json.loads(line)["memory_id"] for line in
             (Path(outcome.delta_dir) / "memories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert added == ["later"] and outcome.reimport_pending
    assert not (root / "restore-intent.json").exists()


def test_a_crash_after_the_write_keeps_the_newer_memories(tmp_path, monkeypatch) -> None:
    """Power lost after the restore committed but before it was recorded: the
    next start must not re-compute "what changed" from the RESTORED store."""
    from superlocalmemory.storage import _restore_boot

    learning_db, memory_db = current_store(tmp_path)
    add_memory(memory_db, "m1", ["Deploys go out on Tuesdays"], fact_ids=["f1"])
    backup._pre_migration_backup(learning_db, memory_db,
                                 backups_root=tmp_path / "pre-migration-snapshots")
    [point] = ur.list_restore_points(tmp_path)
    add_memory(memory_db, "later", ["Written after the copy was taken"], fact_ids=["later-f"])
    ur.request_restore(point.point_id, requested_by="t", data_root=tmp_path,
                       memory_db=memory_db)

    def power_cut(*_a, **_k):
        raise SystemExit("power cut")
    with monkeypatch.context() as m:
        m.setattr(_restore_boot, "_write_outcome", power_cut)
        with pytest.raises(SystemExit):
            ur.perform_pending_restore(tmp_path, memory_db, learning_db)
    assert "later-f" not in fact_ids(memory_db), "the write had committed"

    outcome = ur.perform_pending_restore(tmp_path, memory_db, learning_db)

    assert outcome.status == "restored"
    added = [json.loads(line)["memory_id"] for line in
             (Path(outcome.delta_dir) / "memories.jsonl").read_text(encoding="utf-8").splitlines()]
    assert added == ["later"] and outcome.reimport_pending
