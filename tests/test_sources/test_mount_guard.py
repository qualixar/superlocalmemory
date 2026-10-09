"""An empty or different mount at the folder's path is treated as offline, not as everything deleted."""

from __future__ import annotations

import json
import os

from superlocalmemory.sources.store import SourceStore


def source_row(env, sid):
    media = env.store()
    try:
        return SourceStore(media).get_source(sid)
    finally:
        media.close()


def test_confirm_records_the_device_of_the_folder(env):
    sid = env.add_and_confirm()
    stats = json.loads(source_row(env, sid)["last_scan_stats_json"])
    assert stats["root_dev"] == os.stat(env.root).st_dev
    env.scan(sid)
    assert json.loads(source_row(env, sid)["last_scan_stats_json"])["root_dev"] == os.stat(env.root).st_dev


def test_an_empty_walk_of_a_source_with_files_is_offline_and_tombstones_nothing(env):
    env.write("a.md", "note a")
    env.write("b.md", "note b")
    sid = env.add_and_confirm()
    env.scan(sid)
    for p in list(env.root.iterdir()):
        p.unlink()
    stats = env.scan(sid)
    assert stats.offline is True
    assert {r["state"] for r in env.files(sid).values()} == {"indexed"}
    assert source_row(env, sid)["state"] == "offline" and env.runtime.archived == []


def test_the_files_come_back_and_the_source_resumes(env):
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "a.md").unlink()
    env.scan(sid)
    env.write("a.md", "note a")
    env.scan(sid)
    assert source_row(env, sid)["state"] == "active" and env.files(sid)["a.md"]["state"] == "indexed"


def test_a_different_device_at_the_path_is_offline(env):
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "a.md").unlink()
    env.write("other.md", "a different disk")
    media = env.store()
    with media._write() as conn:
        conn.execute("UPDATE sources SET last_scan_stats_json = ? WHERE source_id = ?",
                     (json.dumps({"root_dev": os.stat(env.root).st_dev + 1}), sid))
    media.close()
    stats = env.scan(sid)
    assert stats.offline is True and "other.md" not in env.files(sid)
    assert env.files(sid)["a.md"]["state"] == "indexed" and env.runtime.archived == []
    assert json.loads(source_row(env, sid)["last_scan_stats_json"])["root_dev"] == os.stat(env.root).st_dev + 1


def test_a_folder_that_became_unsafe_is_offline_and_nothing_is_saved(env):
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / ".ssh").mkdir()
    env.write("b.md", "note b")
    stats = env.scan(sid)
    assert stats.offline is True and stats.offline_reason == "holds_credentials"
    assert env.runtime.contents() == ["note a"] and "b.md" not in env.files(sid)
    row = source_row(env, sid)
    assert row["state"] == "offline" and json.loads(row["last_scan_stats_json"])["offline_reason"] == "holds_credentials"
    (env.root / ".ssh").rmdir()
    env.scan(sid)
    assert source_row(env, sid)["state"] == "active" and env.files(sid)["b.md"]["state"] == "indexed"
