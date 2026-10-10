"""Forgetting the files of a folder that was emptied on purpose."""

from __future__ import annotations

import json
import os

import pytest

from superlocalmemory import sources
from superlocalmemory.sources.store import SourceStore


def row(env, sid):
    media = env.store()
    try:
        return SourceStore(media).get_source(sid)
    finally:
        media.close()


def emptied(env, names=("a.md", "b.md")):
    for n in names:
        env.write(n, f"note {n}")
    sid = env.add_and_confirm()
    env.scan(sid)
    for n in names:
        (env.root / n).unlink()
    assert env.scan(sid).offline_reason == "empty_folder"
    return sid


def reason(env, sid):
    return json.loads(row(env, sid)["last_scan_stats_json"]).get("offline_reason")


def states(env, sid):
    return {r["state"] for r in env.files(sid).values()}


def test_forget_hides_the_files_and_the_folder_stays_active(env):
    sid = emptied(env)
    out = sources.forget_empty(sid)
    assert out == {"source_id": sid, "forgotten": 2, "state": "active"}
    assert row(env, sid)["state"] == "active" and states(env, sid) == {"tombstoned"}
    assert len(env.runtime.archived) == 2 and env.erased == []
    stats = env.scan(sid)
    assert stats.offline is False and row(env, sid)["state"] == "active"
    assert states(env, sid) == {"tombstoned"} and len(env.runtime.archived) == 2


def test_already_tombstoned_rows_are_not_counted(env):
    env.write("a.md", "x")
    env.write("b.md", "y")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "a.md").unlink()
    env.scan(sid)
    (env.root / "b.md").unlink()
    assert env.scan(sid).offline_reason == "empty_folder"
    assert sources.forget_empty(sid)["forgotten"] == 1


@pytest.mark.parametrize("make_reason", ["active", "unreachable", "disk_changed"])
def test_refused_unless_waiting_as_an_empty_folder(env, make_reason):
    sid = emptied(env)
    media = env.store()
    store = SourceStore(media)
    if make_reason == "active":
        store.set_state(sid, "active")
    else:
        store.set_state(sid, "offline", stats={"offline_reason": make_reason})
    media.close()
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "not_empty_folder"
    assert states(env, sid) == {"indexed"} and env.runtime.archived == []


def test_a_file_that_reappeared_stops_it(env):
    sid = emptied(env)
    env.write("c.md", "back")
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "folder_not_empty"
    assert states(env, sid) == {"indexed"} and row(env, sid)["state"] == "offline"
    assert env.runtime.archived == []


def test_a_moved_root_stops_it(env, tmp_path):
    sid = emptied(env)
    other = tmp_path / "other"
    other.mkdir()
    env.root.rmdir()
    env.root.symlink_to(other)
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "root_moved"
    assert states(env, sid) == {"indexed"} and env.runtime.archived == []


def test_another_disk_stops_it(env):
    sid = emptied(env)
    media = env.store()
    with media._write() as conn:
        conn.execute("UPDATE sources SET last_scan_stats_json = ? WHERE source_id = ?",
                     (json.dumps({"offline_reason": "empty_folder", "root_dev": os.stat(env.root).st_dev + 1}), sid))
    media.close()
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "disk_changed" and states(env, sid) == {"indexed"}


def test_remote_access_stops_it(env):
    sid = emptied(env)
    env.remote = True
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "remote_access_on" and states(env, sid) == {"indexed"}


def test_no_writer_stops_it(env):
    sid = emptied(env)
    env.host = env.host.__class__(**{**env.host.__dict__, "runtime": lambda: None})
    sources.configure(env.host)
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty(sid)
    assert e.value.code == "writer_not_ready" and states(env, sid) == {"indexed"}


def test_unknown_source(env):
    with pytest.raises(sources.SourceRefused) as e:
        sources.forget_empty("nope")
    assert e.value.code == "unknown_source"


def test_a_tombstoned_row_no_longer_counts_toward_the_device_overlap(env):
    env.write("a.md", "x")
    env.write("b.md", "y")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "a.md").unlink()
    (env.root / "b.md").unlink()
    env.scan(sid)
    sources.forget_empty(sid)
    media = env.store()
    with media._write() as conn:
        conn.execute("UPDATE sources SET last_scan_stats_json = ? WHERE source_id = ?",
                     (json.dumps({"root_dev": os.stat(env.root).st_dev + 1}), sid))
    media.close()
    env.write("n.md", "new disk")
    stats = env.scan(sid)
    assert stats.offline is False and row(env, sid)["state"] == "active"
