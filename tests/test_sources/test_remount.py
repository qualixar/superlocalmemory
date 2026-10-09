"""A folder that comes back on a new device number keeps being scanned; a different disk does not."""

from __future__ import annotations

import json

from superlocalmemory.sources.store import SourceStore


def _row(env, sid):
    media = env.store()
    try:
        return SourceStore(media).get_source(sid)
    finally:
        media.close()


def _bump_dev(env, sid, by=1):
    media = env.store()
    try:
        store = SourceStore(media)
        stats = json.loads(store.get_source(sid)["last_scan_stats_json"])
        stats["root_dev"] += by
        store.set_state(sid, "active", stats=stats)
        return stats["root_dev"]
    finally:
        media.close()


def test_same_files_on_a_new_device_number_go_on_scanning(env):
    for name in ("a.md", "b.md"):
        env.write(name, f"note {name}")
    sid = env.add_and_confirm()
    env.scan(sid)
    _bump_dev(env, sid)
    env.scan(sid)
    env.write("c.md", "a new note")
    stats = env.scan(sid)
    assert not stats.offline and len(env.runtime.saved) == 3
    row = _row(env, sid)
    import os
    assert row["state"] == "active"
    assert json.loads(row["last_scan_stats_json"])["root_dev"] == os.stat(env.root).st_dev


def test_a_different_disk_with_none_of_the_files_is_offline(env):
    for name in ("a.md", "b.md"):
        env.write(name, f"note {name}")
    sid = env.add_and_confirm()
    env.scan(sid)
    for name in ("a.md", "b.md"):
        (env.root / name).unlink()
    env.write("other.md", "someone else's note")
    _bump_dev(env, sid)
    stats = env.scan(sid)
    assert stats.offline and stats.offline_reason == "disk_changed"
    assert env.runtime.archived == [] and len(env.runtime.saved) == 2
