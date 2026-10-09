"""A targeted scan reads only the named paths but keeps every rule of a full scan."""

from __future__ import annotations

import os

from superlocalmemory.sources.reconcile import scan_source
from superlocalmemory.sources.store import SourceStore


def targeted(env, sid, rels):
    media = env.store()
    try:
        store = SourceStore(media)
        return scan_source(env.host, store, store.get_source(sid), only=frozenset(rels))
    finally:
        media.close()


def test_only_the_named_file_is_saved(env):
    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    env.write("a.md", "two")
    env.write("b.md", "new but not named")
    stats = targeted(env, sid, ["a.md"])
    assert stats.changed == 1 and stats.new == 0
    assert "b.md" not in env.files(sid)


def test_a_deleted_named_file_is_tombstoned_and_others_stay(env):
    env.write("a.md", "one")
    env.write("b.md", "two")
    sid = env.add_and_confirm()
    env.scan(sid)
    os.remove(env.root / "a.md")
    targeted(env, sid, ["a.md"])
    files = env.files(sid)
    assert files["a.md"]["state"] == "tombstoned" and files["b.md"]["state"] == "indexed"


def test_rules_still_apply(env):
    sid = env.add_and_confirm()
    env.write("secret.txt", "key sk-ant-api03-" + "A" * 40)
    env.write(".hidden.md", "x")
    env.write("p.exe", "x")
    os.symlink(env.root / "secret.txt", env.root / "link.md")
    targeted(env, sid, ["secret.txt", ".hidden.md", "p.exe", "link.md"])
    files = env.files(sid)
    assert files["secret.txt"]["state"] == "quarantined"
    assert ".hidden.md" not in files and "p.exe" not in files and "link.md" not in files
    assert env.runtime.saved == []


def test_a_targeted_scan_keeps_last_scan_time_and_stats(env):
    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    media = env.store()
    before = SourceStore(media).get_source(sid)
    media.close()
    env.write("a.md", "two")
    targeted(env, sid, ["a.md"])
    media = env.store()
    after = SourceStore(media).get_source(sid)
    media.close()
    assert after["last_scan_at"] == before["last_scan_at"]
    assert after["last_scan_stats_json"] == before["last_scan_stats_json"]
