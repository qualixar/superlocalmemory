"""The folder must still be the folder that was confirmed; links inside it are skipped, not errors."""

from __future__ import annotations

import os
import threading

from superlocalmemory import sources
from superlocalmemory.sources import locks
from superlocalmemory.sources.store import SourceStore


def test_a_root_swapped_for_a_link_to_a_sibling_is_offline_as_root_moved(env, tmp_path):
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    env.scan(sid)
    other = tmp_path / "other"
    other.mkdir()
    (other / "secret.md").write_text("sibling content")
    os.rename(env.root, tmp_path / "vault_moved")
    os.symlink(other, env.root)
    stats = env.scan(sid)
    assert stats.offline and stats.offline_reason == "root_moved"
    assert not any("sibling content" in c for c in env.runtime.contents())
    assert env.runtime.archived == []
    assert sources.source_report(sid).offline_reason == "root_moved"
    [info] = sources.list_sources("default")
    assert info.offline_reason == "root_moved"


def test_offline_reason_is_empty_for_a_healthy_source(env):
    env.write("a.md", "note a")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert sources.source_report(sid).offline_reason is None
    assert sources.list_sources("default")[0].offline_reason is None


def test_symlinked_files_are_skipped_and_stale_rows_are_hidden(env):
    env.write("real.md", "real note")
    env.write("b.md", "b note v1")
    os.symlink(env.root / "real.md", env.root / "link.md")
    sid = env.add_and_confirm()
    first = env.scan(sid)
    assert first.errors == 0 and "link.md" not in env.files(sid)
    os.unlink(env.root / "b.md")
    os.symlink(env.root / "real.md", env.root / "b.md")
    second = env.scan(sid)
    third = env.scan(sid)
    assert second.errors == 0 and third.errors == 0
    assert env.files(sid)["b.md"]["state"] == "tombstoned"
    assert sources.source_report(sid).skipped_by_rule["symlink_file"] == 2
    assert len(env.runtime.archived) == 1


def test_a_removal_stops_the_scan_before_it_hashes_anything(env, monkeypatch):
    import superlocalmemory.sources.reconcile as rc

    env.write("a.md", "x")
    sid = env.add_and_confirm()
    slept, release = threading.Event(), threading.Event()
    env.host.sleep = lambda s: (slept.set(), release.wait(5))
    hashed = []
    real = rc._digest
    monkeypatch.setattr(rc, "_digest", lambda *a, **k: (hashed.append(1), real(*a, **k))[1])
    t = threading.Thread(target=lambda: env.scan(sid))
    t.start()
    assert slept.wait(5)
    r = threading.Thread(target=lambda: sources.remove_source(sid))
    r.start()
    for _ in range(500):
        if locks.is_removing(sid):
            break
        threading.Event().wait(0.01)
    release.set()
    t.join(10)
    r.join(10)
    assert hashed == [] and env.runtime.saved == []
