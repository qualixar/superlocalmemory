"""Reconciling a folder of notes: first scan, change, move, delete, offline, placeholders."""

from __future__ import annotations

import json
import os
import shutil
from types import SimpleNamespace

import pytest

from superlocalmemory.sources import walk as walk_mod


def prov(row):
    return row["metadata"]["_slm_source"]


def test_first_scan_ingests_text_and_markdown_with_provenance(env):
    env.write("a.md", "# Title\nsome note")
    env.write("sub/b.txt", "plain text")
    env.write("c.exe", "binary")
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert sorted(env.runtime.contents()) == ["# Title\nsome note", "plain text"]
    for row in env.runtime.saved:
        assert row["source_type"] == "folder"
        assert row["actor"] == "test-actor"
        p = prov(row)
        assert p["type"] == "folder" and p["source_id"] == sid and len(p["version"]) == 12
    assert {prov(r)["relpath"] for r in env.runtime.saved} == {"a.md", "sub/b.txt"}
    files = env.files(sid)
    assert files["a.md"]["state"] == "indexed" and files["sub/b.txt"]["state"] == "indexed"
    assert stats.new == 2 and stats.skipped.get("type_not_allowed") == 1


def test_second_scan_changes_nothing(env):
    env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    env.sleeps.clear()
    stats = env.scan(sid)
    assert len(env.runtime.saved) == 1 and stats.new == 0 and stats.changed == 0
    assert env.sleeps == []


def test_change_creates_new_version_and_hides_old(env):
    path = env.write("a.md", "version one")
    sid = env.add_and_confirm()
    env.scan(sid)
    first = env.runtime.saved[0]
    path.write_text("version two")
    os.utime(path, ns=(2_000_000_000_000_000_000,) * 2)
    stats = env.scan(sid)
    assert stats.changed == 1
    assert env.runtime.contents() == ["version one", "version two"]
    assert prov(env.runtime.saved[1])["version"] != prov(first)["version"]
    assert env.runtime.archived == [first["fid"]]
    live = [e for e in json.loads(env.files(sid)["a.md"]["memory_ids_json"]) if not e.get("sup")]
    assert [e["m"] for e in live] == [env.runtime.saved[1]["mid"]]


def test_move_repoints_without_reingest(env):
    path = env.write("old/a.md", "moved text")
    sid = env.add_and_confirm()
    env.scan(sid)
    (env.root / "new").mkdir()
    shutil.move(str(path), str(env.root / "new" / "a.md"))
    stats = env.scan(sid)
    assert stats.moved == 1
    assert len(env.runtime.saved) == 1 and env.runtime.archived == []
    files = env.files(sid)
    assert "old/a.md" not in files and files["new/a.md"]["state"] == "indexed"


def test_delete_hides_now_and_purges_after_grace(env):
    path = env.write("a.md", "gone soon")
    sid = env.add_and_confirm()
    env.scan(sid)
    path.unlink()
    stats = env.scan(sid)
    assert stats.tombstoned == 1
    assert env.runtime.archived == ["f1"]
    row = env.files(sid)["a.md"]
    assert row["state"] == "tombstoned" and row["tombstoned_at"]
    assert env.erased == []
    env.host.purge_after_s = -1.0  # the grace is over
    stats = env.scan(sid)
    assert stats.purged == 1
    assert env.erased and env.erased[0][1] == ("f1",)
    assert "a.md" not in env.files(sid)


def test_old_versions_are_erased_after_grace(env):
    path = env.write("a.md", "one")
    sid = env.add_and_confirm()
    env.scan(sid)
    path.write_text("two")
    os.utime(path, ns=(2_000_000_000_000_000_000,) * 2)
    env.scan(sid)
    env.host.purge_after_s = -1.0
    env.scan(sid)
    assert any(call[1] == ("f1",) for call in env.erased)
    live = json.loads(env.files(sid)["a.md"]["memory_ids_json"])
    assert [e["m"] for e in live] == ["m2"]


def test_unreachable_root_goes_offline_and_tombstones_nothing(env):
    env.write("a.md", "keep me")
    sid = env.add_and_confirm()
    env.scan(sid)
    shutil.rmtree(env.root)
    stats = env.scan(sid)
    assert stats.offline and stats.tombstoned == 0
    assert env.runtime.archived == []
    from superlocalmemory.sources.store import SourceStore
    media = env.store()
    try:
        assert SourceStore(media).get_source(sid)["state"] == "offline"
    finally:
        media.close()
    assert env.files(sid)["a.md"]["state"] == "indexed"


def test_back_online_resumes(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    backup = env.root.parent / "bk"
    env.root.rename(backup)
    env.scan(sid)
    backup.rename(env.root)
    stats = env.scan(sid)
    assert not stats.offline and env.files(sid)["a.md"]["state"] == "indexed"


def test_unreadable_subfolder_is_not_treated_as_deleted(env, monkeypatch):
    env.write("sub/a.md", "inside")
    sid = env.add_and_confirm()
    env.scan(sid)
    real = os.scandir

    def fake(path):
        if str(path).endswith("sub"):
            raise PermissionError(path)
        return real(path)

    monkeypatch.setattr(walk_mod.os, "scandir", fake)
    stats = env.scan(sid)
    assert stats.tombstoned == 0 and env.files(sid)["sub/a.md"]["state"] == "indexed"


def test_placeholder_keeps_memories(env, monkeypatch):
    env.write("a.md", "cloud note")
    sid = env.add_and_confirm()
    env.scan(sid)
    monkeypatch.setattr(walk_mod, "is_placeholder", lambda st: True)
    stats = env.scan(sid)
    assert stats.tombstoned == 0 and stats.placeholders == 1
    assert env.runtime.archived == []
    row = env.files(sid)["a.md"]
    assert row["state"] == "cloud_placeholder" and json.loads(row["memory_ids_json"])
    monkeypatch.undo()
    env.scan(sid)
    assert env.files(sid)["a.md"]["state"] == "indexed"
    assert len(env.runtime.saved) == 1  # same content: not saved again


def test_new_placeholder_is_never_read(env, monkeypatch):
    env.write("a.md", "never read")
    monkeypatch.setattr(walk_mod, "is_placeholder", lambda st: True)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.runtime.saved == []
    assert env.files(sid)["a.md"]["state"] == "cloud_placeholder"


def test_placeholder_attribute_tables():
    win = SimpleNamespace(st_file_attributes=0x400000)
    assert walk_mod.is_placeholder(win)
    assert walk_mod.is_placeholder(SimpleNamespace(st_file_attributes=0x40000))
    assert walk_mod.is_placeholder(SimpleNamespace(st_file_attributes=0x1000))
    assert not walk_mod.is_placeholder(SimpleNamespace(st_file_attributes=0x20))
    assert walk_mod.is_placeholder(SimpleNamespace(st_flags=walk_mod.SF_DATALESS))
    assert not walk_mod.is_placeholder(SimpleNamespace())


def test_stability_check_sleeps_once_per_pass(env):
    for i in range(5):
        env.write(f"n{i}.md", f"note {i}")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.sleeps == [2.0]


def test_file_that_keeps_changing_is_deferred(env, monkeypatch):
    path = env.write("a.md", "in flux")
    sid = env.add_and_confirm()

    def touch(_seconds):
        path.write_text("in flux, longer now")

    env.host.sleep = touch
    stats = env.scan(sid)
    assert stats.deferred == 1 and env.runtime.saved == []


def test_cap_stops_the_walk_and_never_tombstones(env, monkeypatch):
    for i in range(4):
        env.write(f"n{i}.md", f"note {i}")
    sid = env.add_and_confirm()
    env.scan(sid)
    monkeypatch.setattr(walk_mod, "MAX_FILES", 2)
    stats = env.scan(sid)
    assert stats.capped and stats.tombstoned == 0 and env.runtime.archived == []


def test_ignored_and_secret_named_files_are_not_read(env):
    env.write(".hidden.md", "x")
    env.write("node_modules/pkg/readme.md", "x")
    env.write("server.pem", "x")
    env.write(".gitignore", "drafts/\n")
    env.write("drafts/d.md", "x")
    env.write("ok.md", "fine")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert env.runtime.contents() == ["fine"]


def test_symlink_escaping_the_root_is_skipped(env, tmp_path):
    outside = tmp_path / "outside.md"
    outside.write_text("secret outside")
    (env.root / "link.md").symlink_to(outside)
    (env.root / "dirlink").symlink_to(tmp_path, target_is_directory=True)
    env.write("ok.md", "fine")
    sid = env.add_and_confirm()
    stats = env.scan(sid)
    assert env.runtime.contents() == ["fine"]
    assert stats.skipped.get("symlink_escape", 0) >= 1


def test_the_folder_is_never_written(env):
    env.write("a.md", "text")
    env.write("sub/b.md", "more")
    before = sorted((p.relative_to(env.root).as_posix(), p.stat().st_mtime_ns, p.read_bytes())
                    for p in env.root.rglob("*") if p.is_file())
    dirs_before = sorted(p.as_posix() for p in env.root.rglob("*"))
    sid = env.add_and_confirm()
    env.scan(sid)
    env.scan(sid)
    after = sorted((p.relative_to(env.root).as_posix(), p.stat().st_mtime_ns, p.read_bytes())
                   for p in env.root.rglob("*") if p.is_file())
    assert before == after and dirs_before == sorted(p.as_posix() for p in env.root.rglob("*"))


def test_long_markdown_is_split_at_headings(env):
    body = "".join(f"## Section {i}\n" + ("word " * 1500) + "\n" for i in range(5))
    assert len(body) > 24_000
    env.write("long.md", body)
    sid = env.add_and_confirm()
    env.scan(sid)
    assert len(env.runtime.saved) >= 2
    assert all(len(c) <= 24_000 for c in env.runtime.contents())
    assert env.runtime.saved[0]["content"].startswith("## Section 0")


def test_scan_yields_to_recalls(env, monkeypatch):
    calls = []
    monkeypatch.setattr("superlocalmemory.sources.reconcile.yield_to_recalls",
                        lambda *a, **k: calls.append(1))
    env.write("a.md", "x")
    env.write("b.md", "y")
    sid = env.add_and_confirm()
    env.scan(sid)
    assert len(calls) >= 2
