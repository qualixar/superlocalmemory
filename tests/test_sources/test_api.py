"""add, preview, confirm, list, remove, report, hint: the public calls."""

from __future__ import annotations

import json

import pytest

from superlocalmemory import sources
from superlocalmemory.runtimes.features import sources_enabled
from superlocalmemory.sources.roots import RootRefused
from superlocalmemory.sources.store import SourceStore


def test_preview_writes_nothing(env):
    env.write("a.md", "x")
    preview = sources.add_source(env.root, profile_id="default")
    assert preview.files_by_type == {".md": 1}
    assert not (env.data / "media.db").exists()
    assert not (env.data / "features.json").exists()
    assert not sources_enabled(env.data)
    assert preview.source_id and preview.kind == "folder"


def test_preview_counts_skips_quarantine_and_estimates(env):
    env.write("a.md", "x" * 100)
    env.write("b.txt", "y")
    env.write("c.pdf", b"%PDF-1.4" + b"0" * 200_000)
    env.write("keys.md", "AKIA" + "ABCDEFGHIJKLMNOP")
    env.write(".dot.md", "x")
    env.write("d.exe", "x")
    preview = sources.add_source(env.root, profile_id="default")
    assert preview.files_by_type == {".md": 2, ".txt": 1, ".pdf": 1}
    assert preview.skipped_by_rule["dotfile"] == 1
    assert preview.skipped_by_rule["type_not_allowed"] == 1
    assert preview.quarantined_count == 1
    assert preview.est_bytes > 200_000
    assert preview.est_seconds > 0 and preview.estimate_note


def test_preview_warns_over_the_cap(env, monkeypatch):
    from superlocalmemory.sources import walk
    monkeypatch.setattr(walk, "MAX_FILES", 2)
    for i in range(4):
        env.write(f"n{i}.md", "x")
    preview = sources.add_source(env.root, profile_id="default")
    assert preview.capped and any("50,000" in w or "limit" in w for w in preview.warnings)


def test_obsidian_folder_is_detected(env):
    (env.root / ".obsidian").mkdir()
    env.write("a.md", "x")
    assert sources.add_source(env.root, profile_id="default").kind == "obsidian"


def test_add_refuses_unsafe_roots(env, tmp_path):
    from pathlib import Path
    with pytest.raises(RootRefused):
        sources.add_source(Path("/"), profile_id="default")
    with pytest.raises(RootRefused):
        sources.add_source(Path.home(), profile_id="default")
    with pytest.raises(RootRefused):
        sources.add_source(tmp_path / "missing", profile_id="default")


def test_confirm_turns_the_feature_on_and_queues_a_scan(env):
    env.write("a.md", "x")
    preview = sources.add_source(env.root, profile_id="default")
    sources.confirm_source(preview.source_id)
    assert sources_enabled(env.data) and (env.data / "media.db").exists()
    assert env.woken == 1
    media = env.store()
    try:
        jobs = media.list_jobs("default", ["queued"])
        assert [j["kind"] for j in jobs] == ["source_scan"]
        assert json.loads(jobs[0]["payload_json"])["source_id"] == preview.source_id
        assert SourceStore(media).get_source(preview.source_id)["state"] == "active"
    finally:
        media.close()


def test_confirm_is_refused_while_remote_access_is_set_up(env):
    env.remote = True
    preview = sources.add_source(env.root, profile_id="default")
    with pytest.raises(sources.SourceRefused) as exc:
        sources.confirm_source(preview.source_id)
    assert exc.value.code == "remote_access_on"
    assert "remote" in str(exc.value).lower()
    assert not (env.data / "media.db").exists() and not sources_enabled(env.data)


def test_confirm_with_a_raising_remote_check_is_refused(env):
    def boom():
        raise OSError

    env.host.remote_check = boom
    preview = sources.add_source(env.root, profile_id="default")
    with pytest.raises(sources.SourceRefused):
        sources.confirm_source(preview.source_id)


def test_confirm_unknown_id(env):
    with pytest.raises(sources.SourceRefused) as exc:
        sources.confirm_source("nope")
    assert exc.value.code == "unknown_source"


def test_confirm_twice_is_one_source(env):
    preview = sources.add_source(env.root, profile_id="default")
    sources.confirm_source(preview.source_id)
    sources.confirm_source(preview.source_id)
    assert len(sources.list_sources("default")) == 1


def test_list_sources(env):
    env.write("a.md", "x")
    assert sources.list_sources("default") == []
    sid = env.add_and_confirm()
    env.scan(sid)
    [info] = sources.list_sources("default")
    assert info.source_id == sid and info.state == "active" and info.files == {"indexed": 1}
    assert info.root_path == str(env.root.resolve())
    assert sources.list_sources("other") == []


def test_remove_keeps_and_hides(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    sources.remove_source(sid)
    assert env.runtime.archived == ["f1"] and env.erased == []
    assert sources.list_sources("default") == []
    assert env.files(sid)["a.md"]["state"] == "tombstoned"


def test_remove_with_purge_erases(env):
    env.write("a.md", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    sources.remove_source(sid, purge=True)
    assert env.erased and env.erased[0][1] == ("f1",)
    assert env.files(sid) == {}


def test_rescan_returns_a_job(env):
    sid = env.add_and_confirm()
    before = env.woken
    job = sources.rescan(sid)
    assert job["state"] == "queued" and job["job_id"] and env.woken == before + 1


def test_report_lists_problems(env):
    env.write("keys.md", "AKIA" + "ABCDEFGHIJKLMNOP")
    env.write("ok.md", "fine")
    env.write("x.exe", "x")
    sid = env.add_and_confirm()
    env.scan(sid)
    report = sources.source_report(sid)
    assert [q["relpath"] for q in report.quarantined] == ["keys.md"]
    assert report.counts["indexed"] == 1 and report.skipped_by_rule["type_not_allowed"] == 1
    assert report.cloud_only == [] and report.errors == []


def test_hint_is_not_available_yet(env):
    sid = env.add_and_confirm()
    with pytest.raises(sources.HintsNotAvailable):
        sources.hint(sid, ["a.md"])
