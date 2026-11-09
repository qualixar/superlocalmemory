"""LLD-05 §12.2 — Cursor adapter tests (per-project + global)."""

from __future__ import annotations

import os
import sqlite3
import sys
from pathlib import Path

import pytest

from superlocalmemory.core.security_primitives import PathTraversalError
from superlocalmemory.hooks.adapter_base import (
    HARD_BYTES_CAP,
)
from superlocalmemory.hooks.cursor_adapter import (
    CursorAdapter,
    render_cursor,
)


def _parse_frontmatter(text: str) -> dict:
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    block = text[4:end]
    out: dict = {}
    for line in block.splitlines():
        if ":" in line:
            k, v = line.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def _make_adapter(tmp_path: Path, *, scope: str = "project", recall=None,
                  monkeypatch=None) -> CursorAdapter:
    if monkeypatch is not None:
        monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    fn = recall or (lambda q, limit_arg, p: [])
    base = tmp_path if scope == "project" else tmp_path / "home"
    base.mkdir(parents=True, exist_ok=True)
    return CursorAdapter(
        scope=scope, base_dir=base,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fn,
    )


def test_mdc_frontmatter_has_only_verified_fields(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    text = adapter.target_path.read_text(encoding="utf-8")
    keys = set(_parse_frontmatter(text).keys())
    assert keys <= {"description", "alwaysApply", "globs"}
    for banned in ("name", "scope", "type", "internal_type"):
        assert banned not in keys


def test_mdc_frontmatter_valid_yaml(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    text = adapter.target_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    assert "\n---\n" in text
    fm = _parse_frontmatter(text)
    assert fm["alwaysApply"] == "true"
    assert fm["globs"].strip().strip('"') == "**/*"


def test_project_and_global_paths_resolved_safely(tmp_path, monkeypatch,
                                                  fake_recall):
    project = _make_adapter(tmp_path, scope="project", recall=fake_recall,
                            monkeypatch=monkeypatch)
    assert project.target_path.name == "slm-active-brain.mdc"
    assert ".cursor/rules" in str(project.target_path)
    glob = _make_adapter(tmp_path, scope="global", recall=fake_recall,
                         monkeypatch=monkeypatch)
    assert glob.target_path.name == "slm-global.mdc"


def test_symlink_escape_refused(tmp_path, monkeypatch, fake_recall):
    # Pre-create a .cursor/rules/... symlink that escapes the base.
    outside = tmp_path / "outside"
    outside.mkdir()
    target_parent = tmp_path / "base" / ".cursor" / "rules"
    target_parent.mkdir(parents=True)
    if sys.platform == "win32":
        pytest.skip("symlink-escape semantics differ on Windows")
    link = target_parent / "slm-active-brain.mdc"
    link.symlink_to(outside / "gotcha.mdc")
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path / "base",
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    # target_path resolution raises — adapter's sync swallows to False.
    with pytest.raises(PathTraversalError):
        _ = adapter.target_path
    assert adapter.sync() is False


def test_size_cap_4kb_enforced(tmp_path, monkeypatch, fake_recall):
    # Build a monster recall
    def recall(q, limit, pid):
        if "topics" in q:
            return [{"name": "huge_topic_" + "x" * 500, "score": 0.9}
                    for _ in range(50)]
        if "entities" in q:
            return [{"name": "huge_entity_" + "y" * 500, "mentions": 5}
                    for _ in range(50)]
        if "decisions" in q:
            return [{"text": "huge_decision_" + "z" * 500} for _ in range(50)]
        if "memories" in q:
            return [{"text": "huge_mem_" + "w" * 500} for _ in range(50)]
        return []
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=recall,
    )
    adapter.sync()
    body = adapter.target_path.read_bytes()
    assert len(body) <= HARD_BYTES_CAP


def test_atomic_write_no_partial(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    target = adapter.target_path

    real_write = os.write
    state = {"calls": 0}
    def boom(fd, data):
        state["calls"] += 1
        raise OSError("simulated mid-write failure")
    monkeypatch.setattr("os.write", boom)
    with pytest.raises(OSError):
        adapter.sync()
    monkeypatch.setattr("os.write", real_write)
    # Target file must NOT exist — the tempfile wasn't renamed.
    assert not target.exists()


def test_content_hash_skip(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    # Pin timestamp so payload is byte-identical across calls.
    from superlocalmemory.hooks import context_payload as cp
    monkeypatch.setattr(cp, "_now_iso", lambda: "2026-04-18T00:00:00+00:00")
    first = adapter.sync()
    second = adapter.sync()
    assert first is True
    assert second is False


def test_disable_removes_file_and_flags_inactive(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    assert adapter.target_path.exists()
    adapter.disable()
    assert not adapter.target_path.exists()
    assert adapter.is_active() is False


def test_inactive_when_no_cursor_dir(tmp_path, monkeypatch, fake_recall):
    # Ensure no cursor-related dirs exist by pointing HOME to a fresh dir.
    fake_home = tmp_path / "fake_home"
    fake_home.mkdir()
    monkeypatch.setenv("HOME", str(fake_home))
    monkeypatch.setenv("USERPROFILE", str(fake_home))
    # Also isolate Path.home() via monkeypatch on pathlib
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    assert adapter.is_active() is False


def test_file_permissions_0600_on_posix(tmp_path, monkeypatch, fake_recall):
    if sys.platform == "win32":
        pytest.skip("POSIX-only permission bits")
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    mode = adapter.target_path.stat().st_mode & 0o777
    assert mode == 0o600


def test_content_hash_skip_survives_restart(tmp_path, monkeypatch, fake_recall):
    """A3: durable skip driven by cross_platform_sync_log.content_sha256."""
    from superlocalmemory.hooks import context_payload as cp
    monkeypatch.setattr(cp, "_now_iso", lambda: "2026-04-18T00:00:00+00:00")

    adapter1 = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    assert adapter1.sync() is True

    # "Restart": throw away the adapter, build a fresh instance with the same DB.
    adapter2 = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    # Get file mtime before
    mtime_before = adapter2.target_path.stat().st_mtime
    assert adapter2.sync() is False
    mtime_after = adapter2.target_path.stat().st_mtime
    assert mtime_before == mtime_after


def test_sync_log_target_path_sha256_full_length_not_raw(tmp_path, monkeypatch,
                                                         fake_recall):
    """A7: sync log stores SHA-256 (full 64-hex), never the raw absolute path."""
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()

    conn = sqlite3.connect(str(tmp_path / "memory.db"))
    try:
        rows = conn.execute(
            "SELECT target_path_sha256, target_basename FROM "
            "cross_platform_sync_log"
        ).fetchall()
    finally:
        conn.close()

    assert rows, "expected at least one sync log row"
    raw_path = str(adapter.target_path)
    for sha, basename in rows:
        assert len(sha) == 64, f"target_path_sha256 must be 64 hex, got {len(sha)}"
        assert all(c in "0123456789abcdef" for c in sha)
        assert sha != raw_path
        assert os.sep not in sha and "/" not in sha
        assert basename == adapter.target_path.name


def test_render_cursor_outputs_bytes(fake_recall):
    from tests.test_adapters.conftest import make_payload
    out = render_cursor(make_payload())
    assert isinstance(out, bytes)
    assert b"SLM" in out


def test_env_disable_switches_inactive(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    monkeypatch.setenv("SLM_CURSOR_DISABLED", "1")
    assert adapter.is_active() is False


def test_invalid_scope_rejected(tmp_path, fake_recall):
    with pytest.raises(ValueError):
        CursorAdapter(scope="bogus", base_dir=tmp_path,
                      sync_log_db=tmp_path / "memory.db",
                      recall_fn=fake_recall)


def test_disable_on_nonexistent_file_is_safe(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    # Never synced — disable should still run and record a disable row.
    adapter.disable()
    assert adapter.is_active() is False
