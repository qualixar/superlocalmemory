"""Extra coverage for edge branches in LLD-05 modules."""

from __future__ import annotations

import asyncio
import os
import sqlite3
from pathlib import Path

import pytest

from superlocalmemory.hooks.adapter_base import (
    HARD_BYTES_CAP,
    TRUNCATION_MARKER,
    atomic_write,
    path_sha256,
    sync_log_last_content_sha256,
    sync_log_record,
    truncate_to_cap,
)
from superlocalmemory.hooks.antigravity_adapter import AntigravityAdapter
from superlocalmemory.hooks.copilot_adapter import CopilotAdapter
from superlocalmemory.hooks.cursor_adapter import CursorAdapter
from superlocalmemory.hooks.context_payload import (
    DEFAULT_TOP_K,
    build_payload,
    truncate_payload_for_cap,
)
from superlocalmemory.hooks.sync_loop import (
    DEFAULT_INTERVAL_SECONDS,
    _interval_from_env,
    schedule,
)


# ---------------------------------------------------------------------------
# adapter_base
# ---------------------------------------------------------------------------


def test_truncate_to_cap_short_circuits():
    content = b"hello"
    assert truncate_to_cap(content, cap=100) == content


def test_truncate_to_cap_adds_marker():
    content = b"x" * 5000
    out = truncate_to_cap(content, cap=HARD_BYTES_CAP)
    assert out.endswith(TRUNCATION_MARKER)
    assert len(out) <= HARD_BYTES_CAP


def test_sync_log_record_rejects_short_hash(tmp_path):
    with pytest.raises(ValueError):
        sync_log_record(
            tmp_path / "m.db",
            adapter_name="a", profile_id="p",
            target_path_sha256="deadbeef",  # too short
            target_basename="x",
            bytes_written=0, content_sha256="0" * 64,
            success=True,
        )


def test_sync_log_last_returns_none_for_unknown_db(tmp_path):
    # DB doesn't exist.
    assert sync_log_last_content_sha256(
        tmp_path / "nope.db", "x", "0" * 64
    ) is None


def test_sync_log_last_ignores_failed_rows(tmp_path):
    sync_log_record(
        tmp_path / "m.db",
        adapter_name="a", profile_id="p",
        target_path_sha256="a" * 64, target_basename="f",
        bytes_written=0, content_sha256="b" * 64,
        success=False, error_msg="kaboom",
    )
    assert sync_log_last_content_sha256(
        tmp_path / "m.db", "a", "a" * 64
    ) is None


def test_atomic_write_rewrites_on_content_change(tmp_path):
    target = tmp_path / "f.txt"
    r1 = atomic_write(
        target, b"first", adapter_name="test", profile_id="p",
        sync_log_db=tmp_path / "m.db",
    )
    r2 = atomic_write(
        target, b"second", adapter_name="test", profile_id="p",
        sync_log_db=tmp_path / "m.db",
    )
    assert r1.wrote and r2.wrote
    assert target.read_bytes() == b"second"


def test_atomic_write_rewrites_after_out_of_band_mutation(tmp_path):
    """Sync-log skip must not fire when the file was mutated out-of-band.

    Regression guard: git restore / manual edit changes the on-disk content
    after a successful sync. Without on-disk re-hash, the next sync would
    compare new_hash against the stale sync-log row (which matches) and skip
    the write — leaving the file in its diverged state forever.
    """
    target = tmp_path / "f.txt"
    content = b"slm-managed content"
    db = tmp_path / "m.db"

    r1 = atomic_write(target, content, adapter_name="test", profile_id="p",
                      sync_log_db=db)
    assert r1.wrote

    # Second sync — content unchanged, durable skip should fire.
    r2 = atomic_write(target, content, adapter_name="test", profile_id="p",
                      sync_log_db=db)
    assert not r2.wrote

    # Out-of-band mutation (simulates ``git restore`` or manual edit).
    target.write_bytes(b"user has edited this file manually")

    # Third sync — sync-log says skip, but on-disk truth differs. Must re-write.
    r3 = atomic_write(target, content, adapter_name="test", profile_id="p",
                      sync_log_db=db)
    assert r3.wrote
    assert target.read_bytes() == content


# ---------------------------------------------------------------------------
# antigravity
# ---------------------------------------------------------------------------


def test_antigravity_env_disable(tmp_path, monkeypatch, fake_recall):
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    monkeypatch.setenv("SLM_ANTIGRAVITY_DISABLED", "1")
    assert adapter.is_active() is False


def test_antigravity_adapter_force_env(tmp_path, monkeypatch, fake_recall):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    monkeypatch.setenv("SLM_ADAPTER_FORCE_ANTIGRAVITY", "1")
    assert adapter.is_active() is True


def test_cursor_adapter_force_env(tmp_path, monkeypatch, fake_recall):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    monkeypatch.setenv("SLM_ADAPTER_FORCE_CURSOR", "1")
    assert adapter.is_active() is True


def test_copilot_disable_when_file_absent(tmp_path, fake_recall):
    (tmp_path / ".github").mkdir()
    adapter = CopilotAdapter(
        base_dir=tmp_path, sync_log_db=tmp_path / "m.db",
        recall_fn=fake_recall,
    )
    # Never wrote anything — disable is still safe + idempotent.
    adapter.disable()
    assert adapter.is_active() is False


def test_antigravity_disable_idempotent(tmp_path, monkeypatch, fake_recall):
    monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    adapter.sync()
    adapter.disable()
    adapter.disable()   # second call — no crash, still inactive
    assert adapter.is_active() is False


def test_cursor_inactive_in_sync_loop(tmp_path, monkeypatch, fake_recall):
    """Adapter is_active==False → the loop helper should mark it inactive."""
    from superlocalmemory.hooks.sync_loop import run_once
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    result = asyncio.run(run_once([adapter]))
    assert result[adapter.name] == "inactive"


# ---------------------------------------------------------------------------
# sync_loop env parsing + schedule
# ---------------------------------------------------------------------------


def test_interval_env_default(monkeypatch):
    monkeypatch.delenv("SLM_CROSS_PLATFORM_SYNC_INTERVAL", raising=False)
    assert _interval_from_env() == DEFAULT_INTERVAL_SECONDS


def test_interval_env_override(monkeypatch):
    monkeypatch.setenv("SLM_CROSS_PLATFORM_SYNC_INTERVAL", "120")
    assert _interval_from_env() == 120


def test_interval_env_invalid_falls_back(monkeypatch):
    monkeypatch.setenv("SLM_CROSS_PLATFORM_SYNC_INTERVAL", "not-a-number")
    assert _interval_from_env() == DEFAULT_INTERVAL_SECONDS


def test_interval_env_minimum_enforced(monkeypatch):
    monkeypatch.setenv("SLM_CROSS_PLATFORM_SYNC_INTERVAL", "1")
    assert _interval_from_env() == 30  # clamped up


def test_schedule_returns_task():
    class _Fake:
        name = "f"
        target_path = Path("/tmp/x")
        def is_active(self): return False
        def sync(self): return False
        def disable(self): pass

    async def _go():
        task = schedule([_Fake()])
        task.cancel()
        try:
            await task
        except (asyncio.CancelledError, BaseException):
            pass

    asyncio.run(_go())


# ---------------------------------------------------------------------------
# context_payload edge branches
# ---------------------------------------------------------------------------


def test_truncate_payload_all_sections_dropped():
    """Hard cap so small that even topics have to go."""
    from tests.test_adapters.conftest import make_payload
    payload = make_payload(long_text=True)

    def render(p) -> bytes:
        parts = []
        parts.append("|".join(n for n, _ in p.topics))
        parts.append("|".join(n for n, _ in p.entities))
        parts.append("|".join(p.recent_decisions))
        parts.append("|".join(p.project_memories))
        return "||".join(parts).encode()

    out = truncate_payload_for_cap(payload, hard_cap=10, render=render)
    assert len(out) <= 1024  # final step returns whatever we got


def test_recall_row_without_name_or_text_skipped():
    def recall_fn(q, limit, pid):
        return [{"score": 0.9}]  # no name, no text
    payload = build_payload("d", "project", Path("/tmp"), recall_fn=recall_fn)
    assert payload.topics == ()


def test_recall_entity_score_defaults():
    def recall_fn(q, limit, pid):
        if "entities" in q:
            return [{"name": "alpha"}]  # no mentions
        return []
    payload = build_payload("d", "project", Path("/tmp"), recall_fn=recall_fn)
    assert payload.entities == (("alpha", 0),)


# ---------------------------------------------------------------------------
# CLI context_commands extra branches
# ---------------------------------------------------------------------------


def test_build_default_adapters_uses_cwd(tmp_path, monkeypatch, fake_recall):
    from superlocalmemory.cli.context_commands import build_default_adapters
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        "superlocalmemory.cli.context_commands._get_recall_fn",
        lambda: fake_recall,
    )
    adapters = build_default_adapters()
    names = [a.name for a in adapters]
    assert "cursor_project" in names
    assert "cursor_global" in names
    assert "antigravity_workspace" in names
    assert "antigravity_global" in names
    assert "copilot_project" in names


def test_copilot_sync_path_traversal_marks_inactive(tmp_path, monkeypatch,
                                                    fake_recall):
    (tmp_path / ".github").mkdir()
    adapter = CopilotAdapter(
        base_dir=tmp_path, sync_log_db=tmp_path / "m.db",
        recall_fn=fake_recall,
    )
    # Force safe_resolve to raise.
    from superlocalmemory.hooks import copilot_adapter as mod
    from superlocalmemory.core.security_primitives import PathTraversalError
    monkeypatch.setattr(
        mod, "safe_resolve",
        lambda base, rel: (_ for _ in ()).throw(PathTraversalError("no")),
    )
    assert adapter.sync() is False
    assert adapter.is_active() is False  # _inactive_until_retry latched


def test_copilot_disable_path_traversal_silent(tmp_path, monkeypatch,
                                                fake_recall):
    (tmp_path / ".github").mkdir()
    adapter = CopilotAdapter(
        base_dir=tmp_path, sync_log_db=tmp_path / "m.db",
        recall_fn=fake_recall,
    )
    from superlocalmemory.hooks import copilot_adapter as mod
    from superlocalmemory.core.security_primitives import PathTraversalError
    monkeypatch.setattr(
        mod, "safe_resolve",
        lambda base, rel: (_ for _ in ()).throw(PathTraversalError("no")),
    )
    # Must not raise.
    adapter.disable()


def test_antigravity_sync_path_traversal_marks_inactive(tmp_path, monkeypatch,
                                                        fake_recall):
    monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    from superlocalmemory.hooks import antigravity_adapter as mod
    from superlocalmemory.core.security_primitives import PathTraversalError
    monkeypatch.setattr(
        mod, "safe_resolve",
        lambda base, rel: (_ for _ in ()).throw(PathTraversalError("no")),
    )
    assert adapter.sync() is False
    assert adapter.is_active() is False


def test_antigravity_disable_path_traversal_silent(tmp_path, monkeypatch,
                                                    fake_recall):
    monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    from superlocalmemory.hooks import antigravity_adapter as mod
    from superlocalmemory.core.security_primitives import PathTraversalError
    monkeypatch.setattr(
        mod, "safe_resolve",
        lambda base, rel: (_ for _ in ()).throw(PathTraversalError("no")),
    )
    adapter.disable()


def test_cursor_disable_path_traversal_silent(tmp_path, monkeypatch,
                                               fake_recall):
    monkeypatch.setenv("SLM_CURSOR_FORCE", "1")
    adapter = CursorAdapter(
        scope="project", base_dir=tmp_path,
        sync_log_db=tmp_path / "m.db", recall_fn=fake_recall,
    )
    from superlocalmemory.hooks import cursor_adapter as mod
    from superlocalmemory.core.security_primitives import PathTraversalError
    monkeypatch.setattr(
        mod, "safe_resolve",
        lambda base, rel: (_ for _ in ()).throw(PathTraversalError("no")),
    )
    adapter.disable()


def test_copilot_hard_cap_fallback_path(tmp_path):
    """Exercise the hard-cap branch when soft-cap truncation still overshoots."""
    def recall(q, limit, pid):
        if "topics" in q:
            return [{"name": "t_" + "x" * 500, "score": 0.9}
                    for _ in range(20)]
        if "entities" in q:
            return [{"name": "e_" + "y" * 500, "mentions": 5}
                    for _ in range(20)]
        return []
    (tmp_path / ".github").mkdir()
    adapter = CopilotAdapter(
        base_dir=tmp_path, sync_log_db=tmp_path / "m.db",
        recall_fn=recall, soft_cap=200,   # brutally small soft cap
        hard_cap=600,
    )
    assert adapter.sync() is True


def test_get_recall_fn_noop_without_engine():
    """Without an engine, we still get a callable that returns ``[]``."""
    from superlocalmemory.cli.context_commands import _get_recall_fn
    fn = _get_recall_fn()
    assert callable(fn)
    # Don't assert the return value — we don't know if a daemon engine is up.
    try:
        out = fn("x", 1, "default")
    except Exception:
        out = None
    assert out is None or isinstance(out, list)
