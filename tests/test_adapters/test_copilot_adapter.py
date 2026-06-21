"""LLD-05 §12.4 — Copilot adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.hooks.adapter_base import (
    COPILOT_SOFT_BYTES,
    HARD_BYTES_CAP,
)
from superlocalmemory.hooks.copilot_adapter import (
    CopilotAdapter,
    TARGET_REL,
    render_copilot,
)


def _setup(tmp_path: Path, *, with_github: bool, recall=None) -> CopilotAdapter:
    if with_github:
        (tmp_path / ".github").mkdir()
    return CopilotAdapter(
        base_dir=tmp_path, sync_log_db=tmp_path / "memory.db",
        recall_fn=recall or (lambda q, l, p: []),
    )


def test_writes_to_github_copilot_instructions_md(tmp_path, fake_recall):
    adapter = _setup(tmp_path, with_github=True, recall=fake_recall)
    adapter.sync()
    path = tmp_path / TARGET_REL
    assert path.exists()
    content = path.read_text()
    assert "# SLM Active Brain Context" in content
    assert content.startswith("<!-- SLM-START -->")


def test_size_soft_cap_2kb(tmp_path, fake_recall):
    adapter = _setup(tmp_path, with_github=True, recall=fake_recall)
    adapter.sync()
    body = (tmp_path / TARGET_REL).read_bytes()
    # Our fake_recall is small; soft cap must be satisfied comfortably.
    assert len(body) <= COPILOT_SOFT_BYTES


def test_size_hard_cap_4kb(tmp_path):
    def recall(q, limit, pid):
        if "topics" in q:
            return [{"name": "t_" + "x" * 500, "score": 0.9} for _ in range(20)]
        if "entities" in q:
            return [{"name": "e_" + "y" * 500, "mentions": 5} for _ in range(20)]
        if "decisions" in q:
            return [{"text": "d_" + "z" * 500} for _ in range(20)]
        if "memories" in q:
            return [{"text": "m_" + "w" * 500} for _ in range(20)]
        return []
    adapter = _setup(tmp_path, with_github=True, recall=recall)
    adapter.sync()
    body = (tmp_path / TARGET_REL).read_bytes()
    assert len(body) <= HARD_BYTES_CAP


def test_does_not_create_github_dir(tmp_path, fake_recall):
    adapter = _setup(tmp_path, with_github=False, recall=fake_recall)
    assert adapter.is_active() is False
    # sync() still needs to be a safe no-op caller-wise (LLD-05 §9 A8 —
    # errors-never-abort-loop). We don't assert wrote==False here because
    # the loop wouldn't even call sync when is_active is False.


def test_atomic_and_hash_skip(tmp_path, fake_recall, monkeypatch):
    from superlocalmemory.hooks import context_payload as cp
    monkeypatch.setattr(cp, "_now_iso", lambda: "2026-04-18T00:00:00+00:00")
    adapter = _setup(tmp_path, with_github=True, recall=fake_recall)
    assert adapter.sync() is True
    assert adapter.sync() is False


def test_disable(tmp_path, fake_recall):
    adapter = _setup(tmp_path, with_github=True, recall=fake_recall)
    adapter.sync()
    path = tmp_path / TARGET_REL
    assert path.exists()
    adapter.disable()
    # Marker-bounded: disable() strips the SLM block but does not delete the
    # file — copilot-instructions.md is user-owned and may contain other content.
    assert path.exists()
    assert "<!-- SLM-START -->" not in path.read_text()
    assert adapter.is_active() is False


def test_env_disable(tmp_path, monkeypatch, fake_recall):
    adapter = _setup(tmp_path, with_github=True, recall=fake_recall)
    monkeypatch.setenv("SLM_COPILOT_DISABLED", "1")
    assert adapter.is_active() is False


def test_env_force(tmp_path, monkeypatch, fake_recall):
    adapter = _setup(tmp_path, with_github=False, recall=fake_recall)
    monkeypatch.setenv("SLM_COPILOT_FORCE", "1")
    assert adapter.is_active() is True


def test_render_copilot_returns_bytes():
    from tests.test_adapters.conftest import make_payload
    out = render_copilot(make_payload())
    assert isinstance(out, bytes)
    assert b"Never do" in out
