"""LLD-05 §12.3 — Antigravity adapter tests."""

from __future__ import annotations

from pathlib import Path

import pytest

from superlocalmemory.hooks.adapter_base import HARD_BYTES_CAP
from superlocalmemory.hooks.antigravity_adapter import (
    GLOBAL_REL,
    WORKSPACE_REL,
    AntigravityAdapter,
    render_antigravity,
)


def _make_adapter(tmp_path: Path, *, scope: str = "workspace",
                  recall=None, monkeypatch=None) -> AntigravityAdapter:
    if monkeypatch is not None:
        monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    fn = recall or (lambda q, l, p: [])
    return AntigravityAdapter(
        scope=scope, base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fn,
    )


def test_writes_to_singular_agent_skills_path(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    # Must end in the canonical singular path.
    assert str(adapter.target_path).endswith(
        "/.agent/skills/slm-memory-adapter/SKILL.md"
    ) or str(adapter.target_path).endswith(
        "\\.agent\\skills\\slm-memory-adapter\\SKILL.md"
    )


def test_path_never_uses_plural_agents_or_knowledge():
    banned = (".agents/knowledge", ".agents/skills", ".antigravity/knowledge")
    text = WORKSPACE_REL + "|" + GLOBAL_REL
    for b in banned:
        assert b not in text, f"banned path substring {b!r} leaked into rel-path"


def test_global_path_dot_gemini_antigravity_skills():
    assert GLOBAL_REL.startswith(".gemini/antigravity/skills/")
    assert GLOBAL_REL.endswith("/SKILL.md")


def test_skill_md_frontmatter_has_name_and_description(tmp_path, monkeypatch,
                                                       fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    text = adapter.target_path.read_text(encoding="utf-8")
    assert text.startswith("---\n")
    end = text.index("\n---\n", 4)
    block = text[4:end]
    keys = set()
    for line in block.splitlines():
        if ":" in line:
            keys.add(line.split(":", 1)[0].strip())
    assert keys == {"name", "description"}


def test_size_cap_4kb_enforced(tmp_path, monkeypatch):
    def recall(q, limit, pid):
        if "topics" in q:
            return [{"name": "t_" + "x" * 800, "score": 0.9} for _ in range(50)]
        if "entities" in q:
            return [{"name": "e_" + "y" * 800, "mentions": 5} for _ in range(50)]
        if "decisions" in q:
            return [{"text": "d_" + "z" * 800} for _ in range(50)]
        if "memories" in q:
            return [{"text": "m_" + "w" * 800} for _ in range(50)]
        return []
    adapter = _make_adapter(tmp_path, recall=recall, monkeypatch=monkeypatch)
    adapter.sync()
    assert len(adapter.target_path.read_bytes()) <= HARD_BYTES_CAP


def test_content_hash_skip(tmp_path, monkeypatch, fake_recall):
    from superlocalmemory.hooks import context_payload as cp
    monkeypatch.setattr(cp, "_now_iso", lambda: "2026-04-18T00:00:00+00:00")
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    assert adapter.sync() is True
    assert adapter.sync() is False


def test_disable_removes_file(tmp_path, monkeypatch, fake_recall):
    adapter = _make_adapter(tmp_path, recall=fake_recall, monkeypatch=monkeypatch)
    adapter.sync()
    assert adapter.target_path.exists()
    adapter.disable()
    assert not adapter.target_path.exists()
    assert adapter.is_active() is False


def test_invalid_scope(tmp_path, fake_recall):
    with pytest.raises(ValueError):
        AntigravityAdapter(scope="bogus", base_dir=tmp_path,
                           sync_log_db=tmp_path / "memory.db",
                           recall_fn=fake_recall)


def test_inactive_without_gemini_dir(tmp_path, monkeypatch, fake_recall):
    fake_home = tmp_path / "home"
    fake_home.mkdir()
    monkeypatch.setattr(Path, "home", classmethod(lambda cls: fake_home))
    adapter = AntigravityAdapter(
        scope="workspace", base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    assert adapter.is_active() is False


def test_global_scope_writes(tmp_path, monkeypatch, fake_recall):
    home = tmp_path / "home"
    home.mkdir()
    monkeypatch.setenv("SLM_ANTIGRAVITY_FORCE", "1")
    adapter = AntigravityAdapter(
        scope="global", base_dir=home,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=fake_recall,
    )
    assert adapter.sync() is True
    assert adapter.target_path.exists()


def test_code_never_contains_banned_plural_path():
    """Strip the module-level docstring, then ensure banned tokens are absent
    from code / string literals. Tokens in the top-level docstring are
    permitted because they document the rule — they never execute."""
    import ast

    from superlocalmemory.hooks import antigravity_adapter as mod
    src = Path(mod.__file__).read_text(encoding="utf-8")
    tree = ast.parse(src)
    # Drop the module docstring node if present.
    if (tree.body and isinstance(tree.body[0], ast.Expr)
            and isinstance(tree.body[0].value, ast.Constant)
            and isinstance(tree.body[0].value.value, str)):
        tree.body = tree.body[1:]
    code_without_docstring = ast.unparse(tree)
    # Also drop inline-comment lines (they're "#"-prefixed — ast already
    # strips them, but keep defence in depth).
    for banned in (".agents/knowledge", ".agents/skills",
                   ".antigravity/knowledge"):
        assert banned not in code_without_docstring, (
            f"banned token {banned!r} appears in executable code"
        )


def test_render_antigravity_returns_bytes():
    from tests.test_adapters.conftest import make_payload
    assert isinstance(render_antigravity(make_payload()), bytes)
