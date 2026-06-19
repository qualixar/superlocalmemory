"""Non-destructive write tests for CopilotAdapter marker-bounded merge.

Use cases validated here:

1. **Agent-authored project instructions** — an AI agent writes custom
   guidance into ``.github/copilot-instructions.md`` (e.g. "always use
   our error-handling convention"). SLM must not erase that on its next
   sync; only the SLM-managed block should be replaced.

2. **User-curated project conventions** — a developer maintains a
   hand-written ``## Conventions`` section in the instructions file. SLM
   syncs must leave that prose intact.

3. **Disable must not delete the file** — calling ``adapter.disable()``
   should strip the SLM block but preserve everything else; it must not
   ``unlink()`` the file (which was the upstream behaviour before this fix).

4. **Orphaned start marker safety** — if ``<!-- SLM-START -->`` is present
   but ``<!-- SLM-END -->`` is missing (e.g. a truncated write), SLM must
   refuse to write rather than creating a second, broken block.

5. **Idempotent sync** — syncing twice with identical recall data must not
   change the file content.
"""

from __future__ import annotations

from pathlib import Path

from superlocalmemory.hooks.copilot_adapter import CopilotAdapter, TARGET_REL
from superlocalmemory.hooks.memory_protocol import SLM_MARKER_END, SLM_MARKER_START


def _adapter(tmp_path: Path, recall=None) -> CopilotAdapter:
    (tmp_path / ".github").mkdir(exist_ok=True)
    return CopilotAdapter(
        base_dir=tmp_path,
        sync_log_db=tmp_path / "memory.db",
        recall_fn=recall or (lambda q, l, p: []),
    )


def _target(tmp_path: Path) -> Path:
    return tmp_path / TARGET_REL


# ---------------------------------------------------------------------------
# Use case 1 & 2: user / agent content is preserved
# ---------------------------------------------------------------------------

def test_sync_preserves_existing_user_content(tmp_path):
    """User prose before the SLM block must survive a sync."""
    target = _target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# My Project Rules\n\nAlways use our logging library.\n")

    _adapter(tmp_path).sync()

    result = target.read_text()
    assert "# My Project Rules" in result
    assert "Always use our logging library." in result
    assert SLM_MARKER_START in result


def test_sync_preserves_content_after_slm_block(tmp_path):
    """User prose after the SLM block must also survive a sync."""
    target = _target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(
        f"{SLM_MARKER_START}\nold slm content\n{SLM_MARKER_END}\n"
        "\n## Post-SLM user notes\nDo not delete me.\n"
    )

    _adapter(tmp_path).sync()

    result = target.read_text()
    assert "Do not delete me." in result
    assert SLM_MARKER_START in result


def test_sync_replaces_only_slm_block(tmp_path):
    """The SLM block is replaced; surrounding content is unchanged."""
    target = _target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    user_before = "# Agent Rules\n\nUse snake_case.\n"
    user_after = "\n## Team conventions\nReview all PRs.\n"
    target.write_text(
        user_before
        + f"{SLM_MARKER_START}\nstale slm block\n{SLM_MARKER_END}\n"
        + user_after
    )

    _adapter(tmp_path).sync()

    result = target.read_text()
    assert "Use snake_case." in result
    assert "Review all PRs." in result
    assert "stale slm block" not in result
    assert SLM_MARKER_START in result


# ---------------------------------------------------------------------------
# Use case 3: disable must not delete the file
# ---------------------------------------------------------------------------

def test_disable_strips_slm_block_but_keeps_file(tmp_path):
    """disable() must not unlink() the file — it strips the SLM section only."""
    adapter = _adapter(tmp_path)
    adapter.sync()
    target = _target(tmp_path)
    assert target.exists()

    adapter.disable()

    assert target.exists(), "disable() must not delete the file"
    content = target.read_text()
    assert SLM_MARKER_START not in content


def test_disable_preserves_user_content(tmp_path):
    """User content outside the SLM block must survive disable()."""
    target = _target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text("# My notes\n\nKeep this.\n")

    adapter = _adapter(tmp_path)
    adapter.sync()
    adapter.disable()

    assert "Keep this." in target.read_text()


# ---------------------------------------------------------------------------
# Use case 4: orphaned start marker is handled safely
# ---------------------------------------------------------------------------

def test_sync_refuses_on_orphaned_start_marker(tmp_path):
    """If SLM-START has no matching SLM-END, sync must not write."""
    target = _target(tmp_path)
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(f"# Notes\n\n{SLM_MARKER_START}\nno end marker here\n")
    original = target.read_text()

    result = _adapter(tmp_path).sync()

    assert result is False
    assert target.read_text() == original


# ---------------------------------------------------------------------------
# Use case 5: idempotent sync
# ---------------------------------------------------------------------------

def test_sync_is_idempotent(tmp_path):
    """Two syncs with identical recall data must produce identical file content."""
    adapter = _adapter(tmp_path)
    adapter.sync()
    first = _target(tmp_path).read_text()

    adapter.sync()
    second = _target(tmp_path).read_text()

    assert first == second
