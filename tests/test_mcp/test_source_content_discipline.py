# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — F-3 source_content discipline tests

"""F-3: source_content discipline.

TDD RED suite. Tests for:
- Default: source_content ≤ 280 chars preview
- include_source=True returns full content (unless template)
- Template patterns: source_content dropped entirely
- Shared _PROMPT_TEMPLATE_PATTERNS in core/injection.py
"""

from __future__ import annotations

import pytest

from superlocalmemory.core.injection import (
    _PROMPT_TEMPLATE_PATTERNS,
    is_prompt_template,
)
from superlocalmemory.server.recall_serializer import apply_source_content_discipline


# ---------------------------------------------------------------------------
# F-3-A: _PROMPT_TEMPLATE_PATTERNS lives in injection.py
# ---------------------------------------------------------------------------

class TestPromptTemplatePatterns:
    """_PROMPT_TEMPLATE_PATTERNS constant in injection.py."""

    def test_prompt_template_patterns_exists(self) -> None:
        """_PROMPT_TEMPLATE_PATTERNS is importable from injection.py."""
        assert _PROMPT_TEMPLATE_PATTERNS is not None
        assert len(_PROMPT_TEMPLATE_PATTERNS) >= 4

    def test_summarizing_session_pattern(self) -> None:
        """Matches 'You are summarizing a Claude Code session'."""
        text = "You are summarizing a Claude Code session. Do the following:"
        assert is_prompt_template(text) is True

    def test_memory_consolidation_agent_pattern(self) -> None:
        """Matches 'You are a memory consolidation agent'."""
        text = "You are a memory consolidation agent. Compress these facts."
        assert is_prompt_template(text) is True

    def test_apply_maximum_compression_pattern(self) -> None:
        """Matches 'Apply maximum non-destructive compression'."""
        text = "Apply maximum non-destructive compression to the following:"
        assert is_prompt_template(text) is True

    def test_task_notification_pattern(self) -> None:
        """Matches '<task-notification>' XML tag."""
        text = "<task-notification>New task assigned</task-notification>"
        assert is_prompt_template(text) is True

    def test_normal_memory_not_matched(self) -> None:
        """Normal memory content does NOT match template patterns."""
        text = "Varun prefers Python over Java for backend services."
        assert is_prompt_template(text) is False

    def test_empty_string_not_matched(self) -> None:
        """Empty string is not a template match (just empty)."""
        assert is_prompt_template("") is False


# ---------------------------------------------------------------------------
# F-3-B: source_content preview (default ≤ 280 chars)
# ---------------------------------------------------------------------------

class TestSourceContentDiscipline:
    """source_content field handling."""

    def test_short_source_content_unchanged(self) -> None:
        """Source content under 280 chars returned as-is."""
        result = {"fact_id": "f1", "content": "test", "source_content": "short source"}
        out = apply_source_content_discipline(result)
        assert out["source_content"] == "short source"

    def test_long_source_content_truncated_to_280(self) -> None:
        """Source content over 280 chars is truncated to 280 in default mode."""
        long_src = "x" * 500
        result = {"fact_id": "f1", "content": "test", "source_content": long_src}
        out = apply_source_content_discipline(result)
        assert len(out["source_content"]) <= 280

    def test_include_source_returns_full(self) -> None:
        """include_source=True returns full source_content."""
        long_src = "x" * 500
        result = {"fact_id": "f1", "content": "test", "source_content": long_src}
        out = apply_source_content_discipline(result, include_source=True)
        assert out["source_content"] == long_src

    def test_template_source_dropped(self) -> None:
        """Template source_content is dropped entirely (empty string)."""
        template_src = "You are summarizing a Claude Code session. Please output:"
        result = {"fact_id": "f1", "content": "test", "source_content": template_src}
        out = apply_source_content_discipline(result)
        assert out.get("source_content", "") == ""

    def test_template_source_dropped_even_with_include_source(self) -> None:
        """Template source_content is dropped even when include_source=True."""
        template_src = "You are a memory consolidation agent. Compress these:"
        result = {"fact_id": "f1", "content": "test", "source_content": template_src}
        out = apply_source_content_discipline(result, include_source=True)
        assert out.get("source_content", "") == ""

    def test_apply_maximum_compression_dropped(self) -> None:
        """'Apply maximum non-destructive compression' dropped."""
        template_src = "Apply maximum non-destructive compression to the text."
        result = {"fact_id": "f1", "content": "test", "source_content": template_src}
        out = apply_source_content_discipline(result)
        assert out.get("source_content", "") == ""

    def test_task_notification_dropped(self) -> None:
        """'<task-notification>' dropped from source_content."""
        template_src = "<task-notification>Task: store memories</task-notification>"
        result = {"fact_id": "f1", "content": "test", "source_content": template_src}
        out = apply_source_content_discipline(result)
        assert out.get("source_content", "") == ""

    def test_missing_source_content_unchanged(self) -> None:
        """Result without source_content key is unchanged."""
        result = {"fact_id": "f1", "content": "test"}
        out = apply_source_content_discipline(result)
        assert "source_content" not in out or out.get("source_content") == ""

    def test_empty_source_content_unchanged(self) -> None:
        """Empty source_content stays empty."""
        result = {"fact_id": "f1", "content": "test", "source_content": ""}
        out = apply_source_content_discipline(result)
        assert out.get("source_content") == ""
