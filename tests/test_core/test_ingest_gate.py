# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — F-4 Ingest Gate tests

"""F-4: Ingest gate — remember write quality.

TDD suite. Tests for:
- Content > 24000 chars: fact content = head 70% + marker + tail 30%;
  full original preserved in memories. (24K threshold protects normal dense
  session-handoff memories; only pathological pastes are clamped.)
- Content > 1MB: reject (MCP success=False, HTTP 413)
- Prompt-template firewall at ingest
- StoreConfig fields: max_verbatim_chars, max_ingest_bytes
- Kill switch SLM_INGEST_NO_GATE=1
"""

from __future__ import annotations

import os
import pytest

from superlocalmemory.core.config import StoreConfig
from superlocalmemory.core.ingest_gate import (
    apply_ingest_gate,
    IngestGateResult,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _content(n: int) -> str:
    """Generate content of n chars."""
    chunk = "the quick brown fox jumps over the lazy dog "
    return (chunk * (n // len(chunk) + 1))[:n]


# ---------------------------------------------------------------------------
# F-4-A: Config fields on StoreConfig
# ---------------------------------------------------------------------------

class TestStoreConfig:
    """StoreConfig field existence and defaults."""

    def test_store_config_has_max_verbatim_chars(self) -> None:
        """StoreConfig.max_verbatim_chars defaults to 24000 (v3.6.6)."""
        cfg = StoreConfig()
        assert hasattr(cfg, "max_verbatim_chars")
        assert cfg.max_verbatim_chars == 24000

    def test_store_config_has_max_ingest_bytes(self) -> None:
        """StoreConfig.max_ingest_bytes defaults to 1_048_576 (1MB)."""
        cfg = StoreConfig()
        assert hasattr(cfg, "max_ingest_bytes")
        assert cfg.max_ingest_bytes == 1_048_576

    def test_store_config_importable(self) -> None:
        """StoreConfig is importable from superlocalmemory.core.config."""
        from superlocalmemory.core.config import StoreConfig as SC
        assert SC is not None


# ---------------------------------------------------------------------------
# F-4-B: Oversized fact content → head-summary slice
# ---------------------------------------------------------------------------

class TestHeadSummarySlice:
    """Fact content clamped to head 70% + marker + tail 30% for >24000 chars.

    The 24K threshold (≈6K tokens) ensures normal dense memories (6-15K
    session handoffs) pass intact; only pathological pastes are clamped.
    Head+tail preserves OPEN ITEMS that session-close facts put at the end.
    """

    def test_short_content_unchanged(self) -> None:
        """Content ≤ 24000 chars passes through unchanged."""
        content = _content(100)
        result = apply_ingest_gate(content)
        assert result.fact_content == content
        assert result.rejected is False
        assert result.truncated is False

    def test_dense_memory_unchanged(self) -> None:
        """A 15K dense session-handoff memory is stored 100% intact."""
        content = _content(15000)
        result = apply_ingest_gate(content)
        assert result.fact_content == content
        assert result.truncated is False

    def test_exactly_at_limit_unchanged(self) -> None:
        """Content of exactly 24000 chars is unchanged."""
        content = _content(24000)
        result = apply_ingest_gate(content)
        assert result.fact_content == content
        assert result.truncated is False

    def test_over_limit_head_and_tail_preserved(self) -> None:
        """Content > 24000 chars: fact_content keeps head AND tail."""
        content = _content(40000)
        result = apply_ingest_gate(content)
        assert result.truncated is True
        # Head preserved
        assert result.fact_content.startswith(content[:1000])
        # Tail preserved (OPEN ITEMS at the end survive)
        assert result.fact_content.endswith(content[-1000:])
        # Marker present
        assert "content truncated at ingest" in result.fact_content.lower()

    def test_over_limit_fact_content_near_budget(self) -> None:
        """Clamped fact_content is ≤ budget+marker and < original length."""
        content = _content(60000)
        result = apply_ingest_gate(content)
        assert len(result.fact_content) <= 24000 + 80  # budget + marker slack
        assert len(result.fact_content) < len(content)

    def test_over_limit_full_content_preserved(self) -> None:
        """Full original content preserved in result.full_content."""
        content = _content(40000)
        result = apply_ingest_gate(content)
        assert result.full_content == content

    def test_over_limit_marker_present(self) -> None:
        """Clamped content has truncation marker."""
        content = _content(40000)
        result = apply_ingest_gate(content)
        assert "content truncated at ingest" in result.fact_content.lower()


# ---------------------------------------------------------------------------
# F-4-C: 1MB rejection
# ---------------------------------------------------------------------------

class TestMegabyteRejection:
    """Content > 1MB is rejected."""

    def test_under_1mb_not_rejected(self) -> None:
        """Content under 1MB is not rejected."""
        content = _content(500_000)
        result = apply_ingest_gate(content)
        assert result.rejected is False

    def test_over_1mb_rejected(self) -> None:
        """Content over 1MB (by bytes) is rejected."""
        # Generate content > 1MB
        content = "a" * (1_048_576 + 1)
        result = apply_ingest_gate(content)
        assert result.rejected is True

    def test_over_1mb_rejection_has_reason(self) -> None:
        """Rejection includes a reason string."""
        content = "a" * (1_048_576 + 1)
        result = apply_ingest_gate(content)
        assert result.rejection_reason is not None
        assert len(result.rejection_reason) > 0


# ---------------------------------------------------------------------------
# F-4-D: Prompt-template firewall at ingest
# ---------------------------------------------------------------------------

class TestPromptTemplateFirewall:
    """Template content rejected at ingest."""

    def test_summarizing_session_rejected(self) -> None:
        """'You are summarizing a Claude Code session' is rejected as low-quality."""
        content = "You are summarizing a Claude Code session. Output the key facts."
        result = apply_ingest_gate(content)
        assert result.rejected is True
        assert "low" in result.rejection_reason.lower() or "template" in result.rejection_reason.lower()

    def test_memory_consolidation_rejected(self) -> None:
        """'You are a memory consolidation agent' is rejected."""
        content = "You are a memory consolidation agent. Apply deduplication."
        result = apply_ingest_gate(content)
        assert result.rejected is True

    def test_apply_maximum_compression_rejected(self) -> None:
        """'Apply maximum non-destructive compression' is rejected."""
        content = "Apply maximum non-destructive compression to the following memories."
        result = apply_ingest_gate(content)
        assert result.rejected is True

    def test_task_notification_rejected(self) -> None:
        """'<task-notification>' content is rejected."""
        content = "<task-notification>Store memory context</task-notification>"
        result = apply_ingest_gate(content)
        assert result.rejected is True

    def test_normal_content_not_rejected(self) -> None:
        """Normal memory content is NOT rejected by template firewall."""
        content = "Varun prefers TDD for all Python code."
        result = apply_ingest_gate(content)
        assert result.rejected is False


# ---------------------------------------------------------------------------
# F-4-E: Kill switch SLM_INGEST_NO_GATE=1
# ---------------------------------------------------------------------------

class TestIngestGateKillSwitch:
    """Kill switch tests."""

    def test_kill_switch_bypasses_template_rejection(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SLM_INGEST_NO_GATE=1 bypasses template rejection."""
        monkeypatch.setenv("SLM_INGEST_NO_GATE", "1")
        content = "You are summarizing a Claude Code session."
        result = apply_ingest_gate(content)
        assert result.rejected is False

    def test_kill_switch_bypasses_size_clamp(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SLM_INGEST_NO_GATE=1 bypasses the 24K head+tail fact clamp."""
        monkeypatch.setenv("SLM_INGEST_NO_GATE", "1")
        content = _content(40000)
        result = apply_ingest_gate(content)
        assert result.truncated is False
        assert result.fact_content == content

    def test_kill_switch_does_not_bypass_1mb_rejection(
        self, monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        """SLM_INGEST_NO_GATE=1 still rejects content > 1MB (safety)."""
        monkeypatch.setenv("SLM_INGEST_NO_GATE", "1")
        content = "a" * (1_048_576 + 1)
        result = apply_ingest_gate(content)
        # 1MB rejection is a hard safety cap that kill switch does NOT bypass
        assert result.rejected is True
