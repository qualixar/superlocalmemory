# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 — F-5 Core Block Hygiene tests

"""F-5: Core-block hygiene.

TDD RED suite. Tests for:
- dedupe_block_content: normalized-line dedup within block
- filter_low_quality_facts: drop is_low_quality facts before compiling
- per-block char cap of 2000 chars
- Daily recompile registered in MaintenanceScheduler._run()
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.block_hygiene import (
    dedupe_block_content,
    filter_low_quality_block_facts,
    compile_block_content,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_fact_dicts(*contents: str) -> list[dict]:
    return [{"fact_id": f"f{i}", "content": c} for i, c in enumerate(contents)]


# ---------------------------------------------------------------------------
# F-5-A: dedupe_block_content
# ---------------------------------------------------------------------------

class TestDedupeBlockContent:
    """Within-block normalized-line deduplication."""

    def test_no_duplicates_unchanged(self) -> None:
        """Lines without duplicates are returned unchanged."""
        lines = ["Alice likes Python.", "Bob uses TypeScript.", "Carol prefers Go."]
        result = dedupe_block_content(lines)
        assert result == lines

    def test_exact_duplicates_removed(self) -> None:
        """Exact duplicate lines are removed (keep first)."""
        lines = ["Alice likes Python.", "Bob uses Go.", "Alice likes Python."]
        result = dedupe_block_content(lines)
        assert result.count("Alice likes Python.") == 1

    def test_case_normalized_duplicates_removed(self) -> None:
        """Case-normalized duplicates are removed."""
        lines = ["Alice likes Python.", "alice likes python.", "Bob uses Go."]
        result = dedupe_block_content(lines)
        # Only one of the Alice lines should remain
        alice_lines = [l for l in result if "alice" in l.lower() and "python" in l.lower()]
        assert len(alice_lines) == 1

    def test_whitespace_normalized_duplicates_removed(self) -> None:
        """Extra-whitespace duplicates are removed."""
        lines = ["Alice  likes   Python.", "Alice likes Python.", "Bob uses Go."]
        result = dedupe_block_content(lines)
        alice_lines = [l for l in result if "alice" in l.lower()]
        assert len(alice_lines) == 1

    def test_empty_lines_removed(self) -> None:
        """Empty and whitespace-only lines are removed."""
        lines = ["Alice likes Python.", "", "   ", "Bob uses Go."]
        result = dedupe_block_content(lines)
        assert "" not in result
        assert "   " not in result

    def test_order_preserved(self) -> None:
        """Non-duplicate lines preserve their original order."""
        lines = ["C", "A", "B"]
        result = dedupe_block_content(lines)
        assert result == ["C", "A", "B"]


# ---------------------------------------------------------------------------
# F-5-B: filter_low_quality_block_facts
# ---------------------------------------------------------------------------

class TestFilterLowQualityBlockFacts:
    """Filter low-quality facts before block compilation."""

    def test_normal_facts_kept(self) -> None:
        """Normal fact content is kept."""
        facts = _make_fact_dicts(
            "Varun prefers Python.",
            "SLM v3.6.5 ships dedup fixes.",
        )
        result = filter_low_quality_block_facts(facts)
        assert len(result) == 2

    def test_no_data_available_filtered(self) -> None:
        """'No data available' facts are filtered out."""
        facts = _make_fact_dicts("No data available.", "Real fact here.")
        result = filter_low_quality_block_facts(facts)
        assert len(result) == 1
        assert result[0]["content"] == "Real fact here."

    def test_not_detected_yet_filtered(self) -> None:
        """'not detected yet' placeholder filtered."""
        facts = _make_fact_dicts(
            "No behavioral patterns detected yet.",
            "Real behavior observed.",
        )
        result = filter_low_quality_block_facts(facts)
        contents = [f["content"] for f in result]
        assert "Real behavior observed." in contents
        assert all("not detected yet" not in c.lower() for c in contents)

    def test_empty_content_filtered(self) -> None:
        """Facts with empty content are filtered."""
        facts = _make_fact_dicts("", "Real fact.")
        result = filter_low_quality_block_facts(facts)
        assert len(result) == 1

    def test_template_content_filtered(self) -> None:
        """Template content filtered at block compilation."""
        facts = _make_fact_dicts(
            "You are summarizing a Claude Code session.",
            "Real memory content.",
        )
        result = filter_low_quality_block_facts(facts)
        contents = [f["content"] for f in result]
        assert "Real memory content." in contents
        assert not any("summarizing" in c for c in contents)


# ---------------------------------------------------------------------------
# F-5-C: compile_block_content — hard cap 2000 chars
# ---------------------------------------------------------------------------

class TestCompileBlockContent:
    """Block content compilation with hard cap."""

    def test_under_cap_returned_fully(self) -> None:
        """Content under 2000 chars cap returned in full."""
        facts = _make_fact_dicts("Short fact A.", "Short fact B.")
        result = compile_block_content(facts, max_chars=2000)
        assert "Short fact A." in result
        assert "Short fact B." in result

    def test_over_cap_truncated(self) -> None:
        """Compiled content over 2000 chars is truncated."""
        # Create facts that sum > 2000 chars
        big_facts = _make_fact_dicts(*[f"fact_{i} " + "x" * 200 for i in range(15)])
        result = compile_block_content(big_facts, max_chars=2000)
        assert len(result) <= 2000

    def test_dedup_applied_in_compile(self) -> None:
        """compile_block_content deduplicates lines."""
        facts = _make_fact_dicts(
            "Alice likes Python.",
            "Alice likes Python.",  # dup
            "Bob uses Go.",
        )
        result = compile_block_content(facts, max_chars=2000)
        # Count occurrences of "Alice likes Python" in result
        count = result.count("Alice likes Python.")
        assert count == 1

    def test_low_quality_filtered_in_compile(self) -> None:
        """compile_block_content filters low-quality facts."""
        facts = _make_fact_dicts(
            "No data available.",
            "Varun prefers TDD.",
        )
        result = compile_block_content(facts, max_chars=2000)
        assert "No data available." not in result
        assert "Varun prefers TDD." in result

    def test_empty_facts_returns_placeholder(self) -> None:
        """All facts filtered → returns placeholder string."""
        facts = _make_fact_dicts("No data available.")
        result = compile_block_content(facts, max_chars=2000)
        # Should return empty string or placeholder
        assert isinstance(result, str)


# ---------------------------------------------------------------------------
# F-5-D: Daily recompile registered in MaintenanceScheduler._run()
# ---------------------------------------------------------------------------

class TestMaintenanceSchedulerRecompile:
    """Daily recompile triggered in MaintenanceScheduler."""

    def test_run_calls_compile_core_blocks(self) -> None:
        """MaintenanceScheduler._run() calls consolidation_engine.compile_core_blocks_mode_a."""
        from superlocalmemory.core.maintenance_scheduler import MaintenanceScheduler
        from superlocalmemory.core.config import SLMConfig
        from superlocalmemory.storage.models import Mode

        db = MagicMock()
        db.db_path = "/tmp/test.db"
        db.execute.return_value = []
        config = SLMConfig.for_mode(Mode.A)
        scheduler = MaintenanceScheduler(db, config, "default")
        # _run() guards on self._running; put the scheduler in running state so
        # the maintenance body (incl. the F-5 recompile hook) actually executes.
        scheduler._running = True

        compile_called = []

        # _run() does a local `from superlocalmemory.core.block_hygiene import
        # _recompile_core_blocks`, so patch it at its source module.
        with patch(
            "superlocalmemory.core.block_hygiene._recompile_core_blocks",
            side_effect=lambda *a, **kw: compile_called.append(True),
        ):
            # Run the maintenance task directly (no scheduling)
            try:
                scheduler._run()
            except Exception:
                pass  # Other parts may fail in test env

        # NOTE: the test verifies the hook point is called;
        # if the function is not yet there this will fail as RED
        assert compile_called, (
            "MaintenanceScheduler._run() must call _recompile_core_blocks"
        )
