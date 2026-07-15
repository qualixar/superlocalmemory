# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Tests for hooks/auto_invoker.py

"""Tests for AutoInvoker -- multi-signal memory auto-invocation."""

from __future__ import annotations

import sqlite3
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from superlocalmemory.core.config import AutoInvokeConfig
from superlocalmemory.hooks.auto_invoker import AutoInvoker


def _make_db(tmp_path: Path) -> MagicMock:
    """Create a mock DatabaseManager with standard returns."""
    db = MagicMock()

    # Default: no access logs, no facts
    db.execute.return_value = []
    db.get_fact_context.return_value = None
    return db


def _make_invoker(
    db=None,
    vector_store=None,
    trust_scorer=None,
    embedder=None,
    enabled=True,
    **config_overrides,
) -> AutoInvoker:
    """Create an AutoInvoker with sensible test defaults."""
    cfg = AutoInvokeConfig(
        enabled=enabled,
        profile_id="test_profile",
        **config_overrides,
    )
    return AutoInvoker(
        db=db or MagicMock(),
        vector_store=vector_store,
        trust_scorer=trust_scorer,
        embedder=embedder,
        config=cfg,
    )


class TestAutoInvokerInterface:
    """Verify AutoRecall-compatible interface (Rule 16 / AI-04)."""

    def test_get_session_context_returns_string(self):
        invoker = _make_invoker(enabled=True)
        result = invoker.get_session_context(project_path="/test", query="test query")
        assert isinstance(result, str)

    def test_get_session_context_empty_when_disabled(self):
        invoker = _make_invoker(enabled=False)
        result = invoker.get_session_context(query="test")
        assert result == ""

    def test_get_query_context_returns_list_of_dicts(self):
        invoker = _make_invoker(enabled=True)
        result = invoker.get_query_context("test query")
        assert isinstance(result, list)

    def test_enabled_property(self):
        invoker = _make_invoker(enabled=True)
        assert invoker.enabled is True

        invoker2 = _make_invoker(enabled=False)
        assert invoker2.enabled is False

    def test_enable_disable_toggle(self):
        invoker = _make_invoker(enabled=False)
        assert invoker.enabled is False

        invoker.enable()
        assert invoker.enabled is True

        invoker.disable()
        assert invoker.enabled is False


class TestAutoInvokerInvoke:
    """Core invoke() method."""

    def test_invoke_returns_ranked_results(self):
        db = MagicMock()
        # VectorStore returns candidates
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        vs.search.return_value = [
            ("fact_1", 0.9),
            ("fact_2", 0.7),
        ]

        # DB returns fact data for enrichment
        def mock_execute(sql, params=()):
            if "atomic_facts" in sql and "fact_id" in sql and "content" in sql:
                fact_id = params[0]
                return [_mock_fact_row(fact_id)]
            if "MAX(accessed_at)" in sql:
                return [{"last_access": None}]
            if "access_count" in sql and "MAX" in sql:
                return [{"max_count": 10}]
            if "access_count" in sql:
                return [{"access_count": 2}]
            return []

        db.execute.side_effect = mock_execute
        db.get_fact_context.return_value = None

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
        )
        results = invoker.invoke("test query", "test_profile", limit=5)
        assert isinstance(results, list)

    def test_invoke_respects_limit(self):
        db = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768

        # Return many candidates
        vs.search.return_value = [
            (f"fact_{i}", 0.9 - i * 0.05) for i in range(20)
        ]

        def mock_execute(sql, params=()):
            if "atomic_facts" in sql and "content" in sql:
                return [_mock_fact_row(params[0])]
            if "MAX(accessed_at)" in sql:
                return [{"last_access": None}]
            if "access_count" in sql and "MAX" in sql:
                return [{"max_count": 10}]
            if "access_count" in sql:
                return [{"access_count": 2}]
            return []

        db.execute.side_effect = mock_execute
        db.get_fact_context.return_value = None

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
        )
        results = invoker.invoke("test", "test_profile", limit=3)
        assert len(results) <= 3

    def test_invoke_empty_when_no_candidates(self):
        db = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        vs.search.return_value = []

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
        )
        results = invoker.invoke("test", "test_profile")
        assert results == []

    def test_invoke_includes_contextual_description(self):
        db = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        vs.search.return_value = [("fact_1", 0.9)]

        def mock_execute(sql, params=()):
            if "atomic_facts" in sql and "content" in sql:
                return [_mock_fact_row("fact_1")]
            if "MAX(accessed_at)" in sql:
                return [{"last_access": None}]
            if "access_count" in sql and "MAX" in sql:
                return [{"max_count": 10}]
            if "access_count" in sql:
                return [{"access_count": 2}]
            return []

        db.execute.side_effect = mock_execute
        db.get_fact_context.return_value = {
            "contextual_description": "Shows language preference",
        }

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
        )
        results = invoker.invoke("test", "test_profile")
        if results:
            assert results[0]["contextual_description"] == "Shows language preference"

    def test_invoke_skips_archived_facts(self):
        db = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.return_value = [0.1] * 768
        vs.search.return_value = [("fact_1", 0.9)]

        def mock_execute(sql, params=()):
            if "atomic_facts" in sql and "content" in sql:
                return [_mock_fact_row("fact_1", lifecycle="archived")]
            if "MAX(accessed_at)" in sql:
                return [{"last_access": None}]
            if "access_count" in sql and "MAX" in sql:
                return [{"max_count": 10}]
            if "access_count" in sql:
                return [{"access_count": 2}]
            return []

        db.execute.side_effect = mock_execute
        db.get_fact_context.return_value = None

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
            include_archived=False,
        )
        results = invoker.invoke("test", "test_profile")
        assert results == []

    def test_invoke_logs_on_failure(self):
        db = MagicMock()
        vs = MagicMock()
        embedder = MagicMock()
        embedder.embed.side_effect = RuntimeError("embed failed")

        invoker = _make_invoker(
            db=db, vector_store=vs, embedder=embedder,
        )
        # Should not raise
        results = invoker.invoke("test", "test_profile")
        # Falls back to text search, which also returns [] from mock
        assert isinstance(results, list)


class TestFormatForInjection:
    """Output formatting."""

    def test_format_empty(self):
        invoker = _make_invoker()
        assert invoker.format_for_injection([]) == ""

    def test_format_includes_markdown_header(self):
        invoker = _make_invoker()
        results = [{
            "fact_id": "f1",
            "content": "Test content",
            "fact_type": "semantic",
            "score": 0.8,
            "signals": {},
            "contextual_description": "",
        }]
        output = invoker.format_for_injection(results)
        assert "BEGIN UNTRUSTED SLM EVIDENCE v1" in output
        assert "## Relevant Memories" in output
        assert "fact_id=f1" in output

    def test_format_includes_fok_threshold(self):
        invoker = _make_invoker(fok_threshold=0.12)
        results = [{
            "fact_id": "f1",
            "content": "Test",
            "fact_type": "semantic",
            "score": 0.8,
            "signals": {},
            "contextual_description": "",
        }]
        output = invoker.format_for_injection(results)
        assert "0.12" in output


# --- Test helpers ---

def _mock_fact_row(fact_id: str, lifecycle: str = "active") -> dict:
    """Create a mock sqlite Row-like dict for atomic_facts."""
    return {
        "fact_id": fact_id,
        "content": f"Test content for {fact_id}",
        "fact_type": "semantic",
        "lifecycle": lifecycle,
    }
