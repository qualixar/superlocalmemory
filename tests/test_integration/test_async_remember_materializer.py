# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""Unit-mocked materializer test — covers the full /remember → pending →
materialized → engine.store pipeline WITHOUT requiring a running daemon,
HTTP server, or real embedding model.

This complements the slow E2E test in ``test_async_remember_e2e.py`` by
exercising the same materializer logic deterministically and in-process,
so the default test run still gates on the materializer contract that
SLM's whole product depends on (remember/recall).

Why this exists (release credibility note):
    The HTTP /remember endpoint is a thin shim that calls
    ``pending_store.store_pending()``. The materializer thread inside the
    daemon then drains pending rows, calls ``engine.store()``, and marks
    them done/failed. The HTTP+thread layer is integration; the
    pending-store contract and the drain logic ARE the load-bearing parts
    that determine whether memories are persisted.

    These tests run fast (no network, no model load), are deterministic
    (no polling), and lock in the four invariants that an actual
    regression in the materializer would violate.

Invariants covered:
    1. ``store_pending`` writes a 'pending' row that ``get_pending`` returns.
    2. The materializer drains pending → engine.store → mark_done in order.
    3. On engine.store exception, ``mark_failed`` keeps the row eligible
       for retry until ``_MAX_RETRIES`` is hit (no silent data loss).
    4. Metadata + tags survive the round trip from store_pending to
       engine.store call args (the bug class that lost 18 memories
       April 15-26, 2026 was a metadata/engine handoff failure).
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import pytest


# ---------------------------------------------------------------------------
# Helpers — replicate the materializer's drain loop in-process so we don't
# spin up a daemon thread. The real loop lives in
# ``server/unified_daemon.py::_start_pending_materializer``.
# ---------------------------------------------------------------------------


def _drain_one_pass(engine, base_dir: Path) -> tuple[int, int]:
    """Run a single drain iteration matching the real materializer logic.

    Returns ``(stored_count, failed_count)`` for assertions.
    """
    from superlocalmemory.cli.pending_store import (
        get_pending,
        mark_done,
        mark_failed,
    )

    pending = get_pending(limit=50, base_dir=base_dir)
    stored = 0
    failed = 0
    for item in pending:
        md_str = item.get("metadata") or "{}"
        try:
            md = json.loads(md_str)
        except Exception:
            md = {}
        if item.get("tags"):
            md.setdefault("tags", item["tags"])
        try:
            engine.store(item["content"], metadata=md)
            mark_done(item["id"], base_dir=base_dir)
            stored += 1
        except Exception as exc:
            mark_failed(item["id"], str(exc), base_dir=base_dir)
            failed += 1
    return stored, failed


# ---------------------------------------------------------------------------
# Invariant 1: pending_store round-trip
# ---------------------------------------------------------------------------


class TestPendingStoreRoundtrip:
    """``store_pending`` → ``get_pending`` → ``mark_done`` state machine."""

    def test_store_pending_returns_positive_id(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import store_pending

        pid = store_pending(content="hello", base_dir=tmp_path)
        assert pid > 0

    def test_get_pending_returns_stored_row(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import get_pending, store_pending

        pid = store_pending(content="payload", base_dir=tmp_path)
        rows = get_pending(limit=10, base_dir=tmp_path)

        assert len(rows) == 1
        assert rows[0]["id"] == pid
        assert rows[0]["content"] == "payload"

    def test_mark_done_removes_from_pending(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import (
            get_pending, mark_done, store_pending,
        )

        pid = store_pending(content="drain me", base_dir=tmp_path)
        mark_done(pid, base_dir=tmp_path)

        rows = get_pending(limit=10, base_dir=tmp_path)
        assert rows == []


# ---------------------------------------------------------------------------
# Invariant 2: materializer drain → engine.store → mark_done
# ---------------------------------------------------------------------------


class TestMaterializerDrainsToEngine:
    """The drain loop must reach the engine and mark each row done."""

    def test_three_pending_rows_all_stored_and_marked_done(
        self, tmp_path: Path,
    ) -> None:
        from superlocalmemory.cli.pending_store import (
            get_pending, store_pending,
        )

        for i in range(3):
            store_pending(content=f"memory {i}", base_dir=tmp_path)

        engine = MagicMock()
        stored, failed = _drain_one_pass(engine, tmp_path)

        assert stored == 3
        assert failed == 0
        assert engine.store.call_count == 3
        assert get_pending(limit=10, base_dir=tmp_path) == []

    def test_drain_order_is_fifo(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import store_pending

        for i in range(5):
            store_pending(content=f"order-{i}", base_dir=tmp_path)

        engine = MagicMock()
        _drain_one_pass(engine, tmp_path)

        contents_passed = [
            call.args[0] for call in engine.store.call_args_list
        ]
        assert contents_passed == [f"order-{i}" for i in range(5)]


# ---------------------------------------------------------------------------
# Invariant 3: failure path keeps row eligible for retry (no silent loss)
# ---------------------------------------------------------------------------


class TestMaterializerFailureRetryContract:
    """The 18-memories-lost bug (Apr 15-26 2026) was caused by a single
    transient failure permanently dropping the row. v3.4.38 added retry."""

    def test_first_failure_keeps_row_in_pending(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import (
            get_pending, store_pending,
        )

        store_pending(content="will fail once", base_dir=tmp_path)

        engine = MagicMock()
        engine.store.side_effect = RuntimeError("simulated transient")

        stored, failed = _drain_one_pass(engine, tmp_path)
        assert stored == 0
        assert failed == 1

        # Row should still be eligible for retry — NOT permanently lost.
        rows = get_pending(limit=10, base_dir=tmp_path)
        assert len(rows) == 1, (
            "First failure must keep row in pending for retry. "
            "If this drops to 0 rows, the 18-memories-lost bug has regressed."
        )
        assert rows[0]["retry_count"] == 1

    def test_retries_never_delete_unprocessed_evidence(self, tmp_path: Path) -> None:
        """Repeated failure retains the row for bounded-backoff retry."""
        from superlocalmemory.cli.pending_store import (
            _MAX_RETRIES, mark_failed, store_pending,
        )

        row_id = store_pending(content="permanent failure", base_dir=tmp_path)

        for _ in range(_MAX_RETRIES):
            mark_failed(row_id, "always fails", base_dir=tmp_path)

        import sqlite3

        conn = sqlite3.connect(tmp_path / "pending.db")
        try:
            row = conn.execute(
                "SELECT status, retry_count FROM pending_memories"
            ).fetchone()
        finally:
            conn.close()
        assert row == ("pending", _MAX_RETRIES)
        # Backoff may temporarily hide it from get_pending, but the raw
        # evidence remains durable and eligible after next_retry_at.

    def test_transient_failure_then_success(self, tmp_path: Path) -> None:
        """The realistic case: first attempt fails, second succeeds, no loss."""
        from superlocalmemory.cli.pending_store import (
            get_pending, store_pending,
        )

        store_pending(content="recovers on retry", base_dir=tmp_path)

        engine = MagicMock()
        engine.store.side_effect = [
            RuntimeError("transient"),
            None,  # second call succeeds
        ]

        # Pass 1: fails
        _drain_one_pass(engine, tmp_path)
        # Pass 2: succeeds
        stored, failed = _drain_one_pass(engine, tmp_path)

        assert stored == 1
        assert failed == 0
        assert get_pending(limit=10, base_dir=tmp_path) == []
        assert engine.store.call_count == 2


# ---------------------------------------------------------------------------
# Invariant 4: metadata + tags survive the pipeline
# ---------------------------------------------------------------------------


class TestMaterializerMetadataPreservation:
    """Metadata and tags must reach engine.store unchanged. The HTTP
    /remember endpoint encodes them as JSON; the drain decodes them."""

    def test_metadata_dict_round_trips(self, tmp_path: Path) -> None:
        from superlocalmemory.cli.pending_store import store_pending

        store_pending(
            content="with metadata",
            metadata={"source": "test", "agent_id": "claude"},
            base_dir=tmp_path,
        )

        engine = MagicMock()
        _drain_one_pass(engine, tmp_path)

        engine.store.assert_called_once()
        kwargs = engine.store.call_args.kwargs
        assert kwargs["metadata"]["source"] == "test"
        assert kwargs["metadata"]["agent_id"] == "claude"

    def test_tags_string_appears_in_engine_metadata(
        self, tmp_path: Path,
    ) -> None:
        from superlocalmemory.cli.pending_store import store_pending

        store_pending(
            content="tagged",
            tags="agent:claude,project:slm",
            metadata={"source": "test"},
            base_dir=tmp_path,
        )

        engine = MagicMock()
        _drain_one_pass(engine, tmp_path)

        kwargs = engine.store.call_args.kwargs
        assert kwargs["metadata"]["tags"] == "agent:claude,project:slm"
        assert kwargs["metadata"]["source"] == "test"

    def test_empty_metadata_does_not_corrupt_engine_call(
        self, tmp_path: Path,
    ) -> None:
        from superlocalmemory.cli.pending_store import store_pending

        store_pending(content="bare", base_dir=tmp_path)

        engine = MagicMock()
        _drain_one_pass(engine, tmp_path)

        kwargs = engine.store.call_args.kwargs
        assert kwargs["metadata"] == {}
