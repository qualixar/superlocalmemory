# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later

"""End-to-end integration test for the async /remember pipeline.

This test catches the class of bugs that cost 18 production memories
between April 15-26, 2026:

1. Materializer can't see the engine (the `_engine not defined` bug)
2. Embedder returns None on transient failure → NoneType iterable propagates
   up and causes mark_failed → memory permanently lost
3. mark_failed gives up too quickly without retry

Production user flow that must NOT regress:
  POST /remember (async, default mode)
    → store_pending writes to pending.db (status='pending')
    → background materializer reads _engine
    → engine.store() succeeds (or retries on transient errors)
    → mark_done updates pending.db
    → memory appears in memories table
    → recall returns the content

If this test fails, real users are losing data.
"""

from __future__ import annotations

import json
import os
import sqlite3
import time
import urllib.request
from pathlib import Path

import pytest

_DATA_ROOT = Path(os.environ["SLM_DATA_DIR"])
PENDING_DB = _DATA_ROOT / "pending.db"
MEMORY_DB = _DATA_ROOT / "memory.db"
DAEMON_URL = os.environ.get("SLM_TEST_DAEMON_URL", "").rstrip("/")
MATERIALIZER_TIMEOUT_S = 60  # how long we wait for materialization


@pytest.mark.slow
@pytest.mark.skipif(
    not DAEMON_URL,
    reason="requires a fixture-owned SLM_TEST_DAEMON_URL",
)
class TestAsyncRememberE2E:
    """Full pipeline tests — store via async API, verify in DB and recall.

    Marked ``slow`` so the default release test run excludes it
    (pyproject.toml: ``addopts = "-m 'not slow and not ollama and not benchmark'"``).
    Run with ``pytest -m slow`` in CI / pre-release with a warm daemon. The 60s
    materializer poll can transiently fail if the embedding worker is mid-recycle
    or under load — not a release blocker but tracks real materializer health.
    """

    def test_async_remember_persists_to_memory_db(self) -> None:
        """The bug that lost 18 memories: stored returns 'queued' but never
        actually persists. This test catches that regression.
        """
        marker = f"E2E_TEST_{int(time.time() * 1000)}"
        content = f"{marker} — async-pipeline E2E persistence test"

        # Step 1: POST /remember (async — the default, production path)
        req = urllib.request.Request(
            f"{DAEMON_URL}/remember",
            data=json.dumps({"content": content}).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )
        with urllib.request.urlopen(req, timeout=10) as r:
            resp = json.loads(r.read())
        assert resp["ok"] is True, f"remember failed: {resp}"
        assert resp["status"] == "queued", f"unexpected status: {resp}"
        pending_id = resp["pending_id"]

        # Step 2: Wait for materializer to drain
        deadline = time.time() + MATERIALIZER_TIMEOUT_S
        materialized = False
        while time.time() < deadline:
            conn = sqlite3.connect(PENDING_DB, timeout=2)
            row = conn.execute(
                "SELECT status FROM pending_memories WHERE id = ?",
                (pending_id,),
            ).fetchone()
            conn.close()
            if row and row[0] == "done":
                materialized = True
                break
            time.sleep(2)
        assert materialized, (
            f"Memory id={pending_id} stuck in pending after "
            f"{MATERIALIZER_TIMEOUT_S}s — materializer is broken"
        )

        # Step 3: Verify it's actually in memory.db (not just marked done)
        conn = sqlite3.connect(MEMORY_DB, timeout=2)
        rows = conn.execute(
            "SELECT memory_id FROM memories WHERE content LIKE ?",
            (f"%{marker}%",),
        ).fetchall()
        conn.close()
        assert len(rows) >= 1, (
            f"Memory id={pending_id} marked done but NOT in memory.db. "
            "This means the materializer's mark_done is firing but the actual "
            "engine.store() didn't complete — silent data loss."
        )

        # Step 4: Verify it's recallable (the whole point of remembering)
        with urllib.request.urlopen(
            f"{DAEMON_URL}/recall?q={marker}&limit=2",
            timeout=30,
        ) as r:
            recall_resp = json.loads(r.read())
        assert recall_resp["result_count"] >= 1, (
            "Memory persisted but recall returned 0 results — index broken"
        )
        top_result = recall_resp["results"][0]
        assert marker in top_result["content"], (
            f"Recall returned wrong memory: {top_result['content'][:100]}"
        )

    def test_no_engine_not_defined_in_logs(self) -> None:
        """The materializer must not throw NameError on _engine.

        Previously the daemon spammed `materializer loop error: name '_engine'
        is not defined` every 5 seconds, blocking all async materialization.
        """
        log_path = _DATA_ROOT / "logs" / "daemon.log"
        try:
            with open(log_path, "rb") as f:
                # Read last 100 KB only — modern session
                f.seek(0, 2)
                size = f.tell()
                f.seek(max(0, size - 100_000))
                tail = f.read().decode("utf-8", errors="ignore")
        except FileNotFoundError:
            pytest.skip("daemon.log not found")

        bad_lines = [
            line for line in tail.splitlines()
            if "_engine' is not defined" in line
            or "_engine not defined" in line
        ]
        # Allow a few transient errors during startup but not many
        assert len(bad_lines) <= 2, (
            f"Found {len(bad_lines)} '_engine not defined' errors in logs. "
            "The materializer is broken — pending memories will never drain."
        )
