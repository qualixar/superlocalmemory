# tests/optimize/compress/test_ccr_proxy_correctness.py
# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later
#
# WP-10: CCR proxy correctness tests.
# RED phase: tests written before implementation.
# Covers D6 (store-after-success) and D5-B (disable-lossy-on-proxy).

"""CCR proxy correctness tests for WP-10.

Tests:
    test_d6_no_reduction_stores_zero_ccr_rows
    test_d6_reduction_stores_exactly_one_row
    test_d5b_proxy_path_never_lossy
    test_ccr_delete_idempotent

CRIT-2: ccr_count() uses UNFILTERED SELECT COUNT(*) — NOT the TTL-filtered :646 path.
CRIT-3: Singleton reset + temp db per test to prevent bleed.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterator

import pytest

os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _raw_ccr_count(db) -> int:
    """UNFILTERED count of llmcache_ccr_originals rows.

    CRIT-2: Must NOT reuse db.py:646 TTL-filtered count. A fresh no-expiry row
    has ttl_expires=None, so the TTL filter would return 0 and the D6 orphan
    test would falsely pass.
    """
    rows = db._db.execute("SELECT COUNT(*) AS n FROM llmcache_ccr_originals", ())
    return int(dict(rows[0])["n"]) if rows else 0


# ---------------------------------------------------------------------------
# Fake compressor injection
# ---------------------------------------------------------------------------

class _FakeCompressor:
    """Replaces LLMLinguaCompressor in tests.

    compress(text) → str — the real branch is pragma:no cover (optional dep),
    so we inject this fake to exercise the router logic in normal test runs.
    """

    def __init__(self, *, big_reduction: bool) -> None:
        self._big_reduction = big_reduction

    def compress(self, text: str) -> str:
        if self._big_reduction:
            # Return a much shorter string — guaranteed tokens_after < tokens_before
            words = text.split()
            # Keep 10% of words — ensures word count drop (token estimate is word count)
            kept = max(1, len(words) // 10)
            return " ".join(words[:kept])
        else:
            # Return SAME content — no reduction (same word count)
            return text


# ---------------------------------------------------------------------------
# Fixture: reset singletons + temp db
# ---------------------------------------------------------------------------

class _FakeConfig:
    """Minimal config object that enables Layer 2 (prose compression)."""
    compress_enabled: bool = True
    compress_mode: str = "aggressive"
    compress_prose: bool = True
    compress_protect_recent: int = 0


@pytest.fixture
def isolated_router(tmp_path: Path, monkeypatch):
    """Reset CCRStore._instance + CompressRouter._instance singletons.

    Points CacheDB at a temp sqlite so memory.db is NEVER touched (CRIT-3).
    Injects _FakeConfig so compress_prose=True enables Layer 2 branch.
    Returns (router_instance, tmp_cache_db).
    """
    from superlocalmemory.optimize.compress.ccr import CCRStore
    from superlocalmemory.optimize.compress.router import CompressRouter
    from superlocalmemory.optimize.storage import db as _db_mod
    from superlocalmemory.optimize.storage.db import CacheDB

    # Isolate AES key file to tmp (matches conftest pattern)
    monkeypatch.setattr(_db_mod, "_KEY_FILE", tmp_path / "opt-key.bin")
    tmp_db = CacheDB(tmp_path / "llmcache.db")

    # Reset singletons BEFORE creating new instances
    CCRStore._instance = None
    CompressRouter._instance = None

    router = CompressRouter()

    # Wire CCRStore -> tmp_db
    ccr_store_instance = CCRStore()
    ccr_store_instance._db = tmp_db
    router._ccr_store = ccr_store_instance

    # Inject fake config: compress_prose=True so Layer 2 branch is reachable
    class _FakeConfigStore:
        def get(self):
            return _FakeConfig()

    router._config_store = _FakeConfigStore()

    yield router, tmp_db

    # Cleanup: reset singletons after test
    CCRStore._instance = None
    CompressRouter._instance = None


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_d6_no_reduction_stores_zero_ccr_rows(isolated_router) -> None:
    """D6: when fake compressor returns same word count (no reduction),
    strategy must NOT be 'llmlingua2_prose' AND ccr row count must be unchanged.

    This is the orphan bug: the old store-before-compress left a row even when
    there was no reduction (tokens_after_l2 >= tokens_before).
    """
    router, tmp_db = isolated_router

    # Inject no-reduction fake compressor
    router._llmlingua_compressor = _FakeCompressor(big_reduction=False)

    count_before = _raw_ccr_count(tmp_db)

    # Build a long prose text (>500 chars, not JSON, not code)
    prose = (
        "The quick brown fox jumps over the lazy dog. " * 20
    )
    assert len(prose) > 500

    # Call _compress_text with aggressive=True (triggers Layer 2 branch)
    # is_proxy=False here — we test the no-reduction branch directly
    compressed, tokens_before, tokens_after, strategy = router._compress_text(
        prose, aggressive=True, request_id="test-d6-no-red",
        model="gpt-4o", tenant_id="default",
    )

    count_after = _raw_ccr_count(tmp_db)

    assert strategy != "llmlingua2_prose", (
        f"D6: expected no lossy strategy on no-reduction, got {strategy!r}"
    )
    assert count_after == count_before, (
        f"D6 orphan bug: ccr rows changed from {count_before} to {count_after} "
        "even though no reduction occurred — store-before-compress bug not fixed"
    )


def test_d6_reduction_stores_exactly_one_row(isolated_router) -> None:
    """D6: when fake compressor returns big reduction, ccr_count must be +1."""
    router, tmp_db = isolated_router

    # Inject big-reduction fake compressor
    router._llmlingua_compressor = _FakeCompressor(big_reduction=True)

    count_before = _raw_ccr_count(tmp_db)

    prose = (
        "The quick brown fox jumps over the lazy dog. " * 20
    )
    assert len(prose) > 500

    compressed, tokens_before, tokens_after, strategy = router._compress_text(
        prose, aggressive=True, request_id="test-d6-red",
        model="gpt-4o", tenant_id="default",
    )

    count_after = _raw_ccr_count(tmp_db)

    assert strategy == "llmlingua2_prose", (
        f"D6: expected lossy strategy on reduction, got {strategy!r}"
    )
    assert count_after == count_before + 1, (
        f"D6: expected exactly 1 new ccr row on reduction, "
        f"was {count_before} now {count_after}"
    )


def test_d5b_proxy_path_never_lossy(isolated_router) -> None:
    """D5-B: proxy-flagged call must NOT return lossy strategy and must create no CCR row.

    The proxy path goes through compress(ProxyRequest) which calls _compress_text
    with is_proxy=True. On the proxy path, Layer 2 (LLMLingua) is completely skipped.
    """
    router, tmp_db = isolated_router

    # Inject big-reduction fake compressor — but proxy path should skip Layer 2 entirely
    router._llmlingua_compressor = _FakeCompressor(big_reduction=True)

    count_before = _raw_ccr_count(tmp_db)

    prose = (
        "The quick brown fox jumps over the lazy dog. " * 20
    )
    assert len(prose) > 500

    # is_proxy=True — the D5-B guard must skip Layer 2
    compressed, tokens_before, tokens_after, strategy = router._compress_text(
        prose, aggressive=True, request_id="test-d5b",
        model="gpt-4o", tenant_id="default",
        is_proxy=True,
    )

    count_after = _raw_ccr_count(tmp_db)

    assert strategy != "llmlingua2_prose", (
        f"D5-B: proxy path returned lossy strategy {strategy!r} — "
        "Layer 2 must be skipped on proxy path"
    )
    assert count_after == count_before, (
        f"D5-B: proxy path created {count_after - count_before} CCR row(s) — "
        "no CCR rows should be created on the proxy path"
    )


def test_ccr_delete_idempotent(isolated_router) -> None:
    """ccr_delete: double delete must not raise, retrieve after delete returns None."""
    router, tmp_db = isolated_router

    # Store a row directly via CCRStore
    ccr_store = router._ccr_store
    original = b"some original content to test deletion"
    ccr_id = ccr_store.store(original)
    assert ccr_id, "CCRStore.store returned empty ccr_id"

    # Verify it exists
    assert ccr_store.retrieve(ccr_id) == original

    # First delete — must not raise
    ccr_store.delete(ccr_id)

    # Retrieve after delete — must return None
    assert ccr_store.retrieve(ccr_id) is None, (
        "After delete, retrieve must return None"
    )

    # Second delete (idempotent) — must not raise
    ccr_store.delete(ccr_id)  # no raise

    # Still None
    assert ccr_store.retrieve(ccr_id) is None
