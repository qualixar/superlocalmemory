"""Regression test: BoundaryStore._cache LRU cap (WP-B memory leak fix).

TDD RED-first. Verifies:
  1. Saving 50001 distinct ids caps cache at 50000 (oldest evicted).
  2. Newest entry (id 50000) is still present after eviction.
  3. Re-saving an existing id does NOT grow the cache beyond MAX.
  4. Re-saving an existing id moves it to the most-recent position so it
     survives a subsequent eviction.
"""

from __future__ import annotations

import time
from unittest.mock import MagicMock

import pytest

from superlocalmemory.optimize.cache.boundary_store import (
    BoundaryStore,
    PerItemBoundaryRecord,
    _BOUNDARY_CACHE_MAX,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_store() -> BoundaryStore:
    """Return a BoundaryStore backed by a no-op mock DB."""
    db = MagicMock()
    db.boundary_upsert.return_value = None
    db.boundary_get.return_value = None
    return BoundaryStore(db=db)


def _record(entry_id: str) -> PerItemBoundaryRecord:
    return PerItemBoundaryRecord(
        entry_id=entry_id,
        t_hat=0.95,
        gamma_hat=10.0,
        samples=[],
        last_updated=time.time(),
    )


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

class TestBoundaryCacheCap:

    def test_constant_exported(self) -> None:
        """_BOUNDARY_CACHE_MAX must equal 50000."""
        assert _BOUNDARY_CACHE_MAX == 50000

    def test_cache_capped_at_max(self) -> None:
        """Saving MAX+1 distinct ids caps _cache at exactly MAX."""
        store = _make_store()
        for i in range(_BOUNDARY_CACHE_MAX + 1):
            store.save(_record(f"id-{i}"))
        assert len(store._cache) == _BOUNDARY_CACHE_MAX

    def test_oldest_evicted(self) -> None:
        """The FIRST id inserted is evicted when MAX+1 entries are saved."""
        store = _make_store()
        for i in range(_BOUNDARY_CACHE_MAX + 1):
            store.save(_record(f"id-{i}"))
        # id-0 was inserted first — it must be gone
        assert "id-0" not in store._cache

    def test_newest_present_after_eviction(self) -> None:
        """The LAST id inserted must still be in the cache after eviction."""
        store = _make_store()
        for i in range(_BOUNDARY_CACHE_MAX + 1):
            store.save(_record(f"id-{i}"))
        last_id = f"id-{_BOUNDARY_CACHE_MAX}"
        assert last_id in store._cache

    def test_resave_existing_does_not_grow(self) -> None:
        """Re-saving an already-cached id must not grow the cache."""
        store = _make_store()
        for i in range(_BOUNDARY_CACHE_MAX):
            store.save(_record(f"id-{i}"))
        assert len(store._cache) == _BOUNDARY_CACHE_MAX
        # Re-save id-0 — already present, should NOT grow
        store.save(_record("id-0"))
        assert len(store._cache) == _BOUNDARY_CACHE_MAX

    def test_resave_moves_to_recent(self) -> None:
        """Re-saving id-0 makes it the most-recent, so it survives the next eviction."""
        store = _make_store()
        # Fill to MAX
        for i in range(_BOUNDARY_CACHE_MAX):
            store.save(_record(f"id-{i}"))
        # Touch id-0 — should move to end (most recent)
        store.save(_record("id-0"))
        # Add one more new entry — id-1 is now oldest, not id-0
        store.save(_record(f"id-{_BOUNDARY_CACHE_MAX}"))
        # Cache is still at MAX
        assert len(store._cache) == _BOUNDARY_CACHE_MAX
        # id-0 was re-saved after id-1..MAX-1, so id-1 should be evicted
        assert "id-0" in store._cache
        assert "id-1" not in store._cache

    def test_read_hit_moves_to_recent(self) -> None:
        """get() on a cached entry must promote it so it survives the next eviction."""
        store = _make_store()
        # Fill to MAX
        for i in range(_BOUNDARY_CACHE_MAX):
            store.save(_record(f"id-{i}"))
        # Access id-0 via get() — should move it to most-recent
        _ = store.get("id-0")
        # Add new entry — id-1 should be evicted (oldest), not id-0
        store.save(_record(f"id-{_BOUNDARY_CACHE_MAX}"))
        assert "id-0" in store._cache
        assert "id-1" not in store._cache
