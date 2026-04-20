# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory v3.4.22 — LLD-03 §5.2

"""Per-(profile, stratum) posterior LRU cache for the contextual bandit.

LLD reference: ``.backup/active-brain/lld/LLD-03-contextual-bandit-and-ensemble.md``
Section 5.2.

Key design:
  - Loader runs OUTSIDE the lock so DB reads never serialise across strata.
  - ``get`` is thread-safe; concurrent misses may double-load for the same
    key (acceptable because DB read is idempotent and cheap).
  - ``invalidate`` drops a single key — called on every successful bandit
    ``update`` for that stratum (hard rule B5).
  - Max ~256 (profile, stratum) entries ≈ 80 KB RAM steady state.
"""

from __future__ import annotations

import threading
from typing import Callable

_PosteriorMap = dict[str, tuple[float, float]]
_CacheKey = tuple[str, str]


class _BanditCache:
    """Thread-safe LRU keyed by ``(profile_id, stratum)``.

    Values are ``{arm_id: (alpha, beta)}`` dicts. Entries are loaded on demand
    via the caller-supplied ``loader`` and evicted LRU-style once size > max.
    """

    def __init__(self, max_entries: int = 256) -> None:
        if max_entries < 1:
            raise ValueError("max_entries must be >= 1")
        self._store: dict[_CacheKey, _PosteriorMap] = {}
        self._order: list[_CacheKey] = []
        self._lock = threading.Lock()
        self._max = int(max_entries)

    def get(
        self,
        profile_id: str,
        stratum: str,
        loader: Callable[[str, str], _PosteriorMap],
    ) -> _PosteriorMap:
        """Return the posterior map for the key, loading if absent.

        The loader is called exactly once per cache-miss path; the DB read
        runs outside the lock to keep contention bounded.
        """
        key = (profile_id, stratum)
        with self._lock:
            if key in self._store:
                self._touch_locked(key)
                return self._store[key]

        # Miss — load outside the lock.
        data = loader(profile_id, stratum) or {}

        with self._lock:
            # Another thread may have populated in the meantime.
            if key in self._store:
                self._touch_locked(key)
                return self._store[key]
            self._store[key] = data
            self._order.append(key)
            self._evict_if_needed_locked()
        return data

    def invalidate(self, profile_id: str, stratum: str) -> None:
        """Drop a single (profile, stratum) entry. Safe when absent."""
        key = (profile_id, stratum)
        with self._lock:
            self._store.pop(key, None)
            try:
                self._order.remove(key)
            except ValueError:
                pass

    def clear(self) -> None:
        """Drop all entries — used in tests + daemon shutdown."""
        with self._lock:
            self._store.clear()
            self._order.clear()

    def size(self) -> int:
        """Current entry count — primarily for tests / introspection."""
        with self._lock:
            return len(self._store)

    # ------------------------------------------------------------------
    # Internal (lock-held) helpers
    # ------------------------------------------------------------------

    def _touch_locked(self, key: _CacheKey) -> None:
        try:
            self._order.remove(key)
        except ValueError:  # pragma: no cover — invariant: in store => in order
            pass
        self._order.append(key)

    def _evict_if_needed_locked(self) -> None:
        while len(self._order) > self._max:
            oldest = self._order.pop(0)
            self._store.pop(oldest, None)


# Module-level shared cache instance — the bandit module pulls this via
# ``get_shared_cache`` so tests can clear between runs.
_SHARED: _BanditCache | None = None
_SHARED_LOCK = threading.Lock()


def get_shared_cache(max_entries: int = 256) -> _BanditCache:
    """Return the process-wide bandit cache, creating it on first call."""
    global _SHARED
    with _SHARED_LOCK:
        if _SHARED is None:
            _SHARED = _BanditCache(max_entries=max_entries)
        return _SHARED


def reset_shared_cache() -> None:
    """Drop the shared cache — TEST-ONLY helper."""
    global _SHARED
    with _SHARED_LOCK:
        _SHARED = None


__all__ = ("_BanditCache", "get_shared_cache", "reset_shared_cache")
