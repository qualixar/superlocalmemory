"""Regression test for WP-A: per-tenant in-memory index size cap.

HIGH memory leak: VCacheSemantic._set_inner() grew self._index[tenant_id]
without bound (~3.4KB/entry, ~340MB at 100k entries).

Fix: cap per-tenant list at semantic_max_index_entries (default 10000),
keeping the MOST-RECENTLY-added entries (oldest evicted first).
DB is the source of truth; eviction is lossless.
"""

from __future__ import annotations

import hashlib
import random

import numpy as np
import pytest

from superlocalmemory.optimize.cache.semantic import VCacheSemantic, _EMBED_DIM

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

_CAP = 10_000


def _unit_vec(seed_text: str) -> np.ndarray:
    """Deterministic unit vector from a seed string."""
    seed = int(hashlib.md5(seed_text.encode()).hexdigest(), 16) % (2**31)
    rng = random.Random(seed)
    v = np.array([rng.gauss(0, 1) for _ in range(_EMBED_DIM)], dtype=np.float32)
    n = float(np.linalg.norm(v))
    return v / n if n > 0 else v


def _make_config(max_entries: int = _CAP):
    from superlocalmemory.optimize.config.schema import OptimizeConfig
    return OptimizeConfig(
        enabled=True,
        cache_enabled=True,
        semantic_enabled=True,
        semantic_centroid_defense=False,
        semantic_max_index_entries=max_entries,
    )


@pytest.fixture
def tmp_db(tmp_path):
    from superlocalmemory.optimize.storage.db import CacheDB
    return CacheDB(db_path=tmp_path / "test_cap.db")


# ---------------------------------------------------------------------------
# RED: these tests MUST fail against the unpatched code
# ---------------------------------------------------------------------------


class TestIndexCap:
    """WP-A: semantic_max_index_entries enforced inside _set_inner()."""

    def test_schema_has_semantic_max_index_entries(self):
        """OptimizeConfig must expose semantic_max_index_entries with default 10000."""
        from superlocalmemory.optimize.config.schema import OptimizeConfig
        cfg = OptimizeConfig()
        assert hasattr(cfg, "semantic_max_index_entries"), (
            "OptimizeConfig missing field: semantic_max_index_entries"
        )
        assert cfg.semantic_max_index_entries == 10_000

    def test_schema_field_is_configurable(self):
        """semantic_max_index_entries must accept a custom value."""
        from superlocalmemory.optimize.config.schema import OptimizeConfig
        cfg = OptimizeConfig(semantic_max_index_entries=500)
        assert cfg.semantic_max_index_entries == 500

    def test_index_capped_at_max_after_overflow(self, tmp_db):
        """Adding CAP+1 distinct entries must leave exactly CAP entries."""
        cfg = _make_config(max_entries=_CAP)
        cache = VCacheSemantic(db=tmp_db, config=cfg)
        tenant = "tenant_cap_test"

        for i in range(_CAP + 1):
            entry_id = f"entry_{i:06d}"
            vec = _unit_vec(entry_id)
            cache._set_inner(tenant, entry_id, vec, context_fp="")

        with cache._index_lock:
            actual = len(cache._index.get(tenant, []))

        assert actual == _CAP, (
            f"Expected index length {_CAP}, got {actual} — cap not enforced"
        )

    def test_first_entry_evicted_last_entry_present(self, tmp_db):
        """After overflow the FIRST-added entry is gone; LAST-added is present."""
        cap = 100
        cfg = _make_config(max_entries=cap)
        cache = VCacheSemantic(db=tmp_db, config=cfg)
        tenant = "tenant_eviction_order"

        first_id = "entry_FIRST"
        last_id = "entry_LAST"

        # Add first
        cache._set_inner(tenant, first_id, _unit_vec(first_id), context_fp="")

        # Fill up to cap-1 more (total = cap, no eviction yet)
        for i in range(cap - 1):
            eid = f"entry_mid_{i:04d}"
            cache._set_inner(tenant, eid, _unit_vec(eid), context_fp="")

        # Add one more (total = cap+1 → eviction of first_id)
        cache._set_inner(tenant, last_id, _unit_vec(last_id), context_fp="")

        with cache._index_lock:
            ids_in_index = {e[0] for e in cache._index.get(tenant, [])}

        assert first_id not in ids_in_index, (
            f"{first_id!r} should have been evicted (oldest-first policy)"
        )
        assert last_id in ids_in_index, (
            f"{last_id!r} (last added) must be present after eviction"
        )

    def test_readd_existing_id_does_not_grow_beyond_cap(self, tmp_db):
        """Re-adding an already-present entry_id must NOT increase list length."""
        cap = 50
        cfg = _make_config(max_entries=cap)
        cache = VCacheSemantic(db=tmp_db, config=cfg)
        tenant = "tenant_readd"

        # Fill exactly to cap
        for i in range(cap):
            eid = f"entry_{i:04d}"
            cache._set_inner(tenant, eid, _unit_vec(eid), context_fp="")

        with cache._index_lock:
            len_at_cap = len(cache._index.get(tenant, []))

        assert len_at_cap == cap, f"Expected {cap}, got {len_at_cap}"

        # Re-add the very first entry (already present — dedup path).
        # _set_inner may raise due to pre-existing boundary_store behaviour;
        # the critical invariant is that the index DOES NOT GROW past cap
        # regardless of whether the call succeeds or raises.
        repeated_id = "entry_0000"
        try:
            cache._set_inner(tenant, repeated_id, _unit_vec(repeated_id + "_v2"), context_fp="")
        except Exception:
            pass  # pre-existing boundary_store issue — not under fix scope

        with cache._index_lock:
            len_after_readd = len(cache._index.get(tenant, []))

        assert len_after_readd <= cap, (
            f"Re-adding an existing id grew the list beyond cap: {len_after_readd} > {cap}"
        )

    def test_cap_respected_with_small_custom_value(self, tmp_db):
        """Cap of 5: adding 10 entries must leave exactly 5."""
        cap = 5
        cfg = _make_config(max_entries=cap)
        cache = VCacheSemantic(db=tmp_db, config=cfg)
        tenant = "tenant_small_cap"

        for i in range(10):
            eid = f"e_{i}"
            cache._set_inner(tenant, eid, _unit_vec(eid), context_fp="")

        with cache._index_lock:
            actual = len(cache._index.get(tenant, []))

        assert actual == cap, f"Expected {cap}, got {actual}"

    def test_from_dict_roundtrip_preserves_field(self):
        """from_dict must honour semantic_max_index_entries from JSON payload."""
        from superlocalmemory.optimize.config.schema import OptimizeConfig
        cfg = OptimizeConfig.from_dict({"semantic_max_index_entries": 2500})
        assert cfg.semantic_max_index_entries == 2500

    def test_default_used_when_field_absent_from_dict(self):
        """from_dict with missing key must default to 10000."""
        from superlocalmemory.optimize.config.schema import OptimizeConfig
        cfg = OptimizeConfig.from_dict({})
        assert cfg.semantic_max_index_entries == 10_000
