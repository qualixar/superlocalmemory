"""Stage-9 regression: the per-entry caps (WP-A/B) bound depth, but the number
of TENANTS held in memory was unbounded. CentroidStore._centroids and
VCacheSemantic._index must now cap the tenant count and evict lossy-free
(evicted tenants rebuild from the DB on next access).
"""
import numpy as np

from superlocalmemory.optimize.cache.centroid_store import CentroidStore


def test_centroid_store_caps_tenant_count():
    cs = CentroidStore(max_tenants=100)
    for i in range(250):
        cs.update(f"tenant_{i:05d}", np.random.randn(768).astype(np.float32))
    assert len(cs._centroids) == 100
    assert len(cs._counts) == 100
    # newest tenant present, oldest evicted
    assert "tenant_00249" in cs._centroids
    assert "tenant_00000" not in cs._centroids


def test_centroid_store_lru_keeps_recently_updated():
    cs = CentroidStore(max_tenants=10)
    for i in range(10):
        cs.update(f"t{i}", np.random.randn(768).astype(np.float32))
    # touch t0 to make it most-recent
    cs.update("t0", np.random.randn(768).astype(np.float32))
    # add a new tenant -> t1 (now oldest) should be evicted, t0 retained
    cs.update("t_new", np.random.randn(768).astype(np.float32))
    assert "t0" in cs._centroids
    assert "t1" not in cs._centroids
    assert len(cs._centroids) == 10


class _StubDB:
    """No-op DB: _set_inner persists to the DB before touching the index."""

    def get_vector(self, *a, **k):
        return None

    def vec_add(self, *a, **k):
        return None

    def boundary_upsert(self, *a, **k):
        return None

    def boundary_get(self, *a, **k):
        return None

    def get_all_boundaries(self, *a, **k):
        return []

    def get_all_vectors(self, *a, **k):
        return []


class _StubConfig:
    semantic_enabled = False
    semantic_max_index_entries = 10000
    semantic_max_tenants = 50


def test_semantic_index_caps_tenant_shards():
    from superlocalmemory.optimize.cache.semantic import VCacheSemantic

    vs = VCacheSemantic(_StubDB(), _StubConfig())
    for i in range(120):
        vs._set_inner(f"tenant_{i:05d}", f"sem:e{i}", np.random.randn(768).astype(np.float32), "ctx")
    assert len(vs._index) == 50
    # the most-recently-written tenant shard is present
    assert "tenant_00119" in vs._index
    # an early tenant was evicted
    assert "tenant_00000" not in vs._index


def test_config_roundtrips_semantic_max_tenants():
    from superlocalmemory.optimize.config.schema import OptimizeConfig

    cfg = OptimizeConfig.from_dict({"semantic_max_tenants": 777})
    assert cfg.semantic_max_tenants == 777
    assert cfg.as_dict()["semantic_max_tenants"] == 777
    # default when absent
    assert OptimizeConfig.from_dict({}).semantic_max_tenants == 10000
