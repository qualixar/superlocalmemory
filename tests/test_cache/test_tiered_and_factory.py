# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The two-layer cache, the factory, and the compute-once helper."""

import ast
import logging
from pathlib import Path

import pytest

import superlocalmemory.cache as cache_pkg
from superlocalmemory.cache import (
    SqliteDeriveCache, TieredCache, default_cache, derive_cache_path, get_or_compute,
    invalidate_for_model,
)

from .conftest import make_key


@pytest.fixture()
def tiered(tmp_path):
    return TieredCache(SqliteDeriveCache(tmp_path / "derive_cache.db"))


def test_an_l1_hit_does_not_touch_l2(tiered, monkeypatch):
    tiered.put(make_key(1), b"v", kind="text")
    monkeypatch.setattr(tiered.l2, "get", lambda *_a, **_k: pytest.fail("L2 was read"))
    assert tiered.get(make_key(1)) == b"v"


def test_an_l2_hit_is_promoted_to_l1(tiered):
    tiered.l2.put(make_key(1), b"v", kind="json")
    assert tiered.get(make_key(1)) == b"v"
    calls = []
    real = tiered.l2.get
    tiered.l2.get = lambda *a, **k: calls.append(1) or real(*a, **k)
    assert tiered.get(make_key(1)) == b"v"
    assert calls == []


def test_invalidate_clears_both_layers(tiered):
    tiered.put(make_key(1, model="m"), b"v", kind="text")
    tiered.put(make_key(2, model="o"), b"w", kind="text")
    assert tiered.invalidate(model_id="m") == 1
    assert tiered.get(make_key(1, model="m")) is None
    assert tiered.get(make_key(2, model="o")) == b"w"


def test_tiered_stats_shape(tiered):
    tiered.put(make_key(1), b"v", kind="text")
    tiered.get(make_key(1))
    tiered.get(make_key(2))
    s = tiered.stats()
    assert s["backend"] == "tiered"
    assert s["hits"] == 1 and s["misses"] == 1
    assert s["l2"]["backend"] == "sqlite" and "l1" in s


def test_a_second_index_costs_zero_recomputes(tiered):
    calls = []

    def fn():
        calls.append(1)
        return b"derived"

    assert get_or_compute(tiered, make_key(1, model="m1"), "text", fn) == b"derived"
    assert get_or_compute(tiered, make_key(1, model="m1"), "text", fn) == b"derived"
    assert len(calls) == 1
    get_or_compute(tiered, make_key(1, model="m2"), "text", fn)
    assert len(calls) == 2


def test_get_or_compute_survives_a_broken_cache():
    class Broken:
        def get(self, *_a, **_k):
            raise RuntimeError("boom")

        def put(self, *_a, **_k):
            raise RuntimeError("boom")

    assert get_or_compute(Broken(), make_key(1), "text", lambda: b"x") == b"x"


def test_the_path_is_under_the_data_root(tmp_path):
    assert derive_cache_path(tmp_path) == tmp_path / "derive_cache.db"


def test_the_default_backend_is_tiered_and_cached_per_path(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    first = default_cache()
    assert first.stats()["backend"] == "tiered"
    assert default_cache() is first
    assert not (tmp_path / "derive_cache.db").exists()


def test_the_sqlite_backend_and_an_unknown_one(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("SLM_CACHE_BACKEND", "sqlite")
    assert default_cache().stats()["backend"] == "sqlite"
    from superlocalmemory.cache import factory
    factory._reset_for_tests()
    monkeypatch.setenv("SLM_CACHE_BACKEND", "redis")
    with caplog.at_level(logging.WARNING):
        assert default_cache().stats()["backend"] == "tiered"
    assert "redis" in caplog.text


def test_invalidate_for_model_never_creates_the_file(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    assert invalidate_for_model("old-model") == 0
    assert not (tmp_path / "derive_cache.db").exists()


def test_invalidate_for_model_drops_that_models_rows(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    default_cache().put(make_key(1, model="old-model"), b"v", kind="text")
    default_cache().put(make_key(2, model="keep"), b"v", kind="text")
    assert invalidate_for_model("old-model") == 1
    assert default_cache().get(make_key(2, model="keep")) == b"v"


def test_no_serializer_that_runs_code_is_imported():
    banned = {"pickle", "cPickle", "marshal", "shelve", "dill"}
    for path in Path(cache_pkg.__file__).parent.rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            names = []
            if isinstance(node, ast.Import):
                names = [a.name.split(".")[0] for a in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module.split(".")[0]]
            assert not banned & set(names), (path, names)


def test_backups_never_include_the_derive_cache(tmp_path):
    from superlocalmemory.infra.backup import MANAGED_DATABASES
    assert "derive_cache.db" not in MANAGED_DATABASES


def test_invalidate_deriver_drops_one_kind_and_creates_nothing(tmp_path, monkeypatch):
    from superlocalmemory.cache import factory
    from superlocalmemory.cache.keys import CacheKey
    from superlocalmemory.cache.sqlite_store import SqliteDeriveCache

    assert factory.invalidate_deriver("doc.index", tmp_path) == 0
    assert not (tmp_path / factory.FILE_NAME).exists()
    disk = SqliteDeriveCache(tmp_path / factory.FILE_NAME)
    disk.put(CacheKey("a" * 64, "doc.index", "1"), b"{}", kind="json")
    disk.put(CacheKey("b" * 64, "ocr.auto", "1"), b"{}", kind="json")
    assert factory.invalidate_deriver("doc.index", tmp_path) == 1
    assert disk.get(CacheKey("b" * 64, "ocr.auto", "1")) is not None
