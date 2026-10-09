# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An erasure also empties the local derivation cache, and never fails because of it."""

from __future__ import annotations

import pytest

from superlocalmemory.cache import default_cache, derive_cache_path
from superlocalmemory.cache.factory import _reset_for_tests
from superlocalmemory.compliance.gdpr import GDPRCompliance
from superlocalmemory.storage import schema as real_schema
from superlocalmemory.storage.database import DatabaseManager
from superlocalmemory.storage.models import MemoryRecord

from tests.test_cache.conftest import make_key


@pytest.fixture()
def gdpr(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path / "root"))
    _reset_for_tests()
    db = DatabaseManager(tmp_path / "g.db")
    db.initialize(real_schema)
    db.execute("INSERT OR IGNORE INTO profiles (profile_id, name) VALUES ('u', 'U')")
    db.store_memory(MemoryRecord(memory_id="m1", profile_id="u", content="hello"))
    yield GDPRCompliance(db)
    _reset_for_tests()


def _fill() -> None:
    default_cache().put(make_key(1), b"derived from erased text", kind="text")
    assert derive_cache_path().exists()


def test_profile_erasure_empties_the_cache(gdpr):
    _fill()
    gdpr.forget_profile("u")
    assert not derive_cache_path().exists()
    assert default_cache().get(make_key(1)) is None


def test_entity_erasure_empties_the_cache(gdpr):
    _fill()
    gdpr.forget_entity("nobody", "u")
    assert not derive_cache_path().exists()


def test_a_refused_erasure_keeps_the_cache(gdpr):
    _fill()
    with pytest.raises(ValueError):
        gdpr.forget_profile("default")
    assert derive_cache_path().exists()


def test_a_clear_failure_does_not_fail_the_erasure(gdpr, monkeypatch):
    _fill()
    monkeypatch.setattr("superlocalmemory.cache.clear_derived_cache",
                        lambda reason: (_ for _ in ()).throw(RuntimeError("boom")))
    counts = gdpr.forget_profile("u")
    assert counts["memories"] == 1
