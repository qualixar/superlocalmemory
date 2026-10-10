# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Clearing the derivation cache, and the places that must do it."""

import logging
from types import SimpleNamespace

import pytest

from superlocalmemory.cache import (
    clear_derived_cache, default_cache, derive_cache_path, reconcile_redaction_policy,
)
from superlocalmemory.core import derived_cache_policy as policy

from .conftest import make_key


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.delenv("SLM_PII_REDACTION", raising=False)
    return tmp_path


def _fill():
    cache = default_cache()
    cache.put(make_key(1), b"old text", kind="text")
    assert cache.get(make_key(1)) == b"old text"
    return cache


def test_clear_removes_the_file_and_sidecars_and_drops_l1(root):
    cache = _fill()
    path = derive_cache_path()
    for suffix in ("-wal", "-shm"):
        path.with_name(path.name + suffix).write_bytes(b"x")
    assert clear_derived_cache("test") is True
    assert not path.exists()
    assert not path.with_name(path.name + "-wal").exists()
    assert not path.with_name(path.name + "-shm").exists()
    assert cache.get(make_key(1)) is None
    assert default_cache().get(make_key(1)) is None


def test_clear_without_a_file_returns_false_and_creates_nothing(root):
    assert clear_derived_cache("test") is False
    assert list(root.iterdir()) == []


def test_clear_logs_the_reason_but_no_content(root, caplog):
    _fill()
    with caplog.at_level(logging.INFO, logger="superlocalmemory.cache.factory"):
        clear_derived_cache("why not")
    assert "why not" in caplog.text
    assert "old text" not in caplog.text


def test_clear_never_raises(root, monkeypatch, caplog):
    _fill()
    monkeypatch.setattr("superlocalmemory.cache.factory.SqliteDeriveCache.clear",
                        lambda self: (_ for _ in ()).throw(OSError("locked")))
    with caplog.at_level(logging.WARNING):
        assert clear_derived_cache("test") is False
    assert "could not be cleared" in caplog.text


def test_a_changed_redaction_setting_clears_an_existing_cache(root):
    assert reconcile_redaction_policy(False) is False  # no file: nothing happens
    assert list(root.iterdir()) == []
    _fill()  # created under "off"
    assert reconcile_redaction_policy(False) is False
    assert derive_cache_path().exists()
    assert reconcile_redaction_policy(True) is True
    assert not derive_cache_path().exists()


def test_a_cache_created_after_the_check_remembers_the_setting(root):
    reconcile_redaction_policy(True)
    _fill()
    assert reconcile_redaction_policy(True) is False
    assert derive_cache_path().exists()
    assert reconcile_redaction_policy(False) is True


def test_a_legacy_cache_with_no_record_is_cleared_when_redaction_is_on(root):
    _fill()
    import sqlite3
    with sqlite3.connect(str(derive_cache_path())) as conn:
        conn.execute("DELETE FROM cache_meta WHERE key='redaction_policy'")
    assert reconcile_redaction_policy(True) is True


def test_engine_startup_check_uses_config_and_env(root, monkeypatch):
    _fill()
    assert policy.sync_cache_with_redaction(SimpleNamespace(pii_redaction=False)) is False
    monkeypatch.setenv("SLM_PII_REDACTION", "on")
    assert policy.sync_cache_with_redaction(SimpleNamespace(pii_redaction=False)) is True
    assert not derive_cache_path().exists()


def test_engine_initialize_runs_the_startup_check(monkeypatch, tmp_path):
    from superlocalmemory.core.config import SLMConfig
    from superlocalmemory.core.engine import MemoryEngine

    seen = []
    monkeypatch.setattr(policy, "sync_cache_with_redaction", lambda cfg: seen.append(cfg))
    config = SLMConfig.for_mode(__import__("superlocalmemory.storage.models", fromlist=["Mode"]).Mode.A,
                                base_dir=tmp_path)
    engine = MemoryEngine(config)
    try:
        engine.initialize()
    finally:
        getattr(engine, "close", lambda: None)()
    assert seen == [config]


def test_the_erasure_methods_are_wrapped():
    from superlocalmemory.compliance.gdpr import GDPRCompliance

    for name in ("forget_profile", "forget_entity"):
        assert hasattr(getattr(GDPRCompliance, name), "__wrapped__")


def test_the_wrapper_clears_after_a_normal_return_only(root, monkeypatch):
    calls = []
    monkeypatch.setattr(policy, "clear_cache_after_erasure", lambda: calls.append(1))
    assert policy.clears_derived_cache(lambda: {"facts": 1})() == {"facts": 1}
    assert calls == [1]
    assert policy.clears_derived_cache(lambda: {"erasure_aborted": 1})() == {"erasure_aborted": 1}
    assert calls == [1]
    with pytest.raises(ValueError):
        policy.clears_derived_cache(lambda: (_ for _ in ()).throw(ValueError("x")))()
    assert calls == [1]


def test_a_clear_failure_does_not_raise(root, monkeypatch):
    _fill()
    monkeypatch.setattr("superlocalmemory.cache.clear_derived_cache",
                        lambda reason: (_ for _ in ()).throw(RuntimeError("boom")))
    assert policy.clear_cache_after_erasure() is False
    assert policy.clears_derived_cache(lambda: {"facts": 3})() == {"facts": 3}
