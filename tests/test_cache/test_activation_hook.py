# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Switching the embedding model drops the old model's cached derivations."""

import pytest

pytest.importorskip("sqlite_vec")

from tests.test_storage.test_embedding_reindex_units import (  # noqa: E402,F401
    FakeEmbedder, _run, store,
)


def test_the_old_model_is_invalidated_after_a_switch(store, monkeypatch):
    root, db_path = store
    seen = []
    from superlocalmemory.cache import factory
    monkeypatch.setattr(factory, "invalidate_for_model", lambda m: seen.append(m) or 0)
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    assert view["state"] == "activated", view
    assert seen == ["old-model"]


def test_a_cache_failure_never_fails_the_switch(store, monkeypatch):
    root, db_path = store
    from superlocalmemory.cache import factory

    def boom(_m):
        raise RuntimeError("cache exploded")

    monkeypatch.setattr(factory, "invalidate_for_model", boom)
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    assert view["state"] == "activated", view


def test_a_switch_leaves_no_cache_file_behind(store, monkeypatch):
    root, db_path = store
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    assert view["state"] == "activated", view
    assert not (root / "derive_cache.db").exists()
