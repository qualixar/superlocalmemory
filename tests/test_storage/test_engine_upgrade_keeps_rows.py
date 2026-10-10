# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""Upgrading the memory engine never deletes a memory, and one click goes back."""

from __future__ import annotations

import time

import pytest

pytest.importorskip("sqlite_vec")

from superlocalmemory.core import embedding_reindex as er  # noqa: E402
from superlocalmemory.core import embedding_reindex_steps as steps  # noqa: E402
from superlocalmemory.core import engine_upgrade as eu  # noqa: E402
from superlocalmemory.core.config import EmbeddingConfig  # noqa: E402
from superlocalmemory.storage import embedding_spaces as sp  # noqa: E402
from tests.test_storage.test_embedding_reindex_units import FakeEmbedder, store  # noqa: E402,F401

EG2 = "google/embeddinggemma-2"
LIVE = EmbeddingConfig(provider="sentence-transformers", model_name="old-model", dimension=8)


def _rows(db_path):
    conn = sp.connect(db_path)
    try:
        facts = conn.execute("SELECT fact_id, memory_id, profile_id, content FROM atomic_facts "
                             "ORDER BY fact_id").fetchall()
        memories = conn.execute("SELECT memory_id, content FROM memories ORDER BY memory_id").fetchall()
        return [tuple(r) for r in facts], [tuple(r) for r in memories]
    finally:
        conn.close()


def _wait(runner, job_id, seconds=30):
    deadline = time.monotonic() + seconds
    while time.monotonic() < deadline:
        view = runner.status()["job"]
        if view["job_id"] == job_id and view["state"] not in sp.ACTIVE_STATES:
            return view
        time.sleep(0.05)
    raise AssertionError(runner.status())


def test_an_upgrade_keeps_every_row_and_the_previous_space_until_it_is_freed(store, monkeypatch):
    root, db_path = store
    before = _rows(db_path)
    embedders = {EG2: FakeEmbedder(768, "eg2"), "old-model": FakeEmbedder(8, "old")}
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: embedders[cfg.model_name])
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    job = runner.request_switch(eu.upgrade_target(LIVE))
    runner.start()
    try:
        assert _wait(runner, job["job_id"])["state"] == "activated"
        assert _rows(db_path) == before, "the upgrade changed or removed a memory"
        status = runner.status()
        assert status["previous_vectors_kept"] is True and status["previous"] == "old-model::8"
        assert status["live"] == f"{EG2}::768"
        conn = sp.connect(db_path)
        assert conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_MAP}").fetchone()[0] == len(before[0])
        conn.close()

        back = runner.request_rollback()
        assert _wait(runner, back["job_id"])["state"] == "activated"
        assert _rows(db_path) == before, "the rollback changed or removed a memory"
        assert runner.status()["live"] == "old-model::8"
    finally:
        runner.stop()
