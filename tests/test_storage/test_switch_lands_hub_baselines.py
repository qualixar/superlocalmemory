# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""An embedding switch lands with hub baselines for the NEW space.

Measured on the author's 27k-memory store (4.2.1): after `slm embedder switch`
from nomic-embed-text-v1.5 to embeddinggemma-2, `fact_hub_baseline` held 0 rows
(27,236 before). Two causes, both in the swap:

* the vector swap is `ALTER TABLE ... RENAME COLUMN`, and SQLite rewrites a
  trigger that names a renamed column. `trg_hub_baseline_fact_reembedded`
  ("AFTER UPDATE OF embedding") followed the old vectors to `embedding_next`;
  the runner's background clear (`SET embedding_next = NULL`) then fired it
  once per memory and deleted every baseline;
* nothing measured the new space, so the store waited for the maintenance
  backfill (390 s after start, bounded batches) while every memory was judged
  on an imputed constant fitted to the old model.

Here a switch measures every memory in the target space before the swap, in
the same job, and publishes the rows inside the swap's own transaction.
"""

# ruff: noqa: F811 (pytest fixtures are imported and redeclared as parameters)

from __future__ import annotations

import json
import time

import numpy as np
import pytest

pytest.importorskip("sqlite_vec")

from superlocalmemory.core import embedding_reindex as er  # noqa: E402
from superlocalmemory.core import embedding_reindex_steps as steps  # noqa: E402
from superlocalmemory.retrieval import hub_baseline  # noqa: E402
from superlocalmemory.storage import embedding_spaces as sp  # noqa: E402
from tests.test_storage.test_embedding_reindex_units import (  # noqa: E402,F401
    FakeEmbedder,
    _run,
    _store,
    _target,
    store,
)


@pytest.fixture()
def hub_store(store):
    """The unit-test store plus the table and triggers migration M061 gives a real one."""
    import importlib

    m061 = importlib.import_module("superlocalmemory.storage.migrations.M061_fact_hub_baseline")
    root, db_path = store
    conn = sp.connect(db_path)
    m061.apply(conn)
    assert m061.verify(conn)
    conn.close()
    return root, db_path


def _seed_old_space_baselines(db_path) -> None:
    conn = sp.connect(db_path)
    now = time.time()
    with steps.write_txn(conn, db_path):
        for (fact_id,) in conn.execute("SELECT fact_id FROM atomic_facts").fetchall():
            conn.execute(
                "INSERT INTO fact_hub_baseline (fact_id, profile_id, space, probe_version, "
                "baseline_mean, baseline_topk, computed_at) VALUES (?, 'default', "
                "'old-model::8', ?, 0.5, 0.6, ?)", (fact_id, hub_baseline.PROBE_SET_VERSION, now))
    conn.close()


def _settle_fully(runner, deadline_s: float = 30.0) -> None:
    """Until the job is activated AND the runner's background clear has finished."""
    deadline = time.monotonic() + deadline_s
    while time.monotonic() < deadline:
        conn = sp.connect(runner.db_path)
        try:
            job = runner.status()["job"]
            stats = conn.execute(f"SELECT stats FROM {sp.JOBS} ORDER BY job_id DESC LIMIT 1"
                                 ).fetchone()[0]
            if job["state"] == "activated" and json.loads(stats).get("clear") == "done":
                return
        finally:
            conn.close()
        time.sleep(0.05)
    raise AssertionError(f"job did not settle: {runner.status()}")


def _switch_and_settle(root, db_path, monkeypatch, target, embedder, *, kind="switch"):
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: embedder)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    (runner.request_switch(target) if kind == "switch" else runner.request_rollback())
    runner.start()
    try:
        _settle_fully(runner)
    finally:
        runner.stop()


def _rows(db_path) -> dict[str, tuple]:
    conn = sp.connect(db_path)
    try:
        return {r["fact_id"]: (r["space"], r["probe_version"], r["baseline_mean"],
                               r["baseline_topk"])
                for r in conn.execute("SELECT * FROM fact_hub_baseline")}
    finally:
        conn.close()


def test_a_switch_leaves_a_baseline_for_every_memory_in_the_new_space(hub_store, monkeypatch):
    root, db_path = hub_store
    _seed_old_space_baselines(db_path)
    new = FakeEmbedder(4, "new")
    _switch_and_settle(root, db_path, monkeypatch, _target(), new)

    rows = _rows(db_path)
    assert len(rows) == 20, "the switch must not leave the store without baselines"
    assert {r[0] for r in rows.values()} == {"new-model::4"}, \
        "every row must be measured in the NEW space"
    assert {r[1] for r in rows.values()} == {hub_baseline.PROBE_SET_VERSION}


def test_the_stored_baselines_are_the_ones_the_new_vectors_and_probes_give(hub_store, monkeypatch):
    root, db_path = hub_store
    new = FakeEmbedder(4, "new")
    _switch_and_settle(root, db_path, monkeypatch, _target(), new)

    probes = hub_baseline._unit_rows(new.embed_batch(list(hub_baseline.probe_texts())))
    conn = sp.connect(db_path)
    live = {r["fact_id"]: np.frombuffer(r["embedding"], dtype=np.float32)
            for r in conn.execute("SELECT fact_id, embedding FROM atomic_facts")}
    conn.close()
    ids = sorted(live)
    expected = hub_baseline.baselines([live[i] for i in ids], probes)
    stored = _rows(db_path)
    for fact_id, (mean, topk) in zip(ids, expected):
        assert stored[fact_id][2] == pytest.approx(mean, abs=1e-5)
        assert stored[fact_id][3] == pytest.approx(topk, abs=1e-5)


def test_the_background_clear_does_not_erase_what_the_swap_published(hub_store, monkeypatch):
    """The regression itself: the re-embed trigger must watch the LIVE vector column."""
    root, db_path = hub_store
    _switch_and_settle(root, db_path, monkeypatch, _target(), FakeEmbedder(4, "new"))
    conn = sp.connect(db_path)
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = "
                       "'trg_hub_baseline_fact_reembedded'").fetchone()[0]
    assert "embedding_next" not in sql, sql
    fact_id = conn.execute("SELECT fact_id FROM atomic_facts LIMIT 1").fetchone()[0]
    # A real re-embed of one memory still invalidates its baseline.
    with steps.write_txn(conn, db_path):
        conn.execute("UPDATE atomic_facts SET embedding = ? WHERE fact_id = ?",
                     (np.zeros(4, dtype=np.float32).tobytes(), fact_id))
    left = {r[0] for r in conn.execute("SELECT fact_id FROM fact_hub_baseline")}
    conn.close()
    assert fact_id not in left and len(left) == 19


def test_a_rollback_lands_baselines_in_the_space_it_returns_to(hub_store, monkeypatch):
    root, db_path = hub_store
    old = FakeEmbedder(8, "old")
    _switch_and_settle(root, db_path, monkeypatch, _target(), FakeEmbedder(4, "new"))
    assert {r[0] for r in _rows(db_path).values()} == {"new-model::4"}
    _switch_and_settle(root, db_path, monkeypatch, None, old, kind="rollback")
    rows = _rows(db_path)
    assert len(rows) == 20 and {r[0] for r in rows.values()} == {"old-model::8"}


def test_a_switch_whose_probes_cannot_be_embedded_still_lands(hub_store, monkeypatch):
    """Baselines are an accuracy aid. A model that cannot embed the probe texts
    must not fail the switch; the backfill measures the space afterwards."""
    root, db_path = hub_store

    class NoProbes(FakeEmbedder):
        def embed_batch(self, texts):
            if len(texts) == len(hub_baseline.probe_texts()):
                raise RuntimeError("probe batch refused")
            return super().embed_batch(texts)

    view = _run(root, db_path, {"new-model": NoProbes(4, "new")}, monkeypatch)
    assert view["state"] == "activated", view
    assert _rows(db_path) == {}


def test_the_rollback_answer_counts_the_vectors_it_can_reuse(hub_store, monkeypatch):
    root, db_path = hub_store
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: FakeEmbedder(8, "old"))
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    assert runner.request_rollback()["reusable"] == 20


def test_a_store_a_4_2_1_switch_already_damaged_is_repaired(hub_store):
    """The trigger left on `embedding_next` by an earlier switch is rebound."""
    from superlocalmemory.storage.database import DatabaseManager

    root, db_path = hub_store
    conn = sp.connect(db_path)
    with steps.write_txn(conn, db_path):
        conn.execute("ALTER TABLE atomic_facts ADD COLUMN embedding_next BLOB")
        conn.execute("DROP TRIGGER IF EXISTS trg_hub_baseline_fact_reembedded")
        conn.execute("CREATE TRIGGER trg_hub_baseline_fact_reembedded AFTER UPDATE OF "
                     "embedding_next ON atomic_facts BEGIN DELETE FROM fact_hub_baseline "
                     "WHERE fact_id = NEW.fact_id; END")
    conn.close()
    db = DatabaseManager(db_path)
    assert hub_baseline.ensure_reembed_trigger(db) is True
    conn = sp.connect(db_path)
    sql = conn.execute("SELECT sql FROM sqlite_master WHERE name = "
                       "'trg_hub_baseline_fact_reembedded'").fetchone()[0]
    conn.close()
    assert "embedding_next" not in sql
    assert hub_baseline.ensure_reembed_trigger(db) is False  # already right: nothing to do
