# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file

"""The background re-embed's storage guarantees, on a real schema store.

The daemon-level behaviour (recall served throughout, restart, rollback over
HTTP) is in tests/test_integration/test_embedding_switch_e2e.py. These pin the
pieces that test cannot isolate: vec0 renames move the shadow tables, the
erasure trigger never needs sqlite-vec, a failed job leaves the live space
byte-for-byte, the swap moves all four parts of a space together.
"""

from __future__ import annotations

import hashlib
import json
import sqlite3
import time
from pathlib import Path

import numpy as np
import pytest

sqlite_vec = pytest.importorskip("sqlite_vec")

from tests.helpers.env_capabilities import (
    NO_VECTOR_SEARCH_REASON,
    vector_search_available,
)

# The fixtures build stores with sqlite-vec loaded. An interpreter whose sqlite3
# cannot load extensions (the python.org builds on macOS) has the package but
# cannot use it, which an import check does not see.
pytestmark = pytest.mark.skipif(
    not vector_search_available(), reason=NO_VECTOR_SEARCH_REASON,
)

from superlocalmemory.core import embedding_reindex as er  # noqa: E402
from superlocalmemory.core import embedding_reindex_steps as steps  # noqa: E402
from superlocalmemory.storage import embedding_spaces as sp  # noqa: E402


def _t(rows) -> list[tuple]:
    return [tuple(r) for r in rows]


def _vec(text: str, dim: int, salt: str) -> list[float]:
    raw = hashlib.sha256(f"{salt}:{text}".encode()).digest() * (dim // 32 + 1)
    v = np.frombuffer(raw[:dim], dtype=np.uint8).astype(np.float32) - 127.5
    return (v / np.linalg.norm(v)).tolist()


class FakeEmbedder:
    def __init__(self, dim: int, salt: str, fail_after: int | None = None, hook=None):
        self.dim, self.salt, self.fail_after, self.hook = dim, salt, fail_after, hook
        self.calls = 0
        self.closed = False

    def embed_batch(self, texts):
        self.calls += 1
        if self.hook is not None:
            self.hook(self.calls)
        if self.fail_after is not None and self.calls > self.fail_after:
            raise RuntimeError("model server went away")
        return [_vec(t, self.dim, self.salt) for t in texts]

    def shutdown(self):
        self.closed = True


def _store(root: Path, n: int, dim: int = 8, model: str = "old-model") -> Path:
    from superlocalmemory.storage import schema
    from superlocalmemory.storage.database import DatabaseManager

    db_path = root / "memory.db"
    DatabaseManager(db_path).initialize(schema)
    conn = sp.connect(db_path)
    sp.create_vec(conn, "fact_embeddings", dim)
    for statement in __import__("superlocalmemory.storage.embedding_space_swap",
                                fromlist=["x"])._METADATA_DDL:
        conn.execute(statement)
    conn.execute("INSERT INTO memories (memory_id, content) VALUES ('m1', 'x')")
    for i in range(n):
        _add_fact(conn, f"f{i:03d}", f"fixture fact number {i}", dim, model)
    conn.close()
    cfg = {"mode": "a", "embedding": {"provider": "sentence-transformers", "model_name": model,
                                     "dimension": dim}, "embedding_signature": f"{model}::{dim}"}
    (root / "config.json").write_text(json.dumps(cfg))
    return db_path


def _add_fact(conn, fact_id: str, content: str, dim: int = 8, model: str = "old-model") -> None:
    vec = np.asarray(_vec(content, dim, "old"), dtype=np.float32)
    conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content, embedding) "
                 "VALUES (?, 'm1', 'default', ?, ?)", (fact_id, content, vec.tobytes()))
    rowid = int(conn.execute("SELECT COALESCE(MAX(vec_rowid), 0) + 1 FROM embedding_metadata")
                .fetchone()[0])
    conn.execute("INSERT INTO fact_embeddings(rowid, profile_id, embedding) VALUES (?, ?, ?)",
                 (rowid, "default", vec.tobytes()))
    conn.execute("INSERT INTO embedding_metadata (vec_rowid, fact_id, profile_id, model_name, "
                 "dimension) VALUES (?, ?, 'default', ?, ?)", (rowid, fact_id, model, dim))
    conn.execute("INSERT INTO vector_row_map VALUES (?, 'default', ?)", (fact_id, rowid))


@pytest.fixture()
def store(tmp_path, monkeypatch):
    monkeypatch.setenv("SLM_DATA_DIR", str(tmp_path))
    monkeypatch.setattr(steps, "PAUSE_S", 0.0)
    db_path = _store(tmp_path, 20)
    conn = sp.connect(db_path)
    with steps.write_txn(conn, db_path):
        sp.ensure_control_tables(conn)
        sp.write_space(conn, "old-model::8", {"provider": "sentence-transformers",
                                              "model_name": "old-model", "dimension": 8})
    conn.close()
    return tmp_path, db_path


def _target(model="new-model", dim=4):
    from superlocalmemory.core.config import EmbeddingConfig
    return EmbeddingConfig(provider="sentence-transformers", model_name=model, dimension=dim)


def _run(root, db_path, embedders: dict, monkeypatch, kind="switch") -> dict:
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: embedders[cfg.model_name])
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    job = runner.request_switch(_target()) if kind == "switch" else runner.request_rollback()
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            view = runner.status()["job"]
            if view["job_id"] == job["job_id"] and view["state"] not in sp.ACTIVE_STATES:
                return view
            time.sleep(0.05)
        raise AssertionError(f"job did not finish: {runner.status()}")
    finally:
        runner.stop()


def _live_snapshot(db_path) -> tuple:
    conn = sp.connect(db_path)
    try:
        return (sp.vec_dimension(conn, "fact_embeddings"),
                conn.execute("SELECT COUNT(*) FROM fact_embeddings").fetchone()[0],
                sorted(_t(conn.execute("SELECT fact_id, embedding FROM atomic_facts"))),
                sorted(_t(conn.execute("SELECT vec_rowid, fact_id, model_name, dimension "
                                       "FROM embedding_metadata"))))
    finally:
        conn.close()


def test_a_vec0_rename_moves_its_shadow_tables(tmp_path):
    conn = sp.connect(tmp_path / "v.db")
    sp.create_vec(conn, "a_vec", 2)
    conn.execute("INSERT INTO a_vec(rowid, profile_id, embedding) VALUES (1, 'p', ?)",
                 (np.asarray([1, 0], dtype=np.float32).tobytes(),))
    sp.rename_vec(conn, "a_vec", "b_vec")
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert {"b_vec", *(f"b_vec_{s}" for s in sp.VEC_SHADOWS)} <= names
    assert not any(n.startswith("a_vec") for n in names)
    hit = conn.execute("SELECT rowid FROM b_vec WHERE embedding MATCH ? AND profile_id='p' "
                       "AND k=1", (np.asarray([1, 0], dtype=np.float32).tobytes(),)).fetchall()
    assert _t(hit) == [(1,)]


def test_erasure_trigger_works_without_sqlite_vec_and_purges_vectors(store):
    root, db_path = store
    conn = sp.connect(db_path)
    with steps.write_txn(conn, db_path):
        sp.ensure_side_tables(conn)
        sp.create_vec(conn, sp.NEXT_VEC, 4)
        conn.execute(f"INSERT INTO {sp.NEXT_VEC}(rowid, profile_id, embedding) VALUES (7, "
                     "'default', ?)", (np.zeros(4, dtype=np.float32).tobytes(),))
        conn.execute(f"INSERT INTO {sp.NEXT_MAP} VALUES ('f001', 'default', 7, 'h')")
    plain = sqlite3.connect(db_path)  # no sqlite-vec loaded, like many writers
    plain.execute("DELETE FROM atomic_facts WHERE fact_id = 'f001'")
    plain.commit()
    plain.close()
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.NEXT_MAP}").fetchone()[0] == 0
    assert _t(conn.execute(f"SELECT vec_rowid FROM {sp.PURGE}")) == [(7,)]
    with steps.write_txn(conn, db_path):
        assert sp.purge_pending(conn) == 1
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.NEXT_VEC}").fetchone()[0] == 0
    conn.close()


def test_a_switch_moves_all_four_parts_of_the_space_and_keeps_the_previous(store, monkeypatch):
    root, db_path = store
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    assert view["state"] == "activated", view
    dim, count, facts, meta = _live_snapshot(db_path)
    assert (dim, count) == (4, 20)
    assert all(len(blob) == 16 for _fid, blob in facts)
    assert {m[2:] for m in meta} == {("new-model", 4)}
    conn = sp.connect(db_path)
    fisher = conn.execute("SELECT COUNT(*) FROM atomic_facts WHERE length(fisher_mean) = 16"
                          ).fetchone()[0]
    assert fisher == 20, "Fisher vectors stayed in the old dimension"
    assert sp.vec_dimension(conn, sp.PREV_VEC) == 8
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_MAP}").fetchone()[0] == 20
    row = sp.read_space(conn)
    assert (row["live_signature"], row["prev_signature"]) == ("new-model::4", "old-model::8")
    conn.close()
    saved = json.loads((root / "config.json").read_text())
    assert saved["embedding"]["model_name"] == "new-model"
    assert saved["embedding_signature"] == "new-model::4"


def test_a_failing_model_leaves_the_live_space_untouched(store, monkeypatch):
    root, db_path = store
    before = _live_snapshot(db_path)
    monkeypatch.setattr(steps, "RETRIES", 1)
    monkeypatch.setattr(steps, "BATCH", 5)
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new", fail_after=2)}, monkeypatch)
    assert view["state"] == "failed" and "failed" in view["error"]
    assert _live_snapshot(db_path) == before
    conn = sp.connect(db_path)
    assert not sp.table_exists(conn, sp.NEXT_VEC)
    assert sp.read_space(conn)["live_signature"] == "old-model::8"
    conn.close()
    assert json.loads((root / "config.json").read_text())["embedding"]["model_name"] == "old-model"


def test_a_model_of_the_wrong_width_fails_before_staging(store, monkeypatch):
    root, db_path = store
    view = _run(root, db_path, {"new-model": FakeEmbedder(6, "new")}, monkeypatch)
    assert view["state"] == "failed" and "6-dimensional" in view["error"]


def test_writes_and_edits_during_the_job_are_caught_up(store, monkeypatch):
    root, db_path = store
    monkeypatch.setattr(steps, "BATCH", 5)

    def during(call):
        if call == 2:  # mid-bulk: one new fact, one edit, one erasure
            conn = sqlite3.connect(db_path)
            conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                         "VALUES ('late', 'm1', 'default', 'written during the job')")
            conn.execute("UPDATE atomic_facts SET content = 'edited words' WHERE fact_id='f001'")
            conn.execute("DELETE FROM atomic_facts WHERE fact_id = 'f019'")
            conn.commit()
            conn.close()

    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new", hook=during)}, monkeypatch)
    assert view["state"] == "activated", view
    conn = sp.connect(db_path)
    got = dict(_t(conn.execute("SELECT fact_id, embedding FROM atomic_facts")))
    assert "f019" not in got and len(got) == 20
    for fact_id, text in (("late", "written during the job"), ("f001", "edited words")):
        assert np.allclose(np.frombuffer(got[fact_id], dtype=np.float32), _vec(text, 4, "new"))
    meta = _t(conn.execute("SELECT fact_id FROM embedding_metadata"))
    assert len(meta) == 20 and ("f019",) not in meta
    conn.close()


def test_rollback_copies_unchanged_vectors_and_embeds_the_rest(store, monkeypatch):
    root, db_path = store
    old = FakeEmbedder(8, "old")
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    conn = sp.connect(db_path)
    conn.execute("INSERT INTO atomic_facts (fact_id, memory_id, profile_id, content) "
                 "VALUES ('after', 'm1', 'default', 'saved after the switch')")
    conn.close()
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: old)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    job = runner.request_rollback()
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while runner.status()["job"]["state"] in sp.ACTIVE_STATES and time.monotonic() < deadline:
            time.sleep(0.05)
    finally:
        runner.stop()
    view = runner.status()["job"]
    assert view["job_id"] == job["job_id"] and view["state"] == "activated", view
    assert view["copied"] == 20, "unchanged vectors should be copied, not re-embedded"
    dim, count, facts, meta = _live_snapshot(db_path)
    assert (dim, count) == (8, 21) and {m[2:] for m in meta} == {("old-model", 8)}
    conn = sp.connect(db_path)
    assert sp.read_space(conn)["prev_signature"] == "new-model::4"
    states = dict(_t(conn.execute(f"SELECT kind, state FROM {sp.JOBS}")))
    assert states == {"switch": "rolled_back", "rollback": "activated"}
    conn.close()


def test_rollback_is_refused_when_the_previous_model_is_unavailable(store, monkeypatch):
    root, db_path = store
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: None)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    with pytest.raises(er.Refused, match="previous model old-model is not available"):
        runner.request_rollback()


def test_forget_previous_frees_the_tables_and_the_trigger(store, monkeypatch):
    root, db_path = store
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    assert runner.forget_previous()["freed_vectors"] == 20
    conn = sp.connect(db_path)
    names = {r[0] for r in conn.execute("SELECT name FROM sqlite_master")}
    assert not names & {sp.PREV_VEC, sp.PREV_MAP, sp.NEXT_MAP, sp.PURGE, sp.TRIGGER}
    assert sp.read_space(conn)["prev_signature"] is None
    conn.close()
    with pytest.raises(er.Refused):
        runner.forget_previous()


def test_a_second_switch_is_refused_naming_the_running_one(store, monkeypatch):
    from superlocalmemory.storage.embedding_reindex_jobs import JobConflict

    root, db_path = store
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    first = runner.request_switch(_target())
    with pytest.raises(JobConflict) as caught:
        runner.request_switch(_target("third-model", 6))
    assert caught.value.job["job_id"] == first["job_id"]


def test_an_erased_fact_leaves_no_vector_in_staging_or_previous(store, monkeypatch):
    root, db_path = store
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    plain = sqlite3.connect(db_path)
    plain.execute("DELETE FROM atomic_facts WHERE fact_id = 'f005'")
    plain.commit()
    plain.close()
    conn = sp.connect(db_path)
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_MAP} WHERE fact_id='f005'").fetchone()[0] == 0
    before = conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_VEC}").fetchone()[0]
    with steps.write_txn(conn, db_path):
        sp.purge_pending(conn)
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.PREV_VEC}").fetchone()[0] == before - 1
    conn.close()


def test_the_swap_renames_columns_and_the_runner_clears_what_it_replaced(store, monkeypatch):
    """Two switches: the vector columns are swapped by name, the replaced values
    are cleared afterwards, and the space from two switches ago is dropped."""
    root, db_path = store
    probe = sp.connect(db_path)
    seq_before = probe.execute("SELECT COALESCE(MAX(seq), 0) FROM fact_search_changes").fetchone()[0]
    probe.close()
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    conn = sp.connect(db_path)
    changed = conn.execute("SELECT COUNT(DISTINCT fact_id) FROM fact_search_changes WHERE seq > ? "
                           "AND what = 'v'", (seq_before,)).fetchone()[0]
    assert changed == 20, "caches following the change log were not told every vector changed"
    cols = {r[1] for r in conn.execute("PRAGMA table_info(atomic_facts)")}
    assert {"embedding_next", "fisher_mean_next", "fisher_variance_next"} <= cols
    old_in_twin = conn.execute("SELECT COUNT(*) FROM atomic_facts WHERE length(embedding_next) = 32"
                               ).fetchone()[0]
    assert old_in_twin == 20, "after the swap the twins hold the replaced 8-d vectors"
    trigger = conn.execute("SELECT sql FROM sqlite_master WHERE name = "
                           "'trg_atomic_facts_search_change_update_vector'").fetchone()[0]
    assert "embedding," in trigger and "embedding_next" not in trigger
    conn.close()
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: FakeEmbedder(6, "third"))
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    runner.request_switch(_target("third-model", 6))
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while time.monotonic() < deadline:
            conn = sp.connect(db_path)
            try:
                job = runner.status()["job"]
                stats = conn.execute(f"SELECT stats FROM {sp.JOBS} ORDER BY job_id DESC LIMIT 1"
                                     ).fetchone()[0]
                done = (job["state"] == "activated" and json.loads(stats).get("clear") == "done"
                        and not sp.table_exists(conn, sp.TRASH_VEC))
            finally:
                conn.close()
            if done:
                break
            time.sleep(0.05)
    finally:
        runner.stop()
    conn = sp.connect(db_path)
    assert conn.execute("SELECT COUNT(*) FROM atomic_facts WHERE embedding_next IS NOT NULL OR "
                        "fisher_mean_next IS NOT NULL").fetchone()[0] == 0
    assert not sp.table_exists(conn, sp.TRASH_VEC), "the space from two switches ago was kept"
    assert sp.vec_dimension(conn, sp.PREV_VEC) == 4 and sp.vec_dimension(conn, "fact_embeddings") == 6
    conn.close()


def test_a_row_replaced_mid_job_is_staged_again_not_swapped_in_empty(store, monkeypatch):
    """REPLACE deletes and re-inserts without firing delete triggers: the staged
    map still matches the content, only the empty twin shows the row needs work."""
    root, db_path = store
    monkeypatch.setattr(steps, "BATCH", 5)

    def during(call):
        if call == 3:  # f001 was staged by batch 1; replace its row now
            conn = sqlite3.connect(db_path)
            content = conn.execute("SELECT content FROM atomic_facts WHERE fact_id='f001'"
                                   ).fetchone()[0]
            conn.execute("INSERT OR REPLACE INTO atomic_facts (fact_id, memory_id, profile_id, "
                         "content) VALUES ('f001', 'm1', 'default', ?)", (content,))
            conn.commit()
            conn.close()

    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new", hook=during)}, monkeypatch)
    assert view["state"] == "activated", view
    conn = sp.connect(db_path)
    blob = conn.execute("SELECT embedding FROM atomic_facts WHERE fact_id='f001'").fetchone()[0]
    conn.close()
    assert blob is not None and len(blob) == 16, "a replaced row went live with no vector"


def test_background_work_stays_paused_across_every_activation_try(store, monkeypatch):
    """A backoff between tries must not let a long background save back in."""
    from types import SimpleNamespace

    from superlocalmemory.core import embedding_reindex_activate as act
    from superlocalmemory.server.profile_runtime import ProfileRuntime

    root, db_path = store
    runtime = ProfileRuntime("default")
    runner = er.ReindexRunner(db_path=db_path, data_root=root,
                              app_state=SimpleNamespace(profile_runtime=runtime))
    seen: list[tuple[str, bool]] = []
    tries = iter(["ready", "ready", "activated"])

    def fake_activate(_runner, job, _embedder, _target):
        seen.append(("try", runtime.background_paused))
        return {**job, "state": next(tries)}

    def fake_wait(_seconds):
        seen.append(("backoff", runtime.background_paused))
        return False

    monkeypatch.setattr(act, "activate_job", fake_activate)
    monkeypatch.setattr(runner._stop, "wait", fake_wait)
    monkeypatch.setattr(runner, "_catch_up_logged", lambda *a: None)
    runner.clean_marks[1] = 0  # a clean full comparison already happened
    job, handed = runner._activate({"job_id": 1, "state": "ready"}, object(), _target())
    assert handed and job["state"] == "activated"
    assert [k for k, _ in seen] == ["try", "backoff", "try", "backoff", "try"]
    assert all(paused for _, paused in seen), f"background work resumed between tries: {seen}"
    assert not runtime.background_paused


def test_a_copying_batch_yields_to_recall_and_rests(store, monkeypatch):
    """A rollback that only copies vectors never calls the model, so it must
    yield to recall and rest after each batch on its own."""
    root, db_path = store
    _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    yields: list[int] = []
    rests: list[float] = []
    real_sleep = time.sleep
    from types import SimpleNamespace
    monkeypatch.setattr(steps, "_yield_to_recall", lambda: yields.append(1))
    # Only the steps module's clock: the runner loop's own sleeps must not count.
    monkeypatch.setattr(steps, "time", SimpleNamespace(
        perf_counter=time.perf_counter, monotonic=time.monotonic,
        sleep=lambda s: rests.append(s) or real_sleep(0)))
    monkeypatch.setattr(steps, "BATCH", 5)
    old = FakeEmbedder(8, "old")
    monkeypatch.setattr(steps, "build_embedder", lambda cfg: old)
    runner = er.ReindexRunner(db_path=db_path, data_root=root)
    runner.request_rollback()
    runner.start()
    try:
        deadline = time.monotonic() + 30
        while runner.status()["job"]["state"] in sp.ACTIVE_STATES and time.monotonic() < deadline:
            real_sleep(0.05)
    finally:
        runner.stop()
    job = runner.status()["job"]
    # Two calls: the probe at the request and the probe at the job's start;
    # every memory's vector was copied, none embedded.
    assert job["state"] == "activated" and job["copied"] == 20 and old.calls == 2, (job, old.calls)
    assert len(yields) >= 4, "copy-only batches did not yield to recall"
    assert len([r for r in rests if r > 0]) >= 4, "copy-only batches ran back to back"


def test_an_edit_just_before_the_swap_is_caught_by_the_change_log(store, monkeypatch):
    """The full comparison runs before the swap with no lock held; an edit that
    lands after it must still be seen at the swap and staged again."""
    from superlocalmemory.storage import embedding_change_log as change_log

    root, db_path = store
    edited = {"done": False}
    real = er.ReindexRunner._catch_up_logged

    def then_edit(self, job, embedder, dimension):
        real(self, job, embedder, dimension)
        if not edited["done"]:  # right after the last catch-up, before the lock
            edited["done"] = True
            conn = sqlite3.connect(db_path)
            conn.execute("UPDATE atomic_facts SET content = 'reworded at the last moment' "
                         "WHERE fact_id = 'f004'")
            conn.commit()
            conn.close()

    monkeypatch.setattr(er.ReindexRunner, "_catch_up_logged", then_edit)
    view = _run(root, db_path, {"new-model": FakeEmbedder(4, "new")}, monkeypatch)
    assert view["state"] == "activated" and edited["done"], view
    conn = sp.connect(db_path)
    blob = conn.execute("SELECT embedding FROM atomic_facts WHERE fact_id='f004'").fetchone()[0]
    assert np.allclose(np.frombuffer(blob, dtype=np.float32),
                       _vec("reworded at the last moment", 4, "new")), \
        "a fact edited just before the swap went live with its old words' vector"
    assert not change_log.active(conn), "the change log outlived the switch"
    conn.close()


def test_the_store_repair_removes_an_erased_memorys_staged_vector(store):
    """The store repair (``slm db repair``) clears what an erasure queued in a
    switch's staged or previous space, not only the switch's idle pass."""
    from superlocalmemory.storage import integrity_receipts
    from superlocalmemory.storage.integrity_repair import Repair, RunStats

    root, db_path = store
    plain = sqlite3.connect(db_path)
    integrity_receipts.ensure_tables(plain)
    plain.close()
    conn = sp.connect(db_path)
    with steps.write_txn(conn, db_path):
        sp.ensure_side_tables(conn)
        sp.create_vec(conn, sp.NEXT_VEC, 4)
        conn.execute(f"INSERT INTO {sp.NEXT_VEC}(rowid, profile_id, embedding) VALUES (7, "
                     "'default', ?)", (np.zeros(4, dtype=np.float32).tobytes(),))
        conn.execute(f"INSERT INTO {sp.NEXT_MAP} VALUES ('f001', 'default', 7, 'h')")
    plain = sqlite3.connect(db_path)
    plain.execute("DELETE FROM atomic_facts WHERE fact_id = 'f001'")
    plain.commit()
    plain.close()
    stats = RunStats(run_id="r1")
    Repair(db_path)._vectors(stats)
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.NEXT_VEC}").fetchone()[0] == 0
    assert conn.execute(f"SELECT COUNT(*) FROM {sp.PURGE}").fetchone()[0] == 0
    assert stats.done.get("vectors.other_spaces") == 1
    conn.close()
