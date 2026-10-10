# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Spreading activation stores its activations and the next identical recall reads them.

The write is deferred to the shared background writer (``slm-bg-writer``) so
recall never takes the write lock on its own thread. These tests hold the
three halves of that contract:

* written: after a recall, ``activation_cache`` has the activations;
* reused: a second identical recall is answered from the cache, without
  propagating again — through the daemon's own /recall route;
* read-only hot path: the write happens on the background writer, never on
  the thread serving the recall;

plus the 4.1.20 cross-scope key: a cross-scope entry is stored and reused
under its own key, and a key from a different seeding version is not served.

Found while writing these (4.1.19 behaviour, fixed in 4.1.20):

* a PERSONAL-scope hit was returned unfiltered, while the propagation that
  stored it had passed the fail-closed authorization filter. A personal
  propagation can reach another profile's fact through an edge, so the
  repeated question returned a different answer — including facts the first
  answer had removed (``test_each_scope...[False-False]``);
* the bookkeeping writer dropped every job still queued when it was stopped,
  so the last cache write of a short-lived process never landed;
* a failed cache write was logged at DEBUG, i.e. invisibly.

In-process, through the daemon route, the write itself did happen on 4.1.19
(the first test passes there); it was the reuse that was wrong.
"""

from __future__ import annotations

import re
import threading
import zlib

import numpy as np
import pytest

import superlocalmemory.storage.deferred_writes as dw
from superlocalmemory.retrieval import spreading_activation as sa_mod
from superlocalmemory.retrieval.spreading_activation import (
    SpreadingActivation,
    SpreadingActivationConfig,
)
from tests.conftest import force_sync_enrichment
from tests.test_retrieval.cross_scope_fixture import REQ, PartitionedVS, build_store
from tests.test_server.test_per_request_profile import _daemon

FACTS = [
    "Alice leads the Phoenix project at Acme.",
    "The Phoenix project ships its first release in March.",
    "Bob reviews every Phoenix release with Alice.",
    "Acme moved the Phoenix team to Berlin.",
    "Carol joined Acme to work on Phoenix security.",
]
QUESTION = "Who works on the Phoenix project?"


def _word_vector(text: str) -> list[float]:
    """A stable bag-of-words vector: texts that share words have cosine > 0.

    The shared ``mock_embedder`` draws an independent random vector per text
    (seeded by ``hash(text)``, which differs in every process), so the cosine
    between the question and any fact is positive only about half the time.
    Spreading activation drops seeds with cosine <= 0, so in about 1 run in 32
    all five facts were dropped and the channel legitimately found nothing.
    """
    vec = np.zeros(768, dtype=np.float32)
    for word in re.findall(r"[a-z]+", text.lower()):
        vec[zlib.crc32(word.encode()) % 768] += 1.0
    norm = float(np.linalg.norm(vec))
    return (vec / norm if norm else vec).tolist()


@pytest.fixture()
def word_embedder(mock_embedder):
    mock_embedder.embed.side_effect = _word_vector
    return mock_embedder


def _drain() -> None:
    dw._bg_queue.join()


def _rows(db) -> int:
    return int(db.execute("SELECT COUNT(*) AS c FROM activation_cache", ())[0]["c"])


class _Spy:
    """Count propagations and record which thread wrote the cache."""

    def __init__(self, monkeypatch) -> None:
        self.propagations = 0
        self.writer_threads: list[str] = []
        real_prop = SpreadingActivation._propagate
        real_write = SpreadingActivation._cache_results
        spy = self

        def prop(self_, *a, **k):
            spy.propagations += 1
            return real_prop(self_, *a, **k)

        def write(self_, *a, **k):
            spy.writer_threads.append(threading.current_thread().name)
            return real_write(self_, *a, **k)

        monkeypatch.setattr(SpreadingActivation, "_propagate", prop)
        monkeypatch.setattr(SpreadingActivation, "_cache_results", write)


def test_daemon_recall_writes_the_cache_and_the_next_identical_recall_reads_it(
    word_embedder, engine_with_mock_deps, monkeypatch,
) -> None:
    engine = force_sync_enrichment(engine_with_mock_deps)
    for fact in FACTS:
        engine.store(fact)
    spy = _Spy(monkeypatch)
    assert _rows(engine._db) == 0

    with _daemon(engine, profiles=()) as (client, _app):
        first = client.get("/recall", params={"q": QUESTION, "limit": 5})
        assert first.status_code == 200 and first.json()["result_count"] > 0
        _drain()
        written = _rows(engine._db)
        assert written > 0, "the recall left nothing in activation_cache"
        assert spy.propagations == 1

        second = client.get("/recall", params={"q": QUESTION, "limit": 5})
        assert second.status_code == 200
        _drain()

    assert spy.propagations == 1, "the identical recall propagated again instead of reading the cache"
    assert _rows(engine._db) == written, "a cache hit must not write again"
    assert [r["fact_id"] for r in second.json()["results"]] == [
        r["fact_id"] for r in first.json()["results"]
    ]
    assert spy.writer_threads == ["slm-bg-writer"], (
        f"the cache must be written by the background writer only: {spy.writer_threads}"
    )


@pytest.fixture()
def store(tmp_path):
    return build_store(tmp_path / "sa.db")


@pytest.mark.parametrize("include_global,include_shared", [(True, True), (False, False)])
def test_each_scope_is_cached_and_reused_under_its_own_key(
    store, monkeypatch, include_global, include_shared,
) -> None:
    spy = _Spy(monkeypatch)
    ch = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")),
                             SpreadingActivationConfig())
    q = store.queries[0].tolist()
    kw = dict(include_global=include_global, include_shared=include_shared)

    first = ch.search(q, REQ, top_k=10, **kw)
    _drain()
    seeds = ch._seed_search(q, REQ, **kw)
    if include_global or include_shared:
        seeds = ch._merge_cross_scope_seeds(q, seeds, REQ, **kw)
    graph = ch._graph_version(REQ, cross_scope=include_global or include_shared)
    key = ch._compute_query_hash(q, REQ, seeds=seeds, graph=graph, **kw)
    stored = store.db.execute(
        "SELECT COUNT(*) AS c FROM activation_cache WHERE query_hash = ?", (key,),
    )[0]["c"]
    assert first and stored > 0

    second = ch.search(q, REQ, top_k=10, **kw)
    assert second == first
    assert spy.propagations == 1
    assert spy.writer_threads == ["slm-bg-writer"]


def test_a_cross_scope_entry_from_another_seeding_version_is_not_served(
    store, monkeypatch,
) -> None:
    spy = _Spy(monkeypatch)
    ch = SpreadingActivation(store.db, PartitionedVS(store.embs, store.ids("L")),
                             SpreadingActivationConfig())
    q = store.queries[1].tolist()
    ch.search(q, REQ, top_k=10, include_global=True, include_shared=True)
    _drain()
    ch.search(q, REQ, top_k=10, include_global=True, include_shared=True)
    assert spy.propagations == 1, "same version: the entry must be reused"
    monkeypatch.setattr(sa_mod, "_CROSS_SCOPE_SEEDING", "|seeds=next")
    ch.search(q, REQ, top_k=10, include_global=True, include_shared=True)
    _drain()
    assert spy.propagations == 2, "an entry cached by another seeding version was served"


def test_shutdown_finishes_queued_bookkeeping_writes() -> None:
    """A write queued behind a slow one still lands when the writer stops."""
    started, release = threading.Event(), threading.Event()
    done: list[str] = []

    def slow() -> None:
        started.set()
        release.wait(5)
        done.append("slow")

    dw.submit_background(slow)
    assert started.wait(5)
    dw.submit_background(lambda: done.append("queued"))

    stopper = threading.Thread(target=dw._shutdown_background_writer, args=(5,))
    stopper.start()
    release.set()
    stopper.join(10)

    assert done == ["slow", "queued"], f"queued write dropped at shutdown: {done}"
