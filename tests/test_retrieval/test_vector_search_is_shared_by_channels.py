# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Channels asking the same question at once share one vector scan (#304).

Semantic, hopfield and spreading activation each run a nearest-neighbour search
with the same query vector, at the same moment, on every recall. vec0 scans
every vector whatever k is, so three scans cost three times one. Thread dumps
of slow recalls on a 37k-fact store found all three channels inside that one
query. A search already running for the same vector now answers the others.
"""

from __future__ import annotations

import sqlite3
import threading
from pathlib import Path

import numpy as np
import pytest

from superlocalmemory.retrieval import vector_store as vs_mod
from superlocalmemory.retrieval.vector_store import VectorStore, VectorStoreConfig
from superlocalmemory.storage import schema as real_schema

DIM = 4


def _store(tmp_path: Path) -> VectorStore:
    path = tmp_path / "v.db"
    conn = sqlite3.connect(str(path))
    real_schema.create_all_tables(conn)
    conn.commit()
    conn.close()
    store = VectorStore(path, VectorStoreConfig(dimension=DIM, enabled=True))
    if not store.available:
        pytest.skip("sqlite-vec not loadable here")
    rng = np.random.default_rng(7)
    for i in range(120):
        v = rng.normal(size=DIM)
        store.upsert(f"f-{i:03d}", "default", (v / np.linalg.norm(v)).tolist())
    return store


def test_a_small_k_answers_what_it_answered_alone(tmp_path: Path) -> None:
    """Asking with the shared k and cutting gives the same nearest facts."""
    store = _store(tmp_path)
    q = [0.4, -0.1, 0.8, 0.2]
    alone = store._search_rows(store._serialize_f32(q), 10, "default")
    assert store.search(q, top_k=10, profile_id="default") == alone
    assert store.search(q, top_k=3, profile_id="default") == alone[:3]


def test_concurrent_identical_searches_run_one_scan(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    q = [0.1, 0.9, -0.3, 0.2]
    scans: list[int] = []
    entered = threading.Event()
    release = threading.Event()
    real = store._search_rows

    def slow(vec_bytes, k, profile_id):
        scans.append(k)
        entered.set()
        release.wait(5)
        return real(vec_bytes, k, profile_id)

    monkeypatch.setattr(store, "_search_rows", slow)
    out: dict[str, list] = {}

    def ask(name, k):
        out[name] = store.search(q, top_k=k, profile_id="default")

    first = threading.Thread(target=ask, args=("hopfield", 100))
    first.start()
    assert entered.wait(5)
    rest = [threading.Thread(target=ask, args=(n, k))
            for n, k in (("semantic", 50), ("spreading_activation", 10))]
    for t in rest:
        t.start()
    release.set()
    for t in (first, *rest):
        t.join(5)

    assert scans == [vs_mod.SHARED_SEARCH_K]  # one scan answered all three
    assert out["semantic"] == out["hopfield"][:50]
    assert out["spreading_activation"] == out["hopfield"][:10]


def test_a_finished_search_is_not_reused(tmp_path: Path, monkeypatch) -> None:
    """Sharing is only for searches in flight: a later one sees new writes."""
    store = _store(tmp_path)
    q = [0.2, 0.2, 0.2, 0.9]
    before = store.search(q, top_k=5, profile_id="default")
    store.upsert("f-new", "default", (np.array(q) / np.linalg.norm(q)).tolist())
    after = store.search(q, top_k=5, profile_id="default")
    assert after[0][0] == "f-new" and after != before


def test_profiles_and_large_k_are_never_shared(tmp_path: Path, monkeypatch) -> None:
    store = _store(tmp_path)
    seen: list[tuple[int, str | None]] = []
    monkeypatch.setattr(store, "_search_rows",
                        lambda vb, k, pid: seen.append((k, pid)) or [])
    q = [0.5, 0.5, 0.5, 0.5]
    store.search(q, top_k=5, profile_id="other")
    store.search(q, top_k=vs_mod.SHARED_SEARCH_K + 50, profile_id="default")
    assert seen == [(vs_mod.SHARED_SEARCH_K, "other"),
                    (vs_mod.SHARED_SEARCH_K + 50, "default")]
