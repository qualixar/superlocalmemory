"""Small stand-ins for the media tests: a real media store with hand-made vectors,
a memory database with just the tables recall reads, and a query embedder."""

from __future__ import annotations

import json
import sqlite3
import threading

from superlocalmemory.media import open_media_store

DIM = 4


def unit(i: int) -> list[float]:
    v = [0.0] * DIM
    v[i] = 1.0
    return v


def blend(i: int, j: int, weight: float) -> list[float]:
    """Mostly axis ``i``, with ``weight`` of axis ``j``: a controllable distance."""
    v = [0.0] * DIM
    v[i], v[j] = 1.0 - weight, weight
    return v


class MemoryDb:
    """Counts every statement, like the real database's ``execute``."""

    def __init__(self) -> None:
        self.conn = sqlite3.connect(":memory:", check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, memory_id TEXT,"
                          " profile_id TEXT DEFAULT 'default')")
        self.conn.execute("CREATE TABLE memories (memory_id TEXT PRIMARY KEY,"
                          " metadata_json TEXT NOT NULL DEFAULT '{}')")
        self.queries: list[str] = []
        self.db_path = None
        self._lock = threading.Lock()

    def add(self, memory_id: str, fact_id: str, source: dict | None = None) -> None:
        self.conn.execute("INSERT OR IGNORE INTO memories VALUES (?, ?)",
                          (memory_id, json.dumps({"_slm_source": source} if source else {})))
        self.conn.execute("INSERT INTO atomic_facts VALUES (?, ?, 'default')", (fact_id, memory_id))

    def execute(self, sql, params=()):
        with self._lock:
            self.queries.append(sql)
            return self.conn.execute(sql, tuple(params)).fetchall()


def make_store(root, rows):
    """``rows``: [(media_id, kind, anchor_memory_id, vector)] on one profile."""
    store = open_media_store(create=True, data_root=root)
    space = store.ensure_active_space("fake-model", "r1", DIM)
    for n, (media_id, kind, anchor, vector) in enumerate(rows):
        fields = dict(media_id=media_id, profile_id="default", kind=kind, source_sha256=f"{n:064x}",
                      mime="image/png", bytes=10, origin="tool", anchor_memory_id=anchor)
        store.insert_item(**fields)
        store.put_vector(media_id, space, "default", vector)
    return store


class FakeClient:
    """``embed_query`` returns a chosen vector, or None while 'cold'."""

    model_id = "fake-model"

    def __init__(self, vector=None) -> None:
        self.vector = vector
        self.calls: list[tuple[str, float]] = []
        self.warmups = 0

    def embed_query(self, text, *, wait_s=0.3):
        self.calls.append((text, wait_s))
        if self.vector is None:
            self.warmups += 1
        return self.vector


def mid(n: int) -> str:
    return f"{n:032x}"
