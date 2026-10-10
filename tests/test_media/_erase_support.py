"""Shared builders for the erasure, GC and snapshot tests of the media store."""

from __future__ import annotations

import hashlib
import os
import sqlite3
import time
from contextlib import contextmanager
from pathlib import Path

from superlocalmemory.cache.keys import CacheKey
from superlocalmemory.cache.sqlite_store import SqliteDeriveCache
from superlocalmemory.media import files, open_media_store

DIM = 4


class MemDb:
    """The slice of the memory database the erasure code reads."""

    def __init__(self, root: Path) -> None:
        self.db_path = Path(root) / "memory.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(
            "CREATE TABLE IF NOT EXISTS memories (memory_id TEXT PRIMARY KEY, profile_id TEXT,"
            " metadata_json TEXT DEFAULT '{}');"
            "CREATE TABLE IF NOT EXISTS atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT,"
            " memory_id TEXT);")
        self.conn.commit()

    def execute(self, sql, params=()):
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur.fetchall()

    @contextmanager
    def raw_connection(self):
        yield self.conn

    def add_memory(self, memory_id, profile="p1", fact_id=None, media_id=None):
        meta = '{"_slm_source": {"type": "media", "media_id": "%s"}}' % media_id if media_id else "{}"
        self.execute("INSERT INTO memories VALUES (?,?,?)", (memory_id, profile, meta))
        if fact_id:
            self.execute("INSERT INTO atomic_facts VALUES (?,?,?)", (fact_id, profile, memory_id))


def add_image(root, store, *, media_id, profile="p1", memory_id, data=b"img", age_s=3600, cache=True):
    """A stored image: file on disk, row, vector, cached OCR. Returns (stored_sha, relpath)."""
    sha = hashlib.sha256(data).hexdigest()
    tmp = files.tmp_dir(root) / f"{media_id}.bin"
    tmp.write_bytes(data)
    rel = files.place_original(root, tmp, sha, "png")
    path = files.original_path(root, sha, "png")
    old = time.time() - age_s
    os.utime(path, (old, old))
    store.insert_item(media_id=media_id, profile_id=profile, kind="image", source_sha256=media_id * 2 if False else
                      hashlib.sha256(media_id.encode()).hexdigest(), stored_sha256=sha, mime="image/png",
                      bytes=len(data), origin="tool", original_relpath=rel, anchor_memory_id=memory_id,
                      thumb_webp=b"thumb")
    space = store.ensure_active_space("fake", "r1", DIM)
    store.put_vector(media_id, space, profile, [0.1, 0.2, 0.3, 0.4])
    if cache:
        SqliteDeriveCache(Path(root) / "derive_cache.db").put(
            CacheKey(sha, "ocr.auto", "1"), b'{"text": "words"}', kind="json")
    return sha, rel


def cached(root, sha) -> bool:
    return SqliteDeriveCache(Path(root) / "derive_cache.db").get(CacheKey(sha, "ocr.auto", "1")) is not None


def make_root(tmp_path):
    root = tmp_path / "slm"
    root.mkdir()
    return root, MemDb(root), open_media_store(create=True, data_root=root)
