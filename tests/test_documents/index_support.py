"""Builders for the index, lint and erasure tests: a media store with documents and a small memory database."""

from __future__ import annotations

import hashlib
import sqlite3
from contextlib import contextmanager
from pathlib import Path

from superlocalmemory.media import files, open_media_store

_MEMORY_DDL = """
CREATE TABLE memories (memory_id TEXT PRIMARY KEY, profile_id TEXT, metadata_json TEXT DEFAULT '{}');
CREATE TABLE atomic_facts (fact_id TEXT PRIMARY KEY, profile_id TEXT, memory_id TEXT,
  created_at TEXT DEFAULT '2026-01-01T00:00:00Z');
CREATE TABLE canonical_entities (entity_id TEXT PRIMARY KEY, profile_id TEXT, canonical_name TEXT);
CREATE TABLE fact_entity_associations (profile_id TEXT, fact_id TEXT, entity_id TEXT,
  PRIMARY KEY (profile_id, fact_id, entity_id));
CREATE TABLE graph_edges (edge_id TEXT PRIMARY KEY, profile_id TEXT, source_id TEXT, target_id TEXT, edge_type TEXT);
"""


class MemDb:
    """The slice of memory.db the document code reads."""

    def __init__(self, root: Path) -> None:
        self.db_path = Path(root) / "memory.db"
        self.conn = sqlite3.connect(self.db_path)
        self.conn.row_factory = sqlite3.Row
        self.conn.executescript(_MEMORY_DDL)
        self.conn.commit()
        self.queries = 0

    def execute(self, sql, params=()):
        self.queries += 1
        cur = self.conn.execute(sql, params)
        self.conn.commit()
        return cur.fetchall()

    @contextmanager
    def raw_connection(self):
        yield self.conn

    def fact(self, fact_id, memory_id, profile="p1", entities=(), created="2026-01-01T00:00:00Z"):
        self.execute("INSERT OR IGNORE INTO memories(memory_id, profile_id) VALUES (?, ?)", (memory_id, profile))
        self.execute("INSERT INTO atomic_facts(fact_id, profile_id, memory_id, created_at) VALUES (?,?,?,?)",
                     (fact_id, profile, memory_id, created))
        for name in entities:
            eid = f"e-{profile}-{name}"
            self.execute("INSERT OR IGNORE INTO canonical_entities VALUES (?,?,?)", (eid, profile, name))
            self.execute("INSERT INTO fact_entity_associations VALUES (?,?,?)", (profile, fact_id, eid))

    def edge(self, source, target, kind, profile="p1"):
        self.execute("INSERT INTO graph_edges VALUES (?,?,?,?,?)", (f"{source}{target}{kind}", profile, source, target, kind))


def make_root(tmp_path):
    root = tmp_path / "slm"
    root.mkdir()
    return root, MemDb(root), open_media_store(create=True, data_root=root)


def add_document(root, store, db, doc_id, *, profile="p1", title="Doc", pages=(), data=None, with_file=True,
                 created="2026-02-01T00:00:00Z"):
    """A ready document. ``pages`` are (text_origin, entity names, phash) per page; each text page gets one fact."""
    data = data if data is not None else (b"%PDF-" + doc_id.encode())
    sha = hashlib.sha256(data).hexdigest()
    rel = ""
    if with_file:
        tmp = files.tmp_dir(root) / f"{doc_id}.bin"
        tmp.write_bytes(data)
        rel = files.place_original(root, tmp, sha, "pdf")
    store.insert_document(document_id=doc_id, profile_id=profile, sha256=sha, title=title, mime="application/pdf",
                          bytes=len(data), source_relpath=rel)
    space = store.ensure_active_space("fake", "r1", 4)
    for n, (origin, names, phash) in enumerate(pages, 1):
        memory_ids, fact_ids = [], []
        if origin != "none":
            memory_ids, fact_ids = [f"m-{doc_id}-{n}"], [f"f-{doc_id}-{n}"]
            db.fact(fact_ids[0], memory_ids[0], profile, names)
        media_id = hashlib.sha256(f"{doc_id}:{n}".encode()).hexdigest()[:32]
        store.insert_item(media_id=media_id, profile_id=profile, kind="page", source_sha256=sha, phash=phash,
                          mime="image/png", bytes=0, anchor_memory_id=memory_ids[0] if memory_ids else None,
                          document_id=doc_id, page_no=n, origin="document", thumb_webp=b"thumb")
        store.put_vector(media_id, space, profile, [0.1, 0.2, 0.3, 0.4])
        store.put_page(doc_id, n, media_id=media_id, memory_ids=memory_ids, fact_ids=fact_ids,
                       text_origin=origin, char_count=10 if origin != "none" else 0)
    store.update_document(doc_id, state="ready", page_count=len(pages))
    store.refresh_document_counts(doc_id)
    with store._write() as conn:
        conn.execute("UPDATE documents SET created_at = ? WHERE document_id = ?", (created, doc_id))
    return sha, rel
