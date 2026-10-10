"""A media database created before the documents.origin column existed is upgraded in place."""

from __future__ import annotations

import sqlite3
from types import SimpleNamespace

from superlocalmemory.media import open_media_store
from tests.test_documents.support import pdf_input

OLD_DOCS = """CREATE TABLE documents (
      document_id TEXT PRIMARY KEY, profile_id TEXT NOT NULL, sha256 TEXT NOT NULL,
      title TEXT NOT NULL, mime TEXT NOT NULL, bytes INTEGER NOT NULL DEFAULT 0,
      page_count INTEGER NOT NULL DEFAULT 0,
      pages_text_layer INTEGER NOT NULL DEFAULT 0, pages_ocr INTEGER NOT NULL DEFAULT 0,
      pages_empty INTEGER NOT NULL DEFAULT 0, source_id TEXT, source_relpath TEXT,
      memory_id TEXT, fact_ids_json TEXT NOT NULL DEFAULT '[]',
      state TEXT NOT NULL CHECK (state IN ('processing','ready','failed','tombstoned')),
      created_at TEXT NOT NULL, updated_at TEXT NOT NULL, tombstoned_at TEXT)"""


def test_old_documents_table_gains_origin(tmp_path, monkeypatch):
    from superlocalmemory.documents.submit import submit_document
    root = tmp_path / "slm"
    monkeypatch.setenv("SLM_DATA_DIR", str(root))
    s = open_media_store(create=True, data_root=root)
    path = s.path
    s.close()
    c = sqlite3.connect(path)
    c.execute("DROP TABLE documents")
    c.execute(OLD_DOCS)
    for did, src in (("d-user", None), ("d-folder", "s1")):
        c.execute("INSERT INTO documents(document_id, profile_id, sha256, title, mime, source_id, state,"
                  " created_at, updated_at) VALUES (?, 'p1', 'x', 't', 'application/pdf', ?, 'ready', 'n', 'n')",
                  (did, src))
    c.commit()
    c.close()
    s = open_media_store(create=True, data_root=root)
    origins = {r[0]: r[1] for r in s._read().execute("SELECT document_id, origin FROM documents")}
    assert origins == {"d-user": "user", "d-folder": "folder"}
    r = submit_document(pdf_input(("x",)), profile_id="p1", actor_id="a",
                        config=SimpleNamespace(pii_redaction=False), store=s)
    s.close()
    assert r.status == "processing"
