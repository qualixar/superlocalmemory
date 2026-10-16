# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Write one person's words through every path SLM persists them, then look for them.

An erasure is only as complete as the list of places it knows about, and that
list is the thing that drifts: every release adds a sidecar database, a cache or
an index, and the erasure code learns about it later or never. This module is the
other side of that list. It writes a unique marker through each write path and,
after an erasure, reads the BYTES of every file under the data root (and the
backup folder, which is inside it) for that marker. It does not ask the erasure
code what it erased; it asks the disk.

``Universe`` keeps the paths that were written, so a failure names the path that
leaked, not just a file name.
"""

from __future__ import annotations

import sqlite3
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

PROFILE = "erase_me"
OTHER = "keep_me"

#: Files whose bytes are scanned for the marker, in the encodings SLM writes text in.
_ENCODINGS = ("utf-8", "utf-16-le")


def marker(path: str) -> str:
    """A marker unique to one write path, so a leak names where it came from."""
    return f"ZQXERASE-{path}-7731"


def scan(root: Path, needles: list[str],
         binary: dict[str, bytes] | None = None) -> dict[str, list[str]]:
    """``{relative file: [needles found in its bytes]}`` for every file under ``root``.

    Includes SQLite ``-wal``/``-shm`` and journal files, free pages inside a
    database, and the backup folder: a deleted row's bytes stay in the file until
    something overwrites them, and an erasure that leaves them has not erased.
    """
    found: dict[str, list[str]] = {}
    encoded = [(n, [n.encode(enc) for enc in _ENCODINGS]) for n in needles]
    encoded += [(name, [raw]) for name, raw in (binary or {}).items()]
    for path in sorted(root.rglob("*")):
        if path.is_symlink() or not path.is_file():
            continue
        try:
            data = path.read_bytes()
        except OSError:
            continue
        hits = [n for n, forms in encoded if any(f in data for f in forms)]
        if hits:
            found[str(path.relative_to(root))] = hits
    return found


@dataclass
class Universe:
    """What was written, where, and the handles the seeders need."""

    engine: Any
    root: Path
    written: dict[str, str] = field(default_factory=dict)
    ids: dict[str, Any] = field(default_factory=dict)
    #: Raw bytes that must not survive either (a vector is derived from the words).
    binary: dict[str, bytes] = field(default_factory=dict)

    def mark(self, path: str) -> str:
        text = marker(path)
        self.written[path] = text
        return text

    @property
    def needles(self) -> list[str]:
        return list(self.written.values())

    def path_of(self, needle: str) -> str:
        return next(p for p, n in self.written.items() if n == needle)


def _seed_engine(u: Universe) -> None:
    from superlocalmemory.core.engine_ingestion import (
        canonical_store,
        local_trusted_actor_id,
    )

    text = u.mark("engine")
    receipt = canonical_store(
        u.engine,
        f"Alice Zebrowski lives in Lisbon, likes green tea, {text}, and works at Initech.",
        source_type="python-api", trusted_actor_id=local_trusted_actor_id("python-api"),
        require_complete=True, return_receipt=True, profile_id=PROFILE)
    u.ids["fact_ids"] = list(receipt.final_fact_ids)
    _note_vector_bytes(u)


def _note_vector_bytes(u: Universe) -> None:
    """The first bytes of the fact's vector as the index stores it."""
    from superlocalmemory.retrieval import vector_physical

    conn = vector_physical.open_vec(u.engine._db.db_path)
    try:
        row = conn.execute(
            "SELECT embedding FROM fact_embeddings WHERE profile_id = ? LIMIT 1",
            (PROFILE,)).fetchone()
    finally:
        conn.close()
    if row is not None:
        u.binary["vector"] = bytes(row[0])[:64]


def _seed_receipts(u: Universe) -> None:
    """The receipt ledger every accepted write leaves (M032), written by the real runtime."""
    from superlocalmemory.core.remember_runtime import CanonicalRememberRuntime
    from superlocalmemory.storage.admission_journal import Actor, RememberRequest

    text = u.mark("remember_runtime")
    runtime = CanonicalRememberRuntime.for_engine(u.engine)
    actor = Actor("actor", frozenset({PROFILE}), frozenset({"personal"}))
    request = RememberRequest(
        content=f"Bob Quill keeps a pewter kettle in Ostrava. {text}", profile_id=PROFILE,
        source_type="http", idempotency_key="erase-run-1", trusted_actor_id="actor")
    runtime.start()
    try:
        runtime.remember(request, actor, deadline_ms=1_500)
        runtime.wait_for_deferred(5)
    finally:
        runtime.stop()


def _seed_mesh(u: Universe) -> None:
    from superlocalmemory.mesh.broker import MeshBroker

    broker = MeshBroker.for_store(u.engine._db.db_path)  # mesh.db (4.2.8)
    broker.register_peer("sess-erase", profile_id=PROFILE)
    broker.send_message("other", "broadcast", f"mesh note {u.mark('mesh')}", profile_id=PROFILE)


def _seed_pending(u: Universe) -> None:
    from superlocalmemory.cli.pending_store import store_pending

    store_pending(f"queued while offline {u.mark('pending')}", base_dir=u.root, profile_id=PROFILE)


def _seed_recall_queue(u: Universe) -> None:
    from superlocalmemory.core.recall_queue import RecallQueue

    queue = RecallQueue(u.root / "recall_queue.db")
    rid = queue.enqueue(query=f"what did she say {u.mark('recall_query')}", limit_n=5, mode="A",
                        agent_id="auto_recall_hook", session_id="s1")
    claimed = queue.claim_pending()
    queue.complete(rid, received=claimed["received"],
                   result_json='{"results": [{"content": "' + u.mark("recall_result") + '"}]}')
    queue.close()


def _seed_context_cache(u: Universe) -> None:
    from superlocalmemory.core.context_cache import CacheEntry, ContextCache

    cache = ContextCache(home_dir=u.root)
    cache.upsert(CacheEntry(
        session_id="s1", topic_sig="t1", content=f"context {u.mark('context_cache')}",
        fact_ids=["f1"], profile_id=PROFILE, computed_at=int(time.time())))
    cache.close()


def _seed_derive_cache(u: Universe) -> None:
    from superlocalmemory.cache.factory import default_cache
    from superlocalmemory.cache.keys import CacheKey

    key = CacheKey("a" * 64, "ocr", "1", "m", "p")
    default_cache().put(key, f"derived {u.mark('derive_cache')}".encode(), kind="text")


def _seed_learning(u: Universe) -> None:
    from superlocalmemory.learning.database import LearningDatabase
    from superlocalmemory.views import ViewStore

    db = LearningDatabase(u.root / "learning.db")
    db.store_signal(PROFILE, f"query {u.mark('learning_query')}", "fact-1", "click")
    ViewStore(u.root / "learning.db").create(
        PROFILE, name="mine", query=f"saved {u.mark('saved_view')}")


def _seed_code_graph(u: Universe) -> None:
    conn = sqlite3.connect(u.root / "code_graph.db")
    conn.executescript(
        "CREATE TABLE graph_nodes (node_id TEXT PRIMARY KEY, name TEXT);"
        "CREATE TABLE code_memory_links (link_id TEXT PRIMARY KEY, code_node_id TEXT,"
        " slm_fact_id TEXT, enriched_description TEXT);"
        "INSERT INTO graph_nodes VALUES ('n1', 'parse_invoice');")
    fact = (u.ids.get("fact_ids") or ["fact-1"])[0]
    conn.execute("INSERT INTO code_memory_links VALUES ('L1', 'n1', ?, ?)",
                 (fact, f"decision {u.mark('code_link')}"))
    conn.commit()
    conn.close()


def _seed_media(u: Universe) -> None:
    from superlocalmemory.media.store import MediaStore

    store = MediaStore(u.root / "media.db")
    store.insert_item(
        profile_id=PROFILE, kind="image", source_sha256="ab" * 32, mime="image/jpeg",
        bytes=10, origin="tool", thumb_webp=f"thumb {u.mark('media_thumb')}".encode())
    store.close()


def _seed_llm_cache(u: Universe) -> None:
    """Direct use of the Optimize stores, as the MCP tools do for a request of this profile."""
    from superlocalmemory.optimize.storage.db import CacheDB

    db = CacheDB(u.root / "llmcache.db")
    key = "k" * 64
    kwargs: dict[str, Any] = {"profile_id": PROFILE}
    db.set(key, "t" * 64, f"cached {u.mark('mcp_cache')}".encode(), model="mcp-kv",
           ttl_expires=time.time() + 3600, tags=["mcp-kv"], **kwargs)
    db.ccr_put("11111111-1111-4111-8111-111111111111",
               f"original {u.mark('ccr_original')}".encode(), tenant_id="t" * 64, **kwargs)
    u.ids["llm_cache"] = db


def _seed_backup(u: Universe) -> None:
    from superlocalmemory.infra.backup import BackupManager

    u.engine._db.execute("PRAGMA wal_checkpoint(FULL)")
    BackupManager(base_dir=u.root).create_backup(label="erasure-test")


SEEDERS: tuple[Callable[[Universe], None], ...] = (
    _seed_engine, _seed_receipts, _seed_mesh, _seed_pending, _seed_recall_queue,
    _seed_context_cache,
    _seed_derive_cache, _seed_learning, _seed_code_graph, _seed_media, _seed_llm_cache,
    _seed_backup,
)
