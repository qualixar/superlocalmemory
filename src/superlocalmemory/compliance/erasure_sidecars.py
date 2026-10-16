# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V4 | https://qualixar.com | https://varunpratap.com

"""The places outside ``memory.db`` that keep a profile's words, and their erasure.

Each store here was added by a feature, and each was found, by reading the bytes
of the data folder after an erasure, to still hold what the person wrote:

* ``pending.db``: saves queued while the engine was away, as plain text;
* ``admission_journal.db``: the encrypted request of every accepted save (the key
  is on this computer, so it is readable here) and, for a save not yet
  committed, the instruction to write it again;
* ``recall_queue.db``: the prompts the per-prompt hook queued and, for up to an
  hour after each, the memories recalled for them;
* ``hook_spool/``: tool events and reward signals hooks handed to the daemon
  that it has not written yet (4.2.8), with their tool summaries;
* ``write_commits``: the receipt ledger, protected from ordinary deletes;
* ``llmcache.db``: MCP cache values and the reversible originals of compression;
* ``profiles.json``: the workspace's name and description;
* the free pages of ``learning.db`` and ``active_brain_cache.db``, where deleted
  rows stay readable until the file is rewritten.

``erase_profile_sidecars`` removes the profile from all of them and records the
outcome in ``counts``. A failure is a ``*_failed`` marker, never an exception:
the rest of the erasure still has to run, and the final result must say what is
left. Nothing here logs text.
"""

from __future__ import annotations

import logging
import sqlite3
from pathlib import Path
from typing import Any

from superlocalmemory.infra import sqlite_scrub

logger = logging.getLogger(__name__)


def _delete_where_profile(path: Path, table: str, profile_id: str) -> int:
    """Delete a profile's rows from a side database that may not exist yet; returns the count."""
    if not path.exists():
        return 0
    conn = sqlite3.connect(str(path), timeout=30)
    try:
        has = conn.execute("SELECT 1 FROM sqlite_master WHERE type='table' AND name=?",
                           (table,)).fetchone()
        if has is None:
            return 0
        removed = conn.execute(f"DELETE FROM {table} WHERE profile_id = ?",
                               (profile_id,)).rowcount
        conn.commit()
        return int(removed)
    finally:
        conn.close()


def _step(counts: dict, name: str, action: Any) -> None:
    """Run one erasure step; on failure leave ``<name>_failed`` rather than raise."""
    try:
        action()
    except Exception as exc:  # noqa: BLE001
        logger.warning("GDPR erase: %s failed: %s", name, type(exc).__name__)
        counts[f"{name}_failed"] = 1


def _scrub_all(root: Path, counts: dict) -> None:
    names = ["learning.db", "pending.db", "admission_journal.db", "llmcache.db",
             "recall_queue.db"]
    paths = [root / n for n in names]
    cache = "active_brain_cache.db"
    paths.append(root / cache)
    try:
        paths.extend(c / cache for c in root.iterdir() if c.is_dir())
    except OSError:
        pass
    unscrubbed = [p.name for p in paths if p.exists() and not sqlite_scrub.scrub(p)]
    if unscrubbed:
        logger.warning("GDPR erase: could not rewrite %s", ", ".join(unscrubbed))
        counts["sidecar_scrub_failed"] = len(unscrubbed)


def erase_profile_sidecars(
    root: Path, db_path: Path | None, profile_id: str, counts: dict[str, Any],
) -> None:
    """Remove ``profile_id`` from every side store listed in the module docstring."""
    def pending() -> None:
        counts["pending_memories"] = _delete_where_profile(
            root / "pending.db", "pending_memories", profile_id)

    def journal() -> None:
        counts["admission_journal"] = _delete_where_profile(
            root / "admission_journal.db", "admission_journal", profile_id)

    def recall_queue() -> None:
        # Jobs carry the prompt, not a profile, and a finished job carries what was
        # recalled. The queue is transport: a dropped job makes the hook fall back
        # to a direct recall, which is cheaper than keeping someone's words.
        path = root / "recall_queue.db"
        if not path.exists():
            return
        conn = sqlite3.connect(str(path), timeout=30)
        try:
            has = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='recall_requests'").fetchone()
            if has:
                counts["recall_queue"] = conn.execute("DELETE FROM recall_requests").rowcount
                conn.commit()
        finally:
            conn.close()

    def hook_spool() -> None:
        # Spooled hook events carry tool summaries. Removed under the writer's
        # drain lock and before the row sweep below, so none is written back.
        from superlocalmemory.server.hook_event_writer import erase_profile_spool

        counts["hook_spool"] = erase_profile_spool(
            profile_id, db_path=db_path, root=root / "hook_spool")

    def capture() -> None:
        # An opt-in developer corpus of raw proxy traffic. It names no profile, so
        # nothing in it can be selected for removal; say so rather than call the
        # erasure complete or delete a file the person made on purpose.
        path = root / "optimize_capture.jsonl"
        if path.exists() and path.stat().st_size > 0:
            counts["optimize_capture_present"] = 1

    def receipts() -> None:
        if db_path is None:
            return
        from superlocalmemory.storage.write_receipts_erasure import erase_profile_receipts

        counts["write_receipts"] = erase_profile_receipts(db_path, profile_id)

    def optimize() -> None:
        path = root / "llmcache.db"
        if not path.exists():
            return
        from superlocalmemory.optimize.storage import erasure as cache_erasure
        from superlocalmemory.optimize.storage.db import CacheDB

        result = cache_erasure.erase_profile(CacheDB(path), profile_id)
        counts.update(result.as_counts())

    def registry() -> None:
        from superlocalmemory.core import profile_tombstones

        if not profile_tombstones.erase_profile_entry(root, profile_id):
            raise RuntimeError("the workspace entry is still in profiles.json")
        counts["profile_registry"] = 1

    _step(counts, "pending_memories", pending)
    _step(counts, "admission_journal", journal)
    _step(counts, "recall_queue", recall_queue)
    _step(counts, "hook_spool", hook_spool)
    _step(counts, "optimize_capture", capture)
    _step(counts, "write_receipts", receipts)
    _step(counts, "optimize_cache", optimize)
    _step(counts, "profile_registry", registry)
    _scrub_all(root, counts)


def erase_cached_text(root: Path | None, profile_id: str, needles: list[str],
                      counts: dict[str, Any]) -> None:
    """After a targeted erasure (one entity, some facts): drop cached copies of its words.

    ``needles`` are the words that were erased. Only rows of this profile, or of
    an unknown owner, that contain them are removed from the Optimize cache.
    """
    if root is None or not needles:
        return
    path = Path(root) / "llmcache.db"
    if not path.exists():
        return

    def optimize() -> None:
        from superlocalmemory.optimize.storage import erasure as cache_erasure
        from superlocalmemory.optimize.storage.db import CacheDB

        result = cache_erasure.erase_profile(CacheDB(path), profile_id, needles=needles)
        for key, value in result.as_counts().items():
            counts[key] = counts.get(key, 0) + value

    _step(counts, "optimize_cache", optimize)


def needles_from(texts: list[str]) -> list[str]:
    """The strings to look for in cached copies: the start of each erased text.

    A cache holds the text, or a preview of it; the first words are in both.
    """
    out: list[str] = []
    for text in texts:
        head = (text or "").strip()[:48].strip()
        if len(head) >= 8:
            out.append(head)
    return list(dict.fromkeys(out))
