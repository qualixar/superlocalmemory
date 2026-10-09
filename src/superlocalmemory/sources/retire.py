# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Hiding and erasing what a folder file became: replaced versions, deleted files, removed sources."""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.media.store_jobs import utc_stamp
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.ingest import facts_of
from superlocalmemory.sources.store import SourceStore, entries_of

logger = logging.getLogger(__name__)


def _facts(runtime: Any, entries: list[dict[str, Any]]) -> list[str]:
    facts = [f for e in entries for f in e.get("f") or []]
    unknown = [e["m"] for e in entries if e.get("m") and not e.get("f")]
    return sorted({*facts, *facts_of(runtime, unknown)})


def hide_entries(host: SourceHost, runtime: Any, source: dict, entries: list[dict[str, Any]],
                 relpath: str) -> int:
    """Archive the live entries (recall stops showing them) and mark them replaced. Returns failures."""
    live = [e for e in entries if not e.get("sup")]
    failures = 0
    for fact in _facts(runtime, live):
        try:
            runtime.archive_fact(source["profile_id"], fact,
                                 idempotency_key=f"src:{source['source_id'][:12]}:{fact}")
        except Exception as exc:  # noqa: BLE001 - hide what can be hidden; the rest stays visible, not lost
            failures += 1
            logger.warning("a folder memory could not be hidden (%s)", type(exc).__name__)
    now = utc_stamp()
    for entry in live:
        entry["sup"] = now
    return failures


def hide_document(store: SourceStore, runtime: Any, source: dict, row: dict[str, Any]) -> None:
    """Soft-remove the PDF a file became, unless it is shared with a document saved another way."""
    if not row.get("document_id") or row.get("reason") == "shared":
        return
    from superlocalmemory.documents import remove_document

    try:
        remove_document(row["document_id"], source["profile_id"], runtime=runtime, store=store._m)
    except Exception as exc:  # noqa: BLE001
        logger.warning("a folder document could not be hidden (%s)", type(exc).__name__)


def hide_file(host: SourceHost, store: SourceStore, runtime: Any, source: dict, row: dict[str, Any],
              *, tombstone: bool) -> int:
    """Hide everything a file row owns. With ``tombstone`` the row stays, marked deleted."""
    entries = entries_of(row)
    failures = hide_entries(host, runtime, source, entries, row["relpath"])
    hide_document(store, runtime, source, row)
    fields: dict[str, Any] = {"entries": entries}
    if tombstone:
        fields.update(state="tombstoned", tombstoned_at=utc_stamp())
    store.put_file(source["source_id"], row["relpath"], **fields)
    return failures


def _erase(host: SourceHost, runtime: Any, source: dict, entries: list[dict[str, Any]],
           subject: str) -> bool:
    facts = _facts(runtime, entries)
    if not facts:
        return True
    if host.eraser is None:
        return False
    try:
        counts = host.eraser(source["profile_id"], facts, subject) or {}
    except Exception as exc:  # noqa: BLE001 - retried on the next pass
        logger.warning("a folder erasure failed (%s)", type(exc).__name__)
        return False
    return bool(counts.get("erasure_complete"))


def erase_row(host: SourceHost, store: SourceStore, runtime: Any, source: dict, row: dict[str, Any]) -> bool:
    """Hard-erase a file row's memories (and its PDF) and drop the row; False if not complete."""
    subject = f"src:{source['source_id'][:12]}"
    if not _erase(host, runtime, source, entries_of(row), subject):
        return False
    if row.get("document_id") and row.get("reason") != "shared":
        from superlocalmemory.documents import remove_document

        try:
            if not remove_document(row["document_id"], source["profile_id"], hard=True,
                                   eraser=host.eraser, store=store._m):
                return False
        except Exception as exc:  # noqa: BLE001
            logger.warning("a folder document erasure failed (%s)", type(exc).__name__)
            return False
    store.delete_file(source["source_id"], row["relpath"])
    return True


def purge_due(host: SourceHost, store: SourceStore, runtime: Any, source: dict) -> int:
    """Erase tombstoned files and replaced versions that are past the grace period."""
    cutoff = utc_stamp(-host.purge_after_s)
    purged = 0
    for row in store.files(source["source_id"]):
        if row["state"] == "tombstoned":
            if (row["tombstoned_at"] or "9") <= cutoff and erase_row(host, store, runtime, source, row):
                purged += 1
            continue
        entries = entries_of(row)
        due = [e for e in entries if e.get("sup") and e["sup"] <= cutoff]
        if due and _erase(host, runtime, source, due, f"src:{source['source_id'][:12]}"):
            keep = [e for e in entries if e not in due]
            store.put_file(source["source_id"], row["relpath"], entries=keep)
            purged += 1
    return purged
