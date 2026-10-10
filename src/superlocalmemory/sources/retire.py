# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""Hiding and erasing what a folder file became: replaced versions, deleted files, removed sources."""

from __future__ import annotations

import logging
from typing import Any

from superlocalmemory.media.store_jobs import utc_stamp
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.ingest import facts_of, facts_of_keys
from superlocalmemory.sources.store import SourceStore, entries_of, memory_entries

logger = logging.getLogger(__name__)


def _facts(runtime: Any, entries: list[dict[str, Any]], profile_id: str = "") -> list[str]:
    facts = [f for e in entries for f in e.get("f") or []]
    unknown = [e["m"] for e in entries if e.get("m") and not e.get("f")]
    queued = [e["k"] for e in entries if e.get("k") and not e.get("m") and not e.get("f")]
    return sorted({*facts, *facts_of(runtime, unknown), *facts_of_keys(runtime, profile_id, queued)})


# Local hiding goes through ``archive_fact`` (recall already skips archived facts), the same as
# removing a document. The ``hide_sources`` visibility flag is for remote callers only.
def hide_entries(host: SourceHost, runtime: Any, source: dict, entries: list[dict[str, Any]],
                 relpath: str) -> int:
    """Archive the live entries (recall stops showing them) and mark them replaced. Returns failures.

    An entry whose facts could not all be archived gets no ``sup`` time and is flagged ``old``,
    so ``retry_hides`` tries it again on a later pass.
    """
    live = [e for e in memory_entries(entries) if not e.get("sup")]
    failures = 0
    for entry in live:
        ok = True
        for fact in _facts(runtime, [entry], source["profile_id"]):
            try:
                runtime.archive_fact(source["profile_id"], fact,
                                     idempotency_key=f"src:{source['source_id'][:12]}:{fact}")
            except Exception as exc:  # noqa: BLE001 - stays visible, flagged, and retried
                ok = False
                failures += 1
                logger.warning("a folder memory could not be hidden (%s)", type(exc).__name__)
        if ok:
            entry["sup"] = utc_stamp()
            entry.pop("old", None)
        else:
            entry["old"] = True
    return failures


def retry_hides(host: SourceHost, store: SourceStore, runtime: Any, source: dict) -> int:
    """Hide again the replaced or deleted memories that could not be hidden earlier."""
    failures = 0
    for row in store.files(source["source_id"]):
        entries = entries_of(row)
        stale = [e for e in entries if e.get("old") and not e.get("sup")]
        if stale:
            failures += hide_entries(host, runtime, source, stale, row["relpath"])
            store.put_file(source["source_id"], row["relpath"], entries=entries)
    return failures


def hide_document(store: SourceStore, runtime: Any, source: dict, row: dict[str, Any]) -> int:
    """Soft-remove the PDF a file became, unless it is shared with a document saved another way.

    Returns 1 when the document is still not removed afterwards (its pages may still be
    recalled), else 0. A document that was already removed counts as done.
    """
    if not row.get("document_id") or row.get("reason") == "shared":
        return 0
    from superlocalmemory.documents import remove_document

    try:
        remove_document(row["document_id"], source["profile_id"], runtime=runtime, store=store._m)
    except Exception as exc:  # noqa: BLE001 - counted below
        logger.warning("a folder document could not be hidden (%s)", type(exc).__name__)
    document = store._m.get_document(row["document_id"])
    return int(bool(document) and document["state"] != "tombstoned")


def hide_picture(store: SourceStore, row: dict[str, Any]) -> None:
    """Take the picture a file owns out of the library's view, so the same bytes can be saved afresh."""
    if not row.get("media_id") or row.get("reason") == "shared":
        return
    try:
        store._m.set_state(row["media_id"], "tombstoned")
    except Exception as exc:  # noqa: BLE001
        logger.warning("a folder picture could not be hidden (%s)", type(exc).__name__)


def release_copies(store: SourceStore, source: dict, row: dict[str, Any]) -> None:
    """The owner of a picture or document is going: identical copies in the folder take over next pass."""
    if row.get("reason") != "shared" and (row.get("document_id") or row.get("media_id")):
        if store.release_shared(source["source_id"], row.get("sha256"), row["relpath"]):
            store.queue_scan(source["profile_id"], source["source_id"], behind_running=True)


def hide_file(host: SourceHost, store: SourceStore, runtime: Any, source: dict, row: dict[str, Any],
              *, tombstone: bool) -> int:
    """Hide everything a file row owns. With ``tombstone`` the row stays, marked deleted."""
    entries = entries_of(row)
    failures = hide_entries(host, runtime, source, entries, row["relpath"])
    failures += hide_document(store, runtime, source, row)
    hide_picture(store, row)
    release_copies(store, source, row)
    fields: dict[str, Any] = {"entries": entries}
    if tombstone:
        fields.update(state="tombstoned", tombstoned_at=utc_stamp())
        store.delete_links(source["source_id"], row["relpath"])
    store.put_file(source["source_id"], row["relpath"], **fields)
    return failures


def _erase(host: SourceHost, runtime: Any, source: dict, entries: list[dict[str, Any]],
           subject: str) -> bool:
    facts = _facts(runtime, entries, source["profile_id"])
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
        due = [e for e in memory_entries(entries) if e.get("sup") and e["sup"] <= cutoff]
        if due and _erase(host, runtime, source, due, f"src:{source['source_id'][:12]}"):
            keep = [e for e in entries if e not in due]
            store.put_file(source["source_id"], row["relpath"], entries=keep)
            purged += 1
    return purged
