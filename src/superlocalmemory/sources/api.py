# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""The calls behind the public interface: add, preview, confirm, list, remove, forget-empty, rescan, report."""

from __future__ import annotations

import json
import logging
import os
import threading
import time
import uuid
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Literal

from superlocalmemory.sources import host as host_mod
from superlocalmemory.sources import locks, retire
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.ignore import DEFAULT_TYPES, IgnoreRules
from superlocalmemory.sources.preview import SourcePreview, build_preview
from superlocalmemory.sources.report import SourceReport, build_report
from superlocalmemory.sources.roots import check_root
from superlocalmemory.sources.store import SourceStore
from superlocalmemory.sources.walk import walk_tree

logger = logging.getLogger(__name__)

_PENDING_TTL_S = 3600.0
_REMOVE_WAIT_S = 60.0  # a scan is told to stop at its next file; this trips only on one very slow file
_QUICK_WAIT_S = 10.0
_BUSY_MESSAGE = "The folder is busy with a long file; try again in a minute."
_pending: dict[str, "_Pending"] = {}
_pending_lock = threading.Lock()
REMOTE_MESSAGE = ("Folders cannot be connected while remote access is set up, because remote tools "
                  "could read the notes. Turn remote access off first.")


class SourceRefused(ValueError):
    """A request the feature will not carry out. ``code`` is stable for callers and tests."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


class HintsNotAvailable(RuntimeError):
    """File-change hints need the watcher, which this build does not include."""


@dataclass(frozen=True)
class _Pending:
    root: Path
    profile_id: str
    kind: str
    include_types: tuple[str, ...]
    created: float


@dataclass(frozen=True)
class SourceInfo:
    source_id: str
    profile_id: str
    kind: str
    root_path: str
    display_name: str
    state: str
    include_types: tuple[str, ...]
    files: dict[str, int] = field(default_factory=dict)
    last_scan_at: str | None = None
    offline_reason: str | None = None


def _open(create: bool = False) -> tuple[Any, SourceStore | None, SourceHost]:
    from superlocalmemory.media import open_media_store

    host = host_mod.current_host()
    media = open_media_store(create=create, data_root=host.data_root)
    return media, (SourceStore(media) if media is not None else None), host


def add_source(root: Path | str, *, profile_id: str, kind: Literal["folder", "obsidian"] | None = None,
               include_types: tuple[str, ...] = DEFAULT_TYPES) -> SourcePreview:
    """Check the folder and describe what connecting it would do. Saves nothing."""
    real = check_root(root)
    kind = kind or ("obsidian" if (real / ".obsidian").is_dir() else "folder")
    source_id = uuid.uuid4().hex
    types = tuple(t.lower() for t in include_types)
    preview = build_preview(source_id, real, kind, types)
    with _pending_lock:
        now = time.monotonic()
        for key in [k for k, v in _pending.items() if now - v.created > _PENDING_TTL_S]:
            del _pending[key]
        _pending[source_id] = _Pending(real, profile_id, kind, types, now)
    return preview


def confirm_source(source_id: str, *, via: str = "api") -> None:
    """Connect a previewed folder: turn the feature on, record the source and queue its first scan."""
    host = host_mod.current_host()
    with _pending_lock:
        pending = _pending.get(source_id)
    if pending is None:
        raise SourceRefused("unknown_source", "That folder preview has expired. Add the folder again.")
    if host.remote_on():
        raise SourceRefused("remote_access_on", REMOTE_MESSAGE)
    real = check_root(pending.root)
    from superlocalmemory.runtimes.features import enable_sources

    if not enable_sources(source=via, data_root=host.data_root):
        raise SourceRefused("cannot_save", "Couldn't save the setting. Check that the data folder is writable.")
    media, store, _ = _open(create=True)
    try:
        sid = store.create_source(pending.profile_id, pending.kind, str(real), real.name or str(real),
                                  pending.include_types, source_id=source_id)
        _remember_device(store, sid, real)
        store.queue_scan(pending.profile_id, sid)
    finally:
        media.close()
    host.wake()


def _remember_device(store: SourceStore, source_id: str, root: Path) -> None:
    """Note which disk the folder is on, so a different disk mounted there later is noticed."""
    import json
    import os

    source = store.get_source(source_id) or {}
    try:
        known = json.loads(source.get("last_scan_stats_json") or "{}")
    except ValueError:
        known = {}
    if "root_dev" not in known:
        store.set_state(source_id, source.get("state") or "active",
                        stats={**known, "root_dev": os.stat(root).st_dev})


def _offline_reason(source: dict[str, Any]) -> str | None:
    import json

    if source["state"] != "offline":
        return None
    try:
        return json.loads(source.get("last_scan_stats_json") or "{}").get("offline_reason") or None
    except ValueError:
        return None


def list_sources(profile_id: str) -> list[SourceInfo]:
    media, store, _ = _open()
    if media is None:
        return []
    try:
        import json

        return [SourceInfo(
            source_id=s["source_id"], profile_id=s["profile_id"], kind=s["kind"], root_path=s["root_path"],
            display_name=s["display_name"], state=s["state"],
            include_types=tuple(json.loads(s["include_types_json"])), files=store.counts(s["source_id"]),
            last_scan_at=s["last_scan_at"], offline_reason=_offline_reason(s))
            for s in store.list_sources(profile_id)]
    finally:
        media.close()


def _source(store: SourceStore | None, source_id: str) -> dict[str, Any]:
    source = store.get_source(source_id) if store else None
    if source is None or source["state"] == "removed":
        raise SourceRefused("unknown_source", "That folder is not connected.")
    return source


def remove_source(source_id: str, *, purge: bool = False) -> None:
    """Disconnect a folder. Its memories are hidden (kept); with ``purge`` they are erased.

    A scan that is running for the folder is told to stop and finishes its current file first;
    queued scans are cancelled. Once removed, the folder is never set back by a scan.
    """
    media, store, host = _open()
    try:
        source = _source(store, source_id)
        runtime = host.runtime()
        if runtime is None:
            raise SourceRefused("writer_not_ready", "The memory writer is not ready; try again shortly.")
        locks.mark_removing(source_id)
        try:
            store.cancel_scans(source_id)
            with locks.held(source_id, _REMOVE_WAIT_S):
                _clear_source(host, store, runtime, source, purge)
        except locks.SourceBusy:
            raise SourceRefused("source_busy", _BUSY_MESSAGE) from None
        finally:
            locks.clear_removing(source_id)
    finally:
        media.close()


def _clear_source(host: SourceHost, store: SourceStore, runtime: Any, source: dict[str, Any],
                  purge: bool) -> None:
    source_id = source["source_id"]
    for row in store.files(source_id):
        if purge:
            if not retire.erase_row(host, store, runtime, source, row):
                raise SourceRefused("erasure_incomplete", "The erasure was not complete; try again.")
        elif row["state"] != "tombstoned":
            retire.hide_file(host, store, runtime, source, row, tombstone=True)
    if purge:
        store.delete_source_rows(source_id)
    else:
        store.set_state(source_id, "removed")


def _check_still_empty(host: SourceHost, source: dict[str, Any]) -> int:
    """Look at the folder again; return its disk number, or refuse if it is not plainly an empty folder."""
    if host.remote_on():
        raise SourceRefused("remote_access_on", "Folders are paused while remote access is set up. "
                                                "Turn remote access off first.")
    if host.runtime() is None:
        raise SourceRefused("writer_not_ready", "The memory writer is not ready; try again shortly.")
    root = check_root(source["root_path"])
    if os.path.normcase(str(root)) != os.path.normcase(source["root_path"]):
        raise SourceRefused("root_moved", "The folder's path now leads somewhere else.")
    try:
        walked = walk_tree(root, IgnoreRules(root, tuple(json.loads(source["include_types_json"]))))
        dev = os.stat(root).st_dev
    except OSError:
        raise SourceRefused("unreachable", "The folder cannot be read right now.") from None
    if walked.entries or walked.capped:
        raise SourceRefused("folder_not_empty", "The folder has files again; they are read on the next scan.")
    try:
        known = json.loads(source.get("last_scan_stats_json") or "{}").get("root_dev")
    except ValueError:
        known = None
    if isinstance(known, int) and known != dev:
        raise SourceRefused("disk_changed", "Another disk is now at the folder's path.")
    return dev


def forget_empty(source_id: str) -> dict[str, Any]:
    """The folder really is empty: hide the memories of every file it held (kept, not erased).

    Only for a folder waiting as ``offline`` / ``empty_folder``; checked again under the folder's lock.
    """
    from superlocalmemory.sources.reconcile import ScanStats

    media, store, host = _open()
    try:
        _source(store, source_id)
        try:
            with locks.held(source_id, _QUICK_WAIT_S):
                source = _source(store, source_id)
                if source["state"] != "offline" or _offline_reason(source) != "empty_folder":
                    raise SourceRefused("not_empty_folder", "This folder is not waiting as an empty folder.")
                dev = _check_still_empty(host, source)
                runtime = host.runtime()
                rows = [r for r in store.files(source_id) if r["state"] != "tombstoned"]
                for row in rows:
                    retire.hide_file(host, store, runtime, source, row, tombstone=True)
                store.set_state(source_id, "active", stats=ScanStats(root_dev=dev).summary(), scanned=True)
        except locks.SourceBusy:
            raise SourceRefused("source_busy", _BUSY_MESSAGE) from None
        return {"source_id": source_id, "forgotten": len(rows), "state": "active"}
    finally:
        if media is not None:
            media.close()


def rescan(source_id: str) -> dict[str, Any]:
    media, store, host = _open()
    try:
        source = _source(store, source_id)
        job = store.queue_scan(source["profile_id"], source_id)
    finally:
        media.close()
    host.wake()
    return {k: job[k] for k in ("job_id", "state", "done", "total")}


def hint(source_id: str, relpaths: list[str]) -> None:
    """There is no outside hint path in this build: the watcher runs in-process and calls the scanner itself."""
    raise HintsNotAvailable("file-change hints are not available")


def source_report(source_id: str) -> SourceReport:
    media, store, _ = _open()
    try:
        return build_report(store, _source(store, source_id))
    finally:
        if media is not None:
            media.close()


def release_file(source_id: str, relpath: str) -> bool:
    """Let the quarantined content be read on the next scan; a later edit is screened again."""
    media, store, host = _open()
    try:
        source = _source(store, source_id)
        try:
            with locks.held(source_id, _QUICK_WAIT_S):
                row = store.get_file(source_id, relpath)
                if row is None or row["state"] != "quarantined" or not row["sha256"]:
                    return False
                store.put_file(source_id, relpath, state="pending", reason=f"released:{row['sha256']}")
        except locks.SourceBusy:
            raise SourceRefused("source_busy", _BUSY_MESSAGE) from None
        store.queue_scan(source["profile_id"], source_id)
    finally:
        media.close()
    host.wake()
    return True
