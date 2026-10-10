# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
# Part of SuperLocalMemory V3 | https://qualixar.com | https://varunpratap.com

"""One scan pass over one source: the folder is the truth, memory follows it.

Order: refuse while remote access is set up, find what is in the folder, look again at changed
files once (a single wait for the whole pass), then save new and changed files, re-point moved
ones, hide deleted ones and erase what is past its grace period. Nothing in the folder is
ever written, moved or deleted. An unreachable folder tombstones nothing.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable

from superlocalmemory.core.recall_gate import background_work, yield_to_recalls
from superlocalmemory.media.store_jobs import utc_stamp
from superlocalmemory.sources import borrows, ingest, links, locks, obsidian, retire
from superlocalmemory.sources.host import SourceHost
from superlocalmemory.sources.ignore import IgnoreRules, kind_of
from superlocalmemory.sources.roots import RootRefused, check_root
from superlocalmemory.sources.safe_read import open_regular
from superlocalmemory.sources.store import SourceStore, entries_of, memory_entries
from superlocalmemory.sources.walk import Entry, WalkResult, stat_entry, walk_tree

logger = logging.getLogger(__name__)

_QUIET_STATES = ("indexed", "quarantined", "skipped")
_REMOTE_CHECK_EVERY = 25
_PAUSE_REASON = "remote_access_on"


@dataclass
class ScanStats:
    new: int = 0
    changed: int = 0
    moved: int = 0
    unchanged: int = 0
    tombstoned: int = 0
    purged: int = 0
    quarantined: int = 0
    placeholders: int = 0
    deferred: int = 0
    errors: int = 0
    skipped: dict[str, int] = field(default_factory=dict)
    capped: bool = False
    offline: bool = False
    paused: bool = False
    waiting: bool = False
    removed: bool = False
    offline_reason: str = ""
    root_dev: int | None = None

    def summary(self) -> dict[str, Any]:
        out = asdict(self)
        out["paused_reason"] = _PAUSE_REASON if self.paused else None
        return out


@dataclass
class _Pass:
    host: SourceHost
    store: SourceStore
    source: dict[str, Any]
    runtime: Any
    root: Path
    stats: ScanStats
    rows: dict[str, dict[str, Any]]
    vanished: dict[str, list[str]] = field(default_factory=dict)
    names: links.NameIndex | None = None  # Obsidian sources: where embeds can point
    only: frozenset[str] | None = None  # a targeted pass looks at these paths and nothing else
    media_ready: bool = False  # read once per pass: files skipped while media was off are read again

    @property
    def sid(self) -> str:
        return self.source["source_id"]


def _digest(path: Path, limit: int | None = None, file_id: str | None = None) -> tuple[str, bytes | None]:
    """sha256 of a file, read in chunks; the bytes too when ``limit`` allows keeping them.

    Raises OSError for a link, a pipe, or a file that is not the one the walk listed (``file_id``).
    """
    h, kept = hashlib.sha256(), bytearray()
    with open_regular(path, file_id) as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
            if limit is not None and len(kept) <= limit:
                kept += chunk
    return h.hexdigest(), (bytes(kept) if limit is not None and len(kept) <= limit else None)


def _unchanged(row: dict[str, Any] | None, e: Entry, media_ready: bool = False) -> bool:
    if row and row["state"] == "skipped" and row["reason"] == ingest.MEDIA_NOT_READY and media_ready:
        return False  # skipped only because images & documents were off; they are ready now
    return bool(row and row["state"] in _QUIET_STATES
                and (row["size"], row["mtime_ns"], row["file_id"]) == e.signature())


def _stat_fields(e: Entry) -> dict[str, Any]:
    return {"size": e.size, "mtime_ns": e.mtime_ns, "file_id": e.file_id}


def _stable(p: _Pass, candidates: list[Entry]) -> list[Entry]:
    """The candidates whose size and mtime did not move during one shared wait."""
    if not candidates:
        return []
    p.host.sleep(p.host.stability_s)
    steady = []
    for e in candidates:
        if _gone(p):
            return []
        again = stat_entry(p.root, e.relpath)
        if again is not None and again.signature() == e.signature() and not again.placeholder:
            steady.append(e)
        else:
            p.stats.deferred += 1
    return steady


def _placeholder(p: _Pass, e: Entry, row: dict[str, Any] | None) -> None:
    """A cloud-only file is never read and never tombstones what it already gave."""
    p.stats.placeholders += 1
    if row and row["state"] == "cloud_placeholder":
        return
    fields = {} if row else _stat_fields(e)
    p.store.put_file(p.sid, e.relpath, state="cloud_placeholder", reason="cloud_only", **fields)


def _supersede(p: _Pass, row: dict[str, Any] | None, resent: frozenset[str] = frozenset()) -> list[dict[str, Any]]:
    """Hide the old version of a file; returns its entries (marked replaced) to keep for the purge.

    ``resent`` are keys the new save sent again (a retry of the same version): the writer handed
    back the memories it already had, the new save owns them, and they are not hidden.
    """
    if row is None:
        return []
    entries = [e for e in entries_of(row) if e.get("k") not in resent]
    p.stats.errors += retire.hide_entries(p.host, p.runtime, p.source, entries, row["relpath"])
    p.stats.errors += retire.hide_document(p.store, p.runtime, p.source, row)
    retire.hide_picture(p.store, row)
    retire.release_copies(p.store, p.source, row)
    return entries


def _quarantine(p: _Pass, e: Entry, row: dict[str, Any] | None, sha: str, hits: list) -> None:
    kinds = ",".join(sorted({h.kind for h in hits}))
    entries = _supersede(p, row)
    p.store.delete_links(p.sid, e.relpath)
    p.store.put_file(p.sid, e.relpath, sha256=sha, state="quarantined", reason=f"credential:{kinds}",
                     entries=entries, document_id=None, media_id=None, **_stat_fields(e))
    p.stats.quarantined += 1


def _is_note(p: _Pass, relpath: str) -> bool:
    """Markdown and canvas files of an Obsidian vault get properties and links; nothing else does."""
    return p.source["kind"] == "obsidian" and relpath.lower().endswith((".md", ".markdown", ".canvas"))


def _save_number(p: _Pass, e: Entry, row: dict[str, Any] | None, sha: str, kind: str, fresh: bool) -> int:
    """Every save of a path has its own number; only a retry of the same text version keeps it.

    A text file whose last attempt ended in ``error`` on these very bytes is retried under the
    same number, so the parts that did save are re-sent with the same keys and not saved twice.
    ``fresh`` forces a new number (a repeat that pointed at hidden copies is saved again).
    """
    retry = (not fresh and kind == "text" and row is not None
             and row["state"] == "error" and row["sha256"] == sha)
    if retry:
        n = p.store.current_save_n(p.sid, e.relpath)
        if n:
            return n
    return p.store.next_save_n(p.sid, e.relpath)


def _ingest(p: _Pass, e: Entry, sha: str, data: bytes | None, row: dict[str, Any] | None = None,
            *, fresh: bool = False) -> ingest.Ingested:
    """One save of the file, under its save number (see ``_save_number``)."""
    version, kind = sha[:12], kind_of(e.relpath)
    if kind != "text":  # pictures and PDFs are handed over as bytes, never re-opened by path
        data = ingest.load_verified(p.root / e.relpath, e.file_id, sha, kind)
        if data is None:  # edited since the hash: look again next pass
            return ingest.Ingested(retry=True)
    n = _save_number(p, e, row, sha, kind, fresh)
    if kind == "text" and _is_note(p, e.relpath):
        return obsidian.ingest_note(p.host, p.runtime, p.source, e.relpath, data or b"", version, n, p.names)
    if kind == "text":
        return ingest.ingest_text(p.host, p.runtime, p.source, e.relpath, data or b"", version, n)
    if kind == "pdf":
        return ingest.ingest_pdf(p.host, p.source, e.relpath, data, version, n)
    return ingest.ingest_image(p.host, p.runtime, p.source, e.relpath, data, version, n)


def _hidden_copy(p: _Pass, out: ingest.Ingested) -> bool:
    """A repeat that points at memories already archived would leave the file invisible."""
    ids = [x.get("m") or x.get("shared_m") for x in out.entries]
    return out.shared and ingest.any_archived(p.runtime, [i for i in ids if i])


def _save(p: _Pass, e: Entry, row: dict[str, Any] | None, sha: str, data: bytes | None) -> None:
    out = _ingest(p, e, sha, data, row)
    if not out.retry and not out.skip_reason and _hidden_copy(p, out):
        out = _ingest(p, e, sha, data, row, fresh=True)  # once more, under a new save number
    if out.retry:
        p.stats.deferred += 1
        return
    if out.skip_reason:
        p.store.put_file(p.sid, e.relpath, sha256=sha, state="skipped", reason=out.skip_reason,
                         entries=_supersede(p, row), **_stat_fields(e))
        _record_links(p, e, out)
        return
    old = _supersede(p, row, frozenset(x["k"] for x in out.entries if x.get("k")))
    p.store.put_file(p.sid, e.relpath, sha256=sha, state="indexed", reason="shared" if out.shared else None,
                     entries=memory_entries(old) + out.entries, document_id=out.document_id,
                     media_id=out.media_id, **_stat_fields(e))
    _record_links(p, e, out)
    p.stats.changed += 1 if row and row["state"] != "tombstoned" else 0
    p.stats.new += 0 if row and row["state"] != "tombstoned" else 1


def _record_links(p: _Pass, e: Entry, out: ingest.Ingested) -> None:
    if out.links is not None:
        p.store.replace_links(p.sid, e.relpath, out.links)


def _move(p: _Pass, e: Entry, sha: str) -> bool:
    """Re-point a vanished file's row to this path when the bytes are the same.

    The row is the truth: ``_slm_source.relpath`` inside the saved memories keeps the old path.
    """
    olds = p.vanished.get(sha)
    if not olds:
        return False
    old = olds.pop(0)
    p.store.repoint(p.sid, old, e.relpath, state="indexed", **_stat_fields(e))
    p.rows.pop(old, None)
    p.stats.moved += 1
    return True


def _process(p: _Pass, e: Entry, sha: str) -> None:
    row = p.store.get_file(p.sid, e.relpath)  # fresh: an earlier file of this pass may have changed it
    if row and row["sha256"] == sha and row["state"] in ("indexed", "cloud_placeholder", "quarantined"):
        keep = "quarantined" if row["state"] == "quarantined" else "indexed"
        p.store.put_file(p.sid, e.relpath, state=keep, **_stat_fields(e))
        p.stats.unchanged += 1
        return
    if row is None and _move(p, e, sha):
        return
    if e.relpath.lower().endswith(".canvas") and p.source["kind"] != "obsidian":  # plain folders: recorded, not read
        p.store.put_file(p.sid, e.relpath, sha256=sha, state="skipped", reason="canvas_not_supported",
                         **_stat_fields(e))
        return
    data = None
    if kind_of(e.relpath) == "text":
        again, data = _digest(p.root / e.relpath, ingest.SCREEN_BYTES * 20, e.file_id)
        if again != sha or data is None:
            p.stats.deferred += 1
            return
        hits = [] if (row and row.get("reason") == f"released:{sha}") else ingest.screen(data)
        if hits:
            _quarantine(p, e, row, sha, hits)
            return
    _save(p, e, row, sha, data)


def _hash_all(p: _Pass, entries: list[Entry]) -> list[tuple[Entry, str]]:
    out = []
    for e in entries:
        if _gone(p):
            break
        try:
            out.append((e, _digest(p.root / e.relpath, None, e.file_id)[0]))
        except OSError:
            p.stats.errors += 1
    return out


def _index_vanished(p: _Pass, seen: set[str], walked: WalkResult) -> None:
    for rel, row in p.rows.items():
        if rel in seen or row["state"] != "indexed" or not row["sha256"] or walked.under_unreadable(rel):
            continue
        p.vanished.setdefault(row["sha256"], []).append(rel)


def _tombstone_missing(p: _Pass, seen: set[str], walked: WalkResult) -> None:
    for rel, row in list(p.rows.items()):
        if rel in seen or row["state"] == "tombstoned" or walked.under_unreadable(rel):
            continue
        if memory_entries(entries_of(row)) or row.get("document_id"):
            p.stats.errors += retire.hide_file(p.host, p.store, p.runtime, p.source, row, tombstone=True)
            p.stats.tombstoned += 1
        else:
            p.store.delete_file(p.sid, rel)


def _gone(p: _Pass) -> bool:
    """True once the source is being removed or is removed: the scan must write nothing more."""
    if locks.is_removing(p.sid):
        return True
    current = p.store.get_source(p.sid)
    return current is None or current["state"] == "removed"


def _pause(store: SourceStore, source: dict, stats: ScanStats) -> ScanStats:
    stats.paused = True
    store.set_state(source["source_id"], "paused", stats=stats.summary(), scanned=True)
    return stats


def _narrow(p: _Pass, walked: WalkResult) -> list[Entry]:
    """The entries this pass looks at; a targeted pass also keeps only the matching rows."""
    if p.only is None:
        return walked.entries
    p.rows = {r: row for r, row in p.rows.items() if r in p.only}
    return [e for e in walked.entries if e.relpath in p.only]


def _work(p: _Pass, walked: WalkResult, progress: Callable[[int, int], None] | None) -> None:
    p.stats.errors += retire.retry_hides(p.host, p.store, p.runtime, p.source)
    borrows.reset_dead_borrows(p.store, p.runtime, p.sid)
    p.rows = {r["relpath"]: r for r in p.store.files(p.sid)}
    if p.source["kind"] == "obsidian":
        p.names = links.NameIndex([e.relpath for e in walked.entries]
                                  + [r for r, row in p.rows.items() if row["state"] == "indexed"])
    candidates: list[Entry] = []
    entries = _narrow(p, walked)
    for e in entries:
        row = p.rows.get(e.relpath)
        if e.placeholder:
            _placeholder(p, e, row)
        elif _unchanged(row, e, p.media_ready):
            p.stats.unchanged += 1
        else:
            candidates.append(e)
    seen = {e.relpath for e in entries}
    _index_vanished(p, seen, walked)
    hashed = _hash_all(p, _stable(p, candidates))
    for i, (e, sha) in enumerate(hashed):
        if _gone(p):
            p.stats.removed = True
            return
        if i % _REMOTE_CHECK_EVERY == _REMOTE_CHECK_EVERY - 1 and p.host.remote_on():
            p.stats.paused = True
            return
        yield_to_recalls()
        try:
            _process(p, e, sha)
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the pass
            cause = exc.__cause__ if isinstance(exc, ingest.PartialSave) and exc.__cause__ else exc
            if isinstance(exc, ingest.SaveBudgetSpent):
                p.stats.deferred += 1  # the writer is busy: the rest of the file waits, nothing failed
            else:
                logger.warning("a folder file could not be saved (%s)", type(cause).__name__)
                p.stats.errors += 1
            fields: dict[str, Any] = {}
            if isinstance(exc, ingest.PartialSave) and exc.entries:
                # The parts saved before the failure stay owned by the row, so the next save
                # (which supersedes the row) and any removal or purge also reach them.
                kept = p.store.get_file(p.sid, e.relpath)
                fields["entries"] = (entries_of(kept) if kept else []) + exc.entries
            reason = "save_queued" if isinstance(exc, ingest.SaveBudgetSpent) else type(cause).__name__[:60]
            p.store.put_file(p.sid, e.relpath, state="error", reason=reason,
                             sha256=sha, **_stat_fields(e), **fields)
        if progress:
            progress(i + 1, len(hashed))
    if _gone(p):
        p.stats.removed = True
        return
    if not walked.capped:
        _tombstone_missing(p, seen, walked)
    p.stats.purged = retire.purge_due(p.host, p.store, p.runtime, p.source)


def _known_device(source: dict) -> int | None:
    try:
        found = json.loads(source.get("last_scan_stats_json") or "{}").get("root_dev")
    except ValueError:
        return None
    return found if isinstance(found, int) else None


def _holders(store: SourceStore, sid: str) -> list[dict[str, Any]]:
    """Every row, in any state but tombstoned, that holds a memory, a document or a picture.

    A tombstoned row holds nothing that is still on disk, so it never counts toward the guards.
    """
    return [r for r in store.files(sid) if r["state"] != "tombstoned"
            and (entries_of(r) or r.get("document_id") or r.get("media_id"))]


def _device_problem(root: Path, stats: ScanStats, store: SourceStore, sid: str, walked: WalkResult) -> str:
    """"disk_changed" when another disk is now at the folder's path (the disk is noted on the first scan).

    The same files on a new device number (a remount) are the same folder: the number is re-recorded.
    """
    current = os.stat(root).st_dev
    if stats.root_dev is None or stats.root_dev == current:
        stats.root_dev = current
        return ""
    held = _holders(store, sid)
    seen = {(e.relpath, e.size) for e in walked.entries}
    if held and 2 * sum((r["relpath"], r["size"]) in seen for r in held) < len(held):
        return "disk_changed"  # under half of what the folder held is here: another disk
    stats.root_dev = current
    return ""


def _offline(store: SourceStore, source: dict, stats: ScanStats, reason: str) -> ScanStats:
    """An unreachable, emptied or swapped folder: tombstone nothing, try again next pass."""
    stats.offline, stats.offline_reason = True, reason
    store.set_state(source["source_id"], "offline", stats=stats.summary(), scanned=True)
    return stats


def scan_source(host: SourceHost, store: SourceStore, source: dict[str, Any], *,
                progress: Callable[[int, int], None] | None = None,
                only: frozenset[str] | None = None) -> ScanStats:
    """Reconcile one source with its folder; returns what happened.

    ``only`` names the relpaths a watcher saw change. The tree is still walked, so every rule
    (ignore, credential screen, links, remote access) applies; only those paths are read or hidden.
    """
    stats = ScanStats(root_dev=_known_device(source))
    if host.remote_on():
        return _pause(store, source, stats)
    runtime = host.runtime()
    if runtime is None:
        stats.waiting = True
        return stats
    try:
        root = check_root(source["root_path"])  # the rules for a new folder hold on every scan
    except RootRefused as exc:
        return _offline(store, source, stats, exc.code)
    if os.path.normcase(str(root)) != os.path.normcase(source["root_path"]):  # the path now leads somewhere else than the folder confirmed
        return _offline(store, source, stats, "root_moved")
    rules = IgnoreRules(root, tuple(json.loads(source["include_types_json"])))
    try:
        walked = walk_tree(root, rules)
        device = _device_problem(root, stats, store, source["source_id"], walked)
    except OSError:
        return _offline(store, source, stats, "unreachable")
    if device or (not walked.entries and not walked.capped and _holders(store, source["source_id"])):
        return _offline(store, source, stats, device or "empty_folder")
    stats.skipped, stats.capped = walked.skipped, walked.capped
    p = _Pass(host, store, source, runtime, root, stats, {}, only=only, media_ready=ingest.media_ready())
    with background_work():
        _work(p, walked, progress)
    if stats.removed:
        return stats
    if stats.paused:
        return _pause(store, source, stats)
    if only is not None:  # a partial look must not move the full-scan clock or its numbers
        store.set_state(source["source_id"], "active")
        return stats
    store.set_state(source["source_id"], "active", stats=stats.summary(), scanned=True)
    return stats


__all__ = ["ScanStats", "scan_source"]
