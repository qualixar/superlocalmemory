# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""Restore points a person can see, preview and use -- without a terminal.

Every update that changes the store first takes a verified copy (see
``storage/backup.py``). This module turns those copies into *restore points*:

  list_restore_points   every copy, newest first, with what it holds
  preview_restore       what restoring one would change, in counts, reading only
  request_restore       record the request; the restore itself runs at the next
                        start, before anything opens the store
  cancel_restore        withdraw a request that has not run yet
  perform_pending_restore / reimport_delta      (``_restore_boot`` / ``_restore_reimport``)
  prepare_downgrade / cancel_downgrade / downgrade_hold_active   (``_downgrade``)
  run_store_repair_once                         (``upgrade_repair``)

Why the restore waits for a restart: restoring under a running engine is unsafe.
Its caches, projections and open connections would all describe the old file,
and copying a database beside a live ``-wal`` corrupts it. So the request is
recorded and executed on the next start, before migrations and before the
engine opens anything.
"""

from __future__ import annotations

import logging
import re
import shutil
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path

from superlocalmemory.storage import _snapshot_manifest as sm
from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_delta import (
    export_delta,
    export_empty_delta,
    open_compare,
)
from superlocalmemory.storage._restore_types import (
    DELTA_DIR,
    INTENT_NAME,
    SNAPSHOTS_DIR,
    DowngradeReport,
    ReimportReport,
    RestoreIntent,
    RestoreOutcome,
    RestorePoint,
    RestorePreview,
    RestoreRefusedError,
)

logger = logging.getLogger(__name__)

_POINT_ID = re.compile(r"^[0-9A-Za-z][0-9A-Za-z._-]{0,80}$")
_LEGACY = re.compile(r"^memory-(?P<id>\d{8}-\d{6}-pre-\d[0-9A-Za-z.+_-]*)\.db$")
_DELTA_FILES = ("memories.jsonl", "deleted.jsonl", "kind_edits.jsonl", "erasures.jsonl",
                "profiles.jsonl", "meta.json")
#: Room a restore needs beyond the copies: the safety copies of the live store.
_SAFETY_FACTOR = 1.1


def _now_stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _valid_point_id(point_id: str) -> str:
    if not isinstance(point_id, str) or not _POINT_ID.match(point_id) or ".." in point_id:
        raise RestoreRefusedError("That restore point does not exist.")
    return point_id


def _stamp_time(point_id: str) -> str:
    match = re.match(r"^(\d{8})-(\d{6})", point_id)
    if not match:
        return ""
    try:
        moment = datetime.strptime(match.group(1) + match.group(2), "%Y%m%d%H%M%S")
    except ValueError:
        return ""
    return moment.replace(tzinfo=UTC).isoformat(timespec="seconds")


def _point_from(root: Path, point_id: str, memory: Path, learning: Path | None,
                manifest: dict | None) -> RestorePoint:
    from superlocalmemory.storage._restore_preview import (
        copy_schema_version,
        not_restorable_reason,
    )

    files = (manifest or {}).get("files") or []
    sha = {f["snapshot"]: f["sha256"] for f in files if f.get("snapshot") and f.get("sha256")}
    size = memory.stat().st_size + (learning.stat().st_size if learning else 0)
    counts = (manifest or {}).get("counts") or {}
    schema = copy_schema_version([memory, *([learning] if learning else [])])
    reason = not_restorable_reason(schema)
    return RestorePoint(
        point_id=point_id,
        created_at=(manifest or {}).get("created_at") or _stamp_time(point_id),
        reason=(manifest or {}).get("reason") or "update",
        from_version=(manifest or {}).get("from_version"),
        to_version=(manifest or {}).get("to_version"),
        memory_snapshot=memory, learning_snapshot=learning, size_bytes=size,
        sha256=sha or None, facts=counts.get("facts"), legacy=manifest is None,
        repair=(manifest or {}).get("repair"), schema_version=schema,
        restorable=reason is None, not_restorable_reason=reason,
    )


def _copy_of_an_empty_store(memory_snapshot: Path) -> bool:
    """True for a readable copy of a store that had no memory table yet.

    A fresh install takes a copy before its first migration, of a database
    with no tables to restore. It can never be restored (it does not verify),
    so listing it offered a restore point that only ever said "unusable".
    A copy that cannot be read is NOT hidden: damage is something to show.
    """
    try:
        with closing(sm._open_immutable(memory_snapshot)) as conn:
            tables = {r[0] for r in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'")}
    except Exception:  # noqa: BLE001 - unreadable is reported by verify, not hidden
        return False
    return "atomic_facts" not in tables and "memories" not in tables


def list_restore_points(data_root: Path) -> list[RestorePoint]:
    """Every restorable generation, newest first. Reads only.

    Copies of a store that held no memories (taken by a fresh install before
    its first migration) are left out: there is nothing in them to restore.
    """
    root = Path(data_root) / SNAPSHOTS_DIR
    if not root.is_dir() or root.is_symlink():
        return []
    names = {p.name for p in root.iterdir() if p.is_file() and not p.is_symlink()}
    points: list[RestorePoint] = []
    for name in sorted(names):
        stamp = sm.stamp_of(Path(name))
        if stamp and name.startswith("memory-"):
            if _copy_of_an_empty_store(root / name):
                continue
            learning = root / f"learning-{stamp}{sm.SNAPSHOT_SUFFIX}"
            points.append(_point_from(root, stamp, root / name,
                                      learning if learning.name in names else None,
                                      sm.read_manifest(root, stamp)))
            continue
        legacy = _LEGACY.match(name)
        if legacy:
            pid = legacy.group("id")
            learning = root / f"learning-{pid}.db"
            points.append(_point_from(root, pid, root / name,
                                      learning if learning.name in names else None, None))
    return sorted(points, key=lambda p: (p.created_at, p.point_id), reverse=True)


def find_restore_point(point_id: str, data_root: Path) -> RestorePoint:
    point_id = _valid_point_id(point_id)
    for point in list_restore_points(data_root):
        if point.point_id == point_id:
            return point
    raise RestoreRefusedError("That restore point does not exist.")


def verify_point(point: RestorePoint) -> None:
    """Raise ``SnapshotUnusableError`` unless every copy of the point is intact."""
    sha = point.sha256 or {}
    sm.verify_snapshot(point.memory_snapshot, sha.get(point.memory_snapshot.name),
                       require_table="atomic_facts")
    if point.learning_snapshot is not None:
        sm.verify_snapshot(point.learning_snapshot, sha.get(point.learning_snapshot.name))


def _size(*paths: Path) -> int:
    total = 0
    for path in paths:
        for companion in (path, Path(f"{path}-wal"), Path(f"{path}-shm")):
            if companion.exists():
                total += companion.stat().st_size
    return total


def disk_needed(point: RestorePoint, memory_db: Path, learning_db: Path | None) -> int:
    """Staging copy of the point + safety copies of the live store, with headroom."""
    live = _size(memory_db, *( [learning_db] if learning_db else []))
    return int(_SAFETY_FACTOR * live) + point.size_bytes


def free_bytes(data_root: Path) -> int:
    probe = Path(data_root)
    while not probe.exists() and probe.parent != probe:
        probe = probe.parent
    return shutil.disk_usage(str(probe)).free


def preview_restore(point_id: str, *, data_root: Path, memory_db: Path) -> RestorePreview:
    """What restoring ``point_id`` would change. Changes no byte of anything."""
    from superlocalmemory.storage._restore_preview import build_preview

    point = find_restore_point(point_id, data_root)
    return build_preview(point, data_root=Path(data_root), memory_db=Path(memory_db))


def _remove_delta_dir(data_root: Path, delta_dir: Path | str | None) -> None:
    """Delete a delta folder's own files by explicit path, then the folder."""
    if not delta_dir:
        return
    base = (Path(data_root) / DELTA_DIR).resolve()
    target = Path(delta_dir).resolve()
    if target.parent != base or not target.is_dir() or Path(delta_dir).is_symlink():
        return
    for name in _DELTA_FILES:
        (target / name).unlink(missing_ok=True)
    try:
        target.rmdir()
    except OSError:
        pass  # something else is in it: leave it


def request_restore(point_id: str, *, requested_by: str, reimport: bool = True,
                    data_root: Path, memory_db: Path) -> RestoreIntent:
    """Check, export what changed since the copy, and record the request.

    The caller then restarts SuperLocalMemory; the restore runs at that start.
    Raises ``RestoreRefusedError`` (nothing changed) when the copy is not
    intact or the disk is too full.
    """
    preview = preview_restore(point_id, data_root=data_root, memory_db=memory_db)
    if not preview.verified or not preview.disk_ok or not preview.restorable:
        raise RestoreRefusedError(" ".join(preview.problems) or "This copy cannot be used.")
    point = find_restore_point(point_id, data_root)
    data_root = Path(data_root)
    previous = read_json(data_root / INTENT_NAME)
    if previous and previous.get("stage") == "writing":
        raise RestoreRefusedError("A restore is already part-way through; restart "
                                  "SuperLocalMemory to let it finish first.")
    stamp = _now_stamp()
    delta_dir = data_root / DELTA_DIR / stamp
    if preview.live_store_readable:
        with closing(open_compare(Path(memory_db), point.memory_snapshot)) as conn:
            export_delta(conn, delta_dir)
    else:
        export_empty_delta(delta_dir)       # nothing can be read, so nothing is carried
    requested_at = datetime.now(UTC).isoformat(timespec="seconds")
    intent_path = write_json_atomic(data_root / INTENT_NAME, {
        "format": 1, "point_id": point.point_id, "requested_by": str(requested_by)[:200],
        "requested_at": requested_at, "reimport": bool(reimport),
        "delta_dir": str(delta_dir), "stage": "requested", "preview": preview.as_dict(),
    })
    if previous:
        _remove_delta_dir(data_root, previous.get("delta_dir"))
    logger.info("[SLM] Restore of %s requested by %s; it runs at the next start",
                point.point_id, requested_by)
    return RestoreIntent(point_id=point.point_id, requested_by=str(requested_by),
                         requested_at=requested_at, reimport=bool(reimport),
                         delta_dir=delta_dir, intent_path=intent_path,
                         preview=preview.as_dict())


def pending_restore(data_root: Path) -> dict | None:
    return read_json(Path(data_root) / INTENT_NAME)


_PART_WAY = (
    "A restore is part-way through, so it cannot be cancelled now: your memory store may "
    "already hold the restored copy, and the memories written after the copy are kept "
    "aside until it finishes. Restart SuperLocalMemory to let it finish; those memories "
    "are added back then. Nothing was changed.")


def cancel_restore(data_root: Path) -> bool:
    """Withdraw a waiting request. True when there was one.

    Refused (``RestoreRefusedError``) once the restore has started writing: the
    store may already be the copy, and the request's export is then the only
    record of the memories written after it.
    """
    data_root = Path(data_root)
    intent = read_json(data_root / INTENT_NAME)
    path = data_root / INTENT_NAME
    if not path.exists():
        return False
    if intent and intent.get("stage") == "writing":
        raise RestoreRefusedError(_PART_WAY)
    path.unlink()
    if intent:
        _remove_delta_dir(data_root, intent.get("delta_dir"))
    return True


from superlocalmemory.storage._downgrade import (  # noqa: E402
    cancel_downgrade,
    downgrade_hold_active,
    downgrade_status,
    prepare_downgrade,
)
from superlocalmemory.storage._restore_boot import (  # noqa: E402
    perform_pending_restore,
    settle_projections_after_restore,
)
from superlocalmemory.storage._restore_reimport import (  # noqa: E402
    reimport_delta,
    run_pending_reimport,
)
from superlocalmemory.storage.upgrade_repair import run_store_repair_once  # noqa: E402

__all__ = [
    "DowngradeReport", "ReimportReport", "RestoreIntent", "RestoreOutcome", "RestorePoint",
    "RestorePreview", "RestoreRefusedError", "cancel_downgrade", "cancel_restore",
    "downgrade_hold_active", "downgrade_status", "find_restore_point", "list_restore_points",
    "pending_restore", "perform_pending_restore", "prepare_downgrade", "preview_restore",
    "reimport_delta", "request_restore", "run_pending_reimport", "run_store_repair_once",
    "settle_projections_after_restore", "verify_point",
]
