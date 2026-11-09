# Copyright (c) 2026 Varun Pratap Bhardwaj / Qualixar
# Licensed under AGPL-3.0-or-later - see LICENSE file
"""The restore itself: run at start-up, before anything opens the store.

Order, and why each step is where it is:

 1. Refuse while anything else can write: another live daemon (pid file or the
    writer lease's recorded holder), or anyone holding the writer lease -- this
    process included. The lease is then HELD for the whole restore, so nothing
    can start writing part-way through. A refusal keeps the request for the
    next start and changes nothing.
 2. Re-verify the copy (checksum from its manifest, SQLite's own check).
 3. Check the disk can hold a staging copy plus safety copies of the live store.
 4. Export, from the live store as it is NOW, everything that changed since the
    copy, and record in the request that writing has begun and where that
    export is. A restore interrupted after this point re-uses that export: the
    store it would compute a new one from may already be the restored one.
 5. Build a staging copy of the snapshot and apply deletions and erasures to
    it -- erased data never reaches the live store, not even for a moment.
 6. Write the staging copy into the live store through the SQLite backup API,
    one transaction, after a safety copy into ``pre-restore/``. Power lost
    during the write leaves the old store intact (the transaction rolls back)
    and the request in place; the next start runs it again.
 7. Record the outcome, then retire the request (``restore-intent.done-<ts>``).
    Outcome first: a crash between the two re-runs the restore (harmless, the
    same copy is written again with another safety copy) instead of losing the
    note that the newer memories still have to be added back.
"""

from __future__ import annotations

import logging
import os
import sqlite3
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from superlocalmemory.storage._durable_json import read_json, write_json_atomic
from superlocalmemory.storage._restore_types import (
    DELTA_DIR,
    INTENT_NAME,
    OUTCOME_NAME,
    SAFETY_DIR,
    RestoreOutcome,
)

logger = logging.getLogger(__name__)

_STAGING = "staging"
_STAGING_FILES = ("memory.db", "learning.db")


def _stamp() -> str:
    return datetime.now(UTC).strftime("%Y%m%d-%H%M%S-%f")


def _other_writer(memory_db: Path) -> str | None:
    """A description of another live process that can write the store, or None."""
    from superlocalmemory.storage import _boot_snapshot
    from superlocalmemory.storage.migration_runner import _foreign_live_daemon

    pid = _foreign_live_daemon(memory_db)
    if pid is not None:
        return f"SuperLocalMemory is still running (process {pid})"
    holder = _boot_snapshot.lease_holder(memory_db)
    if holder is not None:
        who = "another process" if holder == _boot_snapshot.UNKNOWN_HOLDER else f"process {holder}"
        return f"{who} is writing to your memories"
    return None


def _lease_path(memory_db: Path) -> Path:
    resolved = Path(memory_db).expanduser().resolve()
    return resolved.with_name(f"{resolved.name}.writer.lock")


def _write_outcome(data_root: Path, outcome: RestoreOutcome, **extra: Any) -> None:
    write_json_atomic(Path(data_root) / OUTCOME_NAME, {
        **outcome.as_dict(), "at": datetime.now(UTC).isoformat(timespec="seconds"), **extra})


def _retire_intent(data_root: Path, label: str) -> Path | None:
    intent = Path(data_root) / INTENT_NAME
    if not intent.exists():
        return None
    target = Path(data_root) / f"restore-intent.{label}-{_stamp()}.json"
    os.replace(intent, target)
    return target


#: What a stage means for the store, said truthfully in every message (L1-03).
_UNCHANGED = "Your memories were not changed"
_MAYBE_WRITTEN = ("Your memory store may already hold the restored copy; the memories "
                  "written after the copy are kept aside and are added back when it finishes")


def _refused(data_root: Path, point_id: str | None, reason: str, *,
             stage: str | None = None) -> RestoreOutcome:
    if stage == "writing":
        message = (f"The restore that was part-way through could not continue yet: {reason}. "
                   f"{_MAYBE_WRITTEN}. It is tried again at the next start.")
    else:
        message = (f"The restore did not start: {reason}. {_UNCHANGED}; "
                   "it is tried again at the next start.")
    outcome = RestoreOutcome(status="refused", point_id=point_id, message=message)
    _write_outcome(data_root, outcome)
    logger.warning("[SLM] %s", message)
    return outcome


def _clear_staging(staging_dir: Path, *, remove_dir: bool = False) -> None:
    """Remove this module's own staging files, by explicit path."""
    for name in _STAGING_FILES:
        for suffix in ("", "-wal", "-shm", "-journal"):
            (staging_dir / f"{name}{suffix}").unlink(missing_ok=True)
    if remove_dir:
        try:
            staging_dir.rmdir()          # only if empty
        except OSError:
            pass


def _copy_quiescent(snapshot: Path, dest: Path) -> None:
    """Copy a verified, unchanging snapshot into a new staging file."""
    dest.parent.mkdir(parents=True, exist_ok=True)
    for suffix in ("", "-wal", "-shm", "-journal"):
        Path(f"{dest}{suffix}").unlink(missing_ok=True)
    src = sqlite3.connect(f"{snapshot.absolute().as_uri()}?mode=ro&immutable=1", uri=True)
    try:
        with closing(sqlite3.connect(str(dest))) as dst:
            src.backup(dst)
    finally:
        src.close()


def _quiesce(db: Path) -> None:
    """Fold a staging file's log into it so it can be read immutable."""
    with closing(sqlite3.connect(str(db), isolation_level=None)) as conn:
        if conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal":
            conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")


def perform_pending_restore(
    data_root: Path, memory_db: Path, learning_db: Path, *, config: Any | None = None,
) -> RestoreOutcome | None:
    """Run a requested restore. None when nothing was requested.

    Daemon start step 0 (before migrations and the engine), and ``slm db
    restore`` with the daemon stopped. ``config`` (the loaded settings) lets it
    move vector and graph search back to SQLite until they are rebuilt from the
    restored store; without it ``settle_projections_after_restore`` must be
    called before the backends are promoted again.
    """
    from superlocalmemory.core.file_lock import LockHeldError, exclusive_lock

    data_root, memory_db = Path(data_root), Path(memory_db)
    intent_path = data_root / INTENT_NAME
    if not intent_path.exists():
        return None
    intent = read_json(intent_path)
    if not intent or not intent.get("point_id"):
        _retire_intent(data_root, "invalid")
        outcome = RestoreOutcome(status="invalid", point_id=None,
                                 message="The restore request was unreadable and was set aside.")
        _write_outcome(data_root, outcome)
        return outcome
    point_id = str(intent["point_id"])
    stage = intent.get("stage")
    other = _other_writer(memory_db)
    if other:
        return _refused(data_root, point_id, other, stage=stage)
    try:
        lease = exclusive_lock(_lease_path(memory_db), timeout_s=0.0)
        lease.__enter__()
    except LockHeldError:
        return _refused(data_root, point_id, "the memory store is open for writing",
                        stage=stage)
    try:
        return _perform_locked(data_root, memory_db, Path(learning_db), intent, config)
    finally:
        lease.__exit__(None, None, None)


def _unusable(data_root: Path, intent: dict[str, Any], point_id: str) -> RestoreOutcome:
    """The copy cannot be read (damaged or removed). Said plainly; details logged."""
    _retire_intent(data_root, "failed")
    final = intent.get("final_delta_dir")
    if intent.get("stage") == "writing" and final and Path(final).is_dir():
        # An earlier attempt may have written the store already. The export is the
        # only record of what was written after the copy: hand it to the re-import,
        # which skips every memory the store still has (L1-03).
        outcome = RestoreOutcome(
            status="snapshot_unusable", point_id=point_id, delta_dir=str(final),
            reimport_pending=True,
            message="The copy this restore was using can no longer be read, so the "
                    "restore stopped. An earlier attempt may already have written it; the "
                    "memories written after the copy are added back now, skipping any "
                    "your store still has.")
    else:
        outcome = RestoreOutcome(
            status="snapshot_unusable", point_id=point_id,
            message="The copy cannot be used: it is damaged or was removed. "
                    f"{_UNCHANGED}.")
    _write_outcome(data_root, outcome)
    return outcome


def _too_new(data_root: Path, intent: dict[str, Any], point: Any) -> RestoreOutcome:
    """A copy stamped above what this build reads is never restored by it (L1-04)."""
    reason = point.not_restorable_reason or "This copy was made by a newer version."
    if intent.get("stage") == "writing":
        # Started by a newer build: only that build can finish it. Kept, not dropped.
        return _refused(data_root, point.point_id,
                        "the copy was made by a newer version of SuperLocalMemory, which "
                        "is needed to finish it", stage="writing")
    _retire_intent(data_root, "refused")
    outcome = RestoreOutcome(status="refused", point_id=point.point_id, message=reason)
    _write_outcome(data_root, outcome)
    return outcome


def _perform_locked(data_root: Path, memory_db: Path, learning_db: Path,
                    intent: dict[str, Any], config: Any | None) -> RestoreOutcome:
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage.backup import SnapshotUnusableError

    point_id = str(intent["point_id"])
    try:
        point = ur.find_restore_point(point_id, data_root)
        if not point.restorable:
            return _too_new(data_root, intent, point)
        ur.verify_point(point)
    except (ur.RestoreRefusedError, SnapshotUnusableError) as exc:
        logger.warning("[SLM] Restore point %s cannot be used: %s", point_id, exc)
        return _unusable(data_root, intent, point_id)

    was_writing = intent.get("stage") == "writing"
    progress: dict[str, bool] = {"uncertain": False}
    try:
        written = _stage_and_write(data_root, memory_db, learning_db, intent, point, progress)
    except Exception:  # noqa: BLE001 - see the messages: each says what the store holds
        logger.exception("[SLM] Restore of %s did not complete", point_id)
        _clear_staging(data_root / SAFETY_DIR / _STAGING, remove_dir=True)
        if not was_writing and not progress["uncertain"]:
            # A first attempt that failed changed nothing: set it aside rather than
            # fail the same way at every start. A request already part-way through
            # stays: an earlier attempt may have written the store, and its export
            # is the only record of what must still be added back.
            _retire_intent(data_root, "failed")
            message = f"The restore could not run. {_UNCHANGED}. The details are in the log."
        else:
            message = (f"The restore could not finish this time. {_MAYBE_WRITTEN}. It is "
                       "tried again at the next start; the details are in the log.")
        outcome = RestoreOutcome(status="failed", point_id=point_id, message=message)
        _write_outcome(data_root, outcome)
        return outcome
    if isinstance(written, RestoreOutcome):
        return written
    return _finish(data_root, memory_db, learning_db, intent, point, written, config)


def _request_time_delta(intent: dict[str, Any]) -> Path | None:
    """The export taken when the restore was requested, if it was read from a
    readable store. Used when the store can no longer be read at the restart."""
    from superlocalmemory.storage._restore_delta import delta_from_unreadable_store

    raw = intent.get("delta_dir")
    if not raw or not Path(raw).is_dir() or not (Path(raw) / "meta.json").exists():
        return None
    return None if delta_from_unreadable_store(Path(raw)) else Path(raw)


def _final_delta(data_root: Path, memory_db: Path, learning_db: Path,
                 intent: dict[str, Any], point: Any) -> tuple[Path, str] | RestoreOutcome:
    """(export to use, where it came from: live | request | none).

    A restore interrupted after this point re-uses the export it recorded: the
    store it would compute a new one from may already be the restored one.
    """
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage._restore_damaged import is_unreadable_error
    from superlocalmemory.storage._restore_delta import (
        export_delta,
        export_empty_delta,
        open_compare,
    )

    recorded = Path(intent["final_delta_dir"]) if intent.get("final_delta_dir") else None
    if intent.get("stage") == "writing" and recorded is not None and recorded.is_dir():
        return recorded, str(intent.get("delta_source") or "live")
    needed = ur.disk_needed(point, memory_db, learning_db if learning_db.exists() else None)
    free = ur.free_bytes(data_root)
    if free < needed:
        return _refused(data_root, point.point_id, f"not enough free disk space (needs "
                        f"{needed:,} bytes, {free:,} free)", stage=intent.get("stage"))
    delta, source = data_root / DELTA_DIR / f"{_stamp()}-final", "live"
    try:
        with closing(open_compare(memory_db, point.memory_snapshot)) as conn:
            export_delta(conn, delta)
    except sqlite3.Error as exc:
        if not is_unreadable_error(exc):
            raise
        # L1-02: a store SQLite cannot read must not block the restore. Nothing
        # can be exported from it now; the request-time export, if any, is used.
        logger.warning("[SLM] The live memory store cannot be read (%s: %s); restoring "
                       "the copy in full", type(exc).__name__, exc)
        ur._remove_delta_dir(data_root, delta)
        earlier = _request_time_delta(intent)
        if earlier is not None:
            delta, source = earlier, "request"
        else:
            export_empty_delta(delta)
            source = "none"
    intent.update({"stage": "writing", "final_delta_dir": str(delta), "delta_source": source})
    write_json_atomic(data_root / INTENT_NAME, intent)
    return delta, source


def _write_store(staged: Path, live: Path, progress: dict[str, bool],
                 *, readable: bool) -> tuple[Path, bool]:
    """Write ``staged`` into ``live``. (safety copy, True when ``live`` was damaged)."""
    from superlocalmemory.storage._restore_damaged import live_store_problem, write_over_damaged
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot

    if readable:
        try:
            return restore_pre_migration_snapshot(staged, live), False
        except sqlite3.DatabaseError as exc:
            problem = live_store_problem(live)
            if problem is None:
                raise
            logger.warning("[SLM] %s cannot be read (%s; %s); replacing it with the copy",
                           live.name, exc, problem)
    progress["uncertain"] = True
    return write_over_damaged(staged, live), True


def _stage_and_write(data_root: Path, memory_db: Path, learning_db: Path,
                     intent: dict[str, Any], point: Any,
                     progress: dict[str, bool]) -> dict[str, Any] | RestoreOutcome:
    """Export, stage, write the memory store. Anything raised before the write
    leaves the live store as it was: every step before it touches only new files,
    and the write is one SQLite transaction that rolls back if it does not
    finish. (Over a damaged store the write is a file replacement, flagged in
    ``progress`` so a failure there never claims the store is unchanged.)"""
    from superlocalmemory.storage._restore_delta import (
        apply_delta_to_staging,
        erased_profile_ids,
    )
    from superlocalmemory.storage._restore_learning import profiles_in, replay_erasures

    found = _final_delta(data_root, memory_db, learning_db, intent, point)
    if isinstance(found, RestoreOutcome):
        return found
    delta, source = found
    staging_dir = data_root / SAFETY_DIR / _STAGING
    _clear_staging(staging_dir)
    staged_memory = staging_dir / "memory.db"
    _copy_quiescent(point.memory_snapshot, staged_memory)
    applied = apply_delta_to_staging(staged_memory, delta, data_root=data_root)
    _quiesce(staged_memory)
    staged_learning = None
    if point.learning_snapshot is not None and learning_db.exists():
        staged_learning = staging_dir / "learning.db"
        _copy_quiescent(point.learning_snapshot, staged_learning)
        # L1-05: an erased profile stays erased in learning.db too.
        applied["learning_rows_erased"] = replay_erasures(
            staged_learning, erased_profile_ids(delta, data_root, profiles_in(staged_learning)))
        _quiesce(staged_learning)
    safety, damaged = _write_store(staged_memory, memory_db, progress,
                                   readable=source != "none" and source != "request")
    return {"delta": delta, "applied": applied, "safety": [safety],
            "staged_learning": staged_learning, "source": source, "damaged": damaged}


def _finish_message(point: Any, written: dict[str, Any], learning_note: str) -> str:
    message = (f"Your memories were restored to the copy from "
               f"{point.created_at or point.point_id}. A copy of what was replaced is kept "
               "in the pre-restore folder of your data directory.")
    if written["source"] == "none":
        message += (" Your previous store could not be read, so it was replaced entirely: "
                    "memories written after the copy could not be carried over. The "
                    "damaged store is kept in the same folder.")
    elif written["source"] == "request":
        message += (" Your previous store could not be read at the restart, so it was "
                    "replaced entirely: what it held when you asked for the restore is "
                    "added back; anything written after that could not be carried over. "
                    "The damaged store is kept in the same folder.")
    elif written["damaged"]:
        message += " Your previous store was damaged; it is kept in the same folder."
    return message + learning_note


def _write_learning(data_root: Path, memory_db: Path, learning_db: Path, point_id: str,
                    written: dict[str, Any], progress: dict[str, bool]) -> str | RestoreOutcome:
    """Write the staged learning copy. '' or a note on success; an outcome when the
    memory store had to be put back."""
    from superlocalmemory.storage.backup import restore_pre_migration_snapshot

    staging_dir = data_root / SAFETY_DIR / _STAGING
    try:
        copy, _damaged = _write_store(written["staged_learning"], learning_db, progress,
                                      readable=True)
        written["safety"].append(copy)
        return ""
    except Exception:  # noqa: BLE001 - decided below
        logger.exception("[SLM] learning.db could not be restored")
    if written["damaged"]:
        # The memory store was damaged: there is nothing sound to put back.
        return (" Your learning data could not be restored and was left as it was; "
                "the details are in the log.")
    restore_pre_migration_snapshot(written["safety"][0], memory_db)
    _clear_staging(staging_dir, remove_dir=True)
    _retire_intent(data_root, "failed")
    outcome = RestoreOutcome(
        status="failed", point_id=point_id,
        safety_copies=[str(p) for p in written["safety"]],
        message="The restore could not finish; your memories were put back exactly as "
                "they were before it started. The details are in the log.")
    _write_outcome(data_root, outcome)
    return outcome


def _finish(data_root: Path, memory_db: Path, learning_db: Path, intent: dict[str, Any],
            point: Any, written: dict[str, Any], config: Any | None) -> RestoreOutcome:
    from superlocalmemory.storage import upgrade_restore as ur
    from superlocalmemory.storage._restore_delta import load_kind_edits, load_memories

    staging_dir = data_root / SAFETY_DIR / _STAGING
    delta, safety = written["delta"], written["safety"]
    learning_note = ""
    if written["staged_learning"] is not None:
        result = _write_learning(data_root, memory_db, learning_db, point.point_id,
                                 written, {"uncertain": False})
        if isinstance(result, RestoreOutcome):
            return result
        learning_note = result
    _clear_staging(staging_dir, remove_dir=True)
    projections = (settle_projections_after_restore(config, data_root)
                   if config is not None else "rebuild_needed")
    pending = bool(intent.get("reimport", True)) and bool(
        load_memories(delta) or load_kind_edits(delta))
    outcome = RestoreOutcome(
        status="restored", point_id=point.point_id, safety_copies=[str(p) for p in safety],
        delta_dir=str(delta), reimport_pending=pending, applied=written["applied"],
        projections=projections, message=_finish_message(point, written, learning_note))
    _write_outcome(data_root, outcome, reimport=None)
    _retire_intent(data_root, "done")
    if intent.get("delta_dir") and Path(intent["delta_dir"]) != delta:
        ur._remove_delta_dir(data_root, intent["delta_dir"])   # superseded by the final export
    logger.info("[SLM] Restored %s from %s (applied %s)", memory_db.name, point.point_id,
                written["applied"])
    _settle_retention(data_root, delta if not pending else None)
    return outcome


def _settle_retention(data_root: Path, settled_delta: Path | None) -> None:
    """Bounded retention of deltas and safety copies (L1-12). Never fails a restore."""
    try:
        from superlocalmemory.storage._restore_retention import (
            discard_delta,
            prune_restore_artifacts,
        )

        if settled_delta is not None:
            discard_delta(data_root, settled_delta)
        prune_restore_artifacts(data_root)
    except Exception as exc:  # noqa: BLE001 - housekeeping only
        logger.warning("[SLM] Restore housekeeping skipped: %s", exc)


def settle_projections_after_restore(config: Any, data_root: Path) -> str:
    """Serve search from SQLite until the graph/vector copies are rebuilt.

    The graph and vector projections were built from the store that was just
    replaced. Verifying them needs a stage built from the RESTORED store
    (``ScaleEngineManager.verify(stage_id)`` checks a staged projection against
    the store it was prepared from), so the safe answer is to stop serving the
    stale ones: the state drops to ``local_core`` (SQLite serves every query),
    stages prepared before the restore are retired so they are not resumed, and
    the existing automatic promotion prepares, verifies and promotes a fresh
    projection of the restored store at this or the next start.
    """
    state = str(getattr(config, "scale_engine_state", "") or "local_core").lower()
    if state == "local_core":
        return "serving_from_sqlite"
    try:
        from superlocalmemory.core.scale_engine import ScaleEngineManager

        manager = ScaleEngineManager(config)
        # Disk-only listing: retiring must not depend on loading the vector
        # extension, or stale copies stay "verified" on machines without it.
        for stage in manager.stage_manifests():
            if str(stage.get("state")) in {"prepared", "verified"} and stage.get("stage_id"):
                manager._retire_superseded_stage(str(stage["stage_id"]))
    except Exception as exc:  # noqa: BLE001 - demotion below still makes it safe
        logger.warning("[SLM] Could not retire projection stages after restore: %s", exc)
    config.scale_engine_state = "local_core"
    config.graph_backend = "auto"
    config.vector_backend = "auto"
    save = getattr(config, "save", None)
    if callable(save):
        save()
    return "rebuild_scheduled"


__all__ = ["perform_pending_restore", "settle_projections_after_restore"]
